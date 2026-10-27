"""GLM-5.3 A/B bench runner: C1/C4 x 8k/64k/250k, native /generate, idle-gated.

Per run: wait for engine idle (metrics gauges), send, stream SSE, record
client timings + engine meta_info fields. Unique shuffle per run, fixed
multiset per cell -> identical token counts across arms, no radix hits.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["baseline", "ep2"])
ap.add_argument("--dir", required=True)
ap.add_argument("--base-url", default="http://localhost:8062")
A = ap.parse_args()

# Per-launch nonce: every launch shuffles fresh text so no radix/hicache entry
# from any earlier launch (same arm relaunched) can be hit. Logged for repro.
NONCE = int(time.time()) & 0xFFFFFFF

KEY = os.environ.get("SGLANG_API_KEY", "")
assert KEY, "set SGLANG_API_KEY to the engine's --api-key token"

plan = json.load(open(f"{A.dir}/plan.json"))
vocab = json.load(open(f"{A.dir}/vocab.json"))
LOG = open(f"{A.dir}/{A.arm}.log", "a", buffering=1)
RES = open(f"{A.dir}/results-{A.arm}.jsonl", "a", buffering=1)


def log(msg):
    LOG.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


HDR = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}

METRICS_NAME = None  # which idle instrument


def idle_check():
    """Return (idle: bool, instrument: str)."""
    try:
        req = urllib.request.Request(A.base_url + "/metrics", headers=HDR)
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
        running = queue = 0.0
        for line in body.splitlines():
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
            if not m:
                continue
            name, val = m.group(1), float(m.group(3))
            if "running_reqs" in name:
                running += val
            elif "queue" in name and "req" in name:
                queue += val
        return running == 0 and queue == 0, f"metrics(run={running:g},q={queue:g})"
    except Exception as e:
        # fallback: /get_load (deprecated but present; logs a warning per call)
        req = urllib.request.Request(A.base_url + "/get_load", headers=HDR)
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode())
        return d.get("num_reqs", 1) == 0, f"get_load(num_reqs={d.get('num_reqs')}) err={type(e).__name__}"


def wait_idle(timeout_s=300):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        ok, inst = idle_check()
        if ok:
            return time.perf_counter() - t0, inst
        time.sleep(1.0)
    raise TimeoutError(f"engine not idle after {timeout_s}s (last: {inst})")


def one_stream(text, out_idx, outs, barrier):
    t_send = time.time()
    pc_send = time.perf_counter()
    t_first = t_last = None
    meta = {}
    try:
        body = json.dumps({
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": 512, "ignore_eos": True},
            "stream": True,
        }).encode()
        req = urllib.request.Request(A.base_url + "/generate", data=body, headers=HDR)
        with urllib.request.urlopen(req, timeout=1800) as resp:
            barrier.wait()
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if t_first is None:
                    t_first = time.perf_counter()
                t_last = time.perf_counter()
                obj = json.loads(payload)
                mi = obj.get("meta_info") or {}
                meta.update(mi)
        outs[out_idx] = {
            "t_send": t_send, "pc_send": pc_send, "t_first": t_first, "t_last": t_last,
            "meta": meta, "error": None,
        }
    except BaseException as e:
        outs[out_idx] = {
            "t_send": t_send, "pc_send": pc_send, "t_first": t_first, "t_last": t_last,
            "meta": meta, "error": f"{type(e).__name__}: {e}",
        }


def build_text(cell, seed):
    base = vocab[: cell["n_words"]]
    rnd = random.Random(seed + NONCE)
    rnd.shuffle(base)
    return " ".join(base)


def run_cell(cell):
    recs = []
    for run in cell["runs"]:
        if run["arm"] != A.arm:
            continue
        text = build_text(cell, run["seed"])
        idle_wait, inst = wait_idle()
        time.sleep(2.0)
        t0 = time.perf_counter()
        n = cell["c"]
        outs = [None] * n
        if n == 1:
            b = threading.Barrier(1)
            one_stream(text, 0, outs, b)
        else:
            b = threading.Barrier(n)
            ths = [threading.Thread(target=one_stream, args=(text, i, outs, b)) for i in range(n)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
        wall = time.perf_counter() - t0
        errs = [o["error"] for o in outs if o["error"]]
        streams = []
        for o in outs:
            if o is None or o["error"]:
                streams.append(None)
                continue
            m = o["meta"]
            ctok = m.get("completion_tokens")
            win = (o["t_last"] - o["t_first"]) if (o["t_first"] and o["t_last"]) else None
            rate = (ctok - 1) / win if (ctok and win and win > 0) else None
            streams.append({
                "completion_tokens": ctok,
                "decode_window_s": round(win, 4) if win else None,
                "decode_tok_s": round(rate, 2) if rate else None,
                "spec_accept_length": m.get("spec_accept_length"),
                "spec_accept_rate": m.get("spec_accept_rate"),
            })
        # normalize: ttft per stream relative to that stream's own send
        engine = outs[0]["meta"] if outs[0] else {}
        rec = {
            "arm": A.arm, "c": cell["c"], "ctx": cell["ctx"], "run_idx": run["run_idx"],
            "kind": "warmup" if run["run_idx"] == 0 else "measured",
            "ts": min(o["t_send"] for o in outs if o),
            "wall_s": round(wall, 4),
            "idle_wait_s": round(idle_wait, 2), "idle_inst": inst,
            "streams": streams,
            "aggregate_decode_tok_s": round(sum(s["decode_tok_s"] for s in streams if s and s["decode_tok_s"]), 2),
            "engine": {k: engine.get(k) for k in (
                "prompt_tokens", "completion_tokens", "cached_tokens", "queue_time",
                "forward_entry_time", "prefill_finished_time", "decode_throughput",
                "spec_accept_length", "spec_accept_rate", "e2e_latency", "finish_reason")},
            "errors": errs,
        }
        # per-stream ttft: perf_counter on both sides (epoch t_send is record metadata only)
        for i, s in enumerate(rec["streams"]):
            if s and outs[i] and outs[i]["t_first"]:
                s["ttft_s"] = round(outs[i]["t_first"] - outs[i]["pc_send"], 4)
        recs.append(rec)
        RES.write(json.dumps(rec) + "\n")
        s0 = rec["streams"][0] or {}
        log(f"R c={cell['c']}/{cell['ctx']} run={run['run_idx']} kind={rec['kind']} "
            f"ttft={s0.get('ttft_s')}s wall={rec['wall_s']}s agg={rec['aggregate_decode_tok_s']}t/s "
            f"ptok={rec['engine'].get('prompt_tokens')} cached={rec['engine'].get('cached_tokens')} "
            f"accept={s0.get('spec_accept_length')} idlewait={rec['idle_wait_s']}s errs={len(errs)}")
    return recs


log(f"=== arm={A.arm} base={A.base_url} keylen={len(KEY)} nonce={NONCE} cells={[(c['c'], c['ctx']) for c in plan['cells']]}")
try:
    req = urllib.request.Request(A.base_url + "/flush_cache", data=b"", headers=HDR)
    with urllib.request.urlopen(req, timeout=30) as r:
        log(f"flush_cache -> {r.status}")
except BaseException as e:
    log(f"flush_cache unavailable ({type(e).__name__}: {e}) — launch nonce still guarantees fresh texts")
all_ok = True
summary = []
try:
    for cell in plan["cells"]:
        recs = run_cell(cell)
        meas = [r for r in recs if r["kind"] == "measured" and not r["errors"]]
        if meas:
            ttfts = [r["streams"][0]["ttft_s"] for r in meas]
            aggs = [r["aggregate_decode_tok_s"] for r in meas]
            walls = [r["wall_s"] for r in meas]
            accs = [r["streams"][0].get("spec_accept_length") for r in meas if r["streams"][0].get("spec_accept_length")]
            sm = {"c": cell["c"], "ctx": cell["ctx"],
                  "ttft_s_med": sorted(ttfts)[len(ttfts) // 2], "ttft_s_min": min(ttfts), "ttft_s_max": max(ttfts),
                  "agg_decode_med": sorted(aggs)[len(aggs) // 2], "agg_decode_min": min(aggs), "agg_decode_max": max(aggs),
                  "wall_s_med": sorted(walls)[len(walls) // 2],
                  "spec_accept_med": sorted(accs)[len(accs) // 2] if accs else None}
            summary.append(sm)
            log(f"S c={cell['c']}/{cell['ctx']} ttft_med={sm['ttft_s_med']} agg_med={sm['agg_decode_med']} "
                f"wall_med={sm['wall_s_med']} accept_med={sm['spec_accept_med']}")
except BaseException:
    all_ok = False
    log("FATAL: " + traceback.format_exc())

with open(f"{A.dir}/{'done' if all_ok else 'failed'}-{A.arm}.txt", "w") as f:
    json.dump({"arm": A.arm, "ok": all_ok, "summary": summary}, f, indent=1)
log(f"=== arm={A.arm} {'DONE' if all_ok else 'FAILED'}")
print("done" if all_ok else "failed")
