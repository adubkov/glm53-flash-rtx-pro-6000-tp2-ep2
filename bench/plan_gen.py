"""Generate the GLM A/B bench plan: fixed word multiset per cell, per-run shuffles.

Prompt design: one 8192-word base list (seed 0); each run uses K concatenated,
shuffled copies -> identical token count across runs AND arms, no radix hits.
"""
import json
import os
import random
import sys
import urllib.request

DIR = sys.argv[1] if len(sys.argv) > 1 else "."
BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://localhost:8062")
KEY = os.environ.get("SGLANG_API_KEY", "")
assert KEY, "set SGLANG_API_KEY to the engine's --api-key token"

rng = random.Random(0)
NVOCAB = 270000
words = set()
while len(words) < NVOCAB:
    words.add("".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8)))
vocab = sorted(words)


def generate(text, max_new_tokens=1, stream=False):
    body = json.dumps({
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True},
        "stream": stream,
    }).encode()
    req = urllib.request.Request(BASE_URL + "/generate", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = json.loads(r.read().decode())
    return payload


CAL_WORDS = 4096
probe = generate(" ".join(vocab[:CAL_WORDS]), max_new_tokens=1)
meta = probe.get("meta_info", {})
P_cal = meta.get("prompt_tokens")
if not P_cal:
    print("FATAL: no prompt_tokens in meta_info:", json.dumps(meta)[:400])
    sys.exit(1)
tpw = P_cal / CAL_WORDS
print(f"calibration: {CAL_WORDS} words -> prompt_tokens={P_cal} tpw={tpw:.4f}")

CELLS = [(1, 8192), (1, 65536), (1, 256000), (4, 8192), (4, 65536), (4, 256000)]
ARMS = {"baseline": 0, "ep2": 7777777}
cells = []
for ci, (c, ctx) in enumerate(CELLS):
    n_words = round(ctx / tpw)
    expected = round(n_words * tpw)
    runs = []
    for arm, off in ARMS.items():
        for run_idx in range(4):  # 0=warmup, 1-3 measured
            runs.append({"arm": arm, "run_idx": run_idx, "seed": off + ci * 9973 + run_idx * 104729})
    cells.append({"c": c, "ctx": ctx, "n_words": n_words, "expected_tokens": expected, "runs": runs})
    print(f"cell C{c}/{ctx}: n_words={n_words} expected_tokens~{expected} (off {100*(expected-ctx)/ctx:+.2f}%)")

with open(f"{DIR}/vocab.json", "w") as f:
    json.dump(vocab, f)
with open(f"{DIR}/plan.json", "w") as f:
    json.dump({"tpw": tpw, "cells": cells}, f)
print("plan.json + vocab.json written")
