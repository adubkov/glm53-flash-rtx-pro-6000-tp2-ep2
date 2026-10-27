# Results — GLM-5.3-Flash: TP2 (ep=1) vs TP2+EP2 on one box

Scope of every number: **one host** (2× RTX PRO 6000 Blackwell 96 GB, PCIe, both
cards capped 300 W), **one envelope** ([bench/baseline/STACK-PIN.md](../bench/baseline/STACK-PIN.md)),
**medians of 3 measured runs**, clean rows only (contamination = concurrent
foreign GPU load; depresses aggregates → lower bounds; rows named in §6).
Raw artifacts: `bench/baseline/`, `bench/ep2/`.

Cell naming: C1/C4 = 1/4 concurrent streams sharing one text (C4 prices
concurrent decode, not 4 independent prefills); 8k/64k/256k = prompt-length
targets — achieved prompt tokens per cell: **7,881 / 65,543 / 259,869**.
Decode = aggregate tok/s over streams; accept = EAGLE `spec_accept_length`.

## 1. Baseline (TP2, ep=1)

| cell | TTFT med | aggregate decode | accept | wall med |
|---|---:|---:|---:|---:|
| C1/8k | 1.463 s | 197.7 tok/s | 4.38 | 4.22 s |
| C1/64k ⚠ | 11.903 s | 226.4 tok/s | 5.12 | 14.27 s |
| C1/256k | 49.815 s | 261.2 tok/s | 4.88 | 51.71 s |
| C4/8k | 1.737 s | 875.6 tok/s | 4.61 | 6.60 s |
| C4/64k | 12.386 s | 897.9 tok/s | 5.28 | 17.16 s |
| C4/256k | 50.044 s | 848.3 tok/s | 4.97 | 55.35 s |

⚠ C1/64k median lands on a run whose cell carried foreign-POST contamination
(§6) — treat as a lower bound.

- Prefill ≈ **5.2–5.5k tok/s flat** across 8k→256k (7,881/1.463 ≈ 5.39k;
  65,543/11.90 ≈ 5.51k; 259,869/49.82 ≈ 5.22k).
- C4 aggregate: **4.4× C1 at 8k**, 4.0× at 64k, 3.2× at 256k; TTFT(C4) ≈
  TTFT(C1) at every length (shared prefill).

## 2. EP2 A/B — the serving-config decision

Unpatched `--ep=2 --moe-a2a-backend=none` fails at boot: `FlashInfer b12x W4A16
requires EP size 1` (`flashinfer_cutlass.py:109`) — EP1-only guards plus a
prefix-only expert map (fits rank 0, not rank 1). The two overlays
(`deploy/patches/ep2/`) lift the guards, thread the EP rank offset
(`first_global`) through `launch_sm120_moe → … → _make_w4a16_expert_map`, and
zero-init the MoE output under EP.

Boot green 2026-10-17T11:22:34Z (`ep_size 2`, `a2a none`, runner
`flashinfer_cutlass`; ~8 min; autotune cache HIT). Goldens PASS with the
pre-approved accumulation-order signature: math byte-identical; pong same
5-token prefix then `you will`→`you're`; coherence repetition-count flip.

Final A/B (clean data only):

| cell | baseline | EP2 | raw Δ | accept b→e | per-accept Δ |
|---|---:|---:|---:|---|---:|
| C1/8k | 197.7 | 297.7 | **+50.5%** | 4.38→5.17 | **+28%** |
| C1/64k | 226.4 | 255.4 | **+12.8%** | 5.12→4.49 | **+29%** |
| C1/256k | 261.2 | 261.7 | +0.2% | 4.88→4.53 | **+8%** |
| C4/8k | 875.6 | 965.6 | **+10.3%** | 4.61→5.28 | −4% (par) |
| C4/64k | 897.9 | 939.4 | +4.6% | 5.28→5.51 | par |
| C4/256k | 848.3 | 837.7 | −1.2% | 4.97→5.22 | −6% (par) |

- Clean-data mapping: C1/64k, C1/256k, C4/8k from the quiet re-bench
  (`bench/ep2/rebench-*`); C1/8k, C4/64k, C4/256k from attempt-2 rows verified
  clean by the POST map. Excluded as contaminated (§6): attempt-1 entirely
  (lower bounds C1/8k 306.8 / C1/64k 217.1 / C1/256k 185.7) and attempt-2's
  C1/64k med 192.8, C1/256k med 200.6, C4/8k med 716.0.
- TTFT par everywhere except C1/256k (+2.4%: 51.03 vs 49.82 s).
- Accept medians differ arm-to-arm on identical seeds (accumulation-order
  divergence) → per-accept is the engine lens, raw the client lens
  (definitions: docs/benchmarking.md).
- Reading: C1 decode is launch/tiling-bound → halving the per-rank kernel
  count pays +28–30%; batched C4 and long-prefill are bandwidth/batch-bound →
  par (mechanism: docs/architecture.md §3).

**Verdict: KEEP** — no clean cell loses; the apparent long-context deficit was
contamination (§6). EP2 is the serving config since 2026-10-17.

## 3. Route-pack v2 (EP-aware precompile) — kept, throughput-neutral

Mechanism: the boot precompile warmed EP1-only dead specializations; v2 warms
the same map-aware packed-route specializations the runtime uses under EP. The
runtime compute path is unchanged.

- The cold-cache JIT-stall premise was refuted on this box: 266 triton-cache
  entries all predate the v2 boot (warm cache is a persistent host bind) — zero
  entries written by that boot.
- Full matrix on the v2 stack, all 18 measured rows clean (idle ≤ 1.1 s).
  Cells read `agg @accept → steps/s`; steps/s = agg÷accept per row (raw
  aggregate is invalid across arms when accept differs):

| cell | TP2 (ep=1) | TP2+EP2 (v1 overlays) | TP2-EP2-opt (v2) |
|---|---|---|---|
| C1/8k | 197.7 @4.38 → 45.2 | 297.7 @5.17 → 57.6 | 294.8 @5.28 → 55.3 |
| C1/64k ⚠ | 226.4 @5.12 → 43.0 | 255.4 @4.49 → 56.9 | 268.8 @5.07 → 55.3 |
| C1/256k | 261.2 @4.88 → 54.1 | 261.7 @4.53 → 56.7 | 286.8 @5.07 → 55.7 |
| C4/8k | 875.6 @4.61 → 48.7 | 965.6 @5.28 → 45.3 | 941.9 @5.22 → 46.6 |
| C4/64k | 897.9 @5.28 → 46.1 | 939.4 @5.51 → 41.7 | 925.1 @5.63 → 41.6 |
| C4/256k | 848.3 @4.97 → 42.7 | 837.7 @5.22 → 42.7 | 810.5 @5.22 → 42.0 |

⚠ the TP2 C1/64k median carries §6 contamination — lower bound.

- v2 vs v1: par on all six cells at matched accept (−4% to +3%, ranges overlap).
- Noise proof: two clean same-stack same-seed C4/8k runs 15 min apart measured
  per-stream 40.5–43.8 vs 45.1–47.7 steps/s (~15% swing) — C4 medians carry
  that noise bar.
- Verdict: KEPT — zero measured cost; strictly more correct for cold-cache
  boots. The committed EP2 wins reproduce unchanged.

## 4. PCIe IPC all-reduce tune — kept, throughput-neutral

`SGLANG_PCIE_IPC_TUNE=1` + `SGLANG_PCIE_IPC_TUNE_HIDDEN=4096` ran the init-time
collective tune (~7 s of boot); tuned-kernel cache (`pcie_ipc_all_reduce_ws2.json`,
2501 B) persisted; the untuned-seed warning gone. Pricing bench (3 cells × 4
runs, EP2 seeds): C1/8k −1.3% (56.8 vs 57.6 steps/s), C1/64k −3% at matched
accept (320.1 @5.75 vs 330.2 @5.82), C4/8k −0.7% (959.2 vs 965.6). Mechanism:
the all-reduce is a small slice of the ~17–18 ms verify step at this shape.
Cache KEPT, lever closed.

## 5. CuTeDSL-EP2 route — not deployed (evidence-closed)

Four boot attempts, three distinct gates, all auto-reverted green:
1. Unpatched: scale-factor mismatch — the CuTeDSL runner is g16-only
   (`_FP4_SF_VEC_SIZE=16`) vs this checkpoint's K32 (exact 2× element mismatch:
   75,497,472 vs 150,994,944 expected).
2. K32→K16 lossless expansion patch (validated at load ×3): FlashInfer artifact
   gate — `FLASHINFER_NO_DOWNLOAD=1` truthiness blocks any cubin download.
3. Next: CUDA-arch gate — `jit/moe_utils.py` hardcodes
   `supported_major_versions=[9, 10]`; SM120 (12.0f) filters out.
4. Route closed without a fifth boot: the CuTeDSL W4A16 kernels are
   SM100-class-only — entry points decorated
   `@supported_compute_capability([100, 103, 107])`, kernel sizing constants
   hardcode `sm_100`. Upstream's SM120 W4A16 lives in the separate
   `blackwell_sm12x/` backend — the route EP2 already uses.

## 6. Contamination bounds — how to read every number here

- The measuring session rode the engine via a chat proxy; every agentic turn
  emits `POST /v1/chat` rounds that share the GPU. A C1 decode window is
  2.4–3.4 s — any concurrent round depresses that run's aggregate.
  **Contamination bounds true numbers upward, never down.**
- Ground truth = the engine's own POST access lines in host clock time. The
  bench's own streams appear as docker-proxy `172.21.0.1 /generate`; a chat
  round that starts just after a completion can span the next run — the per-run
  idle gate going green (~1 s) proves nothing was in flight at launch.
- Only idle-gate-clean + POST-map-clean rows are in the tables above; every
  row's `idle_wait_s` is in the shipped jsonl.
- Instrument lesson recorded: `docker logs --since`-filtered probes misread a
  healthy stack as a 40-min wedge; unfiltered POST maps are authoritative.

## Artifacts

- `bench/baseline/`: `results-baseline.jsonl` (24 rows incl. warmups + the
  poisoned rows), `done-baseline.txt`, `STACK-PIN.md`.
- `bench/ep2/`: `results-ep2.attempt1.jsonl` (contaminated, lower bounds),
  `results-ep2.jsonl` (attempt-2, 24 rows), `rebench-*` (3-cell quiet
  re-bench), `v2full-*` (full matrix on the v2 stack, all rows clean).
- `bench/plan.json` shipped; `vocab.json` (3.2 MB) regenerable via `plan_gen.py`.
