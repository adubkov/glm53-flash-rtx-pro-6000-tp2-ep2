# Benchmarking

Method behind [docs/RESULTS.md](RESULTS.md); how to reproduce it.

## Usage

```bash
# engine up first (plan_gen probes it for token calibration)
export SGLANG_API_KEY=<the engine's --api-key token>     # required
export SGLANG_BASE_URL=http://localhost:8062             # optional, this is the default

python3 bench/plan_gen.py bench                        # vocab.json (3.2 MB, regenerable) + plan.json
python3 bench/harness.py --arm baseline --dir bench    # TP2 ep1 seeds
python3 bench/harness.py --arm ep2 --dir bench         # serving config seeds (offset 7777777)
```

- plan_gen: 270k synthetic 8-char words (seed 0, sorted) → per-cell word counts
  calibrated to prompt-token targets via a 4,096-word probe; cells C1/C4 ×
  8k/64k/256k; per arm+cell: 1 warmup + 3 measured runs, seeds offset per arm.
- harness: per run — idle gate (engine `/metrics` gauges running+queue == 0),
  streaming `POST /generate`, records client timings + final-chunk `meta_info`;
  appends to `<arm>.log` and `results-<arm>.jsonl` (one row per run incl.
  warmups; per-row `idle_wait_s` is the purity instrument).
- Arms differ only by seed offset → identical word multiset per cell, so no
  radix/hicache reuse is possible; `flush_cache` at arm start.

## Method

- Metrics from the final chunk's `meta_info` (prompt_tokens, cached_tokens,
  queue_time, forward_entry_time, prefill_finished_time, spec_accept_length) —
  not SSE wall-clock. TTFT = client send → first token. Decode =
  (completion−1)/(last−first) per stream; aggregate = Σ streams.
- C4 = 4 concurrent streams sharing one text → the radix shares the prefill;
  it prices concurrent decode, not 4 independent prefills (4× 256k prefills
  ≈ 1.04M tokens > the 649,984 pool and would queue; with sharing ~260k is
  resident, which is why C4/256k ran without queueing).
- Prompt token counts per cell are fixed across runs and arms (same multiset);
  achieved values are recorded per row (`engine.prompt_tokens`).

## Reading rules

- Medians of the 3 measured runs; run 0 is warmup.
- Contamination (any concurrent decode) depresses aggregate → contaminated
  numbers are **lower bounds**; see docs/RESULTS.md §6 for the bounds
  discipline and the POST-access-log audit.
- Spec decode: aggregate = accept × steps/s. Arms draw different accept lengths
  on identical seeds (accumulation-order divergence), so cross-arm compare
  per-row steps/s (agg÷accept); raw aggregate is the client lens.
- Same-config noise: ~15% swing observed between two clean C4/8k runs 15 min
  apart (docs/RESULTS.md §3).
