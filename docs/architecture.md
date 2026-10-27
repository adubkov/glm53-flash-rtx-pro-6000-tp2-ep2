# Architecture

What runs and why. Numbers in [docs/RESULTS.md](RESULTS.md); pins in
[bench/baseline/STACK-PIN.md](../bench/baseline/STACK-PIN.md).

## 1. Lineage

- Checkpoint: `zai-org/GLM-5.3-Flash-BF16` → community requant (pinned in
  docs/deploy.md §2); never shipped.
- Runtime: pinned public base image; the measured serving build is a local
  stage2 of it (+2 patch layers) — provenance, recipe and licensing in
  docs/deploy.md §3 / LICENSING.md.
- EP2: two full-file bind overlays over image-baked modules (sglang MoE runner +
  flashinfer sm12x dispatch). Same content filed upstream as
  sgl-project/sglang#45298 and flashinfer-ai/flashinfer#6824 — both open and
  unmerged as of 2026-10-20; the overlays are the serving path until merge.

## 2. Quant contract

- ModelOpt MIXED_PRECISION: routed experts **W4A16_NVFP4** (E2M1 weights + per-32
  E4M3 scales, K32); attention + shared projections **FP8_PB_WO**.
- KV cache fp8_e4m3; lm_head fp8 via `SGLANG_LM_HEAD_FP8=1`.
- MoE runner `flashinfer_cutlass` → b12x SM120 path (`launch_sm120_moe`);
  K32 scales native — no scale expansion overlay.

## 3. TP2 + EP2 geometry

- TP2: hidden-state parallel across the two cards;
  `--disable-custom-all-reduce` + PCIe IPC all-reduce (numel cap 786,432;
  init-time tune, tuned-kernel cache persisted in the host cache dir).
- EP2: 288 routed experts → 144/rank at full intermediate width;
  `--moe-a2a-backend=none` — no dispatch layer; the TP all-reduce fuses partials.
- b12x is EP1-only at this runtime revision (two guards + a prefix-only expert
  map that fits rank 0 but not rank 1). The overlays: lift the guards, thread
  `first_global` (= rank × 144) through `launch_sm120_moe → … →
  _make_w4a16_expert_map` (offset-based map with range assertion), zero-init the
  MoE output under EP, and replace the EP1-only route-pack precompile.
- Route-pack v2 (in the sglang overlay): cold-cache insurance, throughput
  neutral (docs/RESULTS.md §3).

## 4. Attention + long-context stack

- Attention backend `dsa` (sparse attention); prefill/decode
  `flashinfer_sparse_mla`; K-pool metadata fusion env on.
- Linear attention (KDA): `triton` backend; `SGLANG_KDA_EXTEND_BLOCK_TOKENS=2048`.
- Hybrid SSM: mamba cache 12 slots, bf16 SSM dtype, 2 states/path.
- Page 64 · hicache 16 GB · chunked prefill 6144 · mem-fraction-static 0.99.
- CUDA graphs: decode full (bs 1–4), prefill breakable (64/128).

## 5. Speculative decoding

- EAGLE: 5 steps, topk 1, 6 draft tokens; `--speculative-adaptive` with
  `deploy/glm53-adaptive.json` — keys "1"/"4" (concurrency), candidate steps
  [3, 5], up_hysteresis 0.0, down_hysteresis −0.25, ceiling_coeff 0.
- EP2 changes accumulation order, so the emitted token stream differs from TP2
  on identical prompts (benign; goldens pass). Accept-length medians differ
  arm-to-arm — docs/RESULTS.md §2 prices raw vs per-accept deltas separately.

## 6. Serving surface

- Native `POST /generate` (streaming; final-chunk `meta_info` carries
  prompt_tokens / cached_tokens / queue_time / spec_accept_length), plus
  OpenAI-compatible endpoints, `/metrics`, `/health_generate`.
- Engine listens :8000 in-container, published on `${GLM53_PORT:-8062}`.
- Multimodal enabled (torchvision processor, CPU preprocessing);
  `--warmups=serving_coverage`.

## 7. EP2 trade-offs (why it is the serving config)

- Wins (raw): C1/8k **+50.5%**, C1/64k **+12.8%**, C4/8k **+10.3%**; par at
  C1/256k and C4/64k; −1.2% C4/256k. Per-accept: +28–29% at short context
  (docs/RESULTS.md §2).
- Costs: two patched files are a maintenance surface on any image bump
  (re-derive per docs/deploy.md §4, flow §8); accumulation-order output flips;
  C1/256k prefill ~+2–3% slower.
