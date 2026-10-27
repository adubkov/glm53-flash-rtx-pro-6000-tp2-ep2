# GLM-5.3-Flash on 2× RTX PRO 6000 — TP2 + EP2 serving recipe

GLM-5.3-Flash (W4A16 NVFP4-K32) on 2× RTX PRO 6000 Blackwell 96 GB (SM120) over
PCIe: **TP2 + EP2** — 288 routed experts split 144/rank through the
flashinfer_cutlass b12x SM120 path, no MoE dispatch layer. EP2 on this runtime
needed two patched modules (filed upstream, unmerged —
sgl-project/sglang#45298, flashinfer-ai/flashinfer#6824); this repo ships them
as bind overlays plus the compose that mounts them, and the measured record.

**Start with [docs/RESULTS.md](docs/RESULTS.md).** Every number there carries
its cell, its arm, and how clean the row was.

## Numbers at a glance

Medians of 3 measured runs · one host · both cards 300 W · clean rows only.
Raw = what a client sees; per-accept = steps/s at matched accept length.

| cell | TP2 (ep=1) | TP2+EP2 | raw Δ | per-accept Δ |
|---|---:|---:|---:|---:|
| C1/8k decode | 197.7 tok/s | 297.7 tok/s | **+50.5%** | +28% |
| C1/64k decode | 226.4 tok/s ⚠ | 255.4 tok/s | **+12.8%** | +29% |
| C1/256k decode | 261.2 tok/s | 261.7 tok/s | +0.2% | +8% |
| C4/8k aggregate | 875.6 tok/s | 965.6 tok/s | **+10.3%** | −4% (par) |
| C4/64k aggregate | 897.9 tok/s | 939.4 tok/s | +4.6% | par |
| C4/256k aggregate | 848.3 tok/s | 837.7 tok/s | −1.2% | −6% (par) |

⚠ = contaminated row: concurrent foreign GPU load depresses aggregates → the
true number is higher (docs/RESULTS.md §6). Here: the baseline C1/64k median.

· prefill (C1, 8k→256k) ~5.2–5.5k tok/s flat · TTFT 64k ~11.8–12.5 s · EAGLE
accept (C1) 4.4–5.3

## Quickstart

1. Checkpoint (docs/deploy.md §2) → 2. `cp deploy/.env.example deploy/.env`,
   fill the five required values → 3. verify overlay shas against your image
   (docs/deploy.md §4 — required before first boot) → 4. `cd deploy && docker
   compose up -d`, verify server_info (§7) → 5. bench it (docs/benchmarking.md).

## Layout

```
deploy/  compose (fail-closed), .env.example, EP2 bind overlays (patches/ep2/),
         adaptive spec-decode config, systemd boot unit
bench/   harness + plan generator + plan.json; baseline/ (TP2 runs, STACK-PIN.md),
         ep2/ (A/B, clean re-bench, route-pack v2 full matrix)
docs/    architecture · deploy · benchmarking · RESULTS
```

## Pins (headline)

| field | value |
|---|---|
| checkpoint | `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO` @ `ee0989a9…` (source: `zai-org/GLM-5.3-Flash-BF16`) |
| runtime | `ghcr.io/ormandj/sglang-glm53-flash-sm120` v0.4.3, digest-pinned; stage2 build path in [docs/deploy.md](docs/deploy.md) §3 |
| serving envelope | pool 649,984 · ctx 524,288 · hicache 16 GB · mamba 12 · EAGLE 5/1/6 adaptive |
| full pin table | [bench/baseline/STACK-PIN.md](bench/baseline/STACK-PIN.md) |

## Upstream (the original work)

| what | link |
|---|---|
| author's repo — image, quantization producer, fix/carry record | https://github.com/ormandj/sglang-glm53-flash-sm120 |
| GLM-5.3-Flash on SM120 upstream tracking | https://github.com/sgl-project/sglang/issues/37813 |
| author's relevant upstream PRs | [sgl#43655](https://github.com/sgl-project/sglang/pull/43655) prefill all-reduce overlap · [sgl#43657](https://github.com/sgl-project/sglang/pull/43657) kda bf16 SSM pool · [sgl#43647](https://github.com/sgl-project/sglang/pull/43647) glm5-next MoE tuning · [sgl#43619](https://github.com/sgl-project/sglang/pull/43619) dsa k-pool prefill split · [flashinfer#4827](https://github.com/flashinfer-ai/flashinfer/pull/4827) SM12x MoE graph-replay workspace · [flashinfer#4049](https://github.com/flashinfer-ai/flashinfer/issues/4049) SM120 MoE tactic bug (closed) |

## License

Our code and docs: **MIT** ([LICENSE](LICENSE)). The checkpoint, the runtime
image, and the overlays' upstream code are not covered by it — read
[LICENSING.md](LICENSING.md) before reusing anything.
