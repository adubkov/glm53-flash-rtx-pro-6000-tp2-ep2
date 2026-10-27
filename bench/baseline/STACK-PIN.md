# Stack pin — the serving stack behind every number in docs/RESULTS.md

One host, 2x RTX PRO 6000 Blackwell 96 GB (SM120), PCIe (no NVLink), both cards
capped 300.00 W. Driver 595.99.02 (open kernel module), kernel 7.0.0-38-generic,
Docker 29.8.2.

## Checkpoint

| field | value |
|---|---|
| served | `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO` @ `ee0989a944b0e213589191d7fca63af825a0741e` (HF; revision probed live 2026-10-20) |
| upstream source | `zai-org/GLM-5.3-Flash-BF16` @ `f12e0fe1f6b2ea274c11a569582edfd99d993c5e` (image label `ai.model.*`; serving host verify record `.source-revision`) |
| identity note | served checkpoint is the ormandj original requant (HF author ormandj, repo created 2026-08-31); nothing locally requantized feeds this stack |

## Runtime image

| field | value |
|---|---|
| public base | `ghcr.io/ormandj/sglang-glm53-flash-sm120@sha256:ec4243f940a179a27fea21895077efd47cd050501f99a1d2a5fecf7df2e7be71` (v0.4.3; GHCR manifest digest probed) |
| stage2 (local, the measured image) | `sha256:536e00320ebedf5d7ff15ea9cf881faddf08f52983165553bf4d4b0d47c1f5cd` = base + 2 patch layers: glm47 nullable parser + routed-expert SwiGLU clamp (shas below) |
| runtime revision | `386684975edf3cbce15c4b12df37908366e2aa8b` ("Build v0.4.3-rc.6 with incorporated sparse MLA fixes"), release candidate rc.6 |
| sglang | head `00143e9c23aee2dead5e6fe217bda4fa8739cb92` · tree `ccd5e98e7952c86fda7f0779269edfc9f6aa06f0` · patch sha256 `a4ac6ed5bfca5cee9c51c72d9a07bc23588daef00823de1d452d24d1a31b47d1` |
| flashinfer | head `2b750b2ebe9265bebae1bb8b37e868e7e3fe2a8a` · tree `dd8ed31d4492b038b9e3c6e67d09a2c26ad80618` · patch sha256 `858d6cde28052b559e55c9866f83925cac7bdc7b7b4bc3150268640f60f5db41` (main snapshot ahead of 0.7.0; source-built; ships `blackwell_sm12x/`) |
| ModelOpt | `0.47.0rc0` (head `022767c7ab3d7d36211affd85e5c496770cde768`) |
| cuDNN | `9.14.0.64-1` |
| cache schema | `v84` |
| stage2 patch shas | glm47 nullable parser `8ed76f9da2aa782b3e9374d00687186624fd383e98acf9ba0d6ac9759e1425d7` · SwiGLU clamp `b6bf7c99f3094c927100c0c933daa63a80ef9cffab44092f3199e2134dbfc0b9` |

## EP2 overlays (deploy/patches/ep2/)

| file | sha256 | binds over (image path) |
|---|---|---|
| `ep2-flashinfer_cutlass.py` | `611668d4d673b04b73d64588685da50e9502307e5d3d3abe500d55ab09785916` | `/opt/sglang-source/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py` |
| `ep2-moe_dispatch.py` | `f5df50557b778d27171d96a39ac3800b0cfa44332531aeb8a2dc17f40bbf22e5` | `/opt/sglang/lib/python3.12/site-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` |
| `b12x-ep2-native-v2.patch` (cumulative pristine→serving diff) | `ca2e039242eac66083c0235c692f8350d2bd72cab359fcf2542e66f866c532b3` | both files above |

Pristine (unpatched) image-baked targets, for overlay verification on any image
(docs/deploy.md 4):

| image-baked file | sha256 |
|---|---|
| sglang `moe_runner/flashinfer_cutlass.py` | `3c1b9db9420135346557a0607c4e63966ce7f664effcb7d0293cbfd45bc3a71d` |
| flashinfer `blackwell_sm12x/moe_dispatch.py` | `12b5b469a28596183ab80e63cc99bd406f0a8117fa509ccbac92e3741eda18c2` |

Both probed on the stage2 image; its two layers touch neither module, so the
overlays apply identically on the public base.

## Serving envelope (fixed point of all numbers)

| field | value |
|---|---|
| topology | TP2 + EP2 — 288 routed experts → 144/rank, `--moe-a2a-backend=none`, runner `flashinfer_cutlass` |
| pool | asked 650,000 → engine-reported 649,984 (page 64) |
| context | 524,288 · max running 16 · chunked prefill 6144 |
| mamba | 12 slots · bf16 SSM · 2 states/path |
| hicache | 16 GB |
| KV / lm_head | fp8_e4m3 · `SGLANG_LM_HEAD_FP8=1` |
| spec decode | EAGLE 5 steps / topk 1 / 6 draft tokens · adaptive steps 3↔5 (`deploy/glm53-adaptive.json`, sha256 `23faa5c717d20bc1638148751ebdf1a474985ebf91fbc3ee4eacd7b6b1240722`) |
| all-reduce | custom AR disabled · PCIe IPC all-reduce (numel cap 786,432) + init-time tune, cache persisted |
