# Deploy

Pins: [bench/baseline/STACK-PIN.md](../bench/baseline/STACK-PIN.md). Facts only.

## 1. Prerequisites

- 2× RTX PRO 6000 Blackwell 96 GB (SM120), PCIe. Tested: driver 595.99.02 (open
  kernel module), Docker 29.8.2 + NVIDIA container toolkit, kernel 7.0.0-38-generic.
- ~166 GiB free for the checkpoint; a persistent cache volume (warm
  triton/cubin/autotune caches save minutes of boot JIT on every start).
- Read access to the pinned HF repos (§2).

## 2. Checkpoint

Served: `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO` @
`ee0989a944b0e213589191d7fca63af825a0741e` — a community requant of
`zai-org/GLM-5.3-Flash-BF16` @ `f12e0fe1f6b2ea274c11a569582edfd99d993c5e`.

```bash
hf download ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO \
  --revision ee0989a944b0e213589191d7fca63af825a0741e --local-dir <MODEL_DIR>
```

Any host path works — compose mounts it read-only at the fixed container path
`/models/glm53-flash-w4a16-e4m3-k32` and points `--model-path` there.

To requant yourself from the pinned BF16 source: `QUANTIZATION.md` + a
fail-closed producer in the runtime repo's `quantization/` (link in §3).

## 3. Image

- compose default: the **public base**
  `ghcr.io/ormandj/sglang-glm53-flash-sm120@sha256:ec4243f9…` (v0.4.3,
  digest-pinned). The EP2 overlays apply to it (§4).
- Runtime author's repo (image + quantization producer; issues/PRs):
  <https://github.com/ormandj/sglang-glm53-flash-sm120>
- Measured serving build (stage2): base + 2 hardening patch layers — glm47
  nullable parser + routed-expert SwiGLU clamp (shas in STACK-PIN.md), built
  locally per the public recipe `helixml/ramjet` `deploy/glm53_flash_sm120/`
  (Apache-2.0, third-party). Serving on the base without those two layers is
  untested by us; if you build stage2, pin your local digest via `SGLANG_IMAGE`
  in `.env`.

## 4. Overlay verification — required before first boot on any image

The EP2 route bind-mounts two full files over image-baked modules. Valid only
if your image's baked files match the pristine shas (probed on the measured
image; its two patch layers touch neither module):

```bash
IMG=${SGLANG_IMAGE:-ghcr.io/ormandj/sglang-glm53-flash-sm120@sha256:ec4243f9...}
docker run --rm --entrypoint sha256sum "$IMG" \
  /opt/sglang-source/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py
docker run --rm --entrypoint sha256sum "$IMG" \
  /opt/sglang/lib/python3.12/site-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py
```

- Expected: `3c1b9db9…` and `12b5b469…` (full values in STACK-PIN.md).
  Mismatch → the image changed those modules: do NOT mount the overlays;
  re-derive by applying `deploy/patches/ep2/b12x-ep2-native-v2.patch` to your
  image's pristine files, verify, and re-pin.
- Overlay side: `shasum -a 256 deploy/patches/ep2/*` must give `611668d4…`,
  `f5df5055…`, `ca2e0392…`.

## 5. Configure

```bash
cd deploy && cp .env.example .env   # fill MODEL_DIR, CACHE_DIR, GPU_ID_0/1, SGLANG_API_KEY
```

compose aborts naming the missing variable instead of starting half-configured.

## 6. Boot

```bash
docker compose up -d
docker compose logs -f        # ready when the server reports fired up; ~8 min incl. warmups
```

Optional: `deploy/glm53-boot.service` (oneshot, opt-in via the
`boot-start.enabled` marker; why `restart: "no"` — see the unit header).

## 7. Verify

```bash
curl -s -H "Authorization: Bearer $SGLANG_API_KEY" localhost:${GLM53_PORT:-8062}/get_server_info \
  | python3 -m json.tool | grep -E '"(ep_size|moe_a2a_backend|moe_runner_backend|context_length|max_total_tokens)"'
```

Expected (measured config): `ep_size: 2`, `moe_a2a_backend: none`,
`moe_runner_backend: flashinfer_cutlass`, `context_length: 524288`,
`max_total_tokens: 649984` (ask 650,000 at page 64). Then
`curl -H "Authorization: Bearer $SGLANG_API_KEY" .../health_generate` → 200.
Capture greedy goldens (a short math probe, a fixed pong prompt, a coherence
probe) as your own cross-arm baseline before trusting any A/B.

## 8. Upgrades

- Any image bump: re-run §4 (pristine shas move if the modules changed),
  re-pin STACK-PIN.md.
- When sgl-project/sglang#45298 + flashinfer-ai/flashinfer#6824 are in your
  image's runtime, drop the two overlay mounts — they exist only to bridge
  the unmerged PRs.
- A runtime bump can move the cache schema (v84) — the first boot then re-pays
  JIT/autotune; keep CACHE_DIR regardless.
