# Licensing and third-party material

The MIT license in `LICENSE` covers **only our own material**. It does not
license the model, the runtime image, or the upstream code the overlays patch.

## What is ours (MIT)

| Path | What it is |
|---|---|
| `deploy/compose.yaml`, `deploy/.env.example`, `deploy/glm53-boot.service` | the serving definition (fail-closed compose, env template, boot unit) |
| `deploy/glm53-adaptive.json` | the adaptive spec-decode config |
| `deploy/patches/ep2/*` | our modifications to two Apache-2.0 upstream files (see below); upstream license governs the files |
| `bench/harness.py`, `bench/plan_gen.py`, `bench/plan.json` | the bench harness and plan |
| `bench/baseline/`, `bench/ep2/` | the measured record |
| `README.md`, `docs/`, `.gitignore` | our write-up |

## What is not ours

### The checkpoint — reference only, zero bytes shipped

The pinned checkpoint ([bench/baseline/STACK-PIN.md](bench/baseline/STACK-PIN.md))
is a community requant of `zai-org/GLM-5.3-Flash-BF16`, pulled — never shipped:
accept its terms yourself; nothing here carries or reconstructs a weight byte.

### The runtime image — pull or build it yourself

`ghcr.io/ormandj/sglang-glm53-flash-sm120` v0.4.3 (digest-pinned in
compose.yaml) and the local stage2 build (docs/deploy.md §3) are
**third-party builds we do not redistribute**. The author's repo —
<https://github.com/ormandj/sglang-glm53-flash-sm120> — is the original work
and its fix/carry record. The stage2 build path runs through Apache-2.0
`helixml/ramjet` (cited by path, nothing shipped). Their own license terms
apply; we make no compliance claim.

### The EP2 overlays — upstream code, our modifications

`deploy/patches/ep2/ep2-flashinfer_cutlass.py` and `ep2-moe_dispatch.py` are
full copies of Apache-2.0 upstream files with our EP2 modifications:

- sglang file: sgl-project/sglang (Apache-2.0), `moe_runner/flashinfer_cutlass.py`
- flashinfer file: flashinfer-ai/flashinfer (Apache-2.0),
  `blackwell_sm12x/moe_dispatch.py`

The Apache-2.0 upstream license governs the files; `b12x-ep2-native-v2.patch`
is the cumulative pristine→serving diff showing exactly what we changed. The
same modifications are filed upstream as sgl-project/sglang#45298 and
flashinfer-ai/flashinfer#6824.

### Third-party runtime dependencies

Inside the image: `torch`, `flashinfer`, `sglang`, `modelopt`. Host side:
`docker`, `python3`, `curl`, and `hf`/`huggingface-cli` for the download. No
vendored source from any of them.

## If you think something here is mislicensed

Open an issue. The digests in `bench/baseline/STACK-PIN.md` exist so a claim
can be checked rather than argued.
