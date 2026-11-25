# hy4-flash-v100

1Cat-vLLM engine for serving [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) on four V100-SXM2-32GB
GPUs, with the `fast14` launch profile. It runs our
[Hy4-flash-W4A16-AWQ](https://huggingface.co/philbert440/Hy4-flash-W4A16-AWQ) checkpoint, which reads the shared
expert's int8 packs from inside the checkpoint, or leoncca's
[Hy4-flash-AWQ-g32](https://huggingface.co/leoncca/Hy4-flash-AWQ-g32) with a separate cache. One request decodes at
about 107 tok/s plain and about 138 tok/s with the built-in DFly2 drafter.

- Recipe: [HY4-V100.md](HY4-V100.md)
- Prebuilt SM70 wheel, and the shared-expert cache for leoncca's checkpoint:
  [release `hy4-fast14-20261125`](https://github.com/philbert440/hy4-flash-v100/releases/tag/hy4-fast14-20261125)
- Model page with measurements: https://philtompkins.com/models/hy4-flash
- Upstream engine: https://github.com/1CatAI/1Cat-vLLM

## Exact base

| | |
|---|---|
| Upstream | 1Cat-vLLM `main` @ [`1fa9c438bc6e593dbfdd9430b23012a82023bed8`](https://github.com/1CatAI/1Cat-vLLM/commit/1fa9c438bc6e593dbfdd9430b23012a82023bed8) (2026-11-19) |
| Hy4-flash support | Leonccaa's port from [ReVolta](https://github.com/Leonccaa/ReVolta) `v0.2.0rc13`, 47 commits applied as one |
| This engine | the `fast14` speed work, squashed, plus the shared-expert cache tool and sidecar loader |
| Tags | `hy4-fast14-20261124` (engine as measured), `hy4-fast14-20261124b` (adds the cache `rebind` fix and the release docs), `hy4-fast14-20261125` (adds the shared-expert sidecar loader; our checkpoint becomes the default) |

The history is four commits, this README, and follow-ups:

1. upstream `main`, unmodified, without upstream history;
2. the ReVolta port, authored by Leonccaa. Its message lists the 47 ReVolta commit ids;
3. the `fast14` engine. Its tree equals the measured engine except for publication edits: a test-harness hook in the
   RPC router is removed, test defaults no longer name local paths, and two documentation paths are generic. No
   kernel or model logic differs;
4. `tools/hy4/shared_w8_cache.py`, which builds the shared-expert int8 cache the fast profiles need;
5. the follow-up: `rebind` accepts a downloaded cache on any copy of the same checkpoint, and the recipe covers the
   prebuilt wheel and the downloadable cache. Engine code is unchanged from `hy4-fast14-20261124`;
6. the sidecar loader: `HY4_W8_SHARED_SIDECAR=1` reads the shared-expert int8 packs from
   `hy4-v100-shared-w8.safetensors` in the checkpoint, checked against the checkpoint's content, with no cache
   directory or rebind. Default off; with the switch off the engine behaves as `hy4-fast14-20261124b`. Also
   `tools/hy4/qpack_shared_sidecar.py`, which writes the sidecar from a verified cache, and
   `tools/hy4/engram_reencode.py`, which re-encodes the Engram tables as Tencent's exact FP8 round trip;
7. the recipe makes our checkpoint and its sidecar the default.

Every speed path sits behind its own environment switch. `VLLM_HY4_PROFILE=reference` turns them all off and
`VLLM_HY4_PROFILE=fast14` turns on the measured set. See `vllm/models/hy_v4_flash/profile.py`.

## Licences and credits

- Engine code: Apache-2.0, from [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) and vLLM. The upstream README is in
  [README.1Cat-vLLM.md](README.1Cat-vLLM.md).
- Hy4-flash support: Leonccaa's [ReVolta](https://github.com/Leonccaa/ReVolta), Apache-2.0. See `NOTICE` and
  `OVERLAY-LEDGER.md`.
- Model: Hy4-flash by the Tencent Hy Team, under the
  [Tencent Hy License, Version 1.0](https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE). The weights this
  engine runs, ours and leoncca's, are quantized derivatives under the same licence; ours keeps leoncca's layout and
  non-expert files. This repo contains no weights. The release's
  shared-expert cache is quantized from those weights, so it is also under the Tencent Hy License; the release
  includes the licence text and a NOTICE.
