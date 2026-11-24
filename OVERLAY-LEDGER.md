# A0 straight-port ledger

Base `1fa9c438bc6e593dbfdd9430b23012a82023bed8`. Donor Leonccaa/ReVolta tag `v0.2.0rc13`, peeled `8b49690a3b3cccd74c2f2b671ebbbc6e4f8e1a9f`. The ordered non-merge patch ledger is the 47 donor commits, listed in order in the port commit message.

| Donor | Resolution | Reason |
| --- | --- | --- |
| `fb2355e58a` AWQ block 1 | Kept the resolved batched-GEMM policy function and trunk-owned diagnostics helpers. Added clamped activation in the existing shared `_silu_and_mul_w13` implementation. | Keep trunk diagnostics and route plumbing. Hy4 needs the clamp before the down projection. |
| `fb2355e58a` AWQ block 2 | Retained the trunk weight-preparation API and policy, added a clamp guard to legacy compact/interleaved W13 preparation. The donor's clean fast single-token guard and E512/g32 metadata shape are retained. | Clamped activations require canonical W13 ordering. Old donor environment reads and int4-zero preparation belong to a different native interface. |
| `39c31a00ff` runner | Kept trunk `_mm_step`. Added the Engram branch to `_maybe_add_ngram_kwargs` at both of its existing live and dummy call sites. The branch emits `lookback_token_ids` and uses the shared raw committed-history buffer. Mapped the two donor tests onto the same hook. | Keep PLE query-offset generation on its own path and preserve its input ownership. |
| `a9f2673064` checkpoint test (3 blocks) | Merged index-time filtering and the table-loading stub into the existing packed/AWQ meta model construction. Kept cache specs and exact load counts. Tests assert that the generic loader visits no Engram tables. | The A1 net probe never reached this divergent donor-history patch. Separate Engram-loading tests own rank-local table loading. |
| `584e67252d` documentation | Carried the donor guide and kept 1Cat's README and deleted-file state. | The support/readme entries describe a released ReVolta setup. |
| `ba4118179e` private tile constant | Removed the donor import and bound resolved `indexer_prefill_tile_mb` during IndexPool construction. The same MiB-derived FP32 output element budget sizes scratch and tiles live GEMM. Direct kernel fixtures use the default resolved policy. | 1Cat owns the field in `Sm70SparseConfig`. |
| A0 launch admission | Added local startup guards for TP4, disabled speculation, and the qualified KV dtypes on CUDA. | The donor config guarded these conditions only when DCP exceeded one. A0's rejection lane uses DCP1. |
| `9756079d9` fixture dependency | Carried the checkpoint coverage context manager without its Vision FC2 type or default-loader hook. | The donor Hy4 NVFP4 test imports it and verifies successful component copies. Trunk has no Vision FC2 method and A0 does not change the common default loader. |
| CPU AWQ resident buffers | Use 1Cat’s IPC-safe allocation context only when the resident buffers target a CUDA device. | CPU tensors do not participate in CUDA IPC, even on a CUDA platform with GPUs masked out. Preserve the CUDA graph path. |
| Pinned Transformers oracle | Map the oracle’s `post_mlp_layernorm`, `pre_expert_proj` and `post_expert_proj` to the corresponding checkpoint parameter names in `test_body.py`. | The pinned upstream oracle uses those names for the same components. Keep weight-copy, shape, routing, and numerical assertions intact. |
| Attribution | Added modified-file notices and the applicable ReVolta notice. | Source authorship and licensing travel with the patch. |

ReVolta's decode layout follows vLLM PR #68296 by Jee Jee Li (`b067b6f6f` in the local upstream checkout). Tencent supplies the model and the Transformers oracle. This local branch is integration evidence. The donor guide describes ReVolta's own tests and measurements.

- Stage 1c retargets the donor scheduler fixture to the shared `_maybe_add_ngram_kwargs` hook. The test still asserts the same lookback IDs across cache hits, preemption and re-entry.

- The full Hy4 suite exposed donor-only fixture methods. The paged fixture binds
  raw allocations through 1Cat's V1 cache reshape and takes indexer dimensions
  and partial RoPE from the pinned Transformers API. The parser fixture calls the
  same reasoning and streaming methods as 1Cat's serving layer. Engram fixtures
  use 1Cat's input-transfer and prefix-restore interfaces and the pinned oracle's
  branch-scale module. All existing assertions and parametrizations are retained.

- ReVolta's sliding-window sizing guard admits replicated owners at DCP2/4.
  Restore that condition in 1Cat's sizing path. Sharded sliding-window owners
  retain their DCP1 requirement. The expanded scheduler page-layout tests
  exercise the Hy4 replicated path and still assert the allocator geometry.

- Retarget Leonccaa's prefill-index tiling fixture to the overlay's explicit
  tile_mb and GEMM sizing helper. Its original 7,168-element cap is smaller
  than one MiB, so the fixture caps the helper's result at that exact value.
  The 128-group gather tile and 64-column GEMM sub-tile are both exercised.

- A0 full-context prefill retains logits, gathered FP16 keys and the FP32
  score tile in the worker workspace. The first profile reserves the same
  upper bound previously charged as a transient peak. The scorer uses the
  original column widths and FP32 GEMM output with contiguous workspace
  views, including the last short tile. The locked-workspace CPU test checks
  the 720 MiB bound at the native limit with one, two and four cache shards.

- A0 NVFP4 loader fixtures assert 1Cat's initial ModelOpt fused method and
  Skinny SM70 admission. The CPU fixture emulates the device capability and
  checks the actual loaded gate/up scales trigger the shared-scale rejection.
  The CUDA fixture resolves the replacement TurboMind method after the weight
  post-load hook and keeps the original clamped-output checks and tolerances.
