# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Checkpoint contract adapted from vllm/models/hy_v4/dfly_config.py and vllm/models/hy_v4_flash/dfly_config.py at upstream 596455c7f16d14e761fbf8b61bf071b2e31c9aa5.  # noqa: E501

"""Admission and checkpoint geometry for the Hy4 Flash SM70 DFly2 path."""

from dataclasses import dataclass

DRAFT_TOKENS = 5
QUERY_WIDTH = DRAFT_TOKENS + 1


def validate_parallel_config(parallel):
    for name in (
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "pipeline_parallel_size",
    ):
        if getattr(parallel, name) != 1:
            raise ValueError(f"Hy4 DFly2 requires {name}=1.")


@dataclass(frozen=True)
class DFlySettings:
    target_layers: tuple[int, ...]
    topk: int
    mask_id: int
    correction_width: int
    ffn_width: int | None
    post_proj_norm: bool

    @classmethod
    def from_checkpoint(cls, root, text):
        from .configs import SPARSE_ATTENTION

        def read(name, default=None):
            return getattr(root, "dfly_" + name, default)

        count = read("num_hidden_layers")
        kv_layers = read("target_kv_layers")
        index_layers = read("target_index_layer")
        if (
            type(count) is not int
            or count < 1
            or not isinstance(kv_layers, (list, tuple))
            or len(kv_layers) != count
            or kv_layers != index_layers
            or any(
                type(layer) is not int
                or not 0 <= layer < len(text.layer_types)
                or text.layer_types[layer] != SPARSE_ATTENTION
                for layer in kv_layers
            )
            or len(set(kv_layers)) != count
        ):
            raise ValueError(
                "DFly2 requires matching sparse dfly_target_kv_layers "
                "and dfly_target_index_layer with one layer per draft layer"
            )
        rope = text.rope_for_layer(kv_layers[0])
        if any(text.rope_for_layer(layer) != rope for layer in kv_layers[1:]):
            raise ValueError("DFly2 paired layers must use the same RoPE")
        for name, default, valid in (
            ("mtp_fusion", "slot0", ("slot0",)),
            ("enable_hidden_correction", True, (True,)),
            ("enable_ihc", False, (False,)),
            ("mlp_type", "dense", ("dense",)),
            ("attn_causal", True, (True,)),
            ("context_local_window", None, (None, 0)),
        ):
            if read(name, default) not in valid:
                raise ValueError(f"Unsupported DFly2 dfly_{name}")
        mask = read("mask_token_id")
        if type(mask) is not int or not 0 <= mask < text.vocab_size:
            raise ValueError("Invalid DFly2 dfly_mask_token_id")
        topk = read("target_index_topk", text.index_topk)
        if (
            topk
            != text.compressed_indexer_group_topk
            * text.compressed_indexer_compress_ratio
        ):
            raise ValueError(
                "DFly2 dfly_target_index_topk must match the target selector"
            )
        correction = read("hidden_correction_intermediate_size", text.hidden_size)
        ffn = read("intermediate_size")
        post_norm = read("attn_post_proj_norm")
        for name, value in (
            ("hidden_correction_intermediate_size", correction),
            ("intermediate_size", ffn),
        ):
            if value is None and name == "intermediate_size":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid DFly2 dfly_{name}")
        if post_norm is not None and type(post_norm) is not bool:
            raise ValueError("Invalid DFly2 dfly_attn_post_proj_norm")
        # This checkpoint omits the flag but stores a post-projection norm in
        # every draft layer. Require all of them when the flag is omitted.
        return cls(
            tuple(kv_layers), topk, mask, correction, ffn, post_norm is not False
        )
