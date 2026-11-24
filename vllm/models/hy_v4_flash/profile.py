# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve the optional SM70 speed paths against the server configuration."""

from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_FAST_DEFAULTS = {
    "VLLM_HY4_EXPERT_KERNEL": "union",
    "VLLM_HY4_EXPERT_UNION_MIN_ROWS": "1",
    "VLLM_HY4_INDEX_BYPASS": True,
    "VLLM_HY4_POSITION_FUSION": True,
    "VLLM_HY4_PARALLEL_ROUTER": True,
    "HY4_F7_ATTN_PREP": True,
    "HY4_F7_ATTN_OUT": True,
    "HY4_F7_IDX_FUSE": True,
    "HY4_F7_HC_BRIDGE": True,
}


_FAST2_ADDITIONS = dict.fromkeys((
    "HY4_DENSE_FP16_STORAGE",
    "HY4_DENSE_SKINNY",
    "HY4_BOUNDARY_ROUTER_INPUT",
    "HY4_BOUNDARY_AR_NORM",
    "HY4_ATTNCORE_PREP",
    "HY4_ATTNCORE_SPLIT",
), True)
_FAST3_ADDITIONS = {
    "VLLM_HY4_EXPERT_UNION_FOLD": True,
    "HY4_W8_ATTN": True,
    "HY4_W8_O_PROJ": True,
    "HY4_W8_LATENT": True,
    "HY4_W8_SHARED": True,
    "HY4_W8_MLP": True,
    "HY4_W8_INDEXER": True,
    "HY4_W8_UV": True,
    "HY4_W8_SHARED_GROUP_SIZE": 32,
}
_FAST4_ADDITIONS = dict.fromkeys((
    "HY4_ATTN3_W8_MERGE",
    "HY4_ATTN3_FP8_PRMT",
    "HY4_NODES_ENGRAM",
    "HY4_NODES_BRIDGE_RMS",
), True)
_FAST3_DEFAULTS = _FAST_DEFAULTS | _FAST2_ADDITIONS | _FAST3_ADDITIONS
_FAST4_DEFAULTS = _FAST3_DEFAULTS | _FAST4_ADDITIONS
_FAST5_ADDITIONS = dict.fromkeys(("HY4_PREP_SPLIT_W8", "HY4_EXPERTS3_RING"), True)
_FAST5_DEFAULTS = _FAST4_DEFAULTS | _FAST5_ADDITIONS
_FAST6_ADDITIONS = {"HY4_BRIDGE_SPLIT_BIG_MIX": True}
_FAST6_DEFAULTS = _FAST5_DEFAULTS | _FAST6_ADDITIONS
_FAST7_ADDITIONS = {"HY4_EXPERTS4_SHARED": True, "HY4_W8_HEAD": True}
_FAST7_DEFAULTS = _FAST6_DEFAULTS | _FAST7_ADDITIONS
_FAST8_ADDITIONS = {"HY4_EXPERTS4B_VERIFY": True}
_FAST8_DEFAULTS = _FAST7_DEFAULTS | _FAST8_ADDITIONS
_FAST9_ADDITIONS = {"HY4_ATTN4B_SWA_TENSOR": True}
_FAST9_DEFAULTS = _FAST8_DEFAULTS | _FAST9_ADDITIONS
_FAST10_ADDITIONS = {
    "VLLM_HY4_DFLY_HEAD": "int8",
    "VLLM_HY4_DFLY_CONFIDENCE_MODE": "variable",
    "VLLM_HY4_DFLY_CONFIDENCE_THRESHOLD": 0.45,
}
_FAST10_DEFAULTS = _FAST9_DEFAULTS | _FAST10_ADDITIONS
_FAST11_ADDITIONS = {
    "HY4_DREV2_DEFER_DRAFTS": True,
    "VLLM_V41_INLINE_DRAFTS": "1",
    "HY4_DREV2_WIDE": True,
}
_FAST11_DEFAULTS = _FAST10_DEFAULTS | _FAST11_ADDITIONS
_FAST13_ADDITIONS = {"HY4_IDX_TC": True, "HY4_DCONF2_LATE_WIDTH": True}
_FAST13_DEFAULTS = _FAST11_DEFAULTS | _FAST13_ADDITIONS
_FAST14_ADDITIONS = {"HY4_PF_ATTN_TC": True, "HY4_PF_ATTN_FIX": 8}
_PROFILE_DEFAULTS = {
    "reference": {},
    "fast": _FAST_DEFAULTS,
    "fast2": _FAST_DEFAULTS | _FAST2_ADDITIONS,
    "fast3": _FAST3_DEFAULTS,
    "fast4": _FAST4_DEFAULTS,
    "fast5": _FAST5_DEFAULTS,
    "fast6": _FAST6_DEFAULTS,
    "fast7": _FAST7_DEFAULTS,
    "fast8": _FAST8_DEFAULTS,
    "fast9": _FAST9_DEFAULTS,
    "fast10": _FAST10_DEFAULTS,
    "fast11": _FAST11_DEFAULTS,
    "fast13": _FAST13_DEFAULTS,
    "fast14": _FAST13_DEFAULTS | _FAST14_ADDITIONS,
}


def _exclusions(
    config: "VllmConfig", defaults: dict | None = None
) -> dict[str, str]:
    from vllm.config.compilation import CUDAGraphMode

    if defaults is None:
        defaults = _FAST_DEFAULTS
    model = config.model_config
    if (
        model is None
        or config.device_config.device_type != "cuda"
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability(70)
        or model.dtype != torch.float16
    ):
        return dict.fromkeys(defaults, "requires SM70 CUDA and FP16")

    text = model.hf_text_config
    parallel = config.parallel_config
    reasons = {}
    if model.uses_mrope:
        for name in ("VLLM_HY4_POSITION_FUSION", "HY4_F7_ATTN_PREP"):
            reasons[name] = "unsupported for image positions"
    if parallel.decode_context_parallel_size > 1:
        reasons["HY4_F7_ATTN_OUT"] = "excluded by cache sharding"

    if config.speculative_config is not None:
        reasons["HY4_F7_HC_BRIDGE"] = "excluded by DFly2"
    elif envs.VLLM_BATCH_INVARIANT:
        reasons["HY4_F7_HC_BRIDGE"] = "excluded by batch-invariant mode"
    elif not (
        getattr(text, "enable_ihc", False)
        and text.hidden_size == 3072
        and text.hc_mult == 4
    ):
        reasons["HY4_F7_HC_BRIDGE"] = "requires four hyper-connection streams"

    if getattr(config.speculative_config, "method", None) != "dfly":
        draft_only = _FAST11_ADDITIONS.keys() | {"HY4_DCONF2_LATE_WIDTH"}
        reasons.update(dict.fromkeys(defaults.keys() & draft_only, "requires DFly2"))

    if not envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK:
        reasons["VLLM_HY4_PARALLEL_ROUTER"] = "grouped topk disabled"
    quant = getattr(model.hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "awq" or quant.get("group_size") != 32:
        reasons["VLLM_HY4_EXPERT_KERNEL"] = "requires native AWQ group size 32"
    if (
        model.enforce_eager
        or config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    ):
        reasons["VLLM_HY4_INDEX_BYPASS"] = "decode graphs disabled"
    elif model.max_model_len <= (
        text.compressed_indexer_group_topk * text.compressed_indexer_compress_ratio
    ):
        reasons["VLLM_HY4_INDEX_BYPASS"] = "all history already fits in the window"
    if "HY4_DENSE_FP16_STORAGE" in defaults:
        if envs.VLLM_BATCH_INVARIANT:
            for name in _FAST2_ADDITIONS:
                if name.startswith(("HY4_DENSE_", "HY4_BOUNDARY_")):
                    reasons[name] = "excluded by batch-invariant mode"
        for name, prerequisite in (
            ("HY4_ATTNCORE_PREP", "HY4_F7_ATTN_PREP"),
            ("HY4_ATTNCORE_SPLIT", "HY4_F7_ATTN_OUT"),
        ):
            if prerequisite in reasons or (
                envs.is_set(prerequisite) and not getattr(envs, prerequisite)
            ):
                reasons[name] = f"requires {prerequisite}"
    if "VLLM_HY4_EXPERT_UNION_FOLD" in defaults and "VLLM_HY4_EXPERT_KERNEL" in reasons:
        reasons["VLLM_HY4_EXPERT_UNION_FOLD"] = reasons["VLLM_HY4_EXPERT_KERNEL"]
    if envs.VLLM_BATCH_INVARIANT:
        for name in (_FAST3_ADDITIONS.keys() | {"HY4_W8_HEAD"}) & defaults.keys():
            if name.startswith("HY4_W8_"):
                reasons[name] = "excluded by batch-invariant mode"
    if (
        "HY4_EXPERTS4_SHARED" in defaults
        and envs.is_set("HY4_W8_SHARED_GROUP_SIZE")
        and envs.HY4_W8_SHARED_GROUP_SIZE != 32
    ):
        reasons["HY4_EXPERTS4_SHARED"] = "requires HY4_W8_SHARED_GROUP_SIZE=32"
    for name, prerequisite in (
        ("HY4_ATTN3_W8_MERGE", "HY4_F7_ATTN_OUT"),
        ("HY4_ATTN3_FP8_PRMT", "HY4_F7_ATTN_OUT"),
        ("HY4_ATTN4B_SWA_TENSOR", "HY4_ATTN3_FP8_PRMT"),
        ("HY4_NODES_BRIDGE_RMS", "HY4_F7_HC_BRIDGE"),
        ("HY4_BRIDGE_SPLIT_BIG_MIX", "HY4_F7_HC_BRIDGE"),
        ("HY4_PREP_SPLIT_W8", "HY4_F7_ATTN_PREP"),
        ("HY4_EXPERTS3_RING", "VLLM_HY4_EXPERT_UNION_FOLD"),
        ("HY4_EXPERTS4_SHARED", "HY4_EXPERTS3_RING"),
        ("HY4_EXPERTS4_SHARED", "HY4_W8_SHARED"),
        ("HY4_EXPERTS4B_VERIFY", "HY4_EXPERTS4_SHARED"),
    ):
        if name in defaults and (
            prerequisite in reasons
            or (envs.is_set(prerequisite) and not getattr(envs, prerequisite))
        ):
            reasons[name] = f"requires {prerequisite}"
    if envs.VLLM_BATCH_INVARIANT and "HY4_NODES_ENGRAM" in defaults:
        reasons["HY4_NODES_ENGRAM"] = "excluded by batch-invariant mode"
    return reasons


def speed_setting(name: str, config: "VllmConfig | None" = None):
    """Preserve explicit switches and reference defaults without mutating env."""
    defaults = _PROFILE_DEFAULTS[envs.VLLM_HY4_PROFILE]
    if name not in defaults or envs.is_set(name):
        return getattr(envs, name)
    if config is None:
        from vllm.config.vllm import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    model = getattr(config, "model_config", None)
    if getattr(getattr(model, "hf_text_config", None), "model_type", None) != "hy_v4_flash":
        return getattr(envs, name)
    if name in _exclusions(config, defaults):
        return getattr(envs, name)
    return defaults[name]


def initialize_profile(config: "VllmConfig") -> None:
    """Log server-wide eligibility and eagerly check the selected native ops."""
    defaults = _PROFILE_DEFAULTS[envs.VLLM_HY4_PROFILE]
    # The compatibility report is also useful when using individual overrides.
    reasons = _exclusions(config, defaults) if defaults else {}
    active = []
    inactive = []
    enabled = set()
    for name in (*_PROFILE_DEFAULTS["fast14"], "HY4_DREV2_WIDE_NORM"):
        if name.startswith("VLLM_HY4_DFLY_") or name == "VLLM_V41_INLINE_DRAFTS":
            continue
        value = speed_setting(name, config)
        if name in ("VLLM_HY4_EXPERT_UNION_MIN_ROWS", "HY4_PF_ATTN_FIX") or name.endswith("_GROUP_SIZE"):
            continue
        if name in reasons:
            inactive.append(f"{name} ({reasons[name]})")
        elif value is True or value == "union":
            enabled.add(name)
            detail = (
                f"=union/min_rows={speed_setting('VLLM_HY4_EXPERT_UNION_MIN_ROWS', config)}"
                if value == "union"
                else ""
            )
            if name == "HY4_PF_ATTN_TC":
                detail = f"/mode={speed_setting('HY4_PF_ATTN_FIX', config)}"
            active.append(f"{name}{detail}")
        else:
            reason = "explicit override" if envs.is_set(name) else "off by profile"
            inactive.append(f"{name} ({reason})")
    if config.speculative_config is None:
        inactive.append("VLLM_HY4_DFLY_VERIFY_INDEX (DFly2 off)")
        inactive.append("VLLM_HY4_DFLY_HEAD (DFly2 off)")
    else:
        active.append(f"VLLM_HY4_DFLY_VERIFY_INDEX={envs.VLLM_HY4_DFLY_VERIFY_INDEX}")
        for name in (*_FAST10_ADDITIONS, "VLLM_V41_INLINE_DRAFTS"):
            active.append(f"{name}={speed_setting(name, config)}")
    logger.info_once(
        "Hy4 profile=%s active=[%s] off=[%s] "
        "scope: small-row kernels, index bypass for short decode graphs; "
        "attention prep takes priority over position fusion",
        envs.VLLM_HY4_PROFILE, ", ".join(active), ", ".join(inactive),
    )
    if defaults:
        _check_native_ops(enabled)
    if {"HY4_BRIDGE_SPLIT_BIG_MIX", "HY4_F7_HC_BRIDGE"} <= enabled:
        from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
            load_hy4_sm70_operator,
        )

        for option in ("f7_hc_bridge", "nodes_hc_rms"):
            if option == "nodes_hc_rms" and "HY4_NODES_BRIDGE_RMS" not in enabled:
                continue
            load_hy4_sm70_operator(option)
            op = getattr(torch.ops.hy4_sm70, option)
            if not any(
                arg.name == "split_big_mix" for arg in op.default._schema.arguments
            ):
                raise RuntimeError(
                    f"{option} requires an updated _sm70_hy4_moe_C extension."
                )
    for override, prerequisite, option, argument in (
        (
            speed_setting("HY4_ATTN4B_SWA_TENSOR", config),
            "HY4_ATTN3_FP8_PRMT",
            "attn_splitk_sm70",
            "swa_tensor",
        ),
        (speed_setting("HY4_ATTNCORE_PREP", config), "HY4_F7_ATTN_PREP", "attn_qkv_prep_sm70", "cache"),
        (
            speed_setting("HY4_PREP_SPLIT_W8", config),
            "HY4_F7_ATTN_PREP",
            "attn_qkv_prep_sm70",
            "split_w8",
        ),
        (
            speed_setting("HY4_ATTNCORE_SPLIT", config),
            "HY4_F7_ATTN_OUT",
            "attn_splitk_sm70",
            "short_splits",
        ),
        (
            speed_setting("HY4_ATTN3_FP8_PRMT", config),
            "HY4_F7_ATTN_OUT",
            "attn_splitk_sm70",
            "prmt_fp8",
        ),
        (
            speed_setting("HY4_ATTN3_W8_MERGE", config),
            "HY4_F7_ATTN_OUT",
            "attn_merge_v_gate_sm70",
            "narrow_w8",
        ),
    ):
        if (
            override
            and prerequisite in enabled
            and config.device_config.device_type == "cuda"
        ):
            from .sm70.f7_attn import load_hy4_f7_attn_operator

            load_hy4_f7_attn_operator(option)
            op = getattr(torch.ops.hy4_f7_attn, option)
            if not any(arg.name == argument for arg in op.default._schema.arguments):
                raise RuntimeError(
                    f"{option} requires an updated _sm70_hy4_attn_C extension."
                )


def _check_native_ops(enabled: set[str]) -> None:
    from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
        load_hy4_sm70_operator,
    )
    from vllm.models.hy_v4_flash.sm70.f7_attn import load_hy4_f7_attn_operator
    from vllm.models.hy_v4_flash.sm70.f7_next import load_hy4_f7_next_operator

    for name, load, operators in (
        ("VLLM_HY4_EXPERT_KERNEL", load_hy4_sm70_operator, ("awq_union_moe_out",)),
        ("VLLM_HY4_EXPERT_UNION_FOLD", load_hy4_sm70_operator, ("awq_union_moe_fold_out",)),
        ("VLLM_HY4_PARALLEL_ROUTER", load_hy4_sm70_operator, ("parallel_top16",)),
        ("HY4_F7_HC_BRIDGE", load_hy4_sm70_operator, ("f7_hc_bridge",)),
        ("HY4_NODES_BRIDGE_RMS", load_hy4_sm70_operator, ("nodes_hc_rms",)),
        ("HY4_DENSE_FP16_STORAGE", load_hy4_sm70_operator, ("dense_gemv",)),
        ("HY4_DENSE_SKINNY", load_hy4_sm70_operator, ("skinny_projection",)),
        ("HY4_BOUNDARY_ROUTER_INPUT", load_hy4_sm70_operator, ("boundary_router_input",)),
        ("HY4_BOUNDARY_AR_NORM", load_hy4_sm70_operator, ("boundary_ar_norm",)),
        ("HY4_EXPERTS3_RING", load_hy4_sm70_operator, ("experts3_ring_fold_out",)),
        ("HY4_EXPERTS4_SHARED", load_hy4_sm70_operator, ("experts4_ring_shared_out",)),
        ("HY4_W8_HEAD", load_hy4_sm70_operator, ("w8_head_projection",)),
        ("HY4_F7_ATTN_PREP", load_hy4_f7_attn_operator,
         ("rope_phases_sm70", "attn_qkv_prep_sm70")),
        ("HY4_F7_ATTN_OUT", load_hy4_f7_attn_operator,
         ("attn_splitk_sm70", "attn_merge_v_gate_sm70")),
        ("HY4_F7_IDX_FUSE", load_hy4_f7_next_operator,
         ("idx_pool_update_sm70", "idx_query_prep_sm70")),
    ):
        if name in enabled:
            for operator in operators:
                try:
                    load(operator)
                except (OSError, RuntimeError) as exc:
                    raise RuntimeError(f"Hy4 fast profile cannot enable {name}: {exc}") from exc
