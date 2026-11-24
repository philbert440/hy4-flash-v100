# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.config.compilation import CUDAGraphMode
from vllm.models.hy_v4_flash import profile


@pytest.fixture
def speed_config(monkeypatch):
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast")
    for key in profile._FAST_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(envs, "VLLM_USE_FUSED_MOE_GROUPED_TOPK", True)
    monkeypatch.setattr(profile.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(profile.current_platform, "is_device_capability", lambda cap: cap == 70)
    return SimpleNamespace(
        device_config=SimpleNamespace(device_type="cuda"),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            uses_mrope=False,
            enforce_eager=False,
            max_model_len=262144,
            hf_config=SimpleNamespace(quantization_config={
                "quant_method": "awq", "group_size": 32,
            }),
            hf_text_config=SimpleNamespace(
                model_type="hy_v4_flash", hidden_size=3072, hc_mult=4, enable_ihc=True,
                compressed_indexer_group_topk=512, compressed_indexer_compress_ratio=4,
            ),
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
    )


@pytest.mark.parametrize("case,disabled", [
    ("plain", set()),
    ("draft", {"HY4_F7_HC_BRIDGE"}),
    ("image", {"HY4_F7_ATTN_PREP", "VLLM_HY4_POSITION_FUSION"}),
    ("shards", {"HY4_F7_ATTN_OUT"}),
    ("short", {"VLLM_HY4_INDEX_BYPASS"}),
    ("eager", {"VLLM_HY4_INDEX_BYPASS"}),
    ("batch_invariant", {"HY4_F7_HC_BRIDGE"}),
    ("router_disabled", {"VLLM_HY4_PARALLEL_ROUTER"}),
    ("non_awq", {"VLLM_HY4_EXPERT_KERNEL"}),
    ("non_sm70", set(profile._FAST_DEFAULTS)),
])
def test_fast_admissions(monkeypatch, speed_config, case, disabled):
    config = speed_config
    if case == "draft":
        config.speculative_config = SimpleNamespace(method="dfly")
    elif case == "image":
        config.model_config.uses_mrope = True
    elif case == "shards":
        config.parallel_config.decode_context_parallel_size = 4
        config.model_config.max_model_len = 1048576
    elif case == "short":
        config.model_config.max_model_len = 2048
    elif case == "eager":
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    elif case == "batch_invariant":
        monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    elif case == "router_disabled":
        monkeypatch.setattr(envs, "VLLM_USE_FUSED_MOE_GROUPED_TOPK", False)
    elif case == "non_awq":
        config.model_config.hf_config.quantization_config = None
    elif case == "non_sm70":
        monkeypatch.setattr(profile.current_platform, "is_device_capability", lambda _: False)
    assert set(profile._exclusions(config)) == disabled
    for key, value in profile._FAST_DEFAULTS.items():
        assert (profile.speed_setting(key, config) == value) == (key not in disabled)


def test_profile_overrides(monkeypatch, speed_config):
    config = speed_config
    monkeypatch.setenv("HY4_F7_IDX_FUSE", "0")
    monkeypatch.setenv("VLLM_HY4_EXPERT_UNION_MIN_ROWS", "2")
    assert profile.speed_setting("HY4_F7_IDX_FUSE", config) is False
    assert profile.speed_setting("VLLM_HY4_EXPERT_UNION_MIN_ROWS", config) == "2"
    monkeypatch.setenv("VLLM_HY4_PROFILE", "reference")
    monkeypatch.setenv("HY4_F7_ATTN_OUT", "1")
    assert profile.speed_setting("HY4_F7_ATTN_OUT", config) is True
    assert profile.speed_setting("HY4_F7_ATTN_PREP", config) is False


def test_fast_refuses_missing_native_operator(monkeypatch, speed_config):
    from vllm.models.hy_v4_flash.sm70 import f7_next

    def missing(operator):
        raise RuntimeError(f"missing {operator}")

    monkeypatch.setattr(f7_next, "load_hy4_f7_next_operator", missing)
    for name in profile._FAST_DEFAULTS:
        if name not in ("HY4_F7_IDX_FUSE", "VLLM_HY4_EXPERT_KERNEL"):
            monkeypatch.setenv(name, "2" if name.endswith("MIN_ROWS") else "0")
    monkeypatch.setenv("VLLM_HY4_EXPERT_KERNEL", "current")
    with pytest.raises(RuntimeError, match="cannot enable HY4_F7_IDX_FUSE: missing idx_pool_update_sm70"):
        profile.initialize_profile(speed_config)


@pytest.mark.parametrize(
    "selection", ["fast2", *profile._FAST2_ADDITIONS]
)
def test_fast2_switch_matrix(monkeypatch, speed_config, selection):
    for name in profile._FAST2_ADDITIONS:
        monkeypatch.delenv(name, raising=False)
        assert profile.speed_setting(name, speed_config) is False

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast2" if selection == "fast2" else "fast")
    if selection != "fast2":
        monkeypatch.setenv(selection, "1")
    for name in profile._FAST2_ADDITIONS:
        assert profile.speed_setting(name, speed_config) is (
            selection == "fast2" or name == selection
        )
    for name, value in profile._FAST_DEFAULTS.items():
        assert profile.speed_setting(name, speed_config) == value

    if selection == "fast2":
        monkeypatch.setenv("HY4_DENSE_FP16_STORAGE", "0")
        assert profile.speed_setting("HY4_DENSE_FP16_STORAGE", speed_config) is False


def test_fast2_refuses_missing_native_operator(monkeypatch, speed_config):
    from vllm.model_executor.layers.quantization import hy4_awq_union_sm70

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast2")

    def load(operator):
        if operator == "boundary_router_input":
            raise RuntimeError(f"missing {operator}")

    monkeypatch.setattr(hy4_awq_union_sm70, "load_hy4_sm70_operator", load)
    with pytest.raises(RuntimeError, match="cannot enable HY4_BOUNDARY_ROUTER_INPUT"):
        profile.initialize_profile(speed_config)


def test_fast3_uses_fast2_defaults_and_allows_startup_overrides(
    monkeypatch, speed_config
):
    for name in profile._FAST3_ADDITIONS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast2")
    for name in profile._FAST3_ADDITIONS:
        expected = 128 if name.endswith("_GROUP_SIZE") else False
        assert profile.speed_setting(name, speed_config) == expected

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast3")
    for name, value in (
        profile._FAST_DEFAULTS | profile._FAST2_ADDITIONS | profile._FAST3_ADDITIONS
    ).items():
        assert profile.speed_setting(name, speed_config) == value

    monkeypatch.setenv("HY4_W8_SHARED_GROUP_SIZE", "32")
    assert profile.speed_setting("HY4_W8_SHARED_GROUP_SIZE", speed_config) == 32
    monkeypatch.setenv("VLLM_HY4_EXPERT_UNION_FOLD", "0")
    assert profile.speed_setting("VLLM_HY4_EXPERT_UNION_FOLD", speed_config) is False
    monkeypatch.setenv("HY4_W8_SHARED", "0")
    assert profile.speed_setting("HY4_W8_SHARED", speed_config) is False

    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting("HY4_F7_HC_BRIDGE", speed_config) is False


def test_w8_checks_the_resolved_startup_switch(monkeypatch, speed_config):
    from vllm.config import vllm as runtime_config
    from vllm.models.hy_v4_flash.sm70 import w8

    monkeypatch.setattr(
        runtime_config, "get_current_vllm_config_or_none", lambda: speed_config
    )
    monkeypatch.delenv("HY4_W8_SHARED", raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast2")
    assert w8.enabled("SHARED") is False
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast3")
    assert w8.enabled("SHARED") is True
    monkeypatch.setenv("HY4_W8_SHARED", "0")
    assert w8.enabled("SHARED") is False


@pytest.mark.parametrize("selection", ["fast3", "fast4", *profile._FAST4_ADDITIONS])
def test_fast4_switch_matrix(monkeypatch, speed_config, selection):
    for key in profile._FAST3_DEFAULTS | profile._FAST4_ADDITIONS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast4" if selection == "fast4" else "fast3")
    if selection in profile._FAST4_ADDITIONS:
        monkeypatch.setenv(selection, "1")
    for key, value in profile._FAST3_DEFAULTS.items():
        assert profile.speed_setting(key, speed_config) == value
    for key in profile._FAST4_ADDITIONS:
        assert profile.speed_setting(key, speed_config) is (
            selection == "fast4" or key == selection
        )


def test_fast4_honors_overrides_and_draft_bridge_exclusion(monkeypatch, speed_config):
    monkeypatch.delenv("HY4_NODES_BRIDGE_RMS", raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast4")
    monkeypatch.setenv("HY4_ATTN3_W8_MERGE", "0")
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting("HY4_ATTN3_W8_MERGE", speed_config) is False
    assert profile.speed_setting("HY4_ATTN3_FP8_PRMT", speed_config) is True
    assert profile.speed_setting("HY4_NODES_BRIDGE_RMS", speed_config) is False
    assert profile.speed_setting("HY4_NODES_ENGRAM", speed_config) is True


@pytest.mark.parametrize("selection", ["fast4", "fast5", *profile._FAST5_ADDITIONS])
def test_fast5_switch_matrix(monkeypatch, speed_config, selection):
    for key in profile._FAST4_DEFAULTS | profile._FAST5_ADDITIONS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast5" if selection == "fast5" else "fast4")
    if selection in profile._FAST5_ADDITIONS:
        monkeypatch.setenv(selection, "1")
    for key, value in profile._FAST4_DEFAULTS.items():
        assert profile.speed_setting(key, speed_config) == value
    for key in profile._FAST5_ADDITIONS:
        assert profile.speed_setting(key, speed_config) is (
            selection == "fast5" or key == selection
        )


def test_fast5_overrides_and_prerequisites(monkeypatch, speed_config):
    for key in profile._FAST5_ADDITIONS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast5")
    monkeypatch.setenv("VLLM_HY4_EXPERT_UNION_FOLD", "0")
    assert profile.speed_setting("HY4_EXPERTS3_RING", speed_config) is False
    monkeypatch.setenv("HY4_PREP_SPLIT_W8", "0")
    assert profile.speed_setting("HY4_PREP_SPLIT_W8", speed_config) is False
    monkeypatch.delenv("HY4_PREP_SPLIT_W8")
    speed_config.model_config.uses_mrope = True
    assert profile.speed_setting("HY4_PREP_SPLIT_W8", speed_config) is False


def test_fast6_bridge_default_and_override(monkeypatch, speed_config):
    switch = "HY4_BRIDGE_SPLIT_BIG_MIX"
    for key in profile._PROFILE_DEFAULTS["fast6"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast5")
    assert profile.speed_setting(switch, speed_config) is False
    monkeypatch.setenv(switch, "1")
    assert profile.speed_setting(switch, speed_config) is True
    monkeypatch.delenv(switch)

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast6")
    for key, value in profile._FAST5_DEFAULTS.items():
        assert profile.speed_setting(key, speed_config) == value
    assert profile.speed_setting(switch, speed_config) is True

    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting(switch, speed_config) is False
    speed_config.speculative_config = None

    monkeypatch.setenv("HY4_F7_HC_BRIDGE", "0")
    assert profile.speed_setting(switch, speed_config) is False
    monkeypatch.delenv("HY4_F7_HC_BRIDGE")
    monkeypatch.setenv(switch, "0")
    assert profile.speed_setting(switch, speed_config) is False


def test_fast7_experts4_default_and_override(monkeypatch, speed_config):
    switch = "HY4_EXPERTS4_SHARED"
    for key in profile._PROFILE_DEFAULTS["fast7"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast6")
    assert profile.speed_setting(switch, speed_config) is False
    monkeypatch.setenv(switch, "1")
    assert profile.speed_setting(switch, speed_config) is True
    monkeypatch.delenv(switch)

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast7")
    for key, value in profile._FAST6_DEFAULTS.items():
        assert profile.speed_setting(key, speed_config) == value
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting(switch, speed_config) is True

    for key, value in (
        ("VLLM_HY4_EXPERT_UNION_FOLD", "0"),
        ("HY4_EXPERTS3_RING", "0"),
        ("HY4_W8_SHARED", "0"),
        ("HY4_W8_SHARED_GROUP_SIZE", "128"),
        (switch, "0"),
    ):
        with monkeypatch.context() as override:
            override.setenv(key, value)
            assert profile.speed_setting(switch, speed_config) is False


def test_fast7_head8_default_and_override(monkeypatch, speed_config):
    switch = "HY4_W8_HEAD"
    monkeypatch.delenv(switch, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast6")
    assert profile.speed_setting(switch, speed_config) is False

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast7")
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting(switch, speed_config) is True
    with monkeypatch.context() as override:
        override.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        assert profile.speed_setting(switch, speed_config) is False
    monkeypatch.setenv(switch, "0")
    assert profile.speed_setting(switch, speed_config) is False
    assert profile.speed_setting("HY4_EXPERTS4_SHARED", speed_config) is True


def test_fast8_experts4b_default_and_override(monkeypatch, speed_config):
    switch = "HY4_EXPERTS4B_VERIFY"
    for key in profile._PROFILE_DEFAULTS["fast8"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("HY4_W8_SHARED_GROUP_SIZE", raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast7")
    assert profile.speed_setting(switch, speed_config) is False

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast8")
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting(switch, speed_config) is True
    for key, value in (
        ("HY4_EXPERTS4_SHARED", "0"),
        ("HY4_EXPERTS3_RING", "0"),
        ("HY4_W8_SHARED_GROUP_SIZE", "128"),
        (switch, "0"),
    ):
        with monkeypatch.context() as override:
            override.setenv(key, value)
            assert profile.speed_setting(switch, speed_config) is False


def test_fast9_swa_tensor_default_and_override(monkeypatch, speed_config):
    switch = "HY4_ATTN4B_SWA_TENSOR"
    for key in profile._PROFILE_DEFAULTS["fast9"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast8")
    assert profile.speed_setting(switch, speed_config) is False
    monkeypatch.setenv(switch, "1")
    assert profile.speed_setting(switch, speed_config) is True
    monkeypatch.delenv(switch)

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast9")
    assert profile.speed_setting(switch, speed_config) is True
    assert profile.speed_setting("HY4_EXPERTS4B_VERIFY", speed_config) is True
    for key in ("HY4_F7_ATTN_OUT", "HY4_ATTN3_FP8_PRMT", switch):
        with monkeypatch.context() as override:
            override.setenv(key, "0")
            assert profile.speed_setting(switch, speed_config) is False


def test_fast10_proposal_head_default_and_override(monkeypatch, speed_config):
    head = "VLLM_HY4_DFLY_HEAD"
    monkeypatch.delenv(head, raising=False)
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast9")
    assert profile.speed_setting(head, speed_config) == "fp32"

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast10")
    assert envs.VLLM_HY4_PROFILE == "fast10"
    assert profile.speed_setting(head, speed_config) == "int8"
    for mode in ("fp16", "fp32"):
        monkeypatch.setenv(head, mode)
        assert profile.speed_setting(head, speed_config) == mode


def test_fast10_confidence_defaults_and_overrides(monkeypatch, speed_config):
    mode = "VLLM_HY4_DFLY_CONFIDENCE_MODE"
    threshold = "VLLM_HY4_DFLY_CONFIDENCE_THRESHOLD"
    monkeypatch.delenv(mode, raising=False)
    monkeypatch.delenv(threshold, raising=False)
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast9")
    assert profile.speed_setting(mode, speed_config) == "off"
    assert profile.speed_setting(threshold, speed_config) == 0.5

    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast10")
    assert profile.speed_setting(mode, speed_config) == "variable"
    assert profile.speed_setting(threshold, speed_config) == 0.45
    monkeypatch.setenv(mode, "off")
    monkeypatch.setenv(threshold, "0.6")
    assert profile.speed_setting(mode, speed_config) == "off"
    assert profile.speed_setting(threshold, speed_config) == 0.6


@pytest.mark.parametrize("name,enabled,disabled", [
    ("HY4_DREV2_WIDE", True, False),
    ("HY4_DREV2_DEFER_DRAFTS", True, False),
    ("VLLM_V41_INLINE_DRAFTS", "1", "0"),
])
def test_fast11_dfly_defaults(monkeypatch, speed_config, name, enabled, disabled):
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast11")
    assert profile.speed_setting(name, speed_config) == disabled
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting(name, speed_config) == enabled
    monkeypatch.setenv(name, "0")
    assert profile.speed_setting(name, speed_config) == disabled


def test_fast13_switches(monkeypatch, speed_config):
    idx = "HY4_IDX_TC"
    late = "HY4_DCONF2_LATE_WIDTH"
    for name in (idx, late):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast11")
    assert not profile.speed_setting(idx, speed_config)
    assert not profile.speed_setting(late, speed_config)
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast13")
    assert profile.speed_setting(idx, speed_config)
    assert not profile.speed_setting(late, speed_config)
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting(late, speed_config)
    for name in (idx, late):
        monkeypatch.setenv(name, "0")
        assert not profile.speed_setting(name, speed_config)


@pytest.mark.parametrize("name,enabled,disabled", [
    ("HY4_PF_ATTN_TC", True, False),
    ("HY4_PF_ATTN_FIX", 8, 0),
])
def test_fast14_prefill_defaults(monkeypatch, speed_config, name, enabled, disabled):
    monkeypatch.delenv(name, raising=False)
    for base in ("reference", "fast13"):
        monkeypatch.setenv("VLLM_HY4_PROFILE", base)
        assert profile.speed_setting(name, speed_config) == disabled
    monkeypatch.setenv("VLLM_HY4_PROFILE", "fast14")
    assert profile.speed_setting(name, speed_config) == enabled
    assert profile.speed_setting(name) == disabled
    speed_config.speculative_config = SimpleNamespace(method="dfly")
    assert profile.speed_setting(name, speed_config) == enabled
    monkeypatch.setenv(name, "0")
    assert profile.speed_setting(name, speed_config) == disabled
