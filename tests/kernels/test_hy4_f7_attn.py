# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU, Meta, admission, boundary, and fallback tests for F7 Stages 1 & 2."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import envs
from vllm.models.hy_v4_flash.sm70.f7_attn import (
    F7_ATTN_HIDDEN,
    F7_ATTN_KV_RANK,
    F7_ATTN_MAX_ROWS,
    F7_ATTN_Q_RANK,
    F7_ATTN_QK_NOPE,
    F7_ATTN_ROPE,
    F7_ATTN_V_DIM,
    F7_FP8_DS_MLA_BYTES,
    can_use_f7_attn_out,
    can_use_f7_attn_prep,
    f7_rope_phase_scope,
    load_hy4_f7_attn_operator,
)


def test_f7_env_defaults_off():
    # Switches default to False unless explicitly enabled in the environment.
    assert isinstance(envs.HY4_F7_ATTN_PREP, bool)
    assert isinstance(envs.HY4_F7_ATTN_OUT, bool)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 6, 12, 18, 24])
@pytest.mark.parametrize("heads", [8, 16])
def test_f7_meta_registration_stage1_and_stage2(m: int, heads: int):
    load_hy4_f7_attn_operator("rope_phases_sm70")
    load_hy4_f7_attn_operator("attn_qkv_prep_sm70")
    load_hy4_f7_attn_operator("attn_splitk_sm70")
    load_hy4_f7_attn_operator("attn_merge_v_gate_sm70")

    pos = torch.empty((m,), dtype=torch.int64, device="meta")
    inv = torch.empty((F7_ATTN_ROPE // 2,), dtype=torch.float32, device="meta")
    cos = torch.empty((m, F7_ATTN_ROPE // 2), dtype=torch.float32, device="meta")
    sin = torch.empty((m, F7_ATTN_ROPE // 2), dtype=torch.float32, device="meta")
    torch.ops.hy4_f7_attn.rope_phases_sm70(pos, inv, cos, sin)

    x = torch.empty((m, F7_ATTN_HIDDEN), dtype=torch.float16, device="meta")
    q_out = torch.empty((m, heads, F7_ATTN_KV_RANK + F7_ATTN_ROPE), dtype=torch.float16, device="meta")
    kv_c_out = torch.empty((m, F7_ATTN_KV_RANK), dtype=torch.float16, device="meta")
    k_pe_out = torch.empty((m, F7_ATTN_ROPE), dtype=torch.float16, device="meta")
    q_c_out = torch.empty((m, F7_ATTN_Q_RANK), dtype=torch.float16, device="meta")
    gate_sig = torch.empty((m, heads * F7_ATTN_V_DIM), dtype=torch.float16, device="meta")
    torch.ops.hy4_f7_attn.attn_qkv_prep_sm70(
        x,
        None,
        cos,
        sin,
        torch.empty((F7_ATTN_Q_RANK, F7_ATTN_HIDDEN), dtype=torch.float16, device="meta"),
        torch.empty((F7_ATTN_KV_RANK + F7_ATTN_ROPE, F7_ATTN_HIDDEN), dtype=torch.float16, device="meta"),
        torch.empty((heads * F7_ATTN_V_DIM, F7_ATTN_HIDDEN), dtype=torch.float16, device="meta"),
        torch.empty((F7_ATTN_Q_RANK,), dtype=torch.float16, device="meta"),
        torch.empty((F7_ATTN_KV_RANK,), dtype=torch.float16, device="meta"),
        torch.empty((heads * (F7_ATTN_QK_NOPE + F7_ATTN_ROPE), F7_ATTN_Q_RANK), dtype=torch.float16, device="meta"),
        torch.empty((heads, F7_ATTN_KV_RANK, F7_ATTN_QK_NOPE), dtype=torch.float16, device="meta"),
        None,
        torch.empty((F7_ATTN_MAX_ROWS, F7_ATTN_Q_RANK), dtype=torch.float16, device="meta"),
        torch.empty((F7_ATTN_MAX_ROWS, F7_ATTN_KV_RANK + F7_ATTN_ROPE), dtype=torch.float16, device="meta"),
        torch.empty((16, F7_ATTN_MAX_ROWS, F7_ATTN_QK_NOPE), dtype=torch.float16, device="meta"),
        torch.empty((36, 4, F7_ATTN_MAX_ROWS, 16), dtype=torch.float32, device="meta"),
        q_out,
        kv_c_out,
        k_pe_out,
        q_c_out,
        gate_sig,
        1e-5,
    )

    ws_splits = 16 if heads == 16 else 32
    state = torch.empty((F7_ATTN_MAX_ROWS, heads, ws_splits, 2), dtype=torch.float32, device="meta")
    acc = torch.empty((F7_ATTN_MAX_ROWS, heads, ws_splits, F7_ATTN_KV_RANK), dtype=torch.float32, device="meta")
    cache_fp8 = torch.empty((8, 64, F7_FP8_DS_MLA_BYTES), dtype=torch.uint8, device="meta")
    indices = torch.empty((m, 512), dtype=torch.int32, device="meta")
    counts = torch.empty((m,), dtype=torch.int32, device="meta")
    sinks = torch.empty((heads,), dtype=torch.float32, device="meta")
    kv_b = torch.empty((heads * (F7_ATTN_QK_NOPE + F7_ATTN_V_DIM), F7_ATTN_KV_RANK), dtype=torch.float16, device="meta")
    w_uv = kv_b.view(heads, -1, F7_ATTN_KV_RANK)[:, F7_ATTN_QK_NOPE:, :]
    o_latent = torch.empty((32, F7_ATTN_MAX_ROWS, F7_ATTN_KV_RANK), dtype=torch.float16, device="meta")
    out = torch.empty((m, heads * F7_ATTN_V_DIM), dtype=torch.float16, device="meta")

    torch.ops.hy4_f7_attn.attn_splitk_sm70(
        q_out, cache_fp8, None, indices, counts, state, acc, 0.0625, ws_splits, 0
    )
    torch.ops.hy4_f7_attn.attn_merge_v_gate_sm70(
        state, acc, counts, sinks, w_uv, gate_sig, o_latent, out, ws_splits
    )
    if m % 6 == 0:
        draft_kv = torch.empty((m // 6, 6, F7_FP8_DS_MLA_BYTES), dtype=torch.uint8, device="meta")
        torch.ops.hy4_f7_attn.attn_splitk_sm70(
            q_out, cache_fp8, draft_kv, indices[: m // 6], counts, state, acc, 0.0625, ws_splits, 6
        )


def _fake_cuda_tensor(shape, dtype=torch.float16, stride=None, is_cuda=True):
    return SimpleNamespace(
        shape=tuple(shape),
        ndim=len(shape),
        dtype=dtype,
        is_cuda=is_cuda,
        is_contiguous=lambda: True,
        stride=lambda d: (
            stride[d] if stride is not None else (1 if d in (-1, len(shape) - 1) else shape[-1])
        ),
    )


def _fake_module(heads: int, prep: bool, out: bool):
    return SimpleNamespace(
        _f7_attn_prep=prep,
        _f7_attn_out=out,
        num_local_heads=heads,
        qk_nope_head_dim=F7_ATTN_QK_NOPE,
        qk_rope_head_dim=F7_ATTN_ROPE,
        kv_lora_rank=F7_ATTN_KV_RANK,
        v_head_dim=F7_ATTN_V_DIM,
        inv_freq=_fake_cuda_tensor((F7_ATTN_ROPE // 2,), torch.float32),
        q_a_layernorm=SimpleNamespace(
            weight=_fake_cuda_tensor((F7_ATTN_Q_RANK,), torch.float16),
            variance_epsilon=1e-5,
        ),
        kv_a_layernorm=SimpleNamespace(
            weight=_fake_cuda_tensor((F7_ATTN_KV_RANK,), torch.float16),
            variance_epsilon=1e-5,
        ),
        q_a_proj=SimpleNamespace(weight=torch.empty(0)),
        kv_a_proj_with_mqa=SimpleNamespace(weight=torch.empty(0)),
        linear_gate=SimpleNamespace(weight=torch.empty(0)),
        q_b_proj=SimpleNamespace(weight=torch.empty(0)),
        kv_b_proj=SimpleNamespace(weight=torch.empty(0)),
    )


def test_stage1_admission_and_image_fallback():
    # CPU tensors must always reject Stage 1 even when switch is True.
    mod = _fake_module(16, prep=True, out=False)
    x_cpu = torch.empty((4, F7_ATTN_HIDDEN), dtype=torch.float16)
    pos_1d = torch.arange(4, dtype=torch.int64)
    pos_3d = torch.arange(12, dtype=torch.int64).view(3, 4)
    assert not can_use_f7_attn_prep(mod, pos_1d, x_cpu)
    assert not can_use_f7_attn_prep(mod, pos_3d, x_cpu)

    # 3-coordinate image positions must explicitly fall back even with CUDA flags.
    x_fake = _fake_cuda_tensor((4, F7_ATTN_HIDDEN), torch.float16)
    pos_3d_fake = _fake_cuda_tensor((3, 4), torch.int64)
    assert not can_use_f7_attn_prep(mod, pos_3d_fake, x_fake)

    # M > 24 must explicitly fall back.
    x_25 = _fake_cuda_tensor((25, F7_ATTN_HIDDEN), torch.float16)
    pos_25 = _fake_cuda_tensor((25,), torch.int64)
    assert not can_use_f7_attn_prep(mod, pos_25, x_25)

    # Default off rejects everything.
    mod_off = _fake_module(16, prep=False, out=False)
    assert not can_use_f7_attn_prep(mod_off, _fake_cuda_tensor((1,), torch.int64), _fake_cuda_tensor((1, F7_ATTN_HIDDEN)))


def test_stage2_admission_dcp_and_cache_fallbacks():
    mod = _fake_module(8, prep=False, out=True)
    for m in (1, 2, 3, 4, 6, 12, 18, 24):
        q = _fake_cuda_tensor((m, 8, 576), torch.float16, stride=(8 * 576, 576, 1))
        cache_fp8 = _fake_cuda_tensor((4, 64, F7_FP8_DS_MLA_BYTES), torch.uint8)
        cache_fp16 = _fake_cuda_tensor((4, 64, 576), torch.float16)
        indices = _fake_cuda_tensor((m, 512), torch.int32)
        counts = _fake_cuda_tensor((m,), torch.int32)
        sinks = _fake_cuda_tensor((8,), torch.float32)
        w_uv = _fake_cuda_tensor((8, 256, 512), torch.float16, stride=(448 * 512, 512, 1))

        assert can_use_f7_attn_out(mod, q, cache_fp8, indices, counts, sinks, None, w_uv)
        assert can_use_f7_attn_out(mod, q, cache_fp16, indices, counts, sinks, None, w_uv)

        # Explicit fallback for four cache shards (DCP: lse is not None or sinks is None).
        lse = _fake_cuda_tensor((m, 32), torch.float32)
        assert not can_use_f7_attn_out(mod, q, cache_fp8, indices, counts, None, lse, w_uv)
        assert not can_use_f7_attn_out(mod, q, cache_fp8, indices, counts, sinks, lse, w_uv)

        # DFly draft_kv admission at multiples of 6.
        if m % 6 == 0:
            draft_kv = _fake_cuda_tensor((m // 6, 6, F7_FP8_DS_MLA_BYTES), torch.uint8)
            assert can_use_f7_attn_out(
                mod, q, cache_fp8, indices, counts, sinks, None, w_uv, draft_kv=draft_kv
            )

    # M = 25 falls back.
    q25 = _fake_cuda_tensor((25, 8, 576), torch.float16, stride=(8 * 576, 576, 1))
    assert not can_use_f7_attn_out(
        mod,
        q25,
        _fake_cuda_tensor((4, 64, F7_FP8_DS_MLA_BYTES), torch.uint8),
        _fake_cuda_tensor((25, 512), torch.int32),
        _fake_cuda_tensor((25,), torch.int32),
        _fake_cuda_tensor((8,), torch.float32),
        None,
        _fake_cuda_tensor((8, 256, 512), torch.float16, stride=(448 * 512, 512, 1)),
    )

    # Default off rejects Stage 2.
    mod_off = _fake_module(8, prep=False, out=False)
    assert not can_use_f7_attn_out(
        mod_off,
        _fake_cuda_tensor((1, 8, 576), torch.float16, stride=(8 * 576, 576, 1)),
        _fake_cuda_tensor((4, 64, F7_FP8_DS_MLA_BYTES), torch.uint8),
        _fake_cuda_tensor((1, 512), torch.int32),
        _fake_cuda_tensor((1,), torch.int32),
        _fake_cuda_tensor((8,), torch.float32),
        None,
        _fake_cuda_tensor((8, 256, 512), torch.float16, stride=(448 * 512, 512, 1)),
    )


def test_f7_rope_phase_scope_isolation():
    with f7_rope_phase_scope() as cache:
        cache["k"] = "v"
        assert cache["k"] == "v"
    # Nested/subsequent scopes start fresh.
    with f7_rope_phase_scope() as cache2:
        assert "k" not in cache2


def test_f7_tensor_version_inference_mode():
    from vllm.models.hy_v4_flash.sm70.f7_attn import _tensor_version

    normal = torch.arange(6, dtype=torch.int64)
    assert isinstance(_tensor_version(normal), int)
    with torch.inference_mode():
        inf = torch.arange(6, dtype=torch.int64)
        assert inf.is_inference()
        assert _tensor_version(inf) is None
