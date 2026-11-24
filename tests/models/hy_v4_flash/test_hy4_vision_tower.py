# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shard convention and shapes for the Hy4 tower. Real weights stay out of pytest."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.models.hy_v4_flash.nvidia.vision import (
    Hy4VisionTransformer,
    column_shard,
    current_tensor_parallel,
    row_shard,
)

pytestmark = pytest.mark.skip_global_cleanup


def _config(**overrides):
    values = dict(
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=16,
        num_hidden_layers=1,
        patch_size=16,
        spatial_merge_size=2,
        num_channels=3,
        out_hidden_size=8,
        max_image_size=64,
        layer_norm_eps=1e-5,
        rms_norm_eps=1e-5,
        interpolate_mode="bilinear",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _tower(**overrides):
    dtype = overrides.pop("dtype", torch.float32)
    tp_size = overrides.pop("tp_size", 1)
    tp_rank = overrides.pop("tp_rank", 0)
    return Hy4VisionTransformer(
        _config(**overrides), dtype=dtype, tp_size=tp_size, tp_rank=tp_rank
    )


def test_column_and_row_slices_rebuild_the_linear():
    weight = torch.randn(32, 32)
    bias = torch.randn(32)
    inputs = torch.randn(2, 32)
    columns = [column_shard(weight, 4, rank) for rank in range(4)]
    column_bias = [column_shard(bias, 4, rank) for rank in range(4)]
    assert torch.equal(torch.cat(columns, dim=0), weight)
    partial = [
        F.linear(inputs, columns[rank], column_bias[rank]) for rank in range(4)
    ]
    assert torch.equal(torch.cat(partial, dim=-1), F.linear(inputs, weight, bias))

    rows = [row_shard(weight, 4, rank) for rank in range(4)]
    assert torch.equal(torch.cat(rows, dim=1), weight)
    wide = inputs.double()
    chunks = wide.chunk(4, dim=-1)
    reduced = sum(
        F.linear(chunks[rank], rows[rank].double(), None) for rank in range(4)
    )
    assert torch.allclose(
        reduced + bias.double(),
        F.linear(wide, weight.double(), bias.double()),
        rtol=0,
        atol=1e-8,
    )


def test_rank_zero_loads_its_slice_and_keeps_the_row_bias_whole():
    full = _tower()
    rank0 = _tower(tp_size=4, tp_rank=0)
    rank0.load_weights(full.state_dict().items())
    full_attn = full.layers[0].self_attn
    shard_attn = rank0.layers[0].self_attn
    assert torch.equal(
        shard_attn.q_proj.weight, column_shard(full_attn.q_proj.weight, 4, 0)
    )
    assert torch.equal(
        shard_attn.q_proj.bias, column_shard(full_attn.q_proj.bias, 4, 0)
    )
    assert torch.equal(shard_attn.o_proj.weight, row_shard(full_attn.o_proj.weight, 4, 0))
    assert shard_attn.o_proj.bias is None
    assert torch.equal(shard_attn.o_proj.row_bias, full_attn.o_proj.bias)
    assert torch.equal(
        rank0.perceive.proj[0].weight, full.perceive.proj[0].weight
    )
    assert torch.equal(
        rank0.embeddings.position_embedding.weight,
        full.embeddings.position_embedding.weight,
    )
    assert rank0.rank_parameter_bytes() < full.rank_parameter_bytes()


def test_one_image_forward_includes_the_newline_slot():
    tower = _tower()
    pixels = torch.randn(4, 3 * 16 * 16)
    features = tower(pixels, torch.tensor([[1, 2, 2]]))
    assert features.shape == (2, 8)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_position_table_stays_fp32_when_the_tower_is_fp16():
    tower = _tower(dtype=torch.float16)
    assert tower.embeddings.position_embedding.weight.dtype == torch.float32
    assert tower.layers[0].self_attn.q_proj.weight.dtype == torch.float16
    features = tower(torch.randn(4, 3 * 16 * 16), torch.tensor([[1, 2, 2]]))
    assert features.dtype == torch.float16
    assert features.shape == (2, 8)


def test_uninitialized_tensor_parallel_is_one_rank():
    assert current_tensor_parallel() == (1, 0)
