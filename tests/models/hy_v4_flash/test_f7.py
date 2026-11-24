# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import envs
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.models.hy_v4_flash.nvidia import hc_bridge, moe
from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashModel

F7_WIDTHS = (1, 2, 3, 4, 6, 12, 18, 24)


@pytest.mark.parametrize("rows", (0, *F7_WIDTHS, 25, 32))
def test_f7_hc_admission(monkeypatch, rows):
    conf = HYV4FlashConfig(hidden_size=3072, hc_mult=4)
    x = torch.empty(rows, 3072, dtype=torch.float16)
    monkeypatch.setattr(envs, "HY4_F7_HC_BRIDGE", True)
    assert not hc_bridge.can_bridge(x, conf)
    monkeypatch.setattr(hc_bridge, "triton_ihc_supported", lambda _: True)
    monkeypatch.setattr(
        hc_bridge.current_platform, "is_device_capability", lambda cap: cap == 70
    )
    assert hc_bridge.can_bridge(x, conf) == (rows in F7_WIDTHS)
    monkeypatch.setattr(envs, "HY4_F7_HC_BRIDGE", False)
    assert not hc_bridge.can_bridge(x, conf)


@pytest.mark.parametrize("rows", F7_WIDTHS)
def test_f7_moe_routes_on_original_input_and_reduces_once(monkeypatch, rows):
    x = torch.zeros(rows, 3072, dtype=torch.float16)
    latent = torch.zeros(rows, 1536, dtype=x.dtype)
    shared = torch.zeros(rows, 384, dtype=x.dtype)
    logits = torch.zeros(rows, 512)
    weights = torch.zeros(rows, 16)
    ids = torch.zeros(rows, 16, dtype=torch.int32)
    routed = latent + 1
    partial = x + 2
    calls = []

    def step(name, expected, result):
        def execute(*args):
            assert len(args) == len(expected)
            assert all(a is b for a, b in zip(args, expected))
            calls.append(name)
            return result

        return execute

    layer = moe.HYV4FlashMoE.__new__(moe.HYV4FlashMoE)
    nn.Module.__init__(layer)
    layer._boundary_router_input = False
    layer.gate = step("gate", (x,), (logits, None))
    down_w, up_w, gu_w, sd_w = object(), object(), object(), object()
    layer.routed_expert_down_proj = SimpleNamespace(weight=down_w)
    layer.routed_expert_up_proj = SimpleNamespace(weight=up_w)
    layer.shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=gu_w),
        down_proj=SimpleNamespace(weight=sd_w),
    )
    layer.experts = SimpleNamespace(
        router=SimpleNamespace(
            select_experts=step("select", (latent, logits), (weights, ids))
        )
    )
    layer.experts.quant_method = SimpleNamespace(
        apply=step(
            "grouped", (layer.experts, latent, weights, ids, None, None), routed
        ),
    )
    monkeypatch.setattr(
        torch.ops.hy4_sm70,
        "f7_moe_in",
        step("fused in", (x, down_w, gu_w), (latent, shared)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.hy4_sm70,
        "f7_moe_out",
        step("fused out", (routed, shared, up_w, sd_w), partial),
        raising=False,
    )
    monkeypatch.setattr(
        moe, "tensor_model_parallel_all_reduce", step("AR", (partial,), x)
    )

    assert layer._f7_forward(x) is x
    assert calls == ["gate", "fused in", "select", "grouped", "fused out", "AR"]
    monkeypatch.setattr(envs, "HY4_F7_MOE_FUSE", True)
    assert not layer._f7_supported(x)


def test_f7_hc_finishes_post_before_engram_and_return():
    trace = []
    layer = HYV4FlashModel.__new__(HYV4FlashModel)
    nn.Module.__init__(layer)
    layer.engram_indices = {1: 0}

    class Decoder(nn.Module):
        hc_attn_layer = hc_mlp_layer = SimpleNamespace()

        def __init__(self, index):
            super().__init__()
            self.index = index
            self.hc_attn_layer = self.hc_mlp_layer = SimpleNamespace(
                prepare_input=lambda x: x + 5,
                post=self.post,
            )

        def post(self, previous, x, previous_post):
            trace.append(("post", self.index))
            return x + previous * previous_post

        def engram_layer(self, x, hashes):
            trace.append(("engram", self.index))
            return x + hashes

        def forward_f7_ihc(self, positions, x, previous, previous_post):
            trace.append(("layer", self.index, previous is not None))
            if previous is not None:
                x = self.post(previous, x, previous_post)
            return x, 2, 3

    layer.layers = nn.ModuleList([Decoder(i) for i in range(3)])
    assert layer._forward_f7_ihc_layers(None, 0, [7]) == 30
    assert trace == [
        ("layer", 0, False),
        ("post", 1),
        ("engram", 1),
        ("layer", 1, False),
        ("layer", 2, True),
        ("post", 2),
        ("post", 2),
    ]


@pytest.mark.parametrize("rows", (0, *F7_WIDTHS, 25, 32))
def test_f7_moe_admission(monkeypatch, rows):
    layer = moe.HYV4FlashMoE.__new__(moe.HYV4FlashMoE)
    nn.Module.__init__(layer)
    layer.experts = SimpleNamespace(
        moe_config=SimpleNamespace(
            tp_size=4, dp_size=1, ep_size=1, pcp_size=1, is_sequence_parallel=False
        ),
        runner=SimpleNamespace(
            _fused_output_is_reduced=False, routed_scaling_factor=1.0
        ),
        router=SimpleNamespace(parallel_top16=True),
        sm70_awq_hy4_union=True,
        sm70_awq_hy4_union_min_rows=1,
    )

    def projection(shape):
        return SimpleNamespace(
            weight=torch.empty(shape, dtype=torch.float16, device="meta"),
            quant_method=moe.UnquantizedLinearMethod(),
        )

    layer.routed_expert_down_proj = projection((1536, 3072))
    layer.routed_expert_up_proj = projection((3072, 1536))
    layer.shared_experts = SimpleNamespace(
        gate_up_proj=projection((768, 3072)),
        down_proj=projection((3072, 384)),
    )
    monkeypatch.setattr(envs, "HY4_F7_MOE_FUSE", True)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(
        moe.current_platform, "is_device_capability", lambda cap: cap == 70
    )
    monkeypatch.setattr(
        moe, "communication_policy", lambda: SimpleNamespace(moe_add_allreduce=False)
    )
    x = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float16,
        ndim=2,
        shape=(rows, 3072),
        is_contiguous=lambda: True,
        storage_offset=lambda: 0,
    )
    assert layer._f7_supported(x) == (rows in F7_WIDTHS)
    layer.experts.moe_config.tp_size = 2
    assert not layer._f7_supported(x)
