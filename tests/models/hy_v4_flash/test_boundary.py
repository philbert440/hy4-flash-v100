# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
    load_hy4_sm70_operator,
)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_hy4_boundary_router_graph(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA required")

    load_hy4_sm70_operator("boundary_router_input")
    rng = torch.Generator(device="cuda").manual_seed(73)
    x = torch.randn((rows, 3072), device="cuda", dtype=torch.float16, generator=rng)
    weight = torch.randn((512, 3072), device="cuda", dtype=torch.float32, generator=rng)
    op = torch.ops.hy4_sm70.boundary_router_input
    op(x, weight)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = op(x, weight)
    x.add_(0.125)
    graph.replay()

    expected = (x.double() @ weight.double().T).float()
    torch.testing.assert_close(logits, expected, rtol=1e-6, atol=2e-4)
