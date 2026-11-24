# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4_IDX_TC native decode scoring and selection against the Triton/torch path."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="CUDA SM70 required",
)

RATIO, TOPK, PAGE, GROUPS = 4, 512, 64, 8192


def _case(ctx, reqs, width, ppb):
    dev = torch.device("cuda")
    live = (ctx - 1) // RATIO
    blocks = ((live + PAGE - 1) // PAGE + ppb) // ppb + 1
    pool = torch.randn(reqs * blocks * ppb + ppb, PAGE, 128, device=dev).half()
    table = torch.zeros(reqs, GROUPS // PAGE, dtype=torch.int32, device=dev)
    table[:, :blocks] = torch.randperm(reqs * blocks, device=dev).view(reqs, blocks).int()
    rows = reqs * width
    q = torch.randn(rows, 16, 128, device=dev).half()
    w = torch.randn(rows, 16, device=dev) * (16 * 128) ** -0.5
    lengths = torch.tensor(
        [ctx - width + 1 + i % width for i in range(rows)], dtype=torch.int32, device=dev
    )
    slots = torch.arange(rows, device=dev)
    slots[-1] = -1  # a padded graph row
    return q, w, pool, table, lengths, slots


@pytest.mark.parametrize(
    "ctx,reqs,width,ppb", [(4096, 1, 1, 1), (2049, 2, 2, 1), (16384, 4, 1, 2), (8192, 1, 6, 1)]
)
def test_idx_tc_matches_triton_and_topk(ctx, reqs, width, ppb):
    from vllm.models.hy_v4_flash.sm70.idx_tc import load_idx_tc
    from vllm.models.hy_v4_flash.sm70.kernels import index_scores

    load_idx_tc()
    ops = torch.ops.hy4_idx_tc
    q, w, pool, table, lengths, slots = _case(ctx, reqs, width, ppb)
    rows = q.shape[0]
    ref = index_scores(q, w, pool, table, 0, GROUPS, RATIO, ppb, logits=torch.empty(rows, GROUPS,
                       device=q.device), lengths=lengths, slots=slots, group_topk=TOPK,
                       query_width=width)
    got = torch.empty_like(ref)
    ops.scores(q, w, pool, table, got, lengths, slots, RATIO, TOPK, ppb, 1, 0, width, True)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    active = (slots >= 0) & (lengths > TOPK * RATIO)
    expect = ref.topk(TOPK - 1, dim=-1, sorted=False).indices.sort(dim=-1).values
    picked = torch.empty_like(expect)
    ops.select(ref, lengths, slots, picked, RATIO, TOPK, 1, 0)
    assert torch.equal(picked[active], expect[active])
