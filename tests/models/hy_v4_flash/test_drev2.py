# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DREV2 switches: deferred DFly draft ids and the 8-row boundary cap."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.hy_v4_flash.nvidia.boundary_ar import small_rows
from vllm.models.hy_v4_flash.sm70.dfly_proposer import DFlyProposer, DraftGraph
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@pytest.mark.parametrize(("wide", "norm", "rows", "exact_rows"),
                         [("0", "0", 4, 4), ("1", "0", 8, 8), ("0", "1", 4, 8), ("1", "1", 8, 8)])
def test_small_rows_cap(monkeypatch, wide, norm, rows, exact_rows):
    monkeypatch.setenv("HY4_DREV2_WIDE", wide)
    monkeypatch.setenv("HY4_DREV2_WIDE_NORM", norm)
    # Eager batches (prefill chunks) never widen.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert small_rows() == small_rows(exact=True) == 4
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert small_rows() == rows
    assert small_rows(exact=True) == exact_rows


def _proposer(proposed):
    proposer = DFlyProposer(
        SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)), "cpu"
    )
    proposer.settings = SimpleNamespace(target_layers=())
    proposer.graph_only = False
    proposer.gate = None
    proposer.log_confidence = False
    replays = []
    inputs = SimpleNamespace(stage=lambda *args: None)
    graph = SimpleNamespace(replay=lambda: replays.append(1))
    proposer.draft_graphs = {len(proposed): DraftGraph(inputs, graph, proposed)}
    return proposer, replays


@pytest.mark.parametrize("defer", [False, True])
def test_deferred_propose_returns_placeholders(defer):
    proposed = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    proposer, replays = _proposer(proposed)
    # Row 0 kept two proposals; row 1 follows a plain decode step.
    proposer.prepare_target([0, 6, 7], [43, 70], [False, False], [5, 0])
    drafts = proposer.propose([[1, 2, 3], [4]], torch.zeros((7, 8)), defer=defer)
    assert replays == [1]
    if defer:
        assert drafts == [[0] * 5, [0] * 5]
        rows, tensor = proposer.deferred
        assert rows == (0, 1) and tensor is proposed
    else:
        assert drafts == proposed.tolist()
        assert proposer.deferred is None


@pytest.mark.parametrize("width", range(6))
def test_deferred_confidence_prefix(width):
    proposed = torch.arange(10, 15).reshape(1, 5)
    proposer, replays = _proposer(proposed)
    proposer.confidence_mode = "variable"
    proposer.confidence_logit_threshold = 0.0
    graph = proposer.draft_graphs[1]
    proposer.draft_graphs[1] = DraftGraph(
        graph.inputs, graph.graph, proposed,
        torch.tensor([[1.0] * width + [-1.0] * (5 - width)]),
        tuple(SimpleNamespace(replay=lambda: replays.append(2)) for _ in range(5)),
    )
    proposer.prepare_target([0, 1], [43], [False], [0])
    assert proposer.propose([[1]], torch.zeros((1, 8)), defer=True) == [[0] * width]
    assert replays == [1] + [2] * width
    if width:
        rows, tensor = proposer.deferred
        assert rows == (0,) and tensor is proposed
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_ids=["a"]),
            input_ids=SimpleNamespace(gpu=torch.zeros(6, dtype=torch.int32)),
            _hy4_dfly_deferred=({"a": 0}, tensor),
        )
        output = SimpleNamespace(scheduled_spec_decode_tokens={"a": [0] * width})
        GPUModelRunner._hy4_write_deferred_drafts(
            runner, output, 1, np.array([width + 1])
        )
        assert runner.input_ids.gpu.tolist() == [0] + list(range(10, 10 + width)) + [0] * (5 - width)
        assert runner._hy4_dfly_deferred is None
    else:
        assert proposer.deferred is None


def test_deferred_drafts_land_on_scheduled_draft_positions():
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(req_ids=["a", "b", "c"]),
        input_ids=SimpleNamespace(gpu=torch.arange(16, dtype=torch.int32)),
        # Rows of the previous draft graph: c drafted in row 0, a in row 1.
        _hy4_dfly_deferred=(
            {"a": 1, "c": 0},
            torch.tensor([[100, 101, 102, 103, 104], [200, 201, 202, 203, 204]]),
        ),
    )
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={
        # a: 1 + 5 tokens; b: real host drafts; c: the scheduler kept three drafts.
        "a": [0] * 5, "b": [7, 8], "c": [0] * 3,
    })
    GPUModelRunner._hy4_write_deferred_drafts(
        runner, scheduler_output, 3, np.array([6, 9, 13])
    )
    expected = torch.arange(16, dtype=torch.int32)
    expected[1:6] = torch.tensor([200, 201, 202, 203, 204])
    expected[10:13] = torch.tensor([100, 101, 102])
    assert torch.equal(runner.input_ids.gpu, expected)
    assert runner._hy4_dfly_deferred is None


def test_deferred_drafts_reject_overlong_schedule():
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(req_ids=["a"]),
        input_ids=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
        _hy4_dfly_deferred=({"a": 0}, torch.zeros((1, 2), dtype=torch.long)),
    )
    with pytest.raises(RuntimeError, match="more drafts"):
        GPUModelRunner._hy4_write_deferred_drafts(
            runner, SimpleNamespace(scheduled_spec_decode_tokens={"a": [0] * 3}),
            1, np.array([4]),
        )


@pytest.mark.parametrize("blocker", [None, "penalty", "bad_words", "budget", "logitsprocs",
                                     "structured", "async", "off"])
def test_defer_only_without_host_draft_consumers(blocker):
    holder = SimpleNamespace(has_tracked_requests=lambda: blocker == "budget")
    runner = SimpleNamespace(
        use_async_scheduling=blocker == "async",
        drafter=SimpleNamespace(defer_drafts=blocker != "off"),
        model_config=SimpleNamespace(
            logits_processors=("x",) if blocker == "logitsprocs" else None),
        input_batch=SimpleNamespace(
            thinking_budget_state_holder=holder,
            no_penalties=blocker != "penalty",
            bad_words_token_ids={0: [[1]]} if blocker == "bad_words" else {},
        ),
    )
    out = SimpleNamespace(has_structured_output_requests=blocker == "structured")
    assert GPUModelRunner._hy4_dfly_can_defer(runner, out) is (blocker is None)
