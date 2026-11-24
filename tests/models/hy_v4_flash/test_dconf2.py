# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCONF2: the late-width schedule equals fast11's confidence-cut schedule."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.hy_v4_flash.sm70.dfly_proposer import DFlyProposer, DraftGraph
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass
class Schedule:
    num_scheduled_tokens: dict
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict
    mixed_decode_tokens: int = 0
    other: list = field(default_factory=list)


class Event:
    waits = 0

    def record(self):
        pass

    def synchronize(self):
        Event.waits += 1


def _proposer(mode, scores, late, monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", Event)
    proposer = DFlyProposer(
        SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)), "cpu"
    )
    proposer.settings = SimpleNamespace(target_layers=())
    proposer.graph_only = False
    proposer.gate = None
    proposer.log_confidence = False
    proposer.confidence_mode = mode
    proposer.confidence_logit_threshold = 0.0
    proposer.late_width = late
    replays = []
    proposed = torch.arange(10, 15).reshape(1, 5)
    proposer.draft_graphs[1] = DraftGraph(
        SimpleNamespace(stage=lambda *args: None),
        SimpleNamespace(replay=lambda: replays.append("score")),
        proposed,
        torch.tensor([scores]),
        tuple(SimpleNamespace(replay=lambda i=i: replays.append(i)) for i in range(5)),
    )
    proposer.prepare_target([0, 1], [43], [False], [0])
    drafts = proposer.propose([[1]], torch.zeros((1, 8)), defer=True)
    return proposer, drafts, replays


def _runner(proposer):
    runner = SimpleNamespace(
        drafter=proposer,
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(req_ids=["a"]),
        input_ids=SimpleNamespace(gpu=torch.zeros(6, dtype=torch.int32)),
        _hy4_dfly_deferred=None,
    )
    if proposer.deferred is not None:
        rows, proposed = proposer.deferred
        proposer.deferred = None
        runner._hy4_dfly_deferred = ({"a": index for index in rows}, proposed)
    return runner


def _engine_schedule(drafts):
    """The scheduler's step for the request's drafts (no truncation)."""
    ids = drafts[0]
    return Schedule({"a": 1 + len(ids)}, 1 + len(ids), {"a": list(ids)} if ids else {})


def _verify(runner, schedule):
    if getattr(runner.drafter, "late", None) is not None:
        schedule = GPUModelRunner._hy4_dconf2_resolve_width(runner, schedule)
    count = len(schedule.scheduled_spec_decode_tokens.get("a", ()))
    if runner._hy4_dfly_deferred is not None:
        GPUModelRunner._hy4_write_deferred_drafts(
            runner, schedule, 1, np.array([count + 1])
        )
    return schedule, runner.input_ids.gpu.tolist()


@pytest.mark.parametrize("mode", ["variable", "skip"])
@pytest.mark.parametrize("cut", range(6))
def test_late_width_schedules_fast11_cut(monkeypatch, mode, cut):
    scores = [1.0] * cut + [-1.0] * (5 - cut)
    base, base_drafts, base_replays = _proposer(mode, scores, False, monkeypatch)
    late, late_drafts, late_replays = _proposer(mode, scores, True, monkeypatch)
    # The late path returns every placeholder and runs every picker.
    assert late_drafts == [[0] * 5]
    assert late_replays == ["score", 0, 1, 2, 3, 4]
    assert base_replays == ["score", *range(len(base_drafts[0]))]
    base_runner, late_runner = _runner(base), _runner(late)
    base_schedule, base_ids = _verify(base_runner, _engine_schedule(base_drafts))
    late_schedule, late_ids = _verify(late_runner, _engine_schedule(late_drafts))
    assert late_ids == base_ids
    assert late_schedule.num_scheduled_tokens == base_schedule.num_scheduled_tokens
    assert late_schedule.total_num_scheduled_tokens == base_schedule.total_num_scheduled_tokens
    assert late_schedule.scheduled_spec_decode_tokens.keys() == (
        base_schedule.scheduled_spec_decode_tokens.keys()
    )
    assert late.late is None and late_runner._hy4_dfly_deferred is None
    unused = 5 - len(base_drafts[0])
    assert getattr(late_runner, "_hy4_dconf2_unused", None) == (
        {"a": unused} if unused else None
    )


def test_late_width_keeps_shorter_scheduler_truncation(monkeypatch):
    late, drafts, _ = _proposer("variable", [1.0] * 4 + [-1.0], True, monkeypatch)
    runner = _runner(late)
    # Near the length limit the scheduler kept two of the five placeholders.
    schedule, ids = _verify(runner, Schedule({"a": 3}, 3, {"a": [0, 0]}))
    assert schedule.scheduled_spec_decode_tokens == {"a": [0, 0]}
    assert ids[:3] == [0, 10, 11]


def test_late_width_dropped_drafts_skip_the_wait(monkeypatch):
    late, _, _ = _proposer("variable", [1.0] * 5, True, monkeypatch)
    runner = _runner(late)
    Event.waits = 0
    schedule = Schedule({"a": 1, "b": 40}, 41, {})
    assert GPUModelRunner._hy4_dconf2_resolve_width(runner, schedule) is schedule
    assert Event.waits == 0 and late.late is None


def test_late_width_needs_one_running_request(monkeypatch):
    proposer, _, _ = _proposer("variable", [1.0] * 2 + [-1.0] * 3, True, monkeypatch)
    assert proposer.late is not None
    proposer.late = proposer.deferred = None
    proposer.prepare_target([0, 1], [43], [False], [0], running=2)
    # Two running requests keep fast11's host read and its fixed five proposals.
    proposer.propose([[1]], torch.zeros((1, 8)), defer=True)
    assert proposer.late is None



def test_late_width_check_without_drafter():
    # Plain servers never set self.drafter; the execute_model entry check must not raise.
    assert GPUModelRunner._hy4_dconf2_late_pending(SimpleNamespace()) is False
    assert GPUModelRunner._hy4_dconf2_late_pending(SimpleNamespace(drafter=None)) is False
    assert GPUModelRunner._hy4_dconf2_late_pending(
        SimpleNamespace(drafter=SimpleNamespace(late=("event", "host")))) is True
