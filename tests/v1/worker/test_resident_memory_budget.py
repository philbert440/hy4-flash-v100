# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from vllm.v1.worker.memory_budget import GPU_STARTUP_MARGIN_BYTES, available_kv_bytes


@dataclass(frozen=True)
class Snapshot:
    total_memory: int
    free_memory: int


@pytest.mark.parametrize(
    "utilization,expected",
    [(0.95, 1_974_718_260), (0.97, 2_656_169_452)],
)
def test_charge_all_resident_memory(utilization, expected):
    # A0 diag-memory2 and the independent fix3 baseline agreed byte for byte:
    # 27,800,168,448 allocated + 788,207,616 stranded + 752,484,352
    # outside Torch (including 664,403,968 present before the first snapshot).
    resident = Snapshot(34_072_559_616, 4_731_699_200)
    graph = 473_956_352  # before/after real graph capture, old startup
    transient = 422_110_208  # recorded forward peak, before logprob profiling
    assert available_kv_bytes(
        resident, utilization, graph, transient, GPU_STARTUP_MARGIN_BYTES
    ) == expected


def test_impossible_budget_is_negative():
    assert available_kv_bytes(Snapshot(1000, 100), 0.95, 15, 30, 20) == -15


@pytest.mark.parametrize(
    "free,utilization,expected",
    [
        (5_467_799_552, 0.95, 2_684_498_740),
        (5_467_799_552, 0.97, 3_365_949_932),
        (5_444_730_880, 0.95, 2_661_430_068),
        (5_444_730_880, 0.97, 3_342_881_260),
    ],
)
def test_fix3_probe_snapshots(free, utilization, expected):
    # Synchronized post-profile snapshots from the FP16 and packed FP8
    # TP4 probes. Charge the larger of the 2,048-token forward and the tiled
    # prompt-logprob peak. Graph profiling released 2 MiB in both modes.
    resident = Snapshot(34_072_559_616, free)
    assert available_kv_bytes(
        resident, utilization, 2_097_152, 920_289_280, GPU_STARTUP_MARGIN_BYTES
    ) == expected


@pytest.mark.parametrize("utilization", [0.99, 0.996, 1.0])
def test_startup_check_allows_capture_to_consume_margin(monkeypatch, utilization):
    from math import ceil
    from types import SimpleNamespace

    from vllm.v1.worker import gpu_worker

    total = 34_072_559_616
    requested = ceil(total * utilization)
    transient = 920_289_280
    minimum = total - requested + transient
    idle = SimpleNamespace(
        total_memory=total, free_memory=minimum,
        torch_allocated_memory=0, torch_memory=0, non_torch_memory=0,
    )
    worker = SimpleNamespace(
        device=None, requested_memory=requested, peak_activation_memory=transient,
    )
    monkeypatch.setattr(gpu_worker, "MemorySnapshot", lambda **_: idle)
    monkeypatch.setattr(gpu_worker.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(gpu_worker.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(gpu_worker.torch.cuda, "memory_snapshot", lambda: [])

    gpu_worker.Worker._check_resident_kv_budget(worker)
    idle.free_memory -= 1
    with pytest.raises(RuntimeError, match="Lower gpu_memory_utilization"):
        gpu_worker.Worker._check_resident_kv_budget(worker)


def test_resident_budget_counts_initial_occupancy_inside_utilization():
    from types import SimpleNamespace

    from vllm.v1.worker.utils import request_memory

    initial = SimpleNamespace(total_memory=1000, free_memory=980, device_="cuda:0")
    config = SimpleNamespace(gpu_memory_utilization=1.0)
    with pytest.raises(ValueError, match="less than desired GPU memory utilization"):
        request_memory(initial, config)
    assert request_memory(initial, config, validate_free=False) == 1000
