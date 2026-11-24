# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil
from typing import Protocol


class DeviceMemorySnapshot(Protocol):
    @property
    def total_memory(self) -> int: ...

    @property
    def free_memory(self) -> int: ...


# Reserve for variability between minimal-cache profiling and serving, plus
# allocator rounding and runtime allocations outside the measured forward.
GPU_STARTUP_MARGIN_BYTES = 150 * (1 << 20)


def available_kv_bytes(
    resident: DeviceMemorySnapshot,
    utilization: float,
    graph_bytes: int,
    transient_bytes: int,
    margin_bytes: int,
) -> int:
    """Size the pool from a synchronized, post-empty-cache driver snapshot.

    Pool = ceil(total * utilization) - resident - graph - transient - margin.
    Resident includes device setup, live tensors, allocator segments and all
    non-Torch usage. Graph accounts for bytes missing from that snapshot.
    """
    used = resident.total_memory - resident.free_memory
    return (
        ceil(resident.total_memory * utilization)
        - used
        - graph_bytes
        - transient_bytes
        - margin_bytes
    )
