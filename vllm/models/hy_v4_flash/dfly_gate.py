# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection
from dataclasses import dataclass
from math import ceil, isclose

from vllm import envs
from vllm.config import CompilationConfig, CUDAGraphMode

from .dfly_config import DRAFT_TOKENS, QUERY_WIDTH

_PROBE_INTERVAL = 8
_MIN_OBSERVATIONS = 24


def draft_max_running() -> int | None:
    limit = envs.VLLM_HY4_DFLY_GATE_MAX_RUNNING
    if limit < 0:
        raise ValueError("VLLM_HY4_DFLY_GATE_MAX_RUNNING must be nonnegative")
    return limit or None


def graph_draft_sizes(compilation: CompilationConfig | None) -> frozenset[int] | None:
    # Preserve the explicitly selected eager and verify-only draft modes.
    if (
        compilation is None
        or compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
        or envs.VLLM_HY4_DFLY_GRAPHS != "both"
    ):
        return None
    return frozenset(
        rows // QUERY_WIDTH
        for rows in compilation.cudagraph_capture_sizes or ()
        if rows % QUERY_WIDTH == 0
    )


def draft_load_allowed(
    running: int, cap: int | None, graph_sizes: Collection[int] | None
) -> bool:
    # With both graphs enabled, draft only at a captured count within the hard cap.
    return (cap is None or running <= cap) and (
        graph_sizes is None or running in graph_sizes
    )


@dataclass
class Acceptance:
    mean: float | None = None
    observations: int = 0
    paused: bool = False
    skipped: int = 0


class DFlyGate:
    def __init__(self, threshold: float):
        if not 0 <= threshold <= DRAFT_TOKENS:
            raise ValueError("VLLM_HY4_DFLY_GATE_THRESHOLD must be between 0 and 5")
        self.threshold = threshold
        # The measured break-even is about 0.9–1.0 accepted proposals.
        # Recover before the next integer so a steady one-token gain can
        # resume at the default threshold. Integer thresholds admit equality.
        self.resume = threshold + min(0.05, (ceil(threshold) - threshold) / 2)
        self.requests: dict[str, Acceptance] = {}

    def observe(self, request_id: str, accepted: int) -> None:
        state = self.requests.setdefault(request_id, Acceptance())
        # While paused, observations are eight steps apart. Weight a fresh
        # probe more heavily than observations from the preceding regime.
        weight = 0.5 if state.paused else 0.125
        state.mean = (
            accepted
            if state.mean is None
            else (1 - weight) * state.mean + weight * accepted
        )
        state.observations = min(_MIN_OBSERVATIONS, state.observations + 1)
        if state.paused:
            state.paused = state.mean < self.resume and not isclose(
                state.mean, self.resume
            )
        elif state.observations == _MIN_OBSERVATIONS:
            state.paused = state.mean < self.threshold and not isclose(
                state.mean, self.threshold
            )

    def allow(self, request_ids: list[str]) -> bool:
        states = [self.requests.setdefault(req, Acceptance()) for req in request_ids]
        # Keep a due probe ready if a paused peer blocks the batch.
        if any(
            state.paused and state.skipped < _PROBE_INTERVAL for state in states
        ):
            for state in states:
                state.skipped = min(_PROBE_INTERVAL, state.skipped + 1)
            return False
        for state in states:
            state.skipped = 0
        return True

    def forget(self, request_ids) -> None:
        for request_id in request_ids:
            self.requests.pop(request_id, None)
