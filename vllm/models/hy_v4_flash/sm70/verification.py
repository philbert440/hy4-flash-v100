# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reconcile Hy4's raw index tail at the next authoritative forward start."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import CircularBufferSpec

from .cache import FlashCache, FlashMetadata, IndexPoolDecodeWorkspace, QuerySpan

if TYPE_CHECKING:
    from .dfly import DraftHistory


@dataclass
class _PendingTail:
    """The ring reflects prefix. Rows past it belong to one target forward."""

    end: int
    rows: dict[str, torch.Tensor] = field(default_factory=dict)
    complete: bool = False


@dataclass
class _RequestTail:
    position: int
    pending: _PendingTail | None = None
    failed: bool = False


class VerificationForward:
    def __init__(self, pending: dict[QuerySpan, _PendingTail]) -> None:
        self.pending = pending
        self.active = True

    def verifies(self, span: QuerySpan) -> bool:
        return span in self.pending

    def stage_tail(
        self,
        span: QuerySpan,
        layer: str,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        if not self.active:
            raise RuntimeError("Hy4 verification metadata used outside its forward")
        pending = self.pending.get(span)
        if pending is None:
            return False
        if layer in pending.rows:
            raise RuntimeError("Duplicate Hy4 verification tail writer")
        coordinates = positions[:, None] if positions.ndim == 1 else positions.T
        pending.rows[layer] = torch.cat((keys.float(), coordinates.float()), dim=-1)
        return True


@dataclass
class VerificationGraph:
    """Address-stable side outputs of the uniform target forward."""

    layers: tuple[str, ...]
    tails: torch.Tensor
    histories: dict[int, "DraftHistory"]
    width: int
    index_workspace: IndexPoolDecodeWorkspace | None = None
    layer_tails: dict[str, torch.Tensor] = field(init=False)

    def __post_init__(self) -> None:
        self.layer_tails = dict(zip(self.layers, self.tails.unbind(), strict=True))

    def capture(self, layer, cache, slots, counts) -> None:
        history = self.histories[layer]
        assert history.cache is cache
        history.slots.copy_(slots)
        history.counts.copy_(counts)


class VerificationState:
    """Worker-owned pending rows, with no reference to a released ring block."""

    def __init__(self) -> None:
        self._requests: dict[str, _RequestTail] = {}
        self._active = False

    def forget(self, request_ids: Collection[str]) -> None:
        """Drop host state on ownership loss. No cache writes are needed."""
        if self._active:
            raise RuntimeError("Cannot release Hy4 tail ownership during forward")
        for request_id in request_ids:
            self._requests.pop(request_id, None)

    def tracks(self, request_id: str) -> bool:
        return request_id in self._requests

    def has_pending(self, request_id: str) -> bool:
        state = self._requests.get(request_id)
        return state is not None and state.pending is not None

    @contextmanager
    def forward(
        self,
        metadata: Mapping[str, FlashMetadata] | None,
        owners: Mapping[str, FlashCache],
        requests: Sequence[tuple[str, QuerySpan]],
        verify: Collection[str],
        finalized_lengths: Mapping[str, int],
        graph: VerificationGraph | None = None,
        *,
        plain_decode_graph: bool = False,
    ) -> Iterator[None]:
        """Enter one target forward with scheduler-derived live spans.

        Every request carries its actual start, including ordinary rows in a
        mixed batch. Metadata can omit host spans for single-row CUDA decode.
        Scope the metadata in place so aliases cannot become separate writers.
        """
        if self._active:
            raise RuntimeError("Expected a single active Hy4 verification forward")
        spans = tuple(span for _, span in requests)
        request_ids = {request_id for request_id, _ in requests}
        begin = 0
        for span in spans:
            if (
                span.begin != begin
                or span.end <= begin
                or span.prefix < 0
                or span.length - span.prefix != span.end - begin
            ):
                raise ValueError("Invalid Hy4 forward spans")
            begin = span.end
        if len(request_ids) != len(requests) or not request_ids.issuperset(verify):
            raise ValueError("Invalid Hy4 verification request IDs")

        if not metadata and (verify or any(self.has_pending(r) for r in request_ids)):
            raise RuntimeError(
                "Hy4 verification and reconciliation require attention metadata"
            )
        layers = {id(layer): layer for layer in (metadata or {}).values()}.values()
        saved = []
        for layer in layers:
            if layer.verification is not None:
                raise RuntimeError("Hy4 metadata already belongs to a forward")
            if layer.num_actual_tokens != begin or (
                layer.spans != spans
                and not (layer.decode and not layer.spans and len(spans) == begin)
            ):
                raise ValueError("Hy4 metadata disagrees with authoritative spans")
            saved.append((layer, layer.spans, layer.decode))

        tails = {
            name: owner.kv_cache
            for name, owner in owners.items()
            if metadata is not None
            and name in metadata
            and isinstance(owner.spec, CircularBufferSpec)
        }

        if graph is not None and (
            len(verify) != len(requests)
            or graph.layer_tails.keys() != tails.keys()
            or any(span.end - span.begin != graph.width for span in spans)
            or graph.tails.shape[1] != begin
        ):
            raise ValueError("Hy4 verification batch disagrees with graph inputs")

        # Check the entire batch before writing any accepted ring rows. The
        # canonical token table must contain the input at the scheduler cursor.
        # After rejection it ends before the optimistic cursor can be used.
        for request_id, span in requests:
            state = self._requests.get(request_id)
            if state is None:
                continue
            start = span.prefix
            if state.failed:
                raise RuntimeError("Hy4 committed index state requires rebuilding")
            if start < state.position:
                raise RuntimeError("Hy4 forward rewinds past the committed index tail")
            pending = state.pending
            if pending is not None and start >= finalized_lengths[request_id]:
                raise RuntimeError("Hy4 forward start exceeds finalized history")
            if start > state.position and (
                pending is None
                or start > pending.end
                or not pending.complete
                or pending.rows.keys() != tails.keys()
            ):
                raise RuntimeError("Hy4 staged index tail does not cover forward start")

        pending_by_span = {}
        tracked = []
        for req, (request_id, span) in enumerate(requests):
            state = self._requests.get(request_id)
            if state is None:
                if request_id not in verify:
                    continue
                state = self._requests[request_id] = _RequestTail(span.prefix)

            if span.prefix > state.position:
                assert state.pending is not None and metadata is not None
                # If a store fails, the old position may already have been
                # overwritten by a new group's members. Do not retry that ring.
                state.failed = True
                for name, tail in tails.items():
                    ratio = tail.shape[1]
                    # Only the new open group can be read from the raw ring.
                    end = span.prefix
                    start = max(state.position, end // ratio * ratio)
                    if start < end:
                        ring = metadata[name].block_table[req, :1]
                        tail[ring, start % ratio : end % ratio] = state.pending.rows[
                            name
                        ][start - state.position : end - state.position]
                state.failed = False

            state.position = span.prefix
            state.pending = None
            if request_id in verify:
                state.pending = pending_by_span[span] = _PendingTail(span.length)
            tracked.append((state, span))

        step = VerificationForward(pending_by_span)
        self._active = True
        try:
            for layer, _, decode in saved:
                if graph is not None or plain_decode_graph:
                    continue
                layer.spans = spans
                layer.decode = decode and not verify
                layer.verification = step
            yield
            if graph is not None:
                # One device copy owns the staging independently of later
                # replays, including when a later batch changes membership.
                for name, rows in zip(graph.layers, graph.tails.clone(), strict=True):
                    for span, pending in pending_by_span.items():
                        pending.rows[name] = rows[span.begin:span.end]
            if any(
                pending.rows.keys() != tails.keys()
                for pending in pending_by_span.values()
            ):
                raise RuntimeError(
                    "Hy4 verification forward missed an index tail writer"
                )
            for state, span in tracked:
                if state.pending is None:
                    state.position = span.length
                else:
                    state.pending.complete = True
        except BaseException:
            # An ordinary eager chunk can overwrite live raw slots. Fail
            # closed if it was interrupted; a staged span can retry its start.
            for state, _ in tracked:
                if state.pending is None:
                    state.failed = True
            raise
        finally:
            step.active = False
            self._active = False
            for layer, original_spans, decode in saved:
                layer.spans = original_spans
                layer.decode = decode
                layer.verification = None
