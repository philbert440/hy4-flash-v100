# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synchronous MRV1 handoff between the target's sparse layers and DFly2."""

import json
import math
from dataclasses import dataclass
from functools import cached_property

import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, cached_file

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.hy_v4_flash.dfly_config import QUERY_WIDTH, DFlySettings
from vllm.models.hy_v4_flash.dfly_gate import (
    DFlyGate,
    draft_load_allowed,
    draft_max_running,
    graph_draft_sizes,
)
from vllm.models.hy_v4_flash.profile import speed_setting
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import FP8_DS_MLA_BYTES, CircularBufferSpec

from .cache import IndexPoolDecodeWorkspace
from .dfly import DFlyModel, DraftHistory, DraftWorkspace
from .dfly_head import DraftHead
from .dfly_weights import DFlyWeights
from .splitk import decode_workspace_specs
from .verification import VerificationGraph

logger = init_logger(__name__)


@dataclass(frozen=True)
class DraftSource:
    request_row: int
    first_candidate: int
    first_query_row: int
    optimistic_anchor: int
    num_drafts: int


class DFlyStep:
    def __init__(self, starts, lengths, discard, draft_counts, max_len, device):
        self.sources: list[DraftSource] = []
        rows = []
        for row, (end, length, skip, count) in enumerate(
            zip(starts[1:], lengths, discard, draft_counts, strict=True)
        ):
            earliest_anchor = int(length) - count
            if skip or earliest_anchor + QUERY_WIDTH > max_len:
                continue
            first = int(end) - count - 1
            if first < starts[row]:
                raise RuntimeError("DFly verification has no target anchor row")
            self.sources.append(DraftSource(row, len(rows), first, int(length), count))
            rows.extend(range(first, int(end)))
        self._rows = rows
        self._device = device
        self.histories: dict[int, DraftHistory] = {}
        self.full_snapshot = False

    @cached_property
    def source_rows(self) -> torch.Tensor:
        return torch.tensor(self._rows, device=self._device, dtype=torch.long)

    def capture(self, layer, cache, slots, counts):
        self.histories[layer] = DraftHistory(
            cache,
            slots.index_select(0, self.source_rows),
            counts.index_select(0, self.source_rows),
        )

    def accepted(self, sampled, max_len):
        for source in self.sources:
            ids = sampled[source.request_row]
            if not ids:
                continue
            accepted = len(ids) - 1
            if accepted > source.num_drafts:
                raise RuntimeError("DFly received more accepted tokens than scheduled")
            anchor = source.optimistic_anchor - source.num_drafts + accepted
            if anchor + QUERY_WIDTH <= max_len:
                snapshot_start = (
                    source.first_query_row
                    if self.full_snapshot
                    else source.first_candidate
                )
                yield (
                    source.request_row,
                    snapshot_start + accepted,
                    source.first_query_row + accepted,
                    anchor,
                    ids[-1],
                )


@dataclass
class DraftGraphInputs:
    anchors: torch.Tensor
    positions: torch.Tensor
    hidden: torch.Tensor
    histories: list[DraftHistory]

    def stage(self, anchors, positions, hidden, source, histories, snapshot) -> None:
        self.anchors.copy_(anchors)
        self.positions.copy_(positions)
        torch.index_select(hidden, 0, source, out=self.hidden)
        for dst, src in zip(self.histories, histories, strict=True):
            assert dst.cache is src.cache
            torch.index_select(src.slots, 0, snapshot, out=dst.slots)
            torch.index_select(src.counts, 0, snapshot, out=dst.counts)


@dataclass
class DraftGraph:
    inputs: DraftGraphInputs
    graph: torch.cuda.CUDAGraph
    proposed: torch.Tensor
    confidence: torch.Tensor | None = None
    token_graphs: tuple[torch.cuda.CUDAGraph, ...] = ()
    # Keep the scoring output alive at the address captured by token_graphs.
    draft_hidden: torch.Tensor | None = None


class DFlyProposer:
    def __init__(self, vllm_config, device):
        self.config = vllm_config
        self.device = device
        self.max_len = vllm_config.model_config.max_model_len
        self.uses_mrope = getattr(vllm_config.model_config, "uses_mrope", False)
        self.step: DFlyStep | None = None
        self.draft_head: DraftHead | None = None
        self.batch_verify_index = envs.VLLM_HY4_DFLY_VERIFY_INDEX == "batched"
        self.max_running = draft_max_running()
        self.graph_only = (
            graph_draft_sizes(getattr(vllm_config, "compilation_config", None))
            is not None
        )
        spec = getattr(vllm_config, "speculative_config", None)
        self.gate = (
            DFlyGate(envs.VLLM_HY4_DFLY_GATE_THRESHOLD)
            if spec is not None and spec.method == "dfly"
            and envs.VLLM_HY4_DFLY_GATE == "ewma"
            else None
        )
        self.confidence_mode = speed_setting(
            "VLLM_HY4_DFLY_CONFIDENCE_MODE", vllm_config
        )
        self.confidence_logit_threshold = -math.inf
        self.variable_verify = self.confidence_mode == "variable"
        if self.confidence_mode != "off":
            if not self.graph_only or vllm_config.parallel_config.data_parallel_size != 1:
                raise ValueError(
                    "DFly confidence gating requires both CUDA graphs and DP=1"
                )
            if self.variable_verify and not speed_setting("HY4_F7_IDX_FUSE", vllm_config):
                raise ValueError("Variable DFly verification requires HY4_F7_IDX_FUSE")
            threshold = speed_setting(
                "VLLM_HY4_DFLY_CONFIDENCE_THRESHOLD", vllm_config
            )
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("DFly confidence threshold must be in [0, 1]")
            if threshold == 1.0:
                self.confidence_logit_threshold = math.inf
            elif threshold > 0.0:
                self.confidence_logit_threshold = math.log(threshold / (1 - threshold))
        self.defer_drafts = speed_setting("HY4_DREV2_DEFER_DRAFTS", vllm_config)
        # HY4_DCONF2_LATE_WIDTH: (event, pinned confidence) of a C1 replay whose
        # width the runner resolves when the next step arrives.
        self.late_width = (
            speed_setting("HY4_DCONF2_LATE_WIDTH", vllm_config)
            and self.confidence_mode != "off"
        )
        self.late: tuple[torch.cuda.Event, torch.Tensor] | None = None
        self._late_host: torch.Tensor | None = None
        self.log_confidence = envs.VLLM_HY4_DFLY_LOG_CONFIDENCE
        self.pending_confidence: dict[str, tuple[int, list[float]]] = {}
        self.plain_histories: dict[int, VerificationGraph] = {}
        self.verify_graphs: dict[int, VerificationGraph] = {}
        self.draft_graphs: dict[int, DraftGraph] = {}
        # (request rows, draft graph output) of a replay whose ids the runner
        # writes into the next verify on the device; see propose(defer=True).
        self.deferred: tuple[tuple[int, ...], torch.Tensor] | None = None

    def load_model(self, target):
        model_config = self.config.model_config
        config = model_config.hf_text_config
        self.settings = DFlySettings.from_checkpoint(model_config.hf_config, config)
        tp = get_tensor_model_parallel_world_size()
        if self.log_confidence:
            self.log_confidence = get_tensor_model_parallel_rank() == 0

        def resolve_shard(filename):
            path = cached_file(
                model_config.model_weights or model_config.model,
                filename,
                revision=model_config.revision,
                cache_dir=self.config.load_config.download_dir,
            )
            if path is None:
                raise ValueError(f"DFly checkpoint file {filename} is unavailable")
            return path

        with open(resolve_shard(SAFE_WEIGHTS_INDEX_NAME)) as file:
            weight_map = json.load(file)["weight_map"]
        weights = DFlyWeights(config, self.settings, weight_map, resolve_shard, tp)
        with (
            set_current_vllm_config(self.config),
            torch.device(self.device),
            set_default_torch_dtype(model_config.dtype),
        ):
            # Keep the module outside the target's FlashCache registry and AWQ
            # loader. The target continues to filter all draft tensors.
            self.backbone = DFlyModel(config, self.settings, weights.ffn_width)
            weights.load_into(self.backbone)
            process_weights_after_loading(self.backbone, model_config, self.device)
        self.backbone.eval()

        if model_config.is_multimodal_model:
            target = target.get_language_model()
        self.embed = target.embed_input_ids
        self.logits = target.compute_logits
        head_mode = speed_setting("VLLM_HY4_DFLY_HEAD", self.config)
        if head_mode != "fp32":
            self.draft_head = DraftHead(target.lm_head, head_mode)
        self.target_attention = {
            layer: target.model.layers[layer].self_attn
            for layer in self.settings.target_layers
        }
        for layer, attn in self.target_attention.items():
            if attn.width != self.settings.topk:
                raise ValueError(
                    f"DFly target selector width mismatch at layer {layer}"
                )
            attn._capture_dfly = True

        packed = self.config.cache_config.cache_dtype == "fp8_ds_mla"
        batch = self.config.scheduler_config.max_num_seqs
        rows = batch * QUERY_WIDTH
        row_bytes = (
            FP8_DS_MLA_BYTES
            if packed
            else config.kv_lora_rank + config.qk_rope_head_dim
        )
        scratch = torch.empty(
            (batch, QUERY_WIDTH, row_bytes),
            dtype=torch.uint8 if packed else model_config.dtype,
            device=self.device,
        )
        offsets = torch.arange(rows, dtype=torch.int32, device=self.device)
        specs = decode_workspace_specs(
            rows,
            config.num_attention_heads // tp,
            self.settings.topk + QUERY_WIDTH,
            config.kv_lora_rank,
        )
        state, accum = (
            torch.empty(shape, dtype=dtype, device=self.device)
            for shape, dtype in specs
        )
        self.workspace = DraftWorkspace(scratch, offsets, state, accum)

    def make_verification_graph(
        self, tokens: int, width: int = QUERY_WIDTH
    ) -> VerificationGraph:
        if not 1 < width <= QUERY_WIDTH or tokens % width:
            raise ValueError(f"Invalid DFly verification shape: {tokens=} {width=}")
        registry = self.config.compilation_config.static_forward_context
        names = tuple(
            name for name, owner in registry.items()
            if isinstance(getattr(owner, "spec", None), CircularBufferSpec)
        )
        tail_width = registry[names[0]].spec.head_size if names else 0
        tails = torch.empty(
            (len(names), tokens, tail_width),
            device=self.device,
            dtype=torch.float32,
        )
        workspace = None
        if self.batch_verify_index:
            indexer = next(iter(self.target_attention.values())).indexer
            select = self.max_len > indexer.width
            workspace = IndexPoolDecodeWorkspace.allocate(
                tokens, indexer.ratio, indexer.dim,
                indexer.max_local_groups if select else 0,
                indexer.history if select else 0,
                self.device, pos_dims=indexer.pos_dims,
            )
        graph = VerificationGraph(
            names,
            tails,
            {
                layer: self._history(attn, tokens)
                for layer, attn in self.target_attention.items()
            },
            width,
            workspace,
        )
        self.verify_graphs[tokens] = graph
        return graph

    def make_plain_snapshot(self, rows: int) -> VerificationGraph:
        # Ordinary decode writes its normal cache. Only borrow the fixed
        # history side outputs of VerificationGraph, with no staging tails.
        snapshot = VerificationGraph(
            (), torch.empty((0, rows, 0), device=self.device),
            {
                layer: self._history(attn, rows)
                for layer, attn in self.target_attention.items()
            },
            1,
        )
        self.plain_histories[rows] = snapshot
        return snapshot

    def _history(self, attn, rows: int) -> DraftHistory:
        return DraftHistory(
            attn.attn.kv_cache,
            torch.zeros((rows, attn.width), device=self.device, dtype=torch.int32),
            torch.zeros(rows, device=self.device, dtype=torch.int32),
        )

    def _draft_positions(self, positions):
        positions = (positions[:, None] + self.workspace.offsets[:QUERY_WIDTH]).flatten()
        return positions.expand(3, -1) if self.uses_mrope else positions

    def _draft_forward(self, anchors, positions, hidden, histories):
        return self.backbone.propose(
            anchors,
            self._draft_positions(positions),
            hidden,
            histories,
            self.workspace,
            self.embed,
            self.logits,
            self.settings.mask_id,
            **({"log_confidence": True} if self.log_confidence else {}),
            **({"pick_token": self.draft_head} if self.draft_head is not None else {}),
        )

    def profile_eager(self) -> None:
        # Verification and drafting are sequential. Exercise the largest
        # scheduled proposer batch without allocating the serving KV pool.
        batch = self.config.scheduler_config.max_num_seqs
        config = self.config.model_config
        scratch = self.workspace.scratch
        histories = [
            DraftHistory(
                scratch.new_zeros((1, attn.attn.spec.block_size, scratch.shape[-1])),
                torch.zeros((batch, attn.width), device=self.device, dtype=torch.int32),
                torch.zeros(batch, device=self.device, dtype=torch.int32),
            )
            for attn in self.target_attention.values()
        ]
        tokens = torch.zeros(batch, device=self.device, dtype=torch.long)
        hidden = torch.zeros(
            (batch, config.hf_text_config.hidden_size),
            device=self.device,
            dtype=config.dtype,
        )
        self._draft_forward(tokens, tokens, hidden, histories)

    @torch.inference_mode()
    def capture_draft_graphs(self, sizes: list[int]) -> None:
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )

        pool = torch.cuda.graph_pool_handle()
        set_graph_pool_id(pool)
        for batch in sorted(sizes, reverse=True):
            inputs = DraftGraphInputs(
                torch.zeros(batch, device=self.device, dtype=torch.long),
                torch.zeros(batch, device=self.device, dtype=torch.long),
                torch.zeros(
                    (batch, self.config.model_config.hf_text_config.hidden_size),
                    device=self.device, dtype=self.config.model_config.dtype,
                ),
                [self._history(attn, batch) for attn in self.target_attention.values()],
            )
            if batch == 1 and self.confidence_mode != "off":
                self.draft_graphs[batch] = self._capture_confidence_draft(inputs, pool)
                continue
            self._draft_forward(
                inputs.anchors, inputs.positions, inputs.hidden, inputs.histories
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=torch.cuda.current_stream()):
                output = self._draft_forward(
                    inputs.anchors, inputs.positions, inputs.hidden, inputs.histories
                )
            proposed, confidence = output if self.log_confidence else (output, None)
            self.draft_graphs[batch] = DraftGraph(inputs, graph, proposed, confidence)

    def _capture_confidence_draft(self, inputs: DraftGraphInputs, pool) -> DraftGraph:
        def score():
            hidden = self.backbone.proposal_hidden(
                inputs.anchors,
                self._draft_positions(inputs.positions),
                inputs.hidden,
                inputs.histories,
                self.workspace,
                self.embed,
                self.settings.mask_id,
            )
            confidence = self.backbone.proposal_confidence(hidden)
            return hidden, get_tp_group().broadcast(confidence)

        warm_hidden, _ = score()
        stream = torch.cuda.current_stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            hidden, confidence = score()

        proposed = inputs.anchors.new_empty((1, QUERY_WIDTH - 1))
        previous = inputs.anchors
        token_graphs = []
        # Keep the backbone's fixed checkpoint geometry and only execute the
        # sequential hidden correction and vocabulary picks for the kept prefix.
        for offset in range(1, QUERY_WIDTH):
            slot = proposed[:, offset - 1]
            # The captured slot views have a different stride from the normal
            # picker output. Compile the matching embedding path before capture.
            slot.copy_(self.backbone.pick_next(
                warm_hidden[:, offset], previous, self.embed, self.logits,
                self.draft_head,
            ))
            token_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(token_graph, pool=pool, stream=stream):
                slot.copy_(self.backbone.pick_next(
                    hidden[:, offset], previous, self.embed, self.logits,
                    self.draft_head,
                ))
            previous = slot
            token_graphs.append(token_graph)
        return DraftGraph(
            inputs, graph, proposed, confidence, tuple(token_graphs), hidden
        )

    def confidence_width(self, scores: list[float]) -> int:
        """Kept proposals for one C1 row's confidence scores."""
        candidates = scores[:1] if self.confidence_mode == "skip" else scores
        for offset, score in enumerate(candidates):
            if score < self.confidence_logit_threshold:
                return offset
        return QUERY_WIDTH - 1

    def late_scores(self) -> list[float]:
        """Wait for the pending late-width confidence copy and return its scores."""
        assert self.late is not None
        (event, host), self.late = self.late, None
        event.synchronize()
        return host[0].tolist()

    def clear_graphs(self) -> None:
        self.verify_graphs.clear()
        self.plain_histories.clear()
        self.draft_graphs.clear()

    def forget_requests(self, request_ids):
        if self.gate is not None:
            self.gate.forget(request_ids)
        if self.log_confidence:
            for request_id in request_ids:
                self.pending_confidence.pop(request_id, None)

    def prepare_target(
        self, starts, lengths, discard, draft_counts, request_ids=(), running=None,
        position_deltas=None,
    ):
        self.step = None
        if all(discard):
            return None
        step = DFlyStep(
            starts, lengths, discard, draft_counts, self.max_len, self.device
        )
        step.running = running
        step.position_deltas = position_deltas
        if self.gate is not None or self.log_confidence:
            step.request_ids = request_ids
            step.observations = tuple(zip(lengths, discard, draft_counts, strict=True))
        if step.sources or self.log_confidence:
            # Log the final verification even when it cannot seed another draft.
            self.step = step
        # Only source rows require target history. Calibration on rank zero
        # must never affect target dispatch.
        return step if step.sources else None

    def load_allows(self, running: int, eligible: int) -> bool:
        if not self.graph_only:
            return draft_load_allowed(running, self.max_running, None)
        return (
            running == eligible
            and draft_load_allowed(running, self.max_running, self.draft_graphs)
            and running * QUERY_WIDTH in self.verify_graphs
        )

    def propose(
        self, sampled: list[list[int]], target_hidden: torch.Tensor, defer: bool = False
    ) -> list[list[int]]:
        step, self.step = self.step, None
        drafts: list[list[int]] = [[] for _ in sampled]
        if step is None:
            return drafts
        if self.gate is not None or self.log_confidence:
            for req, ids, (length, discard, count) in zip(
                step.request_ids, sampled, step.observations, strict=True
            ):
                if discard or not ids:
                    continue
                kept = len(ids) - 1
                if count and self.gate is not None:
                    self.gate.observe(req, kept)
                if self.log_confidence:
                    anchor = int(length) - count - 1
                    pending = self.pending_confidence.pop(req, None)
                    scores = (
                        pending[1][:count]
                        if count and pending is not None and pending[0] == anchor
                        else None
                    )
                    logger.info(
                        "Hy4 DFly2 calibration req=%s anchor=%d proposed=%d "
                        "accepted=%d confidence=%s",
                        req, anchor, count, kept, json.dumps(scores),
                    )
        accepted = list(step.accepted(sampled, self.max_len))
        if not accepted:
            return drafts
        request_rows, snapshot_rows, hidden_rows, anchors, tokens = zip(*accepted)
        running = max(step.running or 0, len(sampled))
        if not self.load_allows(running, len(accepted)):
            return drafts
        if self.gate is not None:
            requests = [step.request_ids[row] for row in request_rows]
            allowed = self.gate.allow(requests)
            if self.log_confidence:
                logger.info(
                    "Hy4 DFly2 gate running=%d eligible=%d skipped=%d",
                    running, len(request_rows), 0 if allowed else len(request_rows),
                )
            if not allowed:
                return drafts
        # DFly proposals are generated text. Each MRoPE plane advances
        # with the request's offset from its logical token cursor.
        rope_positions = anchors
        if step.position_deltas is not None:
            rope_positions = tuple(
                anchor + step.position_deltas[row]
                for row, anchor in zip(request_rows, anchors, strict=True)
            )
        selection = torch.tensor(
            (snapshot_rows, hidden_rows, rope_positions, tokens), device=self.device
        )
        snapshot, source, positions, input_ids = selection.unbind()
        histories = [step.histories[layer] for layer in self.settings.target_layers]
        graph = self.draft_graphs.get(len(accepted))
        if graph is None:
            selected = [
                DraftHistory(
                    history.cache,
                    history.slots.index_select(0, snapshot),
                    history.counts.index_select(0, snapshot),
                )
                for history in histories
            ]
            output = self._draft_forward(
                input_ids, positions, target_hidden.index_select(0, source), selected
            )
            proposed, confidence = output if self.log_confidence else (output, None)
        else:
            graph.inputs.stage(
                input_ids, positions, target_hidden, source, histories, snapshot
            )
            graph.graph.replay()
            proposed, confidence = graph.proposed, graph.confidence
        width = QUERY_WIDTH - 1
        if (
            defer and self.late_width and running == 1 and graph is not None
            and graph.token_graphs and not self.log_confidence
        ):
            # Run every picker without reading the confidence back. The runner
            # cuts the scheduled drafts to confidence_width() of this copy.
            assert confidence is not None
            if self._late_host is None:
                self._late_host = torch.empty(
                    confidence.shape, dtype=confidence.dtype, device="cpu",
                    pin_memory=torch.device(self.device).type == "cuda",
                )
            self._late_host.copy_(confidence, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            for token_graph in graph.token_graphs:
                token_graph.replay()
            self.late = (event, self._late_host)
            self.deferred = (request_rows, proposed)
            for row in request_rows:
                drafts[row] = [0] * width
            return drafts
        if graph is not None and graph.token_graphs:
            assert confidence is not None
            scores = confidence.tolist()
            if running == 1:
                width = self.confidence_width(scores[0])
            for token_graph in graph.token_graphs[:width]:
                token_graph.replay()
        else:
            scores = confidence.tolist() if confidence is not None else None
        if self.log_confidence and scores is not None and width:
            for row, anchor, values in zip(request_rows, anchors, scores, strict=True):
                self.pending_confidence[step.request_ids[row]] = (anchor, values)
        if width:
            if defer and graph is not None:
                # Confidence has fixed the width, but the picker graphs can
                # finish on the device while the host schedules this prefix.
                self.deferred = (request_rows, proposed)
                for row in request_rows:
                    drafts[row] = [0] * width
                return drafts
            if width != QUERY_WIDTH - 1:
                proposed = proposed[:, :width]
            for row, ids in zip(request_rows, proposed.tolist(), strict=True):
                drafts[row] = ids
        return drafts
