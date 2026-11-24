# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.sm70_sparse import sparse_policy
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

from . import w8
from .cache import (
    FlashCache,
    FlashMetadata,
    IndexPoolDecodeWorkspace,
    QuerySpan,
)
from .dcp import select_history
from .decode import (
    gather_decode_group,
    gather_verify_group,
    store_decode_group,
    store_verify_group,
)
from .kernels import (
    allocate_index_prefill_scratch,
    index_prefill_scores,
    index_prefill_workspace_specs,
    index_scores,
)
from ..profile import speed_setting
from .f7_next import (
    can_use_f7_idx_query_prep,
    can_use_f7_idx_update,
    run_f7_idx_query_prep,
    run_f7_idx_update,
)

from .idx_tc import can_use_idx_tc, run_idx_tc
from .rotary import apply_rotary, rotary_phases, span_tokens

if TYPE_CHECKING:
    from .verification import VerificationForward


def pool_page_ids(
    groups: torch.Tensor,
    table: torch.Tensor,
    page: int,
    pages_per_block: int,
) -> torch.Tensor:
    logical_page = groups // page
    return (
        table[logical_page // pages_per_block] * pages_per_block
        + logical_page % pages_per_block
    )


def select_tokens(
    logits: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    history: int,
) -> torch.Tensor:
    """Select complete groups, then append the current group's visible prefix."""
    selected = logits.topk(history, dim=-1, sorted=False).indices.sort(dim=-1).values
    return expand_groups(selected, positions, ratio)


def select_decode_groups(
    logits: torch.Tensor,
    history: int,
    workspace: IndexPoolDecodeWorkspace | None,
) -> torch.Tensor:
    rows = logits.shape[0]
    if workspace is None:
        return torch.cat([
            row.topk(history, sorted=False).indices.sort().values
            for row in logits.split(1)
        ])
    topk_groups = workspace.topk_groups[:rows]
    torch.topk(
        logits,
        history,
        sorted=False,
        out=(workspace.topk_values[:rows], topk_groups),
    )
    sorted_groups = workspace.sorted_groups[:rows]
    torch.sort(topk_groups, out=(sorted_groups, workspace.sort_order[:rows]))
    return sorted_groups


def expand_groups(
    selected: torch.Tensor, positions: torch.Tensor, ratio: int
) -> torch.Tensor:
    current = positions // ratio
    offsets = torch.arange(ratio, device=positions.device)
    recalled = (selected[..., None] * ratio + offsets).flatten(1)
    tail = current[:, None] * ratio + offsets
    return torch.cat(
        (recalled, tail.masked_fill(tail > positions[:, None], -1)), dim=1
    ).int()


def _pool_group(
    indexer: "FlashCompressedIndexer", raw: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    values = indexer.key_value_projection(raw)
    gates = indexer.key_gate_projection(raw) + indexer.intra_group_ape
    return indexer.key_norm((values * gates.softmax(dim=1)).sum(dim=1)).to(dtype)


class FlashCompressedIndexer(nn.Module):
    """Coarse-only Hy4 IndexPool with FP16 pages and an open-group raw ring."""

    def __init__(self, vllm_config: VllmConfig, config, prefix: str, block_size: int):
        super().__init__()
        self.dcp_world = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_group = get_dcp_group() if self.dcp_world > 1 else None
        self.dcp_rank = self.dcp_group.rank_in_group if self.dcp_group else 0
        self.ratio = config.compressed_indexer_compress_ratio
        self.history = config.compressed_indexer_group_topk - 1
        self.width = (self.history + 1) * self.ratio
        self.num_heads = config.compressed_indexer_num_attention_heads
        self.dim = config.compressed_indexer_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.eps = config.rms_norm_eps
        self._f7_idx_fuse = speed_setting("HY4_F7_IDX_FUSE", vllm_config)
        self._idx_tc = speed_setting("HY4_IDX_TC", vllm_config)
        model_config = vllm_config.model_config
        self.pos_dims = (
            3
            if model_config is not None and getattr(model_config, "uses_mrope", False)
            else 1
        )
        rope = getattr(config, "rope_parameters", None) or {}
        self.mrope_section = None
        if isinstance(rope, dict):
            for value in rope.values():
                if isinstance(value, dict) and value.get("mrope_section") is not None:
                    self.mrope_section = tuple(value["mrope_section"])
                    break
        if self.dcp_group is not None:
            assert model_config is not None
            self.max_decode_rows = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = (
            model_config.max_model_len if model_config is not None else 0
        )
        self.max_local_groups = cdiv(self.max_model_len, self.ratio * self.dcp_world)
        self.prefill_tile_mb = sparse_policy(vllm_config).value("indexer_prefill_tile_mb")
        self.logits_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
        for name, in_dim, out_dim in (
            ("query_projection", config.q_lora_rank, self.num_heads * self.dim),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, self.dim),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    in_dim,
                    out_dim,
                    bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        for projection in (
            self.query_projection, self.head_weight_projection, self.token_key_projection
        ):
            w8.enable_linear(projection, "INDEXER")
        self.token_key_norm = RMSNorm(self.dim, self.eps)
        self.key_value_projection = nn.Linear(
            self.dim, self.dim, bias=False, dtype=torch.float32
        )
        self.key_gate_projection = nn.Linear(
            self.dim, self.dim, bias=False, dtype=torch.float32
        )
        self.key_norm = nn.RMSNorm(self.dim, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, self.dim, dtype=torch.float32)
        )
        self.k_cache = FlashCache(
            vllm_config,
            f"{prefix}.k_cache",
            MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=self.dim,
                dtype=torch.float16,
                compress_ratio=self.ratio,
                dcp_sharded=self.dcp_world > 1,
            ),
        )
        self.tail_cache = FlashCache(
            vllm_config,
            f"{prefix}.tail_cache",
            CircularBufferSpec(
                block_size=self.ratio,
                num_kv_heads=1,
                head_size=self.dim + self.pos_dims,
                dtype=torch.float32,
                dcp_sharded=False,
            ),
        )

    def _rotate(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (
                x[..., : -self.rope_dim],
                apply_rotary(x[..., -self.rope_dim :], cos, sin),
            ),
            -1,
        )

    def decode_groups(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: FlashMetadata,
        tail_metadata: FlashMetadata,
        slots: torch.Tensor,
        *,
        bypass_history: bool = False,
    ) -> torch.Tensor | None:
        """Update the decode caches and return the sorted history group IDs."""
        workspace = metadata.index_workspace
        assert workspace is not None
        if can_use_f7_idx_update(self, hidden, positions):
            run_f7_idx_update(
                self, hidden, positions, inv_freq, metadata, tail_metadata, slots, width=1
            )
            if bypass_history:
                return None
            return self._decode_history(
                hidden, q_c, cos, sin, metadata.block_table, metadata.seq_lens, slots, workspace
            )
        keys = self.token_key_norm(self.token_key_projection(hidden)[0])
        rows = keys.shape[0]
        pool, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        page = pool.shape[1]
        pages_per_block = self.k_cache.spec.storage_block_size // page
        table, ring_table = metadata.block_table, tail_metadata.block_table
        raw = workspace.raw[:rows]
        group_positions = workspace.positions[:rows]
        gather_decode_group[(rows,)](
            keys,
            metadata.seq_lens,
            slots,
            ring_table,
            tail,
            raw,
            group_positions,
            ring_table.stride(0),
            tail.stride(0),
            tail.stride(1),
            self.ratio,
            self.dim,
            self.pos_dims,
        )
        pooled = _pool_group(self, raw, keys.dtype)
        if self.pos_dims == 1:
            group_cos, group_sin = rotary_phases(group_positions, inv_freq)
            store_positions = positions
        else:
            group_cos, group_sin = rotary_phases(
                group_positions.transpose(0, 1), inv_freq, self.mrope_section
            )
            store_positions = positions.transpose(0, 1).contiguous()
        store_decode_group[(rows,)](
            keys,
            store_positions,
            self._rotate(pooled, group_cos, group_sin),
            metadata.seq_lens,
            slots,
            table,
            ring_table,
            pool,
            tail,
            table.stride(0),
            ring_table.stride(0),
            pool.stride(0),
            pool.stride(1),
            tail.stride(0),
            tail.stride(1),
            self.ratio,
            self.dim,
            page,
            pages_per_block,
            self.dcp_world,
            self.dcp_rank,
            self.pos_dims,
        )
        if bypass_history:
            return None
        return self._decode_history(
            hidden, q_c, cos, sin, table, metadata.seq_lens, slots, workspace
        )

    def verify_groups(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: FlashMetadata,
        tail_metadata: FlashMetadata,
        lengths: torch.Tensor,
        slots: torch.Tensor,
        staged: torch.Tensor,
        width: int,
        workspace: IndexPoolDecodeWorkspace | None = None,
        *,
        bypass_history: bool = False,
    ) -> torch.Tensor | None:
        """Stage all raw rows and publish only complete position-addressed groups."""
        if can_use_f7_idx_update(
            self, hidden, positions, width=width, staged=staged
        ):
            run_f7_idx_update(
                self,
                hidden,
                positions,
                inv_freq,
                metadata,
                tail_metadata,
                slots,
                staged=staged,
                width=width,
            )
            if bypass_history:
                return None
            return self._decode_history(
                hidden, q_c, cos, sin, metadata.block_table, lengths, slots, workspace,
                query_width=width,
            )
        if workspace is None:
            keys = torch.cat([
                self.token_key_norm(self.token_key_projection(row)[0])
                for row in hidden.split(1)
            ])
        else:
            keys = self.token_key_norm(self.token_key_projection(hidden)[0])
        staged[:, : self.dim].copy_(keys)
        coordinates = positions[:, None] if positions.ndim == 1 else positions.T
        staged[:, self.dim:].copy_(coordinates)

        pool, ring = self.k_cache.kv_cache, self.tail_cache.kv_cache
        table, ring_table = metadata.block_table, tail_metadata.block_table
        batch = hidden.shape[0] // width
        closes = cdiv(width, self.ratio)
        if workspace is None:
            raw = staged.new_empty((batch * closes, self.ratio, self.dim))
            group_pos = positions.new_empty((batch * closes, self.pos_dims))
        else:
            raw = workspace.raw[: batch * closes]
            group_pos = workspace.positions[: batch * closes]
        grid = (batch, closes)
        gather_verify_group[grid](
            staged,
            metadata.seq_lens,
            slots,
            ring_table,
            ring,
            raw,
            group_pos,
            ring_table.stride(0),
            ring.stride(0),
            ring.stride(1),
            width,
            closes,
            self.ratio,
            self.dim,
            self.pos_dims,
        )
        if workspace is None:
            # Match the projection shape used by one-row decode.
            pooled = torch.cat([
                _pool_group(self, group, keys.dtype) for group in raw.split(1)
            ])
        else:
            pooled = _pool_group(self, raw, keys.dtype)
        rope_pos = group_pos.flatten() if self.pos_dims == 1 else group_pos.T
        group_cos, group_sin = rotary_phases(rope_pos, inv_freq, self.mrope_section)
        page = pool.shape[1]
        store_verify_group[grid](
            self._rotate(pooled, group_cos, group_sin),
            metadata.seq_lens,
            slots,
            table,
            pool,
            table.stride(0),
            pool.stride(0),
            pool.stride(1),
            width,
            closes,
            self.ratio,
            self.dim,
            page,
            self.k_cache.spec.storage_block_size // page,
        )
        if bypass_history:
            return None
        return self._decode_history(
            hidden, q_c, cos, sin, table, lengths, slots, workspace, query_width=width
        )

    def _decode_history(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        table: torch.Tensor,
        lengths: torch.Tensor,
        slots: torch.Tensor,
        workspace: IndexPoolDecodeWorkspace | None = None,
        *,
        query_width: int = 1,
    ) -> torch.Tensor | None:
        if self.max_model_len and self.max_model_len <= self.width:
            return None
        rows = hidden.shape[0]
        pool = self.k_cache.kv_cache
        pages_per_block = self.k_cache.spec.storage_block_size // pool.shape[1]
        groups = self.max_local_groups
        logits = None
        if workspace is None:
            # Standalone fixtures may omit the model limit.
            groups = groups or table.shape[1] * self.k_cache.spec.storage_block_size
        elif self.dcp_group is not None:
            (logits,) = current_workspace_manager().get_simultaneous(
                ((rows, groups), torch.float32)
            )
        else:
            groups = workspace.logits.shape[1]
            if groups == 0:
                return None
            logits = workspace.logits[:rows]
        if can_use_f7_idx_query_prep(self, hidden, q_c, cos, sin):
            query, weights = run_f7_idx_query_prep(self, hidden, q_c, cos, sin)
        else:
            if workspace is None:
                # Match the one-token GEMM dispatch for verification keys and
                # queries. Keep the score kernel batched across independent rows.
                query = torch.cat([self.query_projection(row)[0] for row in q_c.split(1)])
                weights = torch.cat(
                    [self.head_weight_projection(row)[0] for row in hidden.split(1)]
                )
            else:
                query = self.query_projection(q_c)[0]
                weights = self.head_weight_projection(hidden)[0]
            query = self._rotate(
                query.view(rows, self.num_heads, self.dim), cos[:, None], sin[:, None]
            )
            weights = weights.float()
            weights *= self.num_heads**-0.5 * self.dim**-0.5
        if (
            self._idx_tc
            and workspace is not None
            and self.dcp_group is None
            and can_use_idx_tc(query, weights, pool, query_width)
        ):
            assert logits is not None
            return run_idx_tc(
                query,
                weights,
                pool,
                table,
                logits,
                lengths,
                slots,
                workspace.sorted_groups[:rows],
                ratio=self.ratio,
                group_topk=self.history + 1,
                pages_per_block=pages_per_block,
                query_width=query_width,
            )
        logits = index_scores(
            query,
            weights,
            pool,
            table,
            0,
            groups,
            self.ratio,
            pages_per_block,
            logits=logits,
            lengths=lengths,
            slots=slots,
            group_topk=self.history + 1,
            world=self.dcp_world,
            rank=self.dcp_rank,
            query_width=query_width,
        )
        if self.dcp_group is not None:
            lengths = lengths[:rows]
            upper = ((lengths - 1) // self.ratio).masked_fill(
                (slots < 0) | (lengths <= self.width), 0
            )
            return select_history(logits, self.local_lengths(upper), self.dcp_group)
        return select_decode_groups(logits, self.history, workspace)

    def profile(self, q: torch.Tensor) -> None:
        if self.max_model_len <= self.width:
            return
        rows = q.shape[0]
        groups = self.max_local_groups
        if self.dcp_group is not None:
            current_workspace_manager().get_simultaneous(
                ((self.max_decode_rows, groups), torch.float32)
            )
        # Profile the deepest permissible score shape while the enclosing
        # forward is live. Eager scores release keys and GEMM output before
        # selection, and all of this scratch dies before the MoE runs.
        logits, keys, scores = allocate_index_prefill_scratch(
            q.device,
            index_prefill_workspace_specs(
                rows,
                groups,
                self.num_heads,
                self.dim,
                self.prefill_tile_mb,
                torch.float16,
                logits_budget=self.logits_budget,
            ),
        )
        # Rotated queries/weights can coexist with either phase. Bound query
        # projection and rotary intermediates by four FP16 query tensors.
        query_and_weights = q.new_empty(
            rows * self.num_heads * (4 * self.dim + 2)
        )
        # Selection expands the groups into tokens. Allow four int64 token
        # matrices for recall, concatenation and intermediates, plus the
        # returned int32 IDs. DCP gathers two packed candidate buffers (a
        # gather and its reshaped output), then sorts indices after the merge.
        expand_bytes = rows * self.width * (4 * 8 + 4)
        dcp_bytes = rows * (self.history + 1) * (16 * self.dcp_world + 24)
        selection_bytes = max(expand_bytes, dcp_bytes)
        extra = q.new_empty(
            max(0, selection_bytes - keys.nbytes - scores.nbytes),
            dtype=torch.uint8,
        )
        del keys, scores, extra, query_and_weights
        if self.dcp_group is not None:
            select_history(
                logits[:rows, None],
                q.new_zeros(rows, dtype=torch.int32),
                self.dcp_group,
            )

    def local_lengths(self, groups: torch.Tensor) -> torch.Tensor:
        return ((groups + self.dcp_world - 1 - self.dcp_rank) // self.dcp_world).int()

    def prepare_groups(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        span: QuerySpan,
        table: torch.Tensor,
        ring_id: torch.Tensor,
        verification: "VerificationForward | None" = None,
    ) -> None:
        """Gather any previous open group before writing the next raw tail."""
        pool, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        page = pool.shape[1]
        pages_per_block = self.k_cache.spec.storage_block_size // page
        defer_tail = verification is not None and verification.stage_tail(
            span, self.tail_cache.prefix, keys, positions
        )
        first, end = span.prefix // self.ratio, span.length // self.ratio
        prev = span.prefix % self.ratio
        if first < end:
            stop = end * self.ratio - span.prefix
            raw = keys[:stop].float()
            if self.pos_dims == 1:
                group_pos = positions[:stop]
                if prev:
                    saved = tail[ring_id, :prev][0]
                    raw = torch.cat((saved[:, : self.dim], raw), dim=0)
                    group_pos = torch.cat(
                        (saved[:, self.dim].to(positions.dtype), group_pos)
                    )
                group_pos = group_pos[:: self.ratio]
            else:
                group_pos = positions[:, :stop]
                if prev:
                    saved = tail[ring_id, :prev][0]
                    raw = torch.cat((saved[:, : self.dim], raw), dim=0)
                    group_pos = torch.cat(
                        (
                            saved[:, self.dim :].to(dtype=torch.long).transpose(0, 1),
                            group_pos,
                        ),
                        dim=1,
                    )
                group_pos = group_pos[:, :: self.ratio]
            raw = raw.reshape(-1, self.ratio, self.dim)
            groups = torch.arange(first, end, device=keys.device)
            if self.dcp_group is not None:
                owner = slice(
                    (self.dcp_rank - first) % self.dcp_world, None, self.dcp_world
                )
                raw = raw[owner]
                group_pos = group_pos[owner] if group_pos.ndim == 1 else group_pos[:, owner]
                groups = groups[owner] // self.dcp_world
            # A verify span can close two groups. Keep each pool projection
            # at the same shape as the one-request decode projection.
            pooled = (
                torch.cat(
                    [_pool_group(self, group, keys.dtype) for group in raw.split(1)]
                )
                if defer_tail and raw.shape[0] > 1
                else _pool_group(self, raw, keys.dtype)
            )
            if self.pos_dims == 1:
                cos, sin = rotary_phases(group_pos, inv_freq)
            else:
                cos, sin = rotary_phases(group_pos, inv_freq, self.mrope_section)
            pages = pool_page_ids(groups, table, page, pages_per_block)
            pool[pages, groups % page] = self._rotate(pooled, cos, sin)
        # Earlier members of the same still-open group are left in the ring.
        tail_start = max(span.prefix, end * self.ratio)
        if not defer_tail and tail_start < span.length:
            start = tail_start - span.prefix
            if self.pos_dims == 1:
                stored = torch.cat(
                    (keys[start:].float(), positions[start:, None].float()), dim=-1
                )
            else:
                stored = torch.cat(
                    (
                        keys[start:].float(),
                        positions[:, start:].transpose(0, 1).float(),
                    ),
                    dim=-1,
                )
            tail[ring_id, tail_start % self.ratio : span.length % self.ratio] = stored

    def forward(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: FlashMetadata,
        tail_metadata: FlashMetadata,
        output: torch.Tensor,
        *,
        decode_verification: bool = False,
    ) -> None:
        verification = metadata.verification if decode_verification else None
        verify_spans = {
            span for span in metadata.spans
            if verification is not None and verification.verifies(span)
        }
        keys = None
        if len(verify_spans) != len(metadata.spans):
            keys = self.token_key_norm(self.token_key_projection(hidden)[0])
        pool = self.k_cache.kv_cache
        page = pool.shape[1]
        pages_per_block = self.k_cache.spec.storage_block_size // page
        columns = torch.arange(self.width, device=positions.device)
        for req, span in enumerate(metadata.spans):
            begin, end, prefix, length = span
            table = metadata.block_table[req]
            if span in verify_spans:
                span_keys = torch.cat([
                    self.token_key_norm(self.token_key_projection(row)[0])
                    for row in hidden[begin:end].split(1)
                ])
            else:
                assert keys is not None
                span_keys = keys[begin:end]
            self.prepare_groups(
                span_keys,
                span_tokens(positions, begin, end),
                inv_freq,
                span,
                table,
                tail_metadata.block_table[req, :1],
                metadata.verification,
            )
            full_end = min(end, begin + max(0, self.width - prefix))
            output[begin:full_end] = columns.masked_fill(
                columns
                > torch.arange(prefix, prefix + full_end - begin, device=hidden.device)[
                    :, None
                ],
                -1,
            )
            if full_end == end:
                continue
            if span in verify_spans:
                rows = end - full_end
                lengths = torch.arange(
                    length - rows + 1, length + 1, device=hidden.device, dtype=torch.int32
                )
                # The packed target slot may be absent on this DCP rank.
                # Every live query still participates in history selection.
                groups = self._decode_history(
                    hidden[full_end:end],
                    q_c[full_end:end],
                    cos[full_end:end],
                    sin[full_end:end],
                    table.expand(rows, -1),
                    lengths,
                    torch.zeros_like(lengths),
                )
                assert groups is not None
                output[full_end:end] = expand_groups(groups, lengths - 1, self.ratio)
                continue
            query = self.query_projection(q_c[full_end:end])[0].view(
                -1, self.num_heads, self.dim
            )
            query = self._rotate(
                query, cos[full_end:end, None], sin[full_end:end, None]
            )
            weights = self.head_weight_projection(hidden[full_end:end])[0].float()
            weights *= self.num_heads**-0.5 * self.dim**-0.5
            first = prefix + full_end - begin
            groups = cdiv(length, self.ratio * self.dcp_world)
            chunk = max(1, self.logits_budget // groups)
            for start in range(0, end - full_end, chunk):
                stop = start + chunk
                q, w = query[start:stop], weights[start:stop]
                logical = torch.arange(
                    first + start, first + start + q.shape[0], device=q.device
                )
                if q.is_cuda:
                    logits = index_prefill_scores(
                        q,
                        w,
                        pool,
                        table,
                        first + start,
                        groups,
                        self.ratio,
                        pages_per_block,
                        world=self.dcp_world,
                        rank=self.dcp_rank,
                        tile_mb=self.prefill_tile_mb,
                    )
                else:
                    g = torch.arange(groups, device=q.device)
                    k = pool[
                        pool_page_ids(g, table, page, pages_per_block), g % page
                    ].float()
                    logits = ((q.float() @ k.T).relu() * w[..., None]).sum(1)
                    logits.masked_fill_(g >= logical[:, None] // self.ratio, -torch.inf)
                if self.dcp_group is not None:
                    local_lengths = self.local_lengths(logical // self.ratio)
                    selected = select_history(logits, local_lengths, self.dcp_group)
                    tokens = expand_groups(selected, logical, self.ratio)
                else:
                    tokens = select_tokens(logits, logical, self.ratio, self.history)
                del logits
                output[full_end + start : full_end + start + q.shape[0]] = tokens
