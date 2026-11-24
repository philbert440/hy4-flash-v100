# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Fusion, correction and backbone adapted from vllm/models/hy_v4/nvidia/dfly.py at upstream 596455c7f16d14e761fbf8b61bf071b2e31c9aa5.  # noqa: E501
# Query loop adapted from vllm/models/hy_v4/nvidia/dfly_speculator.py at upstream 596455c7f16d14e761fbf8b61bf071b2e31c9aa5.  # noqa: E501

"""The DFly2 backbone and its private SM70 query KV."""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward
from vllm.models.hy_v4_flash.dfly_config import QUERY_WIDTH

from .attention import HYV4FlashMLA
from .f7_attn import can_use_f7_attn_out, f7_rope_phase_scope, run_f7_attn_out
from .kernels import (
    _unpack_fp8_ds_mla_torch,
    cache_fp8_ds_mla,
    pack_fp8_ds_mla_torch,
)
from .splitk import paged_attention_splitk


@dataclass
class DraftHistory:
    # Slots are rank-local physical positions in the borrowed, read-only cache.
    cache: torch.Tensor
    slots: torch.Tensor
    counts: torch.Tensor


def _attend_torch(q, history, scratch, sink, scale, latent):
    batch, width = scratch.shape[:2]
    page = history.cache.shape[1]
    slots = history.slots.clamp_min(0)
    keys = torch.cat((history.cache[slots // page, slots % page], scratch), dim=1)
    if keys.dtype == torch.uint8:
        keys = _unpack_fp8_ds_mla_torch(keys, latent)
    keys = keys.float()
    offsets = torch.arange(history.slots.shape[1], device=q.device)
    tail = torch.arange(width, device=q.device)
    live = torch.cat(
        (
            (offsets < history.counts[:, None])[:, None].expand(-1, width, -1),
            (tail[None, :] <= tail[:, None])[None].expand(batch, -1, -1),
        ),
        dim=-1,
    )
    scores = torch.einsum(
        "bqhd,bkd->bqhk", q.reshape(batch, width, *q.shape[1:]).float(), keys
    )
    scores = (scores * scale).masked_fill(~live[:, :, None], -torch.inf)
    sinks = sink.float().view(1, 1, -1, 1).expand(*scores.shape[:-1], 1)
    probs = torch.cat((scores, sinks), dim=-1).softmax(-1)[..., :-1]
    # Mask values as well as scores for unused slots, including padding pages.
    used = live.any(1)
    keys = keys[:, :, :latent].masked_fill(~used[..., None], 0)
    return torch.einsum("bqhk,bkd->bqhd", probs, keys).flatten(0, 1).to(q.dtype)


class DraftWorkspace:
    def __init__(self, scratch, offsets, state=None, accum=None):
        self.scratch = scratch
        self.offsets = offsets
        self.state = state
        self.accum = accum


class DFlyAttention(HYV4FlashMLA):
    def forward(
        self, positions, hidden, history: DraftHistory, workspace: DraftWorkspace
    ):
        if self._f7_attn_prep:
            q, kv_c, k_pe, _, _, _, w_uv, gate_sig = self._project_with_gate(
                positions, hidden
            )
        else:
            q, kv_c, k_pe, _, _, _, w_uv = self._project(positions, hidden)
            gate_sig = None
        scratch = workspace.scratch[: history.slots.shape[0]]
        if scratch.dtype == torch.uint8:
            if q.is_cuda:
                cache_fp8_ds_mla(kv_c, k_pe, scratch, workspace.offsets[: q.shape[0]])
            else:
                scratch.copy_(pack_fp8_ds_mla_torch(kv_c, k_pe).view_as(scratch))
        else:
            torch.cat((kv_c, k_pe), dim=-1, out=scratch.flatten(0, 1))
        if q.is_cuda:
            counts = (
                history.counts[:, None] + workspace.offsets[:QUERY_WIDTH] + 1
            ).flatten()
            if can_use_f7_attn_out(
                self,
                q,
                history.cache,
                history.slots,
                counts,
                self.learnable_sink_param,
                None,
                w_uv,
                draft_kv=scratch,
            ):
                gate_tok = (
                    gate_sig
                    if gate_sig is not None
                    else torch.sigmoid(self.linear_gate(hidden)[0])
                )
                gated = run_f7_attn_out(
                    q,
                    history.cache,
                    history.slots,
                    counts,
                    self.learnable_sink_param,
                    self.scaling,
                    workspace.state,
                    workspace.accum,
                    w_uv,
                    gate_tok,
                    draft_kv=scratch,
                )
                out = self.o_proj(gated)[0]
                return (
                    self.post_proj_norm(out) if self.post_proj_norm is not None else out
                )
            latent = paged_attention_splitk(
                q,
                history.cache,
                history.slots,
                counts,
                self.learnable_sink_param,
                self.scaling,
                workspace.state,
                workspace.accum,
                draft_kv=scratch,
            )
        else:
            latent = _attend_torch(
                q,
                history,
                scratch,
                self.learnable_sink_param,
                self.scaling,
                self.kv_lora_rank,
            )
        return self._project_output(latent, w_uv, hidden, gate_sig=gate_sig)


class DFlyFusion(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.enorm = RMSNorm(hidden_size, eps)
        self.hnorm = RMSNorm(hidden_size, eps)
        self.eh_proj = ReplicatedLinear(2 * hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.empty(hidden_size), requires_grad=False)

    def forward(self, anchors, target_hidden):
        fused, _ = self.eh_proj(
            torch.cat((self.enorm(anchors), self.hnorm(target_hidden)), dim=-1)
        )
        return anchors + self.gate * fused


class DFlyCorrection(HYV4FeedForward):
    def __init__(self, hidden_size: int, intermediate: int, eps: float):
        # The concatenated input is twice the size of the residual output.
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        nn.Module.__init__(self)
        self.hidden_norm = RMSNorm(hidden_size, eps)
        self.embed_norm = RMSNorm(hidden_size, eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * hidden_size, [intermediate, intermediate], bias=False
        )
        self.down_proj = RowParallelLinear(intermediate, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, hidden, previous_embed):
        combined = torch.cat(
            (self.hidden_norm(hidden), self.embed_norm(previous_embed)), -1
        )
        return hidden + super().forward(combined)


class DFlyLayer(nn.Module):
    def __init__(self, config, target_layer, ffn_width, post_proj_norm):
        super().__init__()
        self.self_attn = DFlyAttention(
            config, target_layer, "", quant=None, post_proj_norm=post_proj_norm
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = HYV4FeedForward(config.hidden_size, ffn_width, config.hidden_act)

    def forward(self, positions, hidden, residual, history, workspace):
        if residual is not None:
            hidden = hidden + residual
        output = self.self_attn(
            positions, self.input_layernorm(hidden), history, workspace
        )
        residual = hidden + output
        return self.mlp(self.post_attention_layernorm(residual)), residual


class DFlyModel(nn.Module):
    def __init__(self, config, settings, ffn_width):
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.layers = nn.ModuleList(
            DFlyLayer(config, layer, ffn_width, settings.post_proj_norm)
            for layer in settings.target_layers
        )
        self.final_norm = RMSNorm(hidden, eps)
        self.mtp_fusion = DFlyFusion(hidden, eps)
        self.hidden_correction = DFlyCorrection(hidden, settings.correction_width, eps)
        # Present in the checkpoint. The upstream proposal loop does not use it.
        self.confidence_head = nn.Module()
        self.confidence_head.proj = nn.Linear(hidden, 1)

    def forward(self, embeddings, positions, target_hidden, histories, workspace):
        embeddings[::QUERY_WIDTH] = self.mtp_fusion(
            embeddings[::QUERY_WIDTH], target_hidden
        )
        residual = None
        if self.layers and getattr(self.layers[0].self_attn, "_f7_attn_prep", False):
            with f7_rope_phase_scope():
                for layer, history in zip(self.layers, histories, strict=True):
                    embeddings, residual = layer(
                        positions, embeddings, residual, history, workspace
                    )
        else:
            for layer, history in zip(self.layers, histories, strict=True):
                embeddings, residual = layer(
                    positions, embeddings, residual, history, workspace
                )
        return self.final_norm(embeddings + residual)

    def proposal_hidden(
        self, anchors, positions, target_hidden, histories, workspace, embed, mask_id
    ):
        batch = anchors.numel()
        tokens = anchors.new_full((batch, QUERY_WIDTH), mask_id)
        tokens[:, 0] = anchors
        return self(
            embed(tokens.flatten()), positions, target_hidden, histories, workspace
        ).view(batch, QUERY_WIDTH, -1)

    def pick_next(self, hidden, previous, embed, logits, pick_token):
        corrected = self.hidden_correction(hidden, embed(previous))
        if pick_token is None:
            return logits(corrected).argmax(dim=-1)
        return pick_token(corrected)

    def proposal_confidence(self, hidden):
        return self.confidence_head.proj(hidden[:, 1:]).squeeze(-1)

    def propose(
        self,
        anchors,
        positions,
        target_hidden,
        histories,
        workspace,
        embed,
        logits,
        mask_id,
        *,
        log_confidence: bool = False,
        pick_token=None,
    ):
        hidden = self.proposal_hidden(
            anchors, positions, target_hidden, histories, workspace, embed, mask_id
        )
        previous = anchors
        drafts = []
        for offset in range(1, QUERY_WIDTH):
            previous = self.pick_next(hidden[:, offset], previous, embed, logits, pick_token)
            drafts.append(previous)
        proposed = torch.stack(drafts, dim=-1)
        if log_confidence:
            return proposed, self.proposal_confidence(hidden)
        return proposed
