# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/models/hy_v4/nvidia/vision.py at 596455c7f and transformers/models/hy_v4_vl/modeling_hy_v4_vl.py at b865850d05326cf20e306b31f14b0027de802da3.
"""FP16 Hy4 vision tower. One image, Torch SDPA, no 2D RoPE and no video."""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def column_shard(tensor: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Slice a column-parallel weight or bias on its output axis."""
    span = tensor.shape[0] // tp_size
    start = tp_rank * span
    return tensor[start : start + span]


def row_shard(weight: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Slice a row-parallel weight on its input axis. The bias stays whole."""
    span = weight.shape[1] // tp_size
    start = tp_rank * span
    return weight[:, start : start + span]


def current_tensor_parallel() -> tuple[int, int]:
    """Return (world size, rank). An uninitialized group is one rank."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    except AssertionError:
        return 1, 0


def _grid_rows(grid_thw) -> list[tuple[int, int, int]]:
    if isinstance(grid_thw, torch.Tensor):
        rows = grid_thw.detach().tolist()
    else:
        rows = list(grid_thw)
    return [(int(t), int(h), int(w)) for t, h, w in rows]


def _layer_norm(norm: nn.LayerNorm, hidden: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        hidden.float(),
        norm.normalized_shape,
        norm.weight.float(),
        norm.bias.float(),
        norm.eps,
    )


def _conv2d(conv: nn.Conv2d, hidden: torch.Tensor) -> torch.Tensor:
    bias = None if conv.bias is None else conv.bias.float()
    return F.conv2d(
        hidden.float(),
        conv.weight.float(),
        bias,
        stride=conv.stride,
        padding=conv.padding,
    )


def _frame_lengths(rows: list[tuple[int, int, int]]) -> list[int]:
    # Each frame is its own attention segment. An image has one frame.
    return [h * w for t, h, w in rows for _ in range(t)]


class Hy4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.float() * hidden


class Hy4ShardLinear(nn.Module):
    """Column- or row-parallel linear. Row bias is added once after the all-reduce.

    tp 1 keeps the bias inside the linear, which is the unsharded matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        kind: str,
        tp_size: int,
        tp_rank: int,
    ):
        super().__init__()
        if kind not in ("column", "row"):
            raise ValueError(kind)
        self.kind = kind
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        if kind == "column":
            local_out = out_features // tp_size
            local_in = in_features
        else:
            local_out = out_features
            local_in = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(local_out, local_in))
        if kind == "row" and tp_size > 1:
            self.register_parameter("bias", None)
            self.row_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = nn.Parameter(torch.empty(local_out))
            self.row_bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1] * (self.tp_size if self.kind == "row" else 1)
        bound = 1 / math.sqrt(fan_in) if fan_in else 0
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        if self.row_bias is not None:
            nn.init.uniform_(self.row_bias, -bound, bound)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # FP16 accumulation missed the fixed 2e-2 bound. The matmul runs in FP32.
        bias = None if self.bias is None else self.bias.float()
        output = F.linear(hidden.float(), self.weight.float(), bias)
        if self.row_bias is None:
            return output
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(output) + self.row_bias.float()

    def take(self, leaf: str, tensor: torch.Tensor) -> None:
        if leaf == "weight":
            if self.kind == "column" and tensor.shape != self.weight.shape:
                tensor = column_shard(tensor, self.tp_size, self.tp_rank)
            elif self.kind == "row" and tensor.shape != self.weight.shape:
                tensor = row_shard(tensor, self.tp_size, self.tp_rank)
            target = self.weight
        elif leaf == "bias":
            target = self.bias if self.bias is not None else self.row_bias
            if (
                self.kind == "column"
                and target is not None
                and tensor.shape != target.shape
            ):
                tensor = column_shard(tensor, self.tp_size, self.tp_rank)
        else:
            raise KeyError(leaf)
        target.data.copy_(tensor.detach().to(dtype=target.dtype, device=target.device))


class Hy4VisionAttention(nn.Module):
    def __init__(self, config, tp_size: int, tp_rank: int):
        super().__init__()
        self.num_heads = config.num_attention_heads // tp_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        hidden = config.hidden_size
        local = self.num_heads * self.head_dim
        self.q_proj = Hy4ShardLinear(
            hidden, local * tp_size, kind="column", tp_size=tp_size, tp_rank=tp_rank
        )
        self.k_proj = Hy4ShardLinear(
            hidden, local * tp_size, kind="column", tp_size=tp_size, tp_rank=tp_rank
        )
        self.v_proj = Hy4ShardLinear(
            hidden, local * tp_size, kind="column", tp_size=tp_size, tp_rank=tp_rank
        )
        self.o_proj = Hy4ShardLinear(
            hidden, hidden, kind="row", tp_size=tp_size, tp_rank=tp_rank
        )

    def forward(self, hidden: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        pieces = []
        for chunk in hidden.split(lengths, dim=1):
            batch, seq, _ = chunk.shape
            query = self.q_proj(chunk).view(batch, seq, self.num_heads, self.head_dim)
            key = self.k_proj(chunk).view(batch, seq, self.num_heads, self.head_dim)
            value = self.v_proj(chunk).view(batch, seq, self.num_heads, self.head_dim)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                query, key, value, scale=self.scaling, is_causal=False
            )
            pieces.append(attended.transpose(1, 2).reshape(batch, seq, -1))
        return self.o_proj(torch.cat(pieces, dim=1))


class Hy4VisionMLP(nn.Module):
    def __init__(self, config, tp_size: int, tp_rank: int):
        super().__init__()
        self.dense_h_to_4h = Hy4ShardLinear(
            config.hidden_size,
            config.intermediate_size,
            kind="column",
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.dense_4h_to_h = Hy4ShardLinear(
            config.intermediate_size,
            config.hidden_size,
            kind="row",
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.act = nn.GELU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(hidden)))


class Hy4VisionBlock(nn.Module):
    def __init__(self, config, tp_size: int, tp_rank: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.self_attn = Hy4VisionAttention(config, tp_size, tp_rank)
        self.mlp = Hy4VisionMLP(config, tp_size, tp_rank)

    def forward(self, hidden: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        hidden = hidden.float() + self.self_attn(_layer_norm(self.input_layernorm, hidden), lengths)
        return hidden + self.mlp(_layer_norm(self.post_attention_layernorm, hidden))


class Hy4VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_edge = config.max_image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.position_embedding = nn.Embedding(
            self.position_edge**2 + 1, config.hidden_size
        )

    def _positions(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        edge = self.position_edge
        grid = (
            self.position_embedding.weight[1:]
            .float()
            .reshape(1, edge, edge, -1)
            .permute(0, 3, 1, 2)
        )
        grid = F.interpolate(
            grid,
            scale_factor=((height + 0.1) / edge, (width + 0.1) / edge),
            mode=self.config.interpolate_mode,
            align_corners=False,
        )
        return grid.permute(0, 2, 3, 1).reshape(-1, grid.shape[1]).to(dtype)

    def forward(self, pixel_values: torch.Tensor, rows: list[tuple[int, int, int]]):
        config = self.config
        patches = pixel_values.reshape(
            -1, config.num_channels, config.patch_size, config.patch_size
        )
        weight = self.patch_embedding.weight
        bias = self.patch_embedding.bias
        embedded = F.conv2d(
            patches.float(),
            weight.float(),
            None if bias is None else bias.float(),
            stride=self.patch_embedding.stride,
        )
        embedded = embedded.flatten(2).squeeze(-1)
        positions = []
        for t, height, width in rows:
            one = self._positions(height, width, embedded.dtype)
            if one.shape[0] != height * width:
                raise RuntimeError(
                    f"position grid {tuple(one.shape)} is not {height} by {width}"
                )
            positions.append(one.repeat(t, 1))
        return (embedded + torch.cat(positions, dim=0)).unsqueeze(0)


class Hy4VisionPerceive(nn.Module):
    def __init__(self, config):
        super().__init__()
        merge = config.spatial_merge_size
        hidden = config.hidden_size * merge**2
        self.spatial_merge_size = merge
        self.norm = Hy4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Conv2d(config.hidden_size, hidden, kernel_size=merge, stride=merge),
            nn.GELU(),
            nn.Conv2d(hidden, config.out_hidden_size, kernel_size=1),
        )
        # The checkpoint has no use_image_newline_token flag. The merger still emits one.
        self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))
        nn.init.normal_(self.image_newline, mean=0.0, std=config.out_hidden_size**-0.5)

    def forward(self, hidden: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        hidden = self.norm(hidden.float())
        dtype = hidden.dtype
        height, width = size
        hidden = hidden.reshape(-1, height, width, hidden.shape[-1]).permute(0, 3, 1, 2)
        hidden = _conv2d(self.proj[0], hidden)
        hidden = self.proj[1](hidden)
        hidden = _conv2d(self.proj[2], hidden)
        batch, channels, merged_h, _ = hidden.shape
        newline = self.image_newline.reshape(1, channels, 1, 1).expand(
            batch, channels, merged_h, 1
        )
        hidden = torch.cat((hidden, newline.to(dtype)), dim=-1)
        hidden = hidden.flatten(2).transpose(1, 2)
        return hidden.reshape(-1, channels)


class Hy4VisionTransformer(nn.Module):
    def __init__(self, config, *, dtype=torch.float16, tp_size=None, tp_rank=None):
        super().__init__()
        if tp_size is None or tp_rank is None:
            tp_size, tp_rank = current_tensor_parallel()
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid tensor parallel placement {tp_rank}/{tp_size}")
        for name in ("num_attention_heads", "hidden_size", "intermediate_size"):
            if getattr(config, name) % tp_size:
                raise ValueError(f"{name}={getattr(config, name)} is not divisible by tp {tp_size}")
        self.config = config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.compute_dtype = dtype
        self.embeddings = Hy4VisionEmbeddings(config)
        self.layers = nn.ModuleList(
            [
                Hy4VisionBlock(config, tp_size, tp_rank)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.perceive = Hy4VisionPerceive(config)
        self.to(dtype=dtype)
        self.embeddings.position_embedding.to(dtype=torch.float32)

    def forward(self, pixel_values: torch.Tensor, grid_thw) -> torch.Tensor:
        rows = _grid_rows(grid_thw)
        lengths = _frame_lengths(rows)
        if sum(lengths) != pixel_values.shape[0] or not lengths or min(lengths) <= 0:
            raise ValueError(
                f"pixel_values length {tuple(pixel_values.shape)} does not match grid {rows}"
            )
        hidden = self.embeddings(pixel_values, rows)
        for layer in self.layers:
            hidden = layer(hidden, lengths)
        offset = 0
        features = []
        for t, height, width in rows:
            count = t * height * width
            item = hidden[:, offset : offset + count]
            offset += count
            features.append(self.perceive(item, (height, width)))
        return torch.cat(features, dim=0).to(self.compute_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        shards = {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, Hy4ShardLinear)
        }
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        seen: set[str] = set()
        for name, tensor in weights:
            if name.startswith("vit."):
                name = name[4:]
            if name in seen:
                raise ValueError(f"duplicate vision weight {name}")
            seen.add(name)
            module_name, _, leaf = name.rpartition(".")
            if module_name in shards and leaf in ("weight", "bias"):
                shards[module_name].take(leaf, tensor)
                module = shards[module_name]
                if leaf == "weight":
                    loaded.add(f"{module_name}.weight")
                elif module.bias is not None:
                    loaded.add(f"{module_name}.bias")
                else:
                    loaded.add(f"{module_name}.row_bias")
                continue
            if name not in params:
                raise KeyError(name)
            parameter = params[name]
            parameter.data.copy_(
                tensor.detach().to(dtype=parameter.dtype, device=parameter.device)
            )
            loaded.add(name)
        missing = [name for name in params if name not in loaded]
        if missing:
            raise RuntimeError(f"vision tower left {len(missing)} parameters unloaded: {missing[:6]}")
        return seen

    def rank_parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())
