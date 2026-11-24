# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate the draft checkpoint before allocating and loading its parameters."""

from collections import Counter, defaultdict

import torch
from safetensors import safe_open

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

DRAFT_PREFIX = "model.dfly_layers."


def draft_shapes(config, settings, ffn_width):
    """Global shapes, before the parameter-specific TP loaders shard them."""
    h = config.hidden_size
    q = config.q_lora_rank
    kv = config.kv_lora_rank
    n = config.num_attention_heads
    nope = config.qk_nope_head_dim
    rope = config.qk_rope_head_dim
    v = config.v_head_dim
    correction = settings.correction_width
    shapes = {
        "confidence_head.proj.bias": (1,),
        "confidence_head.proj.weight": (1, h),
        "final_norm.weight": (h,),
        "hidden_correction.down_proj.weight": (h, correction),
        "hidden_correction.embed_norm.weight": (h,),
        "hidden_correction.gate_proj.weight": (correction, 2 * h),
        "hidden_correction.hidden_norm.weight": (h,),
        "hidden_correction.up_proj.weight": (correction, 2 * h),
        "mtp_fusion.eh_proj.weight": (h, 2 * h),
        "mtp_fusion.enorm.weight": (h,),
        "mtp_fusion.gate": (h,),
        "mtp_fusion.hnorm.weight": (h,),
    }
    leaves = {
        "input_layernorm.weight": (h,),
        "mlp.down_proj.weight": (h, ffn_width),
        "mlp.gate_proj.weight": (ffn_width, h),
        "mlp.up_proj.weight": (ffn_width, h),
        "post_attention_layernorm.weight": (h,),
        "self_attn.kv_a_layernorm.weight": (kv,),
        "self_attn.kv_a_proj_with_mqa.weight": (kv + rope, h),
        "self_attn.kv_b_proj.weight": (n * (nope + v), kv),
        "self_attn.learnable_sink_param": (n,),
        "self_attn.linear_gate.weight": (n * v, h),
        "self_attn.o_proj.weight": (h, n * v),
        "self_attn.q_a_layernorm.weight": (q,),
        "self_attn.q_a_proj.weight": (q, h),
        "self_attn.q_b_proj.weight": (n * (nope + rope), q),
    }
    if settings.post_proj_norm:
        leaves["self_attn.post_proj_norm.weight"] = (h,)
    for layer in range(len(settings.target_layers)):
        shapes.update({f"{layer}.{name}": shape for name, shape in leaves.items()})
    return {DRAFT_PREFIX + name: shape for name, shape in shapes.items()}


class DFlyWeights:
    def __init__(self, config, settings, weight_map, resolve_shard, tp):
        self.shards = defaultdict(list)
        for name, filename in weight_map.items():
            if name.startswith(DRAFT_PREFIX):
                self.shards[filename].append(name)
        first_gate = DRAFT_PREFIX + "0.mlp.gate_proj.weight"
        self.paths = {}
        headers = {}
        for filename, names in self.shards.items():
            path = self.paths[filename] = resolve_shard(filename)
            with safe_open(path, framework="pt", device="cpu") as shard:
                shard_names = set(shard.keys())
                for name in names:
                    if name not in shard_names:
                        raise ValueError(f"Missing DFly weight {name} in {filename}")
                    view = shard.get_slice(name)
                    headers[name] = (tuple(view.get_shape()), view.get_dtype())

        self.ffn_width = settings.ffn_width
        if self.ffn_width is None:
            # The pinned checkpoint does not declare its FFN width. Infer it
            # across all FFNs so one corrupt header is reported by its own name.
            widths: Counter[int] = Counter()
            for layer in range(len(settings.target_layers)):
                for projection, axis in (("gate", 0), ("up", 0), ("down", 1)):
                    name = f"{DRAFT_PREFIX}{layer}.mlp.{projection}_proj.weight"
                    shape, _ = headers.get(name, ((), None))
                    if len(shape) == 2 and shape[1 - axis] == config.hidden_size:
                        widths[shape[axis]] += 1
            if not widths:
                raise ValueError(f"Missing or invalid DFly FFN weight {first_gate}")
            self.ffn_width = widths.most_common(1)[0][0]
        if not self.ffn_width or self.ffn_width % tp:
            raise ValueError(
                f"Invalid DFly FFN width for {first_gate}: {self.ffn_width}"
            )
        if settings.correction_width % tp:
            raise ValueError(
                "DFly dfly_hidden_correction_intermediate_size must be divisible by TP"
            )
        expected = draft_shapes(config, settings, self.ffn_width)
        if missing := expected.keys() - headers.keys():
            raise ValueError(f"Missing DFly weights: {sorted(missing)}")
        if unexpected := headers.keys() - expected.keys():
            raise ValueError(f"Unexpected DFly weights: {sorted(unexpected)}")
        for name, shape in expected.items():
            if headers[name] != (shape, "BF16"):
                raise ValueError(
                    f"Invalid DFly weight {name}: expected BF16 {shape}, "
                    f"got {headers[name]}"
                )

    @torch.no_grad()
    def load_into(self, model):
        params = dict(model.named_parameters())
        for filename, names in self.shards.items():
            with safe_open(self.paths[filename], framework="pt", device="cpu") as shard:
                for name in names:
                    local = name.removeprefix(DRAFT_PREFIX)
                    if local.split(".", 1)[0].isdigit():
                        local = "layers." + local
                    weight = shard.get_tensor(name)
                    for projection, piece in (("gate_proj", 0), ("up_proj", 1)):
                        leaf = f".{projection}.weight"
                        if local.endswith(leaf):
                            param = params[
                                local.removesuffix(leaf) + ".gate_up_proj.weight"
                            ]
                            param.weight_loader(param, weight, piece)
                            break
                    else:
                        param = params[local]
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, weight)
