# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).
"""Flash latent experts with HYV4's routing, shared experts and weight format."""

import torch
from torch import nn

from vllm import envs
from vllm.config.execution_policy import communication_policy
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.hy4_awq_union_sm70 import (
    HY4_EXPERTS4B_MAX_TOKENS,
    hy4_union_ring_fold,
    load_hy4_sm70_operator,
)
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.models.hy_v4_flash.profile import speed_setting
from vllm.models.hy_v4_flash.sm70 import w8
from vllm.models.hy_v4_flash.sm70.dense import (
    FP16StorageGate,
    enable_skinny,
    fp16_storage_enabled,
)
from vllm.platforms import current_platform

from .boundary_ar import can_fuse_reduce_norm, reduce_norm, small_rows


class HYV4FlashFeedForward(HYV4FeedForward):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )
        w8.enable_linear(self.gate_up_proj, "MLP")
        w8.enable_linear(self.down_proj, "MLP")
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        return self.post_fc2_norm(result) if self.post_fc2_norm is not None else result


class HYV4FlashMoE(HYV4MoEFused):
    _f7_moe_fuse_override: bool | None = None

    @property
    def _gate_cls(self):
        return FP16StorageGate if fp16_storage_enabled() else super()._gate_cls

    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        latent = config.routed_expert_hidden_size
        down: nn.Module | None = None
        up: nn.Module | None = None
        if latent is not None:
            down = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            up = ReplicatedLinear(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            routed_hidden_size=latent,
            routed_input_transform=down,
            routed_output_transform=up,
        )
        projections = [(down, "LATENT"), (up, "LATENT")]
        if self.shared_experts is not None:
            projections.extend((
                (self.shared_experts.gate_up_proj, "SHARED"),
                (self.shared_experts.down_proj, "SHARED"),
            ))
        for projection, weight_class in projections:
            packed = w8.enable_linear(projection, weight_class)
            if not packed and not envs.HY4_F7_MOE_FUSE:
                enable_skinny(projection)
        self.experts.hy4_expert_kernel = speed_setting("VLLM_HY4_EXPERT_KERNEL")
        self.experts.hy4_expert_union_min_rows = speed_setting(
            "VLLM_HY4_EXPERT_UNION_MIN_ROWS"
        )
        self.experts.hy4_expert_union_fold = speed_setting("VLLM_HY4_EXPERT_UNION_FOLD")
        self.experts.hy4_experts3_ring = speed_setting("HY4_EXPERTS3_RING")
        self._experts4_shared = (
            self.shared_experts is not None and speed_setting("HY4_EXPERTS4_SHARED")
        )
        if self._experts4_shared:
            load_hy4_sm70_operator("experts4_ring_shared_out")
        self._experts4b_verify = self._experts4_shared and speed_setting(
            "HY4_EXPERTS4B_VERIFY"
        )
        router = self.experts.router
        if (
            speed_setting("VLLM_HY4_PARALLEL_ROUTER")
            and envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
            and current_platform.is_cuda()
            and current_platform.is_device_capability(70)
            and isinstance(router, GroupedTopKRouter)
            and router.num_expert_group == router.topk_group == 1
            and router.scoring_func == "sigmoid"
            and router.top_k == 16
            and config.num_experts == 512
        ):
            load_hy4_sm70_operator("parallel_top16")
            router.parallel_top16 = True
        self._boundary_router_input = speed_setting("HY4_BOUNDARY_ROUTER_INPUT")
        self._boundary_ar_norm = speed_setting("HY4_BOUNDARY_AR_NORM")
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def _f7_supported(self, x: torch.Tensor) -> bool:
        enabled = self._f7_moe_fuse_override
        if enabled is None:
            enabled = envs.HY4_F7_MOE_FUSE
        if (
            not enabled
            or envs.VLLM_BATCH_INVARIANT
            or not x.is_cuda
            or x.dtype != torch.float16
            or x.ndim != 2
            or x.shape[1] != 3072
            or not 1 <= x.shape[0] <= 24
            or not x.is_contiguous()
            or x.storage_offset() % 8 != 0
            or not current_platform.is_device_capability(70)
        ):
            return False
        experts = self.experts
        config = experts.moe_config
        runner = experts.runner
        if (
            not getattr(experts, "sm70_awq_hy4_union", False)
            or not getattr(experts.router, "parallel_top16", False)
            or x.shape[0] < experts.sm70_awq_hy4_union_min_rows
            or config.tp_size != 4
            or config.dp_size != 1
            or config.ep_size != 1
            or config.pcp_size != 1
            or config.is_sequence_parallel
            or runner._fused_output_is_reduced
            or runner.routed_scaling_factor != 1.0
            or getattr(runner, "output_transform", None) is not None
            or communication_policy().moe_add_allreduce
            or self.shared_experts is None
        ):
            return False
        projections = (
            (self.routed_expert_down_proj, (1536, 3072)),
            (self.shared_experts.gate_up_proj, (768, 3072)),
            (self.routed_expert_up_proj, (3072, 1536)),
            (self.shared_experts.down_proj, (3072, 384)),
        )
        return all(
            type(proj.quant_method) is UnquantizedLinearMethod
            and proj.weight.dtype == torch.float16
            and proj.weight.shape == shape
            and proj.weight.is_contiguous()
            and proj.weight.storage_offset() % 8 == 0
            for proj, shape in projections
        )

    def _router_logits(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._boundary_router_input
            and not envs.VLLM_BATCH_INVARIANT
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim == 2
            and 1 <= x.shape[0]
            and (x.shape[0] <= 4 or x.shape[0] <= small_rows())
            and x.shape[1] == 3072
            and x.is_contiguous()
            and (weight := self.gate.weight).dtype in (torch.float32, torch.float16)
            and weight.shape == (512, 3072)
            and weight.is_contiguous()
            and x.storage_offset() % 2 == 0
            and weight.storage_offset() % 4 == 0
            and self.gate.bias is None
            and current_platform.is_device_capability(70)
        ):
            load_hy4_sm70_operator("boundary_router_input")
            return torch.ops.hy4_sm70.boundary_router_input(x, weight)
        return super()._router_logits(x)

    def _f7_partials(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the baseline FP32 router product and exact selector. The
        # native AWQ method owns the existing plan, grouped GEMMs and buffers.
        logits = self._router_logits(x)
        shared = self.shared_experts
        assert shared is not None
        latent, shared_act = torch.ops.hy4_sm70.f7_moe_in(
            x, self.routed_expert_down_proj.weight, shared.gate_up_proj.weight
        )
        weights, ids = self.experts.router.select_experts(latent, logits)
        routed = self.experts.quant_method.apply(
            self.experts, latent, weights, ids, None, None
        )
        return torch.ops.hy4_sm70.f7_moe_out(
            routed,
            shared_act,
            self.routed_expert_up_proj.weight,
            shared.down_proj.weight,
        )

    def _experts4_supported(self, x: torch.Tensor) -> bool:
        """Ring-fold rows whose shared expert is the W8 g32 pair EXPERTS4 hosts."""
        experts = self.experts
        shared = self.shared_experts
        max_rows = HY4_EXPERTS4B_MAX_TOKENS if self._experts4b_verify else 4
        if not (
            self._experts4_shared
            and shared is not None
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim == 2
            and x.shape[1] == 3072
            and 1 <= x.shape[0] <= max_rows
            and x.is_contiguous()
            and x.storage_offset() % 8 == 0
            and getattr(experts, "sm70_awq_hy4_union", False)
            and x.shape[0] >= experts.sm70_awq_hy4_union_min_rows
            and hasattr(experts, "_awq_moe_buf_hy4_e4_tickets")
            and hy4_union_ring_fold(experts, x.shape[0], max_rows)
            and experts.runner.routed_scaling_factor == 1.0
            and experts.runner.routed_input_transform is not None
        ):
            return False
        return all(
            getattr(proj, "_hy4_w8_shape", None) == shape
            and getattr(proj.quant_method, "group_size", None) == 32
            and proj.weight.dtype == torch.uint8
            for proj, shape in (
                (shared.gate_up_proj, (768, 3072)),
                (shared.down_proj, (3072, 384)),
            )
        )

    def _experts4_partials(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The runner's partial path without the aux stream: same latent-down,
        # router selection and latent-up calls; the shared expert rides inside
        # the ring W13/W2 launches.
        experts = self.experts
        runner = experts.runner
        shared = self.shared_experts
        assert shared is not None
        logits = self._router_logits(x)
        experts.ensure_moe_quant_config_init()
        latent, _ = runner.apply_routed_input_transform(x)
        weights, ids = experts.router.select_experts(latent, logits)
        result = experts.quant_method.apply_hy4_experts4(
            experts, latent, weights, ids, x,
            shared.gate_up_proj.weight, shared.down_proj.weight,
        )
        if result is None:
            routed = experts.quant_method.apply(
                experts, latent, weights, ids, None, None
            )
            shared_out = shared(x)
        else:
            routed, shared_out = result
        return runner.apply_routed_output_transform(routed), shared_out

    def _experts4b_supported(self, x: torch.Tensor) -> bool:
        """5..8 rows (DFly verify) that super().forward would run unfused."""
        if not (self._experts4b_verify and x.ndim == 2 and x.shape[0] > 4):
            return False
        experts = self.experts
        config = experts.moe_config
        runner = experts.runner
        return (
            config.tp_size == 4
            and config.dp_size == config.ep_size == config.pcp_size == 1
            and not config.is_sequence_parallel
            and not runner._fused_output_is_reduced
            and getattr(runner, "output_transform", None) is None
            and isinstance(experts.router, GroupedTopKRouter)
            and self._experts4_supported(x)
        )

    def _experts4b_forward(self, x: torch.Tensor) -> torch.Tensor:
        # The runner's own combine on the EXPERTS4 partials, as super().forward
        # does: the sum2 all-reduce when it qualifies, else add then reduce.
        runner = self.experts.runner
        routed, shared = self._experts4_partials(x)
        hidden = x.shape[-1]
        result = runner._maybe_sm70_moe_sum2_allreduce(shared, routed, hidden)
        if result is None:
            result = runner._maybe_reduce_final_output(shared + routed, hidden)
        return result

    def _f7_forward(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(self._f7_partials(x))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fused_moe = self._f7_supported(hidden_states)
        if self._boundary_ar_norm and can_fuse_reduce_norm(
            hidden_states, self.combine_output_norm
        ):
            assert self.combine_output_norm is not None
            if fused_moe:
                return reduce_norm(
                    self._f7_partials(hidden_states), self.combine_output_norm
                )
            experts = self.experts
            config = experts.moe_config
            runner = experts.runner
            if (
                config.tp_size == 4
                and config.dp_size == config.ep_size == config.pcp_size == 1
                and not config.is_sequence_parallel
                and not runner._fused_output_is_reduced
                and getattr(runner, "output_transform", None) is None
                and isinstance(experts.router, GroupedTopKRouter)
                and not communication_policy().moe_add_allreduce
            ):
                if self._experts4_supported(hidden_states):
                    routed, shared = self._experts4_partials(hidden_states)
                    return reduce_norm(shared, self.combine_output_norm, routed)
                routed, shared = experts.forward_partials(
                    hidden_states, self._router_logits(hidden_states)
                )
                if shared is None:
                    return reduce_norm(routed, self.combine_output_norm)
                return reduce_norm(shared, self.combine_output_norm, routed)
        if fused_moe:
            result = self._f7_forward(hidden_states)
        elif self._experts4b_supported(hidden_states):
            result = self._experts4b_forward(hidden_states)
        else:
            result = super().forward(hidden_states)
        return (
            self.combine_output_norm(result)
            if self.combine_output_norm is not None
            else result
        )
