# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import (
    DeviceConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.hy_v4_flash.sm70 import indexpool
from vllm.models.hy_v4_flash.sm70.attention import HYV4FlashAttention
from vllm.models.hy_v4_flash.sm70.cache import (
    FlashMetadata,
    IndexPoolDecodeWorkspace,
    QuerySpan,
)
from vllm.models.hy_v4_flash.sm70.kernels import (
    index_scores,
    pack_fp8_ds_mla_torch,
    paged_attention,
)
from vllm.models.hy_v4_flash.sm70.rotary import rotary_phases
from vllm.models.hy_v4_flash.sm70.swa import window_indices
from vllm.models.hy_v4_flash.sm70.verification import VerificationState
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.utils.hashing import sha256
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_max_concurrency_for_kv_cache_config,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowMLASpec
from vllm.v1.request import Request
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup, compressed_kernel_block_size


def bind_runner_cache(owner, raw, cache_dtype, kernel_block_size):
    """Build the same cache view as the 1Cat V1 GPU runner, on this CPU buffer."""
    name = "test_owner"
    runner = object.__new__(GPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[True])
    runner.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    runner.runner_only_attn_layers = set()
    runner.attn_groups = [[AttentionGroup(owner.backend, [name], owner.spec, 0)]]
    return runner._reshape_kv_cache_tensors({name: raw}, [kernel_block_size])[name]


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def text_config():
    # Keep the real token limits, use smaller projection dimensions.
    return SimpleNamespace(
        layer_types=["indexed_attention", "sliding_attention"],
        hidden_size=32,
        q_lora_rank=16,
        kv_lora_rank=16,
        num_attention_heads=4,
        swa_num_attention_heads=8,
        qk_nope_head_dim=12,
        swa_qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        swa_qk_rope_head_dim=8,
        v_head_dim=16,
        swa_v_head_dim=16,
        rms_norm_eps=1e-5,
        sliding_window=512,
        compressed_indexer_compress_ratio=4,
        compressed_indexer_group_topk=512,
        compressed_indexer_num_attention_heads=4,
        compressed_indexer_head_dim=16,
        sandwich_norm=True,
        dsa_skip_attn_post_norm=False,
        rope_for_layer=lambda i: {"rope_theta": (1e7, 1e4)[i]},
    )


@pytest.fixture
def vllm_config():
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


def make_layer(
    config, model_config, idx, rows, monkeypatch, world=1, rank=0, device="cpu"
):
    tp = SimpleNamespace(world_size=world, rank_in_group=rank)
    monkeypatch.setattr(parallel_state, "_TP", tp)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.float16),
        torch.device(device),
    ):
        scratch = torch.empty(rows, 2048, dtype=torch.int32)
        return HYV4FlashAttention(
            config, model_config, idx, scratch, f"model.layers.{idx}.self_attn"
        )


def load_layer(layer, weights):
    params = dict(layer.named_parameters())
    loaded = set()
    for name, weight in weights.items():
        name = {
            "gate_proj.weight": "linear_gate.weight",
            "sinks": "learnable_sink_param",
        }.get(name, name.replace("compressed_indexer.", "indexer."))
        param = params[name]
        getattr(param, "weight_loader", default_weight_loader)(param, weight)
        loaded.add(name)
    assert loaded == set(params)


@pytest.fixture
def hf_layers():
    hf = pytest.importorskip("transformers.models.hy_v4_flash.modeling_hy_v4_flash")
    cfg = text_config()
    hf_cfg = hf.HYV4FlashTextConfig(
        num_hidden_layers=2,
        mlp_layer_types=["dense", "dense"],
        index_kpool=cfg.compressed_indexer_compress_ratio,
        index_topk=cfg.compressed_indexer_group_topk * cfg.compressed_indexer_compress_ratio,
        index_n_heads=cfg.compressed_indexer_num_attention_heads,
        index_head_dim=cfg.compressed_indexer_head_dim,
        rope_parameters={
            layer_type: {"rope_type": "default", "mrope_section": [2, 1, 1]}
            for layer_type in cfg.layer_types
        },
        **{
            k: v
            for k, v in vars(cfg).items()
            if k not in {"rope_for_layer", "sandwich_norm", "dsa_skip_attn_post_norm"}
            and not k.startswith("compressed_indexer_")
        },
    )
    hf_cfg._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(874)
        layers = [hf.HYV4FlashAttention(hf_cfg, i).eval() for i in range(2)]
        with torch.no_grad():
            for layer in layers:
                layer.sinks.uniform_(-2.0, 3.0)
                if layer.compressed_indexer is not None:
                    layer.compressed_indexer.intra_group_ape.normal_(std=0.3)
    return layers, hf.HYV4FlashRotaryEmbedding(hf_cfg)


class PagedLayer:
    """Feed the actual metadata builder and bind shuffled physical pages."""

    def __init__(self, model, config, length, requests=1, swa=None):
        self.layer = model
        self.config = config
        self.device = model.topk_indices_buffer.device
        self.owners = [model.attn]
        if model.indexer is not None:
            self.owners += [model.indexer.k_cache, model.indexer.tail_cache]
        if swa is not None:
            self.owners.append(swa.attn)
        self.tables = {}
        for owner in self.owners:
            spec = owner.spec
            ring = isinstance(spec, CircularBufferSpec)
            span = spec.global_block_size(model.dcp_world, 1)
            blocks = 1 if ring else (length + span - 1) // span
            page = (
                compressed_kernel_block_size(spec)
                if spec.storage_block_size != spec.block_size
                else spec.block_size
            )
            page_ratio = spec.storage_block_size // page
            table = torch.arange(
                1, requests * blocks + 1, device=self.device, dtype=torch.int32
            )
            self.tables[owner.prefix] = table.flip(0).view(requests, blocks)
            shape = owner.backend.get_kv_cache_shape(
                (requests * blocks + 1) * page_ratio,
                page,
                1,
                spec.head_size,
                spec.layout_cache_dtype(config.cache_config.cache_dtype),
            )
            # Pad the page stride without changing the contiguous layout of each row.
            storage = torch.zeros(
                (shape[0], shape[1] + 1, shape[2]), device=self.device, dtype=spec.dtype
            )
            owner.bind_kv_cache(storage[:, : shape[1]])

    @torch.inference_mode()
    def step(self, *requests, state=None, verify=(), request_ids=None):
        # Each entry is (query activations, request prefix length, RoPE offset).
        starts = torch.tensor(
            [0, *(len(x) for x, _, _ in requests)], dtype=torch.int32
        ).cumsum(0)
        lengths = torch.tensor([len(x) + p for x, p, _ in requests], dtype=torch.int32)
        positions = [
            torch.arange(p, p + len(x), device=self.device) for x, p, _ in requests
        ]
        metadata = {}
        for owner in self.owners:
            table = self.tables[owner.prefix]
            span = owner.spec.block_size
            slots = torch.empty(0, dtype=torch.int64, device=self.device)
            if owner is self.layer.attn or isinstance(owner.spec, SlidingWindowMLASpec):
                world = self.layer.dcp_world if owner.spec.dcp_sharded else 1
                rank = self.layer.dcp_rank if owner.spec.dcp_sharded else 0
                slots = torch.cat(
                    [
                        (
                            table[i, p // (span * world)].long() * span
                            + p // world % span
                        ).masked_fill(p % world != rank, -1)
                        for i, p in enumerate(positions)
                    ]
                )
            common = CommonAttentionMetadata(
                query_start_loc=starts.to(self.device),
                query_start_loc_cpu=starts,
                seq_lens=lengths.to(self.device),
                seq_lens_cpu_upper_bound=lengths,
                num_reqs=len(requests),
                num_actual_tokens=int(starts[-1]),
                max_query_len=max(len(x) for x, _, _ in requests),
                max_seq_len=int(lengths.max()),
                block_table_tensor=table,
                slot_mapping=slots,
            )
            builder = owner.backend.get_builder_cls()(
                owner.spec, [owner.prefix], self.config, self.device
            )
            metadata[owner.prefix] = builder.build(0, common)
            assert metadata[owner.prefix].query_start_loc is common.query_start_loc
        phase = torch.cat(
            [p + offset for p, (_, _, offset) in zip(positions, requests)]
        )
        scope = nullcontext()
        if state is not None:
            ids = range(len(requests)) if request_ids is None else request_ids
            spans = [
                (
                    str(i),
                    QuerySpan(
                        int(starts[row]),
                        int(starts[row + 1]),
                        prefix,
                        int(lengths[row]),
                    ),
                )
                for row, (i, (_, prefix, _)) in enumerate(zip(ids, requests))
                if len(requests[row][0])
            ]
            scoped = bool(verify) or any(state.has_pending(i) for i, _ in spans)
            scope = state.forward(
                metadata if scoped else None,
                {o.prefix: o for o in self.owners},
                spans,
                verify,
                {i: span.prefix + 1 for i, span in spans},
            )
        with scope, set_forward_context(metadata, self.config):
            return self.layer(phase, torch.cat([x for x, _, _ in requests]))

    def logical_indices(self, req, rows):
        layer = self.layer
        table = self.tables[layer.attn.prefix][req]
        ids = layer.topk_indices_buffer[rows, : layer.width].clone()
        page = layer.attn.kv_cache.shape[1]
        matches = (ids[..., None] // page == table).int().argmax(-1)
        return (matches * page + ids % page).masked_fill(ids < 0, -1)


def hf_attention(layer, rotary, x, offset=0, selected=None):
    n = len(x)
    positions = torch.arange(n)
    visible = positions[:, None] >= positions[None, :]
    if layer.attention_type == "sliding_attention":
        visible &= positions[:, None] - positions[None, :] < layer.config.sliding_window
    if selected is not None:
        visible &= selected
    mask = torch.zeros(n, n).masked_fill(~visible, -torch.inf)[None, None]
    phases = (positions + offset).expand(3, 1, n)
    cos_sin = rotary(x.float()[None], phases, layer_type=layer.attention_type)
    if selected is not None:
        with patch.object(layer, "compressed_indexer", None):
            return layer(x.float()[None], cos_sin, mask, position_ids=phases)
    return layer(x.float()[None], cos_sin, mask, position_ids=phases)


def verify_boundary_flips(ref, x, visible, picked, logits, positions):
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
        apply_rotary_pos_emb,
    )

    indexer = ref.compressed_indexer

    def rotate(value, phases):
        width = indexer.qk_rope_head_dim
        plain, rope = value.split([value.shape[-1] - width, width], dim=-1)
        rotated, _ = apply_rotary_pos_emb(rope, rope, *phases, unsqueeze_dim=2)
        return torch.cat((plain, rotated), dim=-1)

    x = x.float()[None]
    raw = indexer.token_key_norm(indexer.token_key_projection(x))
    keys = indexer.pool_groups(raw)
    g = keys.shape[1]
    # Absolute rotary offset of 11 above, while grouping follows logical slots.
    phase = (torch.arange(x.shape[1]) + 11)[None, None]
    kg = indexer.rotary_emb(keys, phase[..., : g * 4 : 4], "indexed_attention")
    keys = rotate(keys.unsqueeze(2), kg)
    hidden = x[:, positions]
    q = indexer.query_projection(ref.q_a_layernorm(ref.q_a_proj(hidden))).view(
        1, -1, indexer.n_heads, indexer.head_dim
    )
    qg = indexer.rotary_emb(q, phase[..., positions], "indexed_attention")
    q = rotate(q, qg)
    w = (
        indexer.head_weight_projection(hidden)
        * (indexer.n_heads * indexer.head_dim) ** -0.5
    )
    scores = ((q[0] @ keys[0, :, 0].T).relu() * w[0, :, :, None]).sum(1)
    for row, pos in enumerate(positions.tolist()):
        count = pos // 4
        expected = visible[pos, : count * 4 : 4]
        actual = picked[pos, : count * 4 : 4]
        errors = (logits[row, :count] - scores[row, :count]).abs()
        assert torch.isfinite(errors).all()
        assert errors.norm() / scores[row, :count].norm() < 5e-3
        if not torch.equal(actual, expected):
            assert actual.sum() == expected.sum() == 511
            assert (
                scores[row, :count][expected & ~actual].max()
                - scores[row, :count][actual & ~expected].min()
                <= 2 * errors.max()
            )


def assert_parity(out, expected):
    assert torch.isfinite(out).all()
    comparison = out.float() - expected.float()
    assert comparison.norm() / expected.float().norm() < 5e-3
    torch.testing.assert_close(out.float(), expected.float(), atol=6e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "idx,length,cuts",
    [
        (0, 2048, [1, 3, 4, 259, 2047]),
        (0, 2081, [3, 4, 101, 256, 2048, 2053, 2080]),
        (1, 519, [3, 511, 512, 513]),
    ],
)
@torch.inference_mode()
def test_layer_against_transformers(
    vllm_config, hf_layers, monkeypatch, idx, length, cuts
):
    layers, rotary = hf_layers
    ref = layers[idx]
    model = make_layer(vllm_config, text_config(), idx, length, monkeypatch)
    load_layer(model, ref.state_dict())
    x = torch.randn(length, 32, generator=torch.Generator().manual_seed(607)).half()
    expected, weights = hf_attention(ref, rotary, x, offset=11)
    whole = PagedLayer(model, vllm_config, length)
    select = indexpool.select_tokens
    with patch.object(indexpool, "select_tokens", wraps=select) as called:
        actual = whole.step((x, 0, 11))
    visible = weights[0, 0] > 0
    picked = whole.logical_indices(0, slice(None)).long()
    # Scatter into an extra column for -1 without treating padding as token 0.
    recovered = torch.zeros(length, length + 1, dtype=torch.bool)
    recovered.scatter_(1, picked.masked_fill(picked < 0, length), True)
    recovered = recovered[:, :length]
    if called.call_count:
        (logits, positions, _, _), _ = called.call_args
        assert torch.equal(recovered[:2048], visible[:2048])
        verify_boundary_flips(ref, x, visible, recovered, logits, positions)
        assert (actual.float() - expected[0]).norm() / expected[0].norm() < 5e-3
        expected, _ = hf_attention(ref, rotary, x, offset=11, selected=recovered)
    else:
        assert torch.equal(recovered, visible)
    assert_parity(actual, expected[0])
    if idx == 0:
        pool = model.indexer.k_cache
        first_group = whole.tables[pool.prefix][0, 0]
        closed = pool.kv_cache[first_group, 0].clone()

    paged = PagedLayer(model, vllm_config, length)
    outputs = []
    start = 0
    for end in [*cuts, length]:
        outputs.append(paged.step((x[start:end], start, 11)))
        assert torch.equal(
            paged.logical_indices(0, slice(0, end - start)), picked[start:end]
        )
        if idx == 0 and end in (3, 4):
            first_group = paged.tables[pool.prefix][0, 0]
            if end == 3:
                assert torch.count_nonzero(pool.kv_cache[first_group, 0]) == 0
            else:
                torch.testing.assert_close(pool.kv_cache[first_group, 0], closed)
        start = end
    assert_parity(torch.cat(outputs), actual)

    # The last eight rows sent as one chunk or as one-token decode steps.
    paged = PagedLayer(model, vllm_config, length)
    paged.step((x[:-8], 0, 11))
    decoded = []
    for p in range(length - 8, length):
        decoded.append(paged.step((x[p : p + 1], p, 11)))
        assert torch.equal(paged.logical_indices(0, slice(0, 1)), picked[p : p + 1])
    assert_parity(torch.cat(decoded), actual[-8:])


@torch.inference_mode()
@pytest.mark.parametrize("idx", [0, 1])
def test_tp4_weight_loading_and_output(vllm_config, hf_layers, monkeypatch, idx):
    weights = hf_layers[0][idx].state_dict()
    length = 11
    x = torch.randn(length, 32, generator=torch.Generator().manual_seed(67)).half()
    model = make_layer(vllm_config, text_config(), idx, length, monkeypatch)
    load_layer(model, weights)
    results, head_results, expected_heads = [], [], []
    handle = model.o_proj.register_forward_pre_hook(
        lambda _, args: expected_heads.append(args[0])
    )
    full = PagedLayer(model, vllm_config, length).step((x, 0, 0))
    handle.remove()
    for rank in range(4):
        shard = make_layer(
            vllm_config, text_config(), idx, length, monkeypatch, world=4, rank=rank
        )
        load_layer(shard, weights)
        shard.o_proj.reduce_results = False
        shard.post_proj_norm = None
        handle = shard.o_proj.register_forward_pre_hook(
            lambda _, args: head_results.append(args[0])
        )
        results.append(PagedLayer(shard, vllm_config, length).step((x, 0, 0)))
        handle.remove()
    torch.testing.assert_close(
        torch.cat(head_results, dim=-1), expected_heads[0], atol=0, rtol=0
    )
    reduced = torch.stack(results).float().sum(0).half()
    assert_parity(model.post_proj_norm(reduced), full)


@torch.inference_mode()
def test_mixed_requests_preserve_independent_group_tails(
    vllm_config, hf_layers, monkeypatch
):
    length = 21
    model = make_layer(vllm_config, text_config(), 0, length * 2, monkeypatch)
    load_layer(model, hf_layers[0][0].state_dict())
    rng = torch.Generator().manual_seed(45)
    a, b = torch.randn(2, length, 32, generator=rng).half().unbind()
    pool = model.indexer.k_cache

    def closed_groups(runner, req):
        return pool.kv_cache[runner.tables[pool.prefix][req, 0], : length // 4]

    solo = PagedLayer(model, vllm_config, length)
    expected_a = solo.step((a, 0, 17))
    pool_a = closed_groups(solo, 0).clone()
    solo = PagedLayer(model, vllm_config, length)
    expected_b = solo.step((b, 0, 223))
    pool_b = closed_groups(solo, 0).clone()
    paged = PagedLayer(model, vllm_config, length, requests=2)
    for a0, a1, b0, b1 in [(0, 3, 0, 5), (3, 4, 5, 12), (4, 21, 12, 21)]:
        result = paged.step((a[a0:a1], a0, 17), (b[b0:b1], b0, 223))
        assert_parity(result[: a1 - a0], expected_a[a0:a1])
        assert_parity(result[a1 - a0 :], expected_b[b0:b1])
    torch.testing.assert_close(closed_groups(paged, 0), pool_a, atol=0, rtol=0)
    torch.testing.assert_close(closed_groups(paged, 1), pool_b, atol=0, rtol=0)


def test_window_boundaries():
    rows = window_indices(torch.tensor([510, 511, 512]), 512)
    assert rows[0].tolist() == [*range(511), -1]
    assert rows[1].tolist() == list(range(512))
    assert rows[2].tolist() == list(range(1, 513))


@pytest.mark.parametrize(
    "n_swa,cache_dtype,world",
    [
        (0, "auto", 1),
        (30, "auto", 1),
        (30, "fp8_ds_mla", 1),
        (30, "auto", 2),
        (30, "fp8_ds_mla", 4),
    ],
)
def test_scheduler_page_layout(vllm_config, monkeypatch, n_swa, cache_dtype, world):
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    vllm_config.model_config = SimpleNamespace(max_model_len=8192)
    vllm_config.cache_config.cache_dtype = cache_dtype
    vllm_config.cache_config.gpu_memory_utilization = 0.95
    vllm_config.parallel_config.tensor_parallel_size = 4
    vllm_config.parallel_config.decode_context_parallel_size = world
    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=world, rank_in_group=0)
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    vllm_config.scheduler_config.max_num_batched_tokens = 2048
    vllm_config.scheduler_config.max_num_seqs = 4
    cfg = text_config()
    cfg.kv_lora_rank = 512
    cfg.qk_rope_head_dim = cfg.swa_qk_rope_head_dim = 64
    cfg.compressed_indexer_head_dim = 128
    cfg.layer_types = ["indexed_attention"] * 12 + ["sliding_attention"] * n_swa
    cfg.rope_for_layer = lambda i: {"rope_theta": 1e7 if i < 12 else 1e4}
    for idx in range(12 + n_swa):
        make_layer(vllm_config, cfg, idx, 1, monkeypatch, device="meta")
    layers = vllm_config.compilation_config.static_forward_context
    groups = get_kv_cache_groups(
        vllm_config, {name: layer.spec for name, layer in layers.items()}
    )
    # Use the profiled per-rank KV budget at 0.95 utilization for both layouts.
    cache = get_kv_cache_config_from_groups(vllm_config, groups, int(2.9 * (1 << 30)))
    if n_swa == 30 and world == 1:
        tokens = int(8192 * get_max_concurrency_for_kv_cache_config(vllm_config, cache))
        expected = (1126, 129918) if cache_dtype == "fp8_ds_mla" else (666, 76843)
        assert (cache.num_blocks, tokens) == expected
        print(f"TP4 {cache_dtype}: {cache.num_blocks} pool blocks, {tokens} tokens")
    scheduler_cache = generate_scheduler_kv_cache_config([cache])
    vllm_config.cache_config.enable_prefix_caching = True
    scheduling, hashing = resolve_kv_cache_block_sizes(scheduler_cache, vllm_config)
    assert scheduling == 256
    assert hashing == 4
    coordinator = HybridKVCacheCoordinator(
        scheduler_cache,
        8192,
        2048,
        False,
        True,
        False,
        world,
        1,
        hashing,
    )
    assert len(set(name for group in groups for name in group.layer_names)) == len(
        layers
    )
    assert (
        sum(isinstance(layer.spec, SlidingWindowMLASpec) for layer in layers.values())
        == n_swa
    )
    ring_groups = [
        i for i, group in enumerate(groups) if not group.kv_cache_spec.prefix_cacheable
    ]
    assert len(ring_groups) == 1
    ring = coordinator.single_type_managers[ring_groups[0]]
    assert len(ring.allocate_new_blocks("request", 2053, 2053)) == 1
    assert ring.allocate_new_blocks("request", 2081, 2081) == []
    # Allocate just two real allocator strides, then check each owner can bind.
    for tensor in cache.kv_cache_tensors:
        stride = tensor.size // cache.num_blocks
        raw = torch.empty(stride * 2, dtype=torch.int8)
        for name in tensor.shared_by:
            layer = layers[name]
            spec = layer.spec
            page = (
                compressed_kernel_block_size(spec)
                if spec.storage_block_size != spec.block_size
                else spec.block_size
            )
            shape = layer.backend.get_kv_cache_shape(
                2 * spec.storage_block_size // page,
                page,
                1,
                spec.head_size,
                spec.layout_cache_dtype(cache_dtype),
            )
            layer.bind_kv_cache(bind_runner_cache(layer, raw, cache_dtype, page))
            assert layer.kv_cache.shape == shape
            assert layer.kv_cache.stride(0) * layer.kv_cache.element_size() == stride
            if world > 1:
                replicated = isinstance(
                    spec, (CircularBufferSpec, SlidingWindowMLASpec)
                )
                assert spec.dcp_sharded == (not replicated)
                assert spec.block_size == (
                    4 if isinstance(spec, CircularBufferSpec) else 256 // world
                )
                if spec.storage_block_size != spec.block_size:
                    assert page == 64 // world
            if isinstance(spec, CircularBufferSpec):
                assert spec.dtype == torch.float32
            elif cache_dtype == "fp8_ds_mla" and name.endswith(".attn"):
                assert spec.dtype == torch.uint8
                assert shape[-1] == 656
                assert spec.real_page_size_bytes == spec.block_size * 656
            else:
                assert spec.dtype == torch.float16


@torch.inference_mode()
def test_page_aligned_prefix_reuse(vllm_config, hf_layers, monkeypatch):
    prefix, length = 2048, 2061
    cfg = text_config()
    x = torch.randn(length, 32, generator=torch.Generator().manual_seed(92)).half()
    vllm_config.model_config = SimpleNamespace(max_model_len=4096)
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    models = [make_layer(vllm_config, cfg, i, length, monkeypatch) for i in range(2)]
    for i, model in enumerate(models):
        load_layer(model, hf_layers[0][i].state_dict())
    # Compare against the same split without a cache hit, so the batch shapes match.
    expected = []
    for model in models:
        cold = PagedLayer(model, vllm_config, length)
        cold.step((x[:prefix], 0, 0))
        expected.append(cold.step((x[prefix:], prefix, 0)))
    owners = vllm_config.compilation_config.static_forward_context
    groups = get_kv_cache_groups(
        vllm_config, {name: layer.spec for name, layer in owners.items()}
    )
    cache = get_kv_cache_config_from_groups(vllm_config, groups, 2 << 20)
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([cache]),
        max_model_len=4096,
        max_in_flight_tokens=2048,
        hash_block_size=256,
        enable_caching=True,
    )
    paged = [PagedLayer(model, vllm_config, length) for model in models]
    for tensor in cache.kv_cache_tensors:
        raw = torch.zeros(tensor.size, dtype=torch.int8)
        for name in tensor.shared_by:
            owner = owners[name]
            spec = owner.spec
            page = (
                compressed_kernel_block_size(spec)
                if spec.storage_block_size != spec.block_size
                else spec.block_size
            )
            owner.bind_kv_cache(
                bind_runner_cache(
                    owner, raw, vllm_config.cache_config.cache_dtype, page
                )
            )
    init_none_hash(sha256)
    hasher = get_request_block_hasher(256, sha256)
    first = Request(
        "first",
        list(range(prefix)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=hasher,
    )
    assert manager.allocate_slots(first, prefix) is not None
    group_by_owner = {
        name: i for i, group in enumerate(groups) for name in group.layer_names
    }

    def set_tables(request, runner):
        block_ids = manager.get_block_ids(request.request_id)
        for owner in runner.owners:
            runner.tables[owner.prefix] = torch.tensor(
                [block_ids[group_by_owner[owner.prefix]]], dtype=torch.int32
            )

    for runner in paged:
        set_tables(first, runner)
        runner.step((x[:prefix], 0, 0))
    manager.free(first)
    second = Request(
        "second",
        list(range(length)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=hasher,
    )
    blocks, hit = manager.get_computed_blocks(second)
    assert hit == prefix
    ring_gid = group_by_owner[models[0].indexer.tail_cache.prefix]
    assert not blocks.blocks[ring_gid]
    assert manager.allocate_slots(second, length - hit, hit, blocks) is not None
    for runner, target in zip(paged, expected):
        set_tables(second, runner)
        result = runner.step((x[prefix:], prefix, 0))
        torch.testing.assert_close(result, target, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize("heads,width", [(8, 2048), (16, 512)])
def test_cuda_latent_aggregation(heads, width):
    from vllm import _custom_ops as ops
    from vllm.models.hy_v4_flash.sm70.kernels import paged_attention_torch

    torch.manual_seed(23)
    q = torch.randn(4, heads, 576, device="cuda").half()
    # Include an unscaled QK logit beyond FP16 range and non-zero sinks.
    q[0, :, :512].fill_(40)
    latent = torch.randn(width, 512, device="cuda").half()
    latent[0].fill_(40)
    rotary = torch.randn(width, 64, device="cuda").half()
    sinks = torch.linspace(-2, 5, heads, device="cuda")
    slots = torch.arange(width, device="cuda").reshape(-1, 64).flip(0).flatten()
    cache = torch.zeros(width // 64, 65, 576, device="cuda", dtype=torch.float16)[
        :, :64
    ]
    ops.concat_and_cache_mla(latent, rotary, cache, slots, "auto", sinks[:1])
    torch.testing.assert_close(
        cache[slots // 64, slots % 64], torch.cat((latent, rotary), -1), atol=0, rtol=0
    )
    indices = slots.int().expand(4, -1).contiguous()
    counts = torch.tensor([1, width, width - 3, 511], dtype=torch.int32, device="cuda")
    expected = paged_attention_torch(
        q.cpu(), cache.cpu(), indices.cpu(), counts.cpu(), sinks.cpu(), 1 / 16, 512
    )
    actual = paged_attention(q, cache, indices, counts, sinks, 1 / 16, 512)
    assert_parity(actual.cpu(), expected)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
def test_cuda_weighted_pool_scores():
    torch.manual_seed(41)
    query = torch.randn(7, 16, 128, device="cuda").half()
    weights = torch.randn(7, 16, device="cuda") / 32
    pool = torch.randn(9, 65, 128, device="cuda").half()[:, :64]
    table = torch.tensor([8, 2, 6, 0, 4, 3, 1, 5, 7], device="cuda", dtype=torch.int32)
    groups = torch.arange(522)
    keys = pool.cpu()[table.cpu()[groups // 64], groups % 64].float()
    expected = ((query.cpu().float() @ keys.T).relu() * weights.cpu()[..., None]).sum(1)
    expected.masked_fill_(groups >= (2054 + torch.arange(7)[:, None]) // 4, -torch.inf)
    actual = index_scores(query, weights, pool, table, 2054, 522, 4, 1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=1e-4)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_decode_metadata_reuses_runner_buffers(vllm_config, world):
    from vllm.models.hy_v4_flash.sm70.cache import FlashMetadataBuilder
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    vllm_config.parallel_config.decode_context_parallel_size = world
    spec = MLAAttentionSpec(
        block_size=256 // world,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float16,
        dcp_sharded=world > 1,
    )
    builder = FlashMetadataBuilder(spec, ["attn"], vllm_config, torch.device("cpu"))
    assert builder.get_cudagraph_support(vllm_config, spec) == (
        AttentionCGSupport.UNIFORM_BATCH
    )
    starts = torch.arange(4, dtype=torch.int32)
    lengths = torch.ones(3, dtype=torch.int32)
    table = torch.zeros(3, 10, dtype=torch.int32)
    slots = torch.full((3,), -1, dtype=torch.int64)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=lengths,
        num_reqs=3,
        num_actual_tokens=3,
        max_query_len=1,
        max_seq_len=2304,
        block_table_tensor=table,
        slot_mapping=slots,
        is_dummy_batch=True,
    )
    # Exercise the CUDA decode branch with CPU storage and no host reads.
    with (
        patch.object(torch.Tensor, "is_cuda", True),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError),
        patch.object(torch.Tensor, "item", side_effect=AssertionError),
    ):
        captured = builder.build_for_cudagraph_capture(common)
    assert captured.decode and captured.num_actual_tokens == 3
    assert not captured.spans
    for rows, batch_lens in [(3, [2049, 512, 0]), (2, [520, 0])]:
        valid = len(batch_lens) - 1
        starts[valid + 1 :].fill_(valid)
        lengths[:rows].copy_(torch.tensor(batch_lens))
        table[:valid, 0].fill_(7)
        slots[:valid].fill_(7 * spec.block_size)
        table[valid:].zero_()
        slots[valid:].fill_(-1)
        step = common.replace(
            query_start_loc=starts[: rows + 1],
            query_start_loc_cpu=starts[: rows + 1],
            seq_lens=lengths[:rows],
            block_table_tensor=table[:rows],
            slot_mapping=slots[:rows],
            num_reqs=rows,
            num_actual_tokens=rows,
            max_seq_len=max(batch_lens),
            is_dummy_batch=False,
        )
        with (
            patch.object(torch.Tensor, "is_cuda", True),
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError),
            patch.object(torch.Tensor, "item", side_effect=AssertionError),
        ):
            replay = builder.build(0, step)
        assert replay.decode and not replay.spans
        for field in ("seq_lens", "query_start_loc", "block_table", "slot_mapping"):
            assert (
                getattr(replay, field).data_ptr() == getattr(captured, field).data_ptr()
            )
        assert replay.num_actual_tokens == rows
        assert replay.seq_lens[-1] == 0
        assert replay.query_start_loc[-1] == replay.query_start_loc[-2]
        assert replay.slot_mapping[-1] == -1
        assert replay.block_table[-1].eq(0).all()
    one_token_prefill = common.replace(
        seq_lens_cpu_upper_bound=lengths,
        is_dummy_batch=False,
        is_prefilling=torch.tensor([True, False, False]),
    )
    assert not builder.build(0, one_token_prefill).decode


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_ds_mla"])
@torch.inference_mode()
def test_cuda_decode_graph_replays_eager(
    vllm_config, monkeypatch, workspace_init, cache_dtype
):
    config = text_config()
    config.qk_rope_head_dim = config.swa_qk_rope_head_dim = 16
    vllm_config.cache_config.cache_dtype = cache_dtype
    prefixes: tuple[int, ...] = (2051, 512)
    requests: tuple[tuple[int, ...], ...] = ((0, 1), (1,))
    if cache_dtype == "fp8_ds_mla":
        config.kv_lora_rank = 512
        config.qk_rope_head_dim = config.swa_qk_rope_head_dim = 64
        config.compressed_indexer_head_dim = 128
        config.num_attention_heads = 8
        config.swa_num_attention_heads = 16
        prefixes = (2047, 2051, 510, 511)
        requests = ((1, 2, 3, 0), (3,))
    vllm_config.model_config = SimpleNamespace(
        max_model_len=2304, hf_text_config=config
    )
    device = torch.device("cuda")
    rows = 4
    torch.manual_seed(79)
    layers = [
        PagedLayer(
            make_layer(
                vllm_config, config, idx, sum(prefixes), monkeypatch, device=device
            ),
            vllm_config,
            2304,
            requests=rows,
        )
        for idx in range(2)
    ]
    _check_cuda_decode_graph(vllm_config, config, layers, prefixes, {rows: requests})


def _check_cuda_decode_graph(vllm_config, config, layers, prefixes, replay_batches):
    from vllm.v1.worker.workspace import lock_workspace

    device = layers[0].device
    rows = max(replay_batches)
    world = vllm_config.parallel_config.decode_context_parallel_size
    for paged in layers:
        for weight in paged.layer.parameters():
            weight.normal_(std=0.15)
        paged.step(
            *(
                (torch.randn(n, config.hidden_size, device=device).half(), 0, 0)
                for n in prefixes
            )
        )

    owners = [owner for paged in layers for owner in paged.owners]
    initial = [owner.kv_cache.clone() for owner in owners]
    starts = torch.empty(rows + 1, dtype=torch.int32)
    starts_gpu = torch.empty_like(starts, device=device)
    lengths = torch.empty(rows, dtype=torch.int32, device=device)
    positions = torch.empty(rows, dtype=torch.int64, device=device)
    inputs = torch.randn(rows, config.hidden_size, device=device).half()
    tables = {
        owner.prefix: paged.tables[owner.prefix]
        for paged in layers
        for owner in paged.owners
    }
    stable_tables = {name: torch.empty_like(table) for name, table in tables.items()}
    slots = {
        owner.prefix: torch.empty(rows, dtype=torch.int64, device=device)
        for owner in owners
    }
    builders = {
        owner.prefix: owner.backend.get_builder_cls()(
            owner.spec, [owner.prefix], vllm_config, device
        )
        for owner in owners
    }

    def stage(requests, batch_size, dummy=False):
        starts.copy_(
            torch.arange(rows + 1).clamp_max(batch_size if dummy else len(requests))
        )
        starts_gpu.copy_(starts)
        lengths.fill_(1 if dummy else 0)
        positions.zero_()
        for row, req in enumerate(requests):
            lengths[row] = prefixes[req] + 1
            positions[row] = prefixes[req]
        metadata = {}
        for owner in owners:
            name = owner.prefix
            table = stable_tables[name][:batch_size]
            table.zero_()
            slots[name].fill_(-1)
            for row, req in enumerate(requests):
                table[row].copy_(tables[name][req])
                if name in main_prefixes:
                    page = owner.kv_cache.shape[1]
                    pos = prefixes[req]
                    if owner.spec.dcp_sharded:
                        if pos % world != parallel_state.get_dcp_group().rank_in_group:
                            continue
                        pos //= world
                    slots[name][row].copy_(table[row, pos // page] * page + pos % page)
            common = CommonAttentionMetadata(
                query_start_loc=starts_gpu[: batch_size + 1],
                query_start_loc_cpu=starts[: batch_size + 1],
                seq_lens=lengths[:batch_size],
                num_reqs=batch_size,
                num_actual_tokens=batch_size,
                max_query_len=1,
                max_seq_len=max(
                    (prefixes[req] + 1 for req in requests),
                    default=vllm_config.model_config.max_model_len,
                ),
                block_table_tensor=table,
                slot_mapping=slots[name][:batch_size],
                is_dummy_batch=dummy,
            )
            builder = builders[name]
            metadata[name] = (
                builder.build_for_cudagraph_capture(common)
                if dummy
                else builder.build(0, common)
            )
        return metadata

    def forward(metadata, batch_size):
        with set_forward_context(metadata, vllm_config):
            return tuple(
                paged.layer(positions[:batch_size], inputs[:batch_size])
                for paged in layers
            )

    main_prefixes = {paged.layer.attn.prefix for paged in layers}
    positions.zero_()
    forward(None, rows)
    lock_workspace()
    for batch_size, replay_requests in replay_batches.items():
        expected_steps = []
        for requests in replay_requests:
            eager = forward(stage(requests, batch_size), batch_size)
            expected_caches = [owner.kv_cache.clone() for owner in owners]
            expected_steps.append((requests, eager, expected_caches))
            for owner, before in zip(owners, initial):
                owner.kv_cache.copy_(before)

        dummy = stage((), batch_size, dummy=True)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward(dummy, batch_size)
        torch.cuda.current_stream().wait_stream(stream)
        capture = (
            parallel_state.graph_capture(
                device, parallel_state.GraphCaptureContext(stream)
            )
            if world > 1
            else nullcontext()
        )
        graph = torch.cuda.CUDAGraph()
        with capture, torch.cuda.graph(graph, stream=stream):
            captured = forward(dummy, batch_size)
        torch.cuda.current_stream().wait_stream(stream)
        for owner, before in zip(owners, initial):
            assert torch.equal(
                owner.kv_cache.view(torch.uint8), before.view(torch.uint8)
            )

        for requests, eager, expected_caches in expected_steps:
            metadata = stage(requests, batch_size)
            graph.replay()
            for actual, expected in zip(captured, eager):
                assert torch.isfinite(expected).all()
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            for paged in layers:
                layer = paged.layer
                pad = slice(len(requests), batch_size)
                local_width = layer.width // layer.dcp_world
                assert layer.topk_indices_buffer[pad, :local_width].eq(-1).all()
                counts = metadata[layer.attn.prefix].counts
                assert counts[pad].eq(0).all()
                expected_counts = [
                    min(prefixes[req] + 1, layer.width) for req in requests
                ]
                expected_counts = [
                    (n + layer.dcp_world - 1 - layer.dcp_rank) // layer.dcp_world
                    for n in expected_counts
                ]
                assert counts[: len(requests)].tolist() == expected_counts
            for owner, before, expected in zip(owners, initial, expected_caches):
                assert torch.equal(
                    owner.kv_cache.view(torch.uint8), expected.view(torch.uint8)
                )
                pages_per_block = (
                    owner.spec.storage_block_size // owner.kv_cache.shape[1]
                )
                assert torch.equal(
                    owner.kv_cache[:pages_per_block], before[:pages_per_block]
                )
                owner.kv_cache.copy_(before)


@torch.inference_mode()
def _cuda_hy4_dcp_worker(rank, world, cache_dtype, rendezvous, check):
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    parallel_config = ParallelConfig(
        tensor_parallel_size=4, decode_context_parallel_size=world
    )
    vc = VllmConfig(
        device_config=DeviceConfig(device=device), parallel_config=parallel_config
    )
    with set_current_vllm_config(vc):
        parallel_state.init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=rendezvous,
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=parallel_config.tensor_parallel_size,
            decode_context_model_parallel_size=parallel_config.decode_context_parallel_size,
        )
    init_workspace_manager(device)
    try:
        check(rank, world, cache_dtype, device)
    finally:
        reset_workspace_manager()
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


def _check_cuda_hy4_dcp(rank, world, cache_dtype, device):
    from vllm.v1.worker.workspace import lock_workspace

    config = text_config()
    config.num_attention_heads = 32
    config.kv_lora_rank = 512
    config.qk_rope_head_dim = config.swa_qk_rope_head_dim = 64
    config.compressed_indexer_num_attention_heads = 16
    config.compressed_indexer_head_dim = 128
    torch.manual_seed(45)
    batches = [
        ((2047, 0), (1, 0)),
        ((3, 2047), (1, 1)),  # Mixed prefill and decode, closing the old ring.
        ((1, 2050), (1, 2)),
        ((1, 2051), (1, 3)),  # Group close also occurs on the decode path.
    ]
    inputs = [
        tuple(
            (torch.randn(n, config.hidden_size, device=device).half(), p, 0)
            for n, p in batch
        )
        for batch in batches
    ]
    reference: list[torch.Tensor] = []
    actual: list[torch.Tensor] = []
    weights = None
    for size, results in ((1, reference), (world, actual)):
        vc = VllmConfig(device_config=DeviceConfig(device=device))
        vc.parallel_config.decode_context_parallel_size = size
        vc.model_config = SimpleNamespace(max_model_len=2304, hf_text_config=config)
        vc.cache_config.cache_dtype = cache_dtype
        vc.scheduler_config.max_num_seqs = 2
        with (
            set_current_vllm_config(vc),
            set_default_torch_dtype(torch.float16),
            device,
        ):
            indices = torch.empty(2048, 2048, dtype=torch.int32)
            sparse = HYV4FlashAttention(
                vc, config, 0, indices, "model.layers.0.self_attn"
            )
            swa = HYV4FlashAttention(vc, config, 1, indices, "model.layers.1.self_attn")
        if weights is None:
            for weight in sparse.parameters():
                weight.normal_(std=0.12)
            sparse.q_b_proj.weight.add_(rank * 0.03)
            sparse.learnable_sink_param.add_(rank * 0.25)
            weights = sparse.state_dict()
        else:
            sparse.load_state_dict(weights)
        if size > 1:
            with set_forward_context(None, vc):
                dummy = torch.cat([x for x, _, _ in inputs[0]])
                sparse(torch.arange(len(dummy), device=device), dummy)
            lock_workspace()
        paged = PagedLayer(sparse, vc, 2304, requests=2, swa=swa)
        for batch in inputs:
            results.append(paged.step(*batch))
    for out, expected in zip(actual, reference):
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.accelerator.device_count() < 4,
    reason="TP4 requires four CUDA GPUs",
)
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_ds_mla"])
def test_cuda_hy4_dcp_matches_tp4_eager(world, cache_dtype, tmp_path):
    torch.multiprocessing.spawn(
        _cuda_hy4_dcp_worker,
        args=(
            world,
            cache_dtype,
            (tmp_path / "rendezvous").as_uri(),
            _check_cuda_hy4_dcp,
        ),
        nprocs=4,
    )


def _check_cuda_hy4_dcp_graph(rank, world, cache_dtype, device):
    config = text_config()
    config.num_attention_heads = 32
    config.swa_num_attention_heads = 64
    config.kv_lora_rank = 512
    config.qk_rope_head_dim = config.swa_qk_rope_head_dim = 64
    config.compressed_indexer_num_attention_heads = 16
    config.compressed_indexer_head_dim = 128
    vc = VllmConfig(device_config=DeviceConfig(device=device))
    vc.parallel_config.decode_context_parallel_size = world
    vc.model_config = SimpleNamespace(max_model_len=2304, hf_text_config=config)
    vc.cache_config.cache_dtype = cache_dtype
    vc.scheduler_config.max_num_seqs = 4
    prefixes = (2047, 2051, 510, 511)
    torch.manual_seed(79)
    with (
        set_current_vllm_config(vc),
        set_default_torch_dtype(torch.float16),
        device,
    ):
        layers = [
            PagedLayer(
                HYV4FlashAttention(
                    vc,
                    config,
                    idx,
                    torch.empty(sum(prefixes), 2048, dtype=torch.int32),
                    f"model.layers.{idx}.self_attn",
                ),
                vc,
                2304,
                requests=4,
            )
            for idx in range(2)
        ]
    _check_cuda_decode_graph(
        vc,
        config,
        layers,
        prefixes,
        {1: ((0,), (1,)), 2: ((2, 3), (1, 0)), 4: ((1, 2, 3, 0), (3,))},
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.accelerator.device_count() < 4,
    reason="TP4 requires four CUDA GPUs",
)
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_ds_mla"])
def test_cuda_hy4_dcp_graph_replays_eager(world, cache_dtype, tmp_path):
    torch.multiprocessing.spawn(
        _cuda_hy4_dcp_worker,
        args=(
            world,
            cache_dtype,
            (tmp_path / "rendezvous").as_uri(),
            _check_cuda_hy4_dcp_graph,
        ),
        nprocs=4,
    )


def test_index_prefill_profiles_transient_scratch_at_depth(monkeypatch):
    from vllm.models.hy_v4_flash.sm70 import kernels
    from vllm.v1.worker import workspace

    heads, dim, ratio = 16, 128, 4
    allocations = []
    allocate = kernels.allocate_index_prefill_scratch

    def record(device, specs):
        buffers = allocate(device, specs)
        allocations.append(sum(t.nbytes for t in buffers))
        return buffers

    monkeypatch.setattr(indexpool, "allocate_index_prefill_scratch", record)
    monkeypatch.setattr(indexpool, "select_history", lambda *args: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    cases = (
        (1_048_576, 1, 2048, 512, 192, 720),
        (262_144, 1, 2048, 512, 192, 720),
        (1_048_576, 2, 2048, 512, 192, 720),
        (1_048_576, 4, 2048, 512, 192, 720),
        (1_048_576, 4, 2048, 512, 256, 784),
        (1_048_576, 4, 2048, 384, 512, 912),
    )
    for length, world, rows, logits_mb, tile_mb, scratch_mb in cases:
        device = torch.device("meta")
        manager = workspace.WorkspaceManager(device)
        monkeypatch.setattr(workspace, "_manager", manager)
        groups = (length + ratio * world - 1) // (ratio * world)
        logits_budget = logits_mb * 2**18
        layer = SimpleNamespace(
            dcp_group=object() if world > 1 else None,
            dcp_world=world,
            max_decode_rows=4,
            max_model_len=length,
            max_local_groups=groups,
            width=2048,
            history=511,
            num_heads=heads,
            dim=dim,
            prefill_tile_mb=tile_mb,
            logits_budget=logits_budget,
        )
        query = torch.empty((rows, heads, dim), dtype=torch.float16, device=device)
        indexpool.FlashCompressedIndexer.profile(layer, query)

        assert allocations[-1] == scratch_mb * 2**20
        reserved = manager._current_workspaces[0]
        if world == 1:
            assert reserved is None
        else:
            assert reserved.nbytes == layer.max_decode_rows * groups * 4

        for served in {length, min(length, 262_144), min(length, 283_392)}:
            live_groups = (served + ratio * world - 1) // (ratio * world)
            chunk = max(1, min(rows, logits_budget // live_groups))
            for live_rows in (1, chunk - 1, chunk):
                specs = kernels.index_prefill_workspace_specs(
                    live_rows, live_groups, heads, dim, tile_mb, torch.float16
                )
                assert sum(t.nbytes for t in allocate(device, specs)) <= allocations[-1]


@pytest.mark.parametrize("idx", [0, 1])
@torch.inference_mode()
def test_mixed_verification_uses_accepted_state(
    vllm_config, hf_layers, monkeypatch, idx
):
    cfg = text_config()
    cfg.compressed_indexer_group_topk = 4
    rng = torch.Generator().manual_seed(761)
    prefix = (510, 507, 5)
    widths = (6, 6, 3)
    accepted = (1, 5, 3)
    length = 530
    model = make_layer(vllm_config, cfg, idx, sum(prefix), monkeypatch)
    load_layer(model, hf_layers[0][idx].state_dict())
    x = torch.randn(3, length, cfg.hidden_size, generator=rng).half()
    replay = torch.randn(3, 5, cfg.hidden_size, generator=rng).half()
    offsets = (17, 223, 41)

    def seed():
        runner = PagedLayer(model, vllm_config, length, requests=3)
        runner.step(*((x[i, :p], 0, offsets[i]) for i, p in enumerate(prefix)))
        return runner

    reference = seed()
    accepted_outputs = [[] for _ in prefix]
    for n in range(max(accepted)):
        reference_rows = [
            (
                x[i, p + n : p + min(n + 1, accepted[i])],
                p + min(n, accepted[i]),
                offsets[i],
            )
            for i, p in enumerate(prefix)
        ]
        output = reference.step(*reference_rows)
        row = 0
        for i, (values, _, _) in enumerate(reference_rows):
            if len(values):
                accepted_outputs[i].append(output[row : row + len(values)])
                row += len(values)
    replay_steps = [
        [
            (replay[i, n : n + 1], p + accepted[i] + n, offsets[i])
            for i, p in enumerate(prefix)
        ]
        for n in range(replay.shape[1])
    ]
    reference_steps = []
    for rows in replay_steps:
        output = reference.step(*rows)
        slots = model.topk_indices_buffer[: len(output), : model.width].clone()
        reference_steps.append((output, slots))

    verified = seed()
    state = VerificationState()
    verified_output = verified.step(
        *((x[i, p : p + widths[i]], p, offsets[i]) for i, p in enumerate(prefix)),
        state=state,
        verify=("0", "1"),
    )
    first = 0
    for width, count, outputs in zip(widths, accepted, accepted_outputs):
        torch.testing.assert_close(
            verified_output[first : first + count],
            torch.cat(outputs),
            atol=0,
            rtol=0,
        )
        first += width
    for rows, (expected, expected_slots) in zip(replay_steps, reference_steps):
        actual = verified.step(*rows, state=state)
        actual_slots = model.topk_indices_buffer[: len(actual), : model.width]
        assert torch.equal(actual_slots, expected_slots)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("accepted", range(7))
def test_verification_window_reclamation_uses_processed_cursor(
    vllm_config, monkeypatch, accepted
):
    cfg = text_config()
    vllm_config.model_config = SimpleNamespace(max_model_len=2048)
    for idx in range(2):
        make_layer(vllm_config, cfg, idx, 1, monkeypatch)
    owners = vllm_config.compilation_config.static_forward_context
    groups = get_kv_cache_groups(
        vllm_config, {name: layer.spec for name, layer in owners.items()}
    )
    cache = get_kv_cache_config_from_groups(vllm_config, groups, 2 << 20)
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([cache]),
        max_model_len=2048,
        max_in_flight_tokens=2048,
        hash_block_size=256,
        enable_caching=True,
    )
    init_none_hash(sha256)
    hasher = get_request_block_hasher(256, sha256)
    prefix = 765
    request = Request(
        "verify",
        list(range(prefix)),
        SamplingParams(max_tokens=30),
        None,
        block_hasher=hasher,
    )
    assert manager.allocate_slots(request, prefix) is not None
    request.num_computed_tokens = prefix
    request.append_output_token_ids([1001])
    assert manager.allocate_slots(request, 6) is not None

    def blocks(spec):
        return next(
            m
            for m in manager.coordinator.single_type_managers
            if isinstance(m.kv_cache_spec, spec)
        ).req_to_blocks[request.request_id]

    window = blocks(SlidingWindowMLASpec)
    ring = blocks(CircularBufferSpec)[0]
    saved = window[0]
    assert saved.ref_cnt == 1
    request.num_computed_tokens += 6
    request.num_in_flight_tokens = 6
    assert manager.allocate_slots(request, 1) is not None
    assert window[0] is saved and saved.ref_cnt == 1
    # The fourth page was allocated optimistically. Before acceptance it cannot
    # appear in the prefix cache for this request.
    assert window[3].block_hash is None

    request.num_in_flight_tokens = 0
    request.num_computed_tokens = prefix + accepted
    if accepted:
        request.append_output_token_ids(list(range(1002, 1002 + accepted)))
    assert manager.allocate_slots(request, 1) is not None
    if accepted < 2:
        assert window[0] is saved and saved.ref_cnt == 1
    else:
        assert window[0].is_null and saved.ref_cnt == 0
    assert window[3].block_hash is None
    assert blocks(CircularBufferSpec)[0] is ring
    manager.free(request)
    assert ring.ref_cnt == 0 and ring.block_hash is None


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@torch.inference_mode()
def test_sm70_tensor_prefill_paged_tail(monkeypatch, packed, partial):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA required")

    from vllm.models.hy_v4_flash.sm70 import kernels
    from vllm.models.hy_v4_flash.sm70.f7_attn import load_hy4_f7_attn_operator

    load_hy4_f7_attn_operator("paged_prefill_sm70")
    gen = torch.Generator(device="cuda").manual_seed(1704)
    kv = torch.randn((64, 576), generator=gen, dtype=torch.float16, device="cuda")
    cache = pack_fp8_ds_mla_torch(kv[:, :512], kv[:, 512:]) if packed else kv
    cache = cache.view(2, 32, -1)
    # Keep the head stride different from a token-major tensor.
    q = torch.randn(
        (8, 3, 576), generator=gen, dtype=torch.float16, device="cuda"
    ).transpose(0, 1)
    slots = torch.arange(51, 34, -1, dtype=torch.int32, device="cuda").expand(3, -1)
    counts = torch.tensor([0, 1, 17], dtype=torch.int32, device="cuda")
    sinks = None if partial else torch.zeros(8, dtype=torch.float32, device="cuda")
    expected_lse = torch.empty((3, 8), device="cuda") if partial else None
    actual_lse = torch.empty_like(expected_lse) if partial else None

    expected = paged_attention(q, cache, slots, counts, sinks, 0.0625, 512, expected_lse)
    monkeypatch.setattr(kernels, "_paged_attention", object())
    actual = paged_attention(
        q, cache, slots, counts, sinks, 0.0625, 512, actual_lse, tensor_cores=True
    )
    torch.testing.assert_close(actual, expected, rtol=0.007, atol=0.0007)
    assert torch.count_nonzero(actual[0]).item() == 0
    if partial:
        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
        assert torch.isneginf(actual_lse[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("prefix", "accepted"), [(254, 2), (255, 1), (257, 0), (511, 3)]
)
@torch.inference_mode()
def test_cuda_native_fp8_verification_rejection(
    vllm_config, monkeypatch, workspace_init, prefix, accepted
):
    from vllm.models.hy_v4_flash.sm70 import attention

    cfg = text_config()
    cfg.kv_lora_rank = 512
    cfg.qk_rope_head_dim = cfg.swa_qk_rope_head_dim = 64
    cfg.compressed_indexer_head_dim = 128
    cfg.num_attention_heads = 8
    cfg.swa_num_attention_heads = 16
    vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
    vllm_config.model_config = SimpleNamespace(max_model_len=768, hf_text_config=cfg)
    device = torch.device("cuda")
    width = 6
    end = prefix + accepted
    gen = torch.Generator(device=device).manual_seed(613)
    inputs = torch.randn(
        prefix + width + 8,
        cfg.hidden_size,
        device=device,
        dtype=torch.float16,
        generator=gen,
    )
    replay = torch.randn(
        8, cfg.hidden_size, device=device, dtype=torch.float16, generator=gen
    )
    real_writer = attention.cache_fp8_ds_mla
    writes = []

    def check_writer(latent, rope, cache, slots):
        assert cache.dtype == torch.uint8 and cache.shape[-1] == 656
        expected = torch.full_like(cache, 0xA5)
        real_writer(latent, rope, cache, slots)
        for row in range(len(slots)):
            real_writer(
                latent[row : row + 1],
                rope[row : row + 1],
                expected,
                slots[row : row + 1],
            )
        page = cache.shape[1]
        rows = cache[slots // page, slots % page]
        assert torch.equal(rows, expected[slots // page, slots % page])
        assert torch.equal(rows[:, 528:].contiguous().view(torch.float16), rope)
        writes.append((slots.clone(), rows.clone()))

    model = make_layer(vllm_config, cfg, 0, prefix, monkeypatch, device=device)
    for weight in model.parameters():
        weight.normal_(std=0.12, generator=gen)

    def seed():
        paged = PagedLayer(model, vllm_config, 768)
        paged.step((inputs[:prefix], 0, 0))
        return paged

    reference = seed()
    for pos in range(prefix, end):
        reference.step((inputs[pos : pos + 1], pos, 0))
    expected_outputs = [
        reference.step((row[None], end + i, 0)) for i, row in enumerate(replay)
    ]
    expected_cache = model.attn.kv_cache.clone()
    actual = seed()
    initial = model.attn.kv_cache.clone()
    monkeypatch.setattr(attention, "cache_fp8_ds_mla", check_writer)
    state = VerificationState()
    actual.step(
        (inputs[prefix : prefix + width], prefix, 0), state=state, verify=("0",)
    )
    assert len(writes) == 1 and writes[0][0].numel() == width
    cache = model.attn.kv_cache
    table = actual.tables[model.attn.prefix][0]
    page = cache.shape[1]
    positions = torch.arange(prefix, prefix + width, device=device)
    slots = table[positions // page] * page + positions % page
    assert torch.equal(writes[0][0], slots)
    assert not torch.equal(
        cache[slots[accepted:] // page, slots[accepted:] % page],
        initial[slots[accepted:] // page, slots[accepted:] % page],
    )
    span_bytes = writes[0][1].clone()
    for i, row in enumerate(replay):
        output = actual.step((row[None], end + i, 0), state=state)
        # Eager reconciliation and stock decode use distinct reduction
        # kernels. Keep the FP16 numerical check separate from byte checks.
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected_outputs[i], atol=2e-3, rtol=2e-3)
        later = slots[accepted + i + 1 :]
        assert torch.equal(
            cache[later // page, later % page], span_bytes[accepted + i + 1 :]
        )
    assert len(writes) == 9
    # Replayed positions use identical one-row projections and the native
    # writer on both sides, including every previously rejected slot.
    rewritten = torch.arange(end, end + len(replay), device=device)
    replay_slots = table[rewritten // page] * page + rewritten % page
    assert torch.equal(
        cache[replay_slots // page, replay_slots % page],
        expected_cache[replay_slots // page, replay_slots % page],
    )


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_ds_mla"])
@pytest.mark.parametrize(
    "idx,prefix",
    [(0, p) for p in (*range(252, 256), *range(2046, 2050), *range(2100, 2104))]
    + [(1, p) for p in range(508, 512)],
)
@torch.inference_mode()
def test_verify_attention_matches_one_row_decode(
    vllm_config, monkeypatch, workspace_init, device_type, cache_dtype, idx, prefix
):
    from vllm.v1.worker.workspace import init_workspace_manager

    if device_type == "cpu":
        if not HAS_TRITON or os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("Triton CPU interpreter required")
    elif not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device(device_type)
    if device_type == "cpu":
        init_workspace_manager(device)

    cfg = text_config()
    cfg.hidden_size, cfg.q_lora_rank, cfg.kv_lora_rank = 3072, 1024, 512
    cfg.num_attention_heads, cfg.swa_num_attention_heads = 32, 64
    cfg.qk_rope_head_dim = cfg.swa_qk_rope_head_dim = 64
    cfg.qk_nope_head_dim = cfg.swa_qk_nope_head_dim = 192
    cfg.v_head_dim = cfg.swa_v_head_dim = 256
    cfg.compressed_indexer_num_attention_heads = 16
    cfg.compressed_indexer_head_dim = 128
    vllm_config.cache_config.cache_dtype = cache_dtype
    vllm_config.model_config = SimpleNamespace(max_model_len=2176, hf_text_config=cfg)
    model = make_layer(vllm_config, cfg, idx, 6, monkeypatch, world=4, device=device)
    runner = PagedLayer(model, vllm_config, 2176)
    owners = {owner.prefix: owner for owner in runner.owners}
    gen = torch.Generator(device=device).manual_seed(615)

    def normal(*shape):
        return torch.randn(*shape, device=device, generator=gen).half()

    # Pin Q/KV projection inputs separately from the attention reduction.
    hidden, q_c = normal(6, 3072), normal(6, 1024)
    q = normal(6, model.num_local_heads, 576)
    kv, rope = normal(prefix + 6, 512), normal(prefix + 6, 64)
    model.learnable_sink_param.copy_(
        torch.linspace(-2, 5, model.num_local_heads, device=device)
    )
    cache = model.attn.kv_cache
    table = runner.tables[model.attn.prefix]
    page = cache.shape[1]
    pos = torch.arange(prefix + 6, device=device)
    slots = (table[0, pos // page] * page + pos % page).long()
    cos, sin = rotary_phases(pos[prefix:], model.inv_freq)
    written = (
        pack_fp8_ds_mla_torch(kv, rope)
        if cache_dtype == "fp8_ds_mla"
        else torch.cat((kv, rope), dim=-1)
    )

    if model.indexer is not None:
        for param in model.indexer.parameters():
            param.normal_(std=0.03, generator=gen)
        # Force exact top-k ties in an above-width case as well.
        if prefix == 2100:
            model.indexer.head_weight_projection.weight.zero_()
        pool = model.indexer.k_cache.kv_cache
        pool.copy_(normal(*pool.shape))
        tail = model.indexer.tail_cache.kv_cache
        tail.copy_(normal(*tail.shape))
        tail[..., -1] = prefix - prefix % 4 + torch.arange(4, device=device)
    seed = {name: owner.kv_cache.clone() for name, owner in owners.items()}

    def write(begin, end):
        target = slots[begin:end]
        cache[target // page, target % page] = written[begin:end]

    def metadata(start, end, decode):
        rows = end - start
        span = QuerySpan(0, rows, start, end)
        lengths = torch.tensor([end], dtype=torch.int32, device=device)
        starts = torch.tensor([0, rows], dtype=torch.int32, device=device)
        result = {}
        for name in owners:
            result[name] = FlashMetadata(
                (span,), runner.tables[name], slots[start:end], rows, starts,
                lengths, decode, torch.empty_like(starts)[:rows]
                if decode else None, None,
            )
        if decode and model.indexer is not None:
            builder = model.indexer.k_cache.backend.get_builder_cls()
            kwargs = (model.indexer.k_cache.spec, [model.indexer.k_cache.prefix])
            if device_type == "cuda":
                workspace = builder(*kwargs, vllm_config, device).index_workspace
            else:
                history, groups = 511, 2176 // 4
                workspace = IndexPoolDecodeWorkspace(
                    torch.empty(1, 4, 128),
                    torch.empty(1, dtype=torch.int64),
                    torch.empty(1, groups),
                    torch.empty(1, history),
                    *(torch.empty(1, history, dtype=torch.int64) for _ in range(3)),
                )
            result[model.indexer.k_cache.prefix].index_workspace = workspace
        return result, span

    indices = model.topk_indices_buffer[:, :model.width]

    def read(q, metadata, indices, counts):
        return model._read_cache(
            q, cache, indices, counts, model.learnable_sink_param, None,
            metadata[model.attn.prefix],
        )

    write(0, prefix)
    verify, span = metadata(prefix, prefix + 6, False)
    state = VerificationState()
    with state.forward(verify, owners, [("r", span)], {"r"}, {"r": prefix + 1}):
        counts = model._eager_slots(
            hidden, q_c, pos[prefix:], cos, sin, verify, indices, page,
            decode_verification=True,
        )
        write(prefix, prefix + 6)
        actual = read(q, verify, indices, counts)
    old = paged_attention(
        q, cache, indices, counts, model.learnable_sink_param, model.scaling, 512
    )
    actual_slots = indices.clone()
    actual_counts = counts.clone()
    if model.indexer is not None:
        actual_pool = pool.clone()
        assert torch.equal(tail, seed[model.indexer.tail_cache.prefix])

    from vllm.models.hy_v4_flash.sm70.verification import VerificationGraph

    tail_names = (model.indexer.tail_cache.prefix,) if model.indexer else ()
    graph = VerificationGraph(
        tail_names, torch.empty((len(tail_names), 6, 129), device=device), {}, 6,
    )
    fixed, _ = metadata(prefix, prefix + 6, False)
    for entry in fixed.values():
        entry.slot_mapping = entry.slot_mapping.clone()
    stable_q, stable_hidden, stable_q_c = q.clone(), hidden.clone(), q_c.clone()
    stable_pos = pos[prefix:].clone()
    stable_cos, stable_sin = cos.clone(), sin.clone()

    def verify_forward():
        counts = model._verification_slots(
            stable_hidden, stable_q_c, stable_pos, stable_cos, stable_sin,
            fixed, indices, page, graph,
        )
        out = model._read_cache(
            stable_q, cache, indices, counts, model.learnable_sink_param, None,
            fixed[model.attn.prefix], verification_width=6,
            batch_verification=graph.index_workspace is not None,
        )
        return out, counts

    cuda_graph = None
    for replay in range(2):
        # Mutate graph inputs in place between executions. The second input
        # also shifts the cursor and cache write addresses by one row.
        start = prefix + replay
        if replay:
            stable_q.neg_()
            stable_hidden.neg_()
            stable_q_c.neg_()
            stable_pos.add_(1)
            next_cos, next_sin = rotary_phases(stable_pos, model.inv_freq)
            stable_cos.copy_(next_cos)
            stable_sin.copy_(next_sin)
        current, current_span = metadata(start, start + 6, False)
        live = stable_pos
        live_slots = (table[0, live // page] * page + live % page).long()
        for name, owner in owners.items():
            owner.kv_cache.copy_(seed[name])
            if model.indexer is not None and name == tail_names[0]:
                tail[..., -1] = start - start % 4 + torch.arange(4, device=device)
            current[name].slot_mapping = live_slots
            fixed[name].slot_mapping.copy_(live_slots)
            fixed[name].seq_lens.copy_(current[name].seq_lens)
        write(0, prefix + 6)
        cache[live_slots[-1] // page, live_slots[-1] % page] = written[-1]
        verifier = VerificationState()
        with verifier.forward(
            current, owners, [("r", current_span)], {"r"}, {"r": start + 1},
        ):
            eager_counts = model._eager_slots(
                stable_hidden, stable_q_c, stable_pos, stable_cos, stable_sin,
                current, indices, page, decode_verification=True,
            )
            eager_out = read(stable_q, current, indices, eager_counts)
            eager_slots = indices.clone()
        if model.indexer is not None:
            eager_pool = pool.clone()
            ring_before = tail.clone()
            pool.copy_(seed[model.indexer.k_cache.prefix])
        if device_type == "cuda":
            if replay == 0:
                verify_forward()
                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph):
                    graph_out, graph_counts = verify_forward()
            cuda_graph.replay()
        else:
            graph_out, graph_counts = verify_forward()
        assert torch.equal(graph_counts, eager_counts)
        assert torch.equal(indices, eager_slots)
        assert torch.equal(graph_out.view(torch.int16), eager_out.view(torch.int16))
        if model.indexer is not None:
            assert torch.equal(pool, eager_pool)
            assert torch.equal(tail, ring_before)
            expected_tail = verifier._requests["r"].pending.rows[tail_names[0]]
            assert torch.equal(graph.tails[0], expected_tail)

    if device_type == "cpu" and model.indexer is not None:
        indexer = model.indexer
        projection_names = (
            "token_key_projection", "query_projection", "head_weight_projection",
            "key_gate_projection", "key_value_projection",
        )
        receipts = {name: [] for name in projection_names}
        measurements = []

        def record(name, out):
            value = out[0] if isinstance(out, tuple) else out
            receipts[name].append(value.float().clone())

        hooks = [
            getattr(indexer, name).register_forward_hook(
                lambda _module, _args, out, name=name: record(name, out)
            )
            for name in projection_names
        ]
        for _ in range(2):
            pool.copy_(seed[indexer.k_cache.prefix])
            output, counts = verify_forward()
            measurements.append({
                name: torch.cat(pieces) for name, pieces in receipts.items() if pieces
            })
            for pieces in receipts.values():
                pieces.clear()
            if graph.index_workspace is None:
                reference = output.clone()
                reference_counts = counts.clone()
                graph.index_workspace = IndexPoolDecodeWorkspace.allocate(
                    6, 4, 128, 2176 // 4, 511, device
                )
        for hook in hooks:
            hook.remove()
        assert torch.equal(counts, reference_counts)
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)
        differences = {
            name: (values - measurements[1][name]).abs().max().item()
            for name, values in measurements[0].items()
        }
        assert all(delta <= 0.004 for delta in differences.values()), differences
        print(f"Batched index {prefix=} {cache_dtype=} max_abs={differences}")

    elif device_type == "cuda" and model.indexer is not None:
        from vllm.models.hy_v4_flash.sm70.dfly_proposer import DFlyProposer

        # Exercise the server's switch and the target attention entry point,
        # with the same stable tensors and metadata used in graph capture.
        monkeypatch.setattr(model, "_capture_dfly", True)

        def captured_verification(mode):
            monkeypatch.setenv("VLLM_HY4_DFLY_VERIFY_INDEX", mode)
            drafter = DFlyProposer(vllm_config, device)
            drafter.target_attention = {model.layer_idx: model}
            verify_graph = drafter.make_verification_graph(6)
            assert (verify_graph.index_workspace is not None) == (mode == "batched")

            def forward():
                with set_forward_context(fixed, vllm_config):
                    context = get_forward_context()
                    context.additional_kwargs["hy4_verify_graph"] = verify_graph
                    return model._aggregate(
                        stable_q, kv[prefix:prefix + 6], rope[prefix:prefix + 6],
                        stable_hidden, stable_q_c, stable_pos, stable_cos,
                        stable_sin, fixed,
                    )

            pool.copy_(seed[model.indexer.k_cache.prefix])
            with patch.object(
                model.indexer, "verify_groups", wraps=model.indexer.verify_groups
            ) as verify_groups:
                forward()
                cuda_verify = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_verify):
                    output = forward()
                assert verify_groups.call_count == 2
                assert verify_groups.call_args.args[-1] is verify_graph.index_workspace
            pool.copy_(seed[model.indexer.k_cache.prefix])
            cuda_verify.replay()
            assert torch.equal(tail, ring_before)
            history = verify_graph.histories[model.layer_idx]
            return (
                output.clone(),
                history.slots.clone(),
                history.counts.clone(),
                pool.clone(),
                verify_graph.tails.clone(),
            )

        rowwise = captured_verification("rowwise")
        batched = captured_verification("batched")
        assert torch.equal(rowwise[1], batched[1])
        assert torch.equal(rowwise[2], batched[2])
        for index in (0, 3, 4):
            torch.testing.assert_close(
                rowwise[index], batched[index], atol=0.004, rtol=0.002
            )

    for name, owner in owners.items():
        owner.kv_cache.copy_(seed[name])
    write(0, prefix)
    expected = []
    for i in range(6):
        row = slice(i, i + 1)
        decode, _ = metadata(prefix + i, prefix + i + 1, True)
        counts = model._decode_slots(
            hidden[row], q_c[row], pos[prefix:][row], cos[row], sin[row],
            decode, indices[:1], page,
        )
        write(prefix + i, prefix + i + 1)
        assert torch.equal(actual_slots[row], indices[:1])
        assert torch.equal(actual_counts[row], counts)
        expected.append(read(q[row], decode, indices[:1], counts))
    expected = torch.cat(expected)
    if model.indexer is not None:
        assert torch.equal(pool, actual_pool)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    old_delta = (old.float() - expected.float()).abs().flatten(1).amax(dim=1)
    assert (old_delta > 0).all()
    print(f"{idx=} {prefix=} {cache_dtype=} old_max_abs={old_delta.tolist()}")


@pytest.mark.parametrize("rows", [6, 12, 24])
@pytest.mark.parametrize("columns", [544, 65536])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_verify_selects_same_group_ids(rows, columns, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.models.hy_v4_flash.sm70.indexpool import select_decode_groups

    history = 511
    gen = torch.Generator().manual_seed(17)
    scores = torch.randn(rows, columns, generator=gen)
    # Complete ties, ties at the selection boundary, and unique scores.
    for row in range(rows):
        live = history - 1 + row % 6
        scores[row, live:] = -torch.inf
        if row % 3 == 0:
            scores[row, :live] = 0
        elif row % 3 == 1:
            scores[row, :live] = scores[row, :live].round()
    # Exercise the full score width as well as the 2,048-token crossing.
    scores[-1] = torch.randint(-3, 4, (columns,), generator=gen)
    scores = scores.to(device)
    workspace = IndexPoolDecodeWorkspace.allocate(
        rows, 4, 128, columns, history, scores.device
    )
    expected = select_decode_groups(scores, history, None)
    actual = select_decode_groups(scores, history, workspace)
    assert torch.equal(actual, expected)
