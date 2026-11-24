# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.kernels import test_hy4_flash_paged as paged_fixture
from vllm.config import (
    CUDAGraphMode,
    CompilationMode,
    DeviceConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.config import HYV4FlashForCausalLMConfig
from vllm.models.hy_v4_flash.configs import SPARSE_ATTENTION, HYV4FlashConfig
from vllm.models.hy_v4_flash.dfly_config import QUERY_WIDTH, DFlySettings
from vllm.models.hy_v4_flash.sm70.dfly import (
    DFlyModel,
    DraftHistory,
    DraftWorkspace,
    _attend_torch,
)
from vllm.models.hy_v4_flash.sm70.dfly_proposer import (
    DFlyProposer,
    DFlyStep,
    DraftGraphInputs,
)
from vllm.models.hy_v4_flash.sm70.dfly_weights import (
    DRAFT_PREFIX,
    DFlyWeights,
    draft_shapes,
)
from vllm.models.hy_v4_flash.sm70.kernels import (
    cache_fp8_ds_mla,
    pack_fp8_ds_mla_torch,
)
from vllm.models.hy_v4_flash.sm70.splitk import (
    decode_workspace_specs,
    paged_attention_splitk,
)
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def settings():
    model = SimpleNamespace(
        hidden_size=32,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=16,
        num_attention_heads=4,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        vocab_size=48,
        index_topk=16,
        compressed_indexer_group_topk=4,
        compressed_indexer_compress_ratio=4,
        layer_types=[SPARSE_ATTENTION] * 5,
        rope_for_layer=lambda _: {"rope_theta": 1e7},
    )
    root = SimpleNamespace(
        dfly_num_hidden_layers=5,
        dfly_target_index_layer=list(range(5)),
        dfly_target_kv_layers=list(range(5)),
        dfly_mask_token_id=47,
        dfly_intermediate_size=64,
    )
    return model, DFlySettings.from_checkpoint(root, model)


@pytest.fixture
def cpu_setup(monkeypatch):
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(cfg), set_default_torch_dtype(torch.float16):
        yield


def test_weight_mapping(settings, tmp_path, cpu_setup):
    cfg, spec = settings
    shapes = draft_shapes(cfg, spec, spec.ffn_width)
    assert len(shapes) == 87
    # Each checkpoint item gets a distinct value so the fused loader's order
    # and both shards can be verified independently.
    tensors = {
        name: torch.full(shape, ordinal, dtype=torch.bfloat16)
        for ordinal, (name, shape) in enumerate(shapes.items(), 1)
    }
    path = tmp_path / "draft.safetensors"
    save_file(tensors, path)
    weights = DFlyWeights(
        cfg, spec, dict.fromkeys(shapes, path.name), lambda _: path, 1
    )
    draft = DFlyModel(cfg, spec, weights.ffn_width)
    weights.load_into(draft)
    params = dict(draft.named_parameters())
    touched = set()
    for name, tensor in tensors.items():
        local = name.removeprefix(DRAFT_PREFIX)
        if local[0].isdigit():
            local = "layers." + local
        for index, projection in enumerate(("gate_proj", "up_proj")):
            if f".{projection}." in local:
                local = local.replace(f".{projection}.", ".gate_up_proj.")
                param = params[local].chunk(2)[index]
                break
        else:
            param = params[local]
        torch.testing.assert_close(param, tensor.to(param.dtype))
        touched.add(local)
    assert touched == params.keys()

    bad_name = DRAFT_PREFIX + "4.mlp.up_proj.weight"
    with pytest.raises(ValueError, match=bad_name):
        DFlyWeights(
            cfg,
            spec,
            dict.fromkeys(shapes.keys() - {bad_name}, path.name),
            lambda _: path,
            1,
        )
    tensors[bad_name] = tensors[bad_name][:-1].contiguous()
    save_file(tensors, path)
    with pytest.raises(ValueError, match=bad_name):
        DFlyWeights(cfg, spec, dict.fromkeys(shapes, path.name), lambda _: path, 1)


def test_checkpoint_headers():
    path = os.getenv("HY4_DFLY_CHECKPOINT")
    if path is None:
        pytest.skip("set HY4_DFLY_CHECKPOINT to the local AWQ checkpoint")
    path = Path(path)
    cfg = HYV4FlashConfig.from_dict(json.loads((path / "config.json").read_text()))
    settings = DFlySettings.from_checkpoint(cfg, cfg)
    weights = DFlyWeights(
        cfg,
        settings,
        json.loads((path / "model.safetensors.index.json").read_text())["weight_map"],
        lambda name: path / name,
        4,
    )
    assert sum(map(len, weights.shards.values())) == 87
    assert weights.ffn_width == 9216
    assert settings.target_layers == (25, 29, 33, 37, 41)


def test_snapshot_does_not_alias_shared_selection():
    step = DFlyStep(
        [0, 9, 15, 16], [9, 58, 98], [True, False, False], [0, 5, 0], 100, "cpu"
    )
    assert step.source_rows.tolist() == [9, 10, 11, 12, 13, 14]
    slots = torch.arange(32).view(16, 2)
    counts = torch.arange(16)
    cache = torch.empty(0)
    step.capture(25, cache, slots, counts)
    slots.add_(100)
    counts.add_(100)
    step.capture(29, cache, slots, counts)
    assert step.histories[25].slots.tolist() == [[i, i + 1] for i in range(18, 30, 2)]
    assert step.histories[25].counts.tolist() == list(range(9, 15))
    assert step.histories[29].slots.tolist() == [[i, i + 1] for i in range(118, 130, 2)]
    assert step.histories[25].cache is cache
    assert list(step.accepted([[], [4], []], 100)) == [(1, 0, 9, 53, 4)]
    assert list(step.accepted([[], [4, 5, 6], []], 100)) == [(1, 2, 11, 55, 6)]
    assert list(step.accepted([[], list(range(6)), []], 100)) == [(1, 5, 14, 58, 5)]


@torch.inference_mode()
def test_live_target_layers_snapshot(monkeypatch):
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config = paged_fixture.text_config()
    config.layer_types = [SPARSE_ATTENTION, SPARSE_ATTENTION]
    layers = [
        paged_fixture.make_layer(cfg, config, i, 6, monkeypatch) for i in range(2)
    ]
    layers[1].topk_indices_buffer = layers[0].topk_indices_buffer
    step = DFlyStep([0, 6], [6], [False], [5], 100, "cpu")
    ordinary_context = paged_fixture.set_forward_context

    @contextmanager
    def capture_context(metadata, config):
        with ordinary_context(metadata, config):
            get_forward_context().additional_kwargs["hy4_dfly"] = step
            yield

    monkeypatch.setattr(paged_fixture, "set_forward_context", capture_context)
    for n, layer in enumerate(layers):
        for p in layer.parameters():
            p.fill_(0.03)
        layer._capture_dfly = True
        fixture = paged_fixture.PagedLayer(layer, cfg, 512)
        table = fixture.tables[layer.attn.prefix]
        # Use a different physical page table for the paired layers.
        if n:
            table.copy_(table.flip(1))
        fixture.step((torch.ones(6, 32).half(), 0, 0))
        history = step.histories[n]
        assert history.cache is layer.attn.kv_cache
        assert history.counts.tolist() == list(range(1, 7))
        page = int(table[0, 0]) * history.cache.shape[1]
        for row in range(6):
            assert history.slots[row, : row + 1].tolist() == list(
                range(page, page + row + 1)
            )
    assert not torch.equal(step.histories[0].slots, step.histories[1].slots)
    assert not torch.equal(step.histories[0].slots, layers[0].topk_indices_buffer)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_draft_fixed_inputs_match_eager(settings, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config, settings = settings
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    cfg = VllmConfig(device_config=DeviceConfig(device=device))
    batch, rows = 2, 2 * QUERY_WIDTH
    gen = torch.Generator(device=device).manual_seed(7963)
    with (
        set_current_vllm_config(cfg),
        set_default_torch_dtype(torch.float16),
        torch.device(device),
    ):
        drafter = DFlyProposer(
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)), device
        )
        drafter.settings = settings
        drafter.backbone = DFlyModel(config, settings, settings.ffn_width).eval()
        embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32,
        )
        for parameter in drafter.backbone.parameters():
            parameter.normal_(std=0.1, generator=gen)
        drafter.embed = embeddings
        drafter.logits = lm_head
        specs = decode_workspace_specs(
            rows, config.num_attention_heads, settings.topk + QUERY_WIDTH,
            config.kv_lora_rank,
        )
        drafter.workspace = DraftWorkspace(
            torch.zeros(batch, QUERY_WIDTH, config.kv_lora_rank + config.qk_rope_head_dim),
            torch.arange(rows, dtype=torch.int32),
            *(torch.empty(shape, dtype=dtype) for shape, dtype in specs),
        )
        histories = [
            DraftHistory(
                torch.randn(4, 16, drafter.workspace.scratch.shape[-1], generator=gen),
                torch.zeros(batch, settings.topk, dtype=torch.int32),
                torch.zeros(batch, dtype=torch.int32),
            )
            for _ in settings.target_layers
        ]
        inputs = DraftGraphInputs(
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch, config.hidden_size),
            histories,
        )
        stable_hidden, stable_logits = [], []

        def forward():
            return drafter._draft_forward(
                inputs.anchors, inputs.positions, inputs.hidden, inputs.histories,
            )

        cuda_graph = None
        for iteration in range(2):
            anchors = torch.tensor([1, 2]) + iteration * 3
            positions = torch.tensor([12, 24]) + iteration
            hidden = torch.randn_like(inputs.hidden)
            snapshot = [
                DraftHistory(
                    history.cache,
                    torch.randint(0, 64, (batch, settings.topk), dtype=torch.int32),
                    torch.tensor([8, 12], dtype=torch.int32) - index - iteration,
                )
                for index, history in enumerate(histories)
            ]
            expected_hidden, expected_logits = [], []

            def trace_logits(value):
                logits = lm_head(value.float())
                expected_hidden.append(value)
                expected_logits.append(logits)
                return logits

            drafter.logits = trace_logits
            expected = drafter._draft_forward(anchors, positions, hidden, snapshot)
            inputs.stage(anchors, positions, hidden, torch.arange(batch),
                         snapshot, torch.arange(batch))

            def record_logits(value):
                logits = lm_head(value.float())
                stable_hidden.append(value)
                stable_logits.append(logits)
                return logits

            drafter.logits = record_logits
            if device == "cuda":
                if cuda_graph is None:
                    forward()
                    stable_hidden.clear()
                    stable_logits.clear()
                    cuda_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(cuda_graph):
                        actual = forward()
                cuda_graph.replay()
            else:
                stable_hidden.clear()
                stable_logits.clear()
                actual = forward()
            assert torch.equal(actual, expected)
            assert len(stable_logits) == QUERY_WIDTH - 1
            for x, y in zip(
                stable_hidden + stable_logits, expected_hidden + expected_logits,
                strict=True,
            ):
                assert torch.equal(
                    x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
                )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_causal_scratch_reader(packed, device):
    if device == "cpu" and os.getenv("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton interpreter on CPU")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    latent = 512 if packed else 16
    width = QUERY_WIDTH
    # Deliberately use different history and scratch page widths. One request
    # has no history. Others use shuffled slots and partially filled tiles.
    rope = 64 if packed else 16
    history = torch.randn(4, 17, latent + rope).half()
    scratch = torch.randn(3, width, latent + rope).half()
    if packed:
        history = pack_fp8_ds_mla_torch(
            history[..., :latent].contiguous(), history[..., latent:].contiguous()
        )
        projected = scratch.to(device).flatten(0, 1)
        scratch = pack_fp8_ds_mla_torch(
            scratch[..., :latent].contiguous(), scratch[..., latent:].contiguous()
        )
    history, scratch = history.to(device)[:, :16], scratch.to(device)
    if packed and device == "cuda":
        native = torch.empty_like(scratch)
        cache_fp8_ds_mla(
            projected[:, :latent],
            projected[:, latent:],
            native,
            torch.arange(3 * width, device=device, dtype=torch.int32),
        )
        assert torch.equal(native, scratch)
        scratch = native
    slots = torch.tensor(
        [[63, 0, 4] + list(range(17, 42)), [-1] * 28, [3, 19, 0] + [-1] * 25],
        device=device,
        dtype=torch.int32,
    )
    counts = torch.tensor([28, 0, 3], device=device, dtype=torch.int32)
    snapshot = DraftHistory(history, slots, counts)
    before = history.clone()
    q = torch.randn(3 * width, 4, latent + rope).half().to(device)
    sink = torch.randn(4).to(device)
    expected = _attend_torch(q, snapshot, scratch, sink, 0.08, latent)
    specs = decode_workspace_specs(q.shape[0], q.shape[1], 28 + width, latent)
    state, accum = (
        torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs
    )
    live = (counts[:, None] + torch.arange(1, width + 1, device=device)).flatten().int()
    actual = paged_attention_splitk(
        q, history, slots, live, sink, 0.08, state, accum, draft_kv=scratch
    )
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.004)
    assert torch.equal(before, history)


def test_startup(settings, monkeypatch):
    text, _ = settings
    text.model_type = "hy_v4_flash"
    root = SimpleNamespace(
        dfly_num_hidden_layers=5,
        dfly_target_index_layer=list(range(5)),
        dfly_target_kv_layers=list(range(5)),
        dfly_mask_token_id=47,
    )
    target = SimpleNamespace(
        model="/checkpoint",
        hf_text_config=text,
        hf_config=root,
        dtype=torch.float16,
        enforce_eager=False,
        verify_with_parallel_config=lambda _: None,
    )
    parallel = ParallelConfig()
    kwargs = dict(target_model_config=target, target_parallel_config=parallel)
    spec = SpeculativeConfig(method="dfly", num_speculative_tokens=5, **kwargs)
    assert spec.draft_model_config is target
    assert spec.draft_parallel_config is parallel
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        SpeculativeConfig(method="dfly", num_speculative_tokens=4, **kwargs)
    with pytest.raises(ValueError, match="method='dfly'"):
        SpeculativeConfig(method="ngram", num_speculative_tokens=5, **kwargs)
    with pytest.raises(ValueError, match="draft_sample_method"):
        SpeculativeConfig(
            method="dfly",
            num_speculative_tokens=5,
            draft_sample_method="probabilistic",
            **kwargs,
        )
    with pytest.raises(ValueError, match="Hy4 Flash target"):
        SpeculativeConfig(method="dfly", num_speculative_tokens=5)
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.scheduler_config.async_scheduling = False
    cfg.speculative_config = spec
    cfg.model_config = target
    cfg.parallel_config = parallel
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", False)
    verify = HYV4FlashForCausalLMConfig.verify_and_update_config
    monkeypatch.setenv("HY4_F7_HC_BRIDGE", "1")
    with pytest.raises(ValueError, match="does not support HY4_F7_HC_BRIDGE=1"):
        verify(cfg)
    monkeypatch.setenv("HY4_F7_HC_BRIDGE", "0")
    monkeypatch.setenv("VLLM_HY4_DFLY_GRAPHS", "off")
    verify(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    monkeypatch.setenv("VLLM_HY4_DFLY_GRAPHS", "both")
    cfg.compilation_config.cudagraph_mode = None
    cfg.compilation_config.mode = CompilationMode.NONE
    cfg.compilation_config.cudagraph_capture_sizes = None
    cfg.compilation_config.max_cudagraph_capture_size = None
    cfg.scheduler_config.max_num_seqs = 4
    verify(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert cfg.compilation_config.cudagraph_capture_sizes == [6, 12, 18, 24]
    cfg.model_config.enforce_eager = True
    verify(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    parallel.decode_context_parallel_size = 2
    with pytest.raises(ValueError, match="decode_context_parallel_size"):
        SpeculativeConfig(method="dfly", num_speculative_tokens=5, **kwargs)
    with pytest.raises(ValueError, match="decode_context_parallel_size"):
        verify(cfg)
    parallel.decode_context_parallel_size = 1
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", True)
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER"):
        verify(cfg)


def test_graph_budget_charges_retained_driver_memory_once():
    from vllm.v1.worker.memory_budget import GPU_STARTUP_MARGIN_BYTES, available_kv_bytes

    mib = 1 << 20
    total, resident = 30_000 * mib, 27_532 * mib
    live = resident + 90 * mib
    snapshot = SimpleNamespace(total_memory=total, free_memory=total - resident)
    transient = 400 * mib
    assert available_kv_bytes(
        snapshot, 1.0, live - resident, transient, GPU_STARTUP_MARGIN_BYTES
    ) == 1828 * mib


def test_confidence_logs_against_verification_result(monkeypatch):
    from vllm.models.hy_v4_flash.sm70 import dfly_proposer

    monkeypatch.setenv("VLLM_HY4_DFLY_LOG_CONFIDENCE", "1")
    monkeypatch.setenv("VLLM_HY4_DFLY_GATE", "off")
    proposer = DFlyProposer(
        SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)), "cpu"
    )
    proposer.settings = SimpleNamespace(target_layers=())
    proposals = torch.tensor([[10, 11, 12, 13, 14]])
    confidence = torch.tensor([[1., 2., 3., 4., 5.]])
    monkeypatch.setattr(proposer, "_draft_forward", lambda *_: (proposals, confidence))
    log = []
    monkeypatch.setattr(
        dfly_proposer.logger, "info", lambda fmt, *args: log.append(fmt % args)
    )

    proposer.prepare_target([0, 1], [8], [False], [0], ["r"])
    assert proposer.propose([[9]], torch.zeros((1, 3))) == [proposals[0].tolist()]
    # The next verify schedules the anchor itself plus five proposal rows.
    proposer.prepare_target([0, 6], [14], [False], [5], ["r"])
    proposer.propose([[10, 11, 99]], torch.zeros((6, 3)))
    assert log == [
        "Hy4 DFly2 calibration req=r anchor=7 proposed=0 accepted=0 confidence=null",
        "Hy4 DFly2 calibration req=r anchor=8 proposed=5 accepted=2 "
        "confidence=[1.0, 2.0, 3.0, 4.0, 5.0]",
    ]


@pytest.mark.parametrize("image", [False, True], ids=["text", "vision"])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_ds_mla"])
def test_dfly_refuses_sharded_cache(settings, image, cache_dtype):
    from vllm.model_executor.models.config import (
        HYV4FlashVLForConditionalGenerationConfig,
    )

    text, _ = settings
    text.model_type = "hy_v4_flash"
    parallel = ParallelConfig()
    parallel.tensor_parallel_size = 4
    parallel.decode_context_parallel_size = 4
    model = SimpleNamespace(hf_text_config=text)
    message = "Hy4 DFly2 requires decode_context_parallel_size=1."
    with pytest.raises(ValueError, match=message):
        SpeculativeConfig(
            method="dfly", num_speculative_tokens=5,
            target_model_config=model, target_parallel_config=parallel,
        )

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.speculative_config = SimpleNamespace(method="dfly")
    config.model_config = model
    config.parallel_config = parallel
    config.cache_config.cache_dtype = cache_dtype
    config_cls = (
        HYV4FlashVLForConditionalGenerationConfig if image else HYV4FlashForCausalLMConfig
    )
    with pytest.raises(ValueError, match=message):
        config_cls.verify_and_update_config(config)


def test_image_draft_positions_use_logical_anchor_plus_rope_delta(monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, uses_mrope=True)
    )
    proposer = DFlyProposer(config, "cpu")
    proposer.settings = SimpleNamespace(target_layers=(), mask_id=0)
    proposer.workspace = SimpleNamespace(offsets=torch.arange(6))
    proposer.embed = proposer.logits = None
    proposer.graph_only = False
    proposer.gate = None
    proposer.log_confidence = False

    def backbone(anchors, positions, *_):
        torch.testing.assert_close(anchors, torch.tensor([12, 21]))
        expected = torch.cat((torch.arange(23, 29), torch.arange(66, 72)))
        torch.testing.assert_close(positions, expected.expand(3, -1))
        return torch.full((2, 5), 42)

    proposer.backbone = SimpleNamespace(propose=backbone)
    # The first request has an image delta of -17 and kept two proposals.
    # The second has an image delta of -4 and follows a plain decode step.
    proposer.prepare_target(
        [0, 6, 7], [43, 70], [False, False], [5, 0],
        position_deltas=(-17, -4),
    )
    assert proposer.propose([[10, 11, 12], [21]], torch.zeros((7, 8))) == [
        [42] * 5, [42] * 5,
    ]
