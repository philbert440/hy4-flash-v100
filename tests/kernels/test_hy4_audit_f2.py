# SPDX-License-Identifier: Apache-2.0
"""Audit probes for accepted-prefix visibility and ownership."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.test_hy4_flash_paged import make_layer, text_config
from tests.kernels.test_hy4_flash_verification import forward_scope, ring_spec
from vllm.config import DeviceConfig, VllmConfig
from vllm.models.hy_v4_flash.sm70.cache import QuerySpan
from vllm.models.hy_v4_flash.sm70.indexpool import (
    FlashCompressedIndexer,
    pool_page_ids,
    select_tokens,
)
from vllm.models.hy_v4_flash.sm70.swa import window_indices
from vllm.models.hy_v4_flash.sm70.verification import VerificationState
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request

RATIO = 4
PAGE = 64
WINDOW = 512
POOL_BLOCK = 256


def new_indexer(blocks=4, pool_block_size=POOL_BLOCK):
    return SimpleNamespace(
        ratio=RATIO,
        pos_dims=1,
        mrope_section=None,
        dim=8,
        k_cache=SimpleNamespace(
            kv_cache=torch.zeros(blocks + 2, PAGE, 8, dtype=torch.float16),
            spec=SimpleNamespace(storage_block_size=pool_block_size),
        ),
        tail_cache=SimpleNamespace(
            prefix="tail",
            spec=ring_spec(8),
            kv_cache=torch.full((blocks + 2, RATIO, 9), -7.0),
        ),
        dcp_group=None,
        dcp_world=1,
        dcp_rank=0,
        key_value_projection=lambda rows: rows,
        key_gate_projection=torch.zeros_like,
        intra_group_ape=torch.zeros(RATIO, 8),
        key_norm=lambda rows: rows,
        _rotate=lambda rows, *_: rows,
    )


def keys(start, stop, offset=1.0):
    positions = torch.arange(start, stop)
    return (positions[:, None] + offset).expand(-1, 8).half().contiguous(), positions


def prepare(
    indexer,
    table,
    start,
    stop,
    *,
    verify=False,
    ring_id=None,
    offset=1.0,
    finalized=None,
    fail=False,
    request="request",
):
    ring_id = table[:1] if ring_id is None else ring_id
    inputs, positions = keys(start, stop, offset)
    span = QuerySpan(0, stop - start, start, stop)
    scope, metadata = nullcontext(), None
    if verify and not hasattr(indexer, "verification"):
        indexer.verification = VerificationState()
    if hasattr(indexer, "verification"):
        scope, metadata = forward_scope(
            indexer,
            [(request, span)],
            ring_id[None],
            (request,) if verify else (),
            None if finalized is None else {request: finalized},
        )
    with scope:
        FlashCompressedIndexer.prepare_groups(
            indexer,
            inputs,
            positions,
            torch.ones(32),
            span,
            table,
            ring_id,
            None if metadata is None else metadata.verification,
        )
        if fail:
            raise RuntimeError("injected forward failure")


def test_live_ring_and_next_close_match_one_token_reference():
    """The builder matrix compares only ring[:committed % 4]. Slots past that
    still have to survive, because a later member write must not be the first
    thing that repairs a clobber."""
    for residue in range(RATIO):
        for width in range(1, 7):
            prefix = 8 + residue
            for accepted in range(width + 1):
                actual, reference = new_indexer(), new_indexer()
                for indexer in (actual, reference):
                    indexer.tail_cache.kv_cache[:] = -7
                table = torch.tensor([1, 2, 3, 4])
                ring = torch.tensor([1])
                prepare(actual, table, 0, prefix, ring_id=ring)
                prepare(reference, table, 0, prefix, ring_id=ring)
                before = actual.tail_cache.kv_cache[1].clone()
                prepare(
                    actual, table, prefix, prefix + width, verify=True, ring_id=ring
                )
                assert torch.equal(actual.tail_cache.kv_cache[1], before)
                end = prefix + accepted
                for pos in range(prefix, end):
                    prepare(reference, table, pos, pos + 1, ring_id=ring)
                closed = torch.arange(0, end // RATIO)
                if closed.numel():
                    pages = pool_page_ids(closed, table, PAGE, POOL_BLOCK // PAGE)
                    assert torch.equal(
                        actual.k_cache.kv_cache[pages, closed % PAGE],
                        reference.k_cache.kv_cache[pages, closed % PAGE],
                    ), (residue, width, accepted, end)
                # One chunk closes the next group. The reference still takes
                # those tokens one at a time, so leftover ring slots would show
                # up here if the close read them.
                close_at = (end // RATIO + 1) * RATIO
                prepare(actual, table, end, close_at, ring_id=ring, offset=20)
                for pos in range(end, close_at):
                    prepare(reference, table, pos, pos + 1, ring_id=ring, offset=20)
                groups = torch.arange(0, close_at // RATIO)
                pages = pool_page_ids(groups, table, PAGE, POOL_BLOCK // PAGE)
                assert torch.equal(
                    actual.k_cache.kv_cache[pages, groups % PAGE],
                    reference.k_cache.kv_cache[pages, groups % PAGE],
                ), (residue, width, accepted, end)


def test_rejected_close_is_invisible_until_replay_and_topk_does_not_keep_it():
    """A group that closes inside the rejected suffix is written immediately.
    Queries that have not crossed that group must not select it, including
    across replay steps that do not close it again. Passing the optimistic
    length must be rejected before the reader runs."""
    prefix = 16  # first sparse query; groups 0..3 are complete
    width = 6
    # Span positions 16..21 close group 4 (16..19) and leave 20, 21 open.
    actual, reference = new_indexer(blocks=8), new_indexer(blocks=8)
    table = torch.arange(1, 5)
    ring = torch.tensor([1])
    prepare(actual, table, 0, prefix, ring_id=ring)
    prepare(reference, table, 0, prefix, ring_id=ring)
    prepare(actual, table, prefix, prefix + width, verify=True, ring_id=ring, offset=50)
    with pytest.raises(RuntimeError, match="exceeds finalized history"):
        prepare(
            actual,
            table,
            prefix + width,
            prefix + width + 1,
            ring_id=ring,
            finalized=prefix + 1,
        )
    # Group 4 was published from rejected tokens. The one-token reference
    # never closed it, so the row differs. That is the stale candidate.
    pages = pool_page_ids(torch.tensor([4]), table, PAGE, POOL_BLOCK // PAGE)
    stale = actual.k_cache.kv_cache[pages, 4 % PAGE].clone()
    fresh = reference.k_cache.kv_cache[pages, 4 % PAGE].clone()
    assert not torch.equal(stale, fresh)

    def select_at(pos):
        # Eager prefill rule from indexpool.py: a group is visible only when
        # its id is strictly below the query position // ratio. The query is
        # the rejected group key, so that group wins whenever the mask lets
        # it through.
        g = torch.arange(8)
        k = actual.k_cache.kv_cache[
            pool_page_ids(g, table, PAGE, POOL_BLOCK // PAGE), g % PAGE
        ].float()
        logits = (k[4] @ k.T).relu()
        logits = logits.masked_fill(g >= pos // RATIO, float("-inf"))
        return select_tokens(logits[None], torch.tensor([pos]), RATIO, 3)

    def recalled(selected):
        # select_tokens appends the open group after history * ratio columns.
        return selected[:, : 3 * RATIO]

    kept = select_at(prefix - 1)
    assert (recalled(kept) // RATIO != 4).all()
    # Several replay steps that do not close group 4 (close is position 19).
    for pos in range(prefix, prefix + 3):
        prepare(actual, table, pos, pos + 1, ring_id=ring, offset=7)
        prepare(reference, table, pos, pos + 1, ring_id=ring, offset=7)
        selected = select_at(pos)
        assert (recalled(selected) // RATIO != 4).all(), pos
        assert torch.equal(actual.k_cache.kv_cache[pages, 4 % PAGE], stale), pos
    # Even after initial reconciliation the tracked cursor cannot skip the close.
    with pytest.raises(RuntimeError, match="does not cover forward start"):
        prepare(actual, table, 20, 21, ring_id=ring)
    # Replay through the close rewrites the key to the accepted-only value.
    for pos in range(prefix + 3, 20):
        prepare(actual, table, pos, pos + 1, ring_id=ring, offset=7)
        prepare(reference, table, pos, pos + 1, ring_id=ring, offset=7)
    assert torch.equal(
        actual.k_cache.kv_cache[pages, 4 % PAGE],
        reference.k_cache.kv_cache[pages, 4 % PAGE],
    )
    # select_tokens returns a fresh row of width (history + 1) * ratio.
    # Copying it over a poisoned buffer replaces the live row completely.
    # A padding row the forward does not write keeps the stale candidate.
    width_cols = (3 + 1) * RATIO
    buffer = torch.full((2, width_cols), 4, dtype=torch.int32)
    buffer[0].copy_(select_at(20)[0])
    assert (recalled(buffer[:1]) // RATIO == 4).any()
    buffer[0].copy_(select_at(prefix - 1)[0])
    assert (buffer[0] // RATIO != 4).all()
    assert (buffer[1] == 4).all()


def test_pooled_page_with_four_subpages_does_not_alias_a_committed_group():
    """Production pool blocks hold four 64-row pages. A rejected close in a
    later page of the same block must not replace a committed group's row."""
    indexer = new_indexer(blocks=8, pool_block_size=POOL_BLOCK)
    # storage block 0 covers logical pool pages 0..3, groups 0..255.
    table = torch.tensor([1, 0, 0, 0])
    ring = torch.tensor([2])
    prepare(indexer, table, 0, 4, ring_id=ring, offset=3)
    committed_row = indexer.k_cache.kv_cache[
        pool_page_ids(torch.tensor([0]), table, PAGE, 4), 0
    ].clone()
    # Positions 252..257 close group 63 (252..255), which lives in page 0
    # row 63, and start group 64 on the next page. Accept none.
    prepare(indexer, table, 252, 258, verify=True, ring_id=ring, offset=9)
    assert torch.equal(
        indexer.k_cache.kv_cache[pool_page_ids(torch.tensor([0]), table, PAGE, 4), 0],
        committed_row,
    )
    rejected = indexer.k_cache.kv_cache[
        pool_page_ids(torch.tensor([63]), table, PAGE, 4), 63 % PAGE
    ]
    assert not torch.equal(rejected, torch.zeros_like(rejected))


def _manager(monkeypatch):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = SimpleNamespace(max_model_len=4096)
    cfg = text_config()
    for idx in range(2):
        make_layer(config, cfg, idx, 1, monkeypatch)
    owners = config.compilation_config.static_forward_context
    groups = get_kv_cache_groups(
        config, {name: layer.spec for name, layer in owners.items()}
    )
    cache = get_kv_cache_config_from_groups(config, groups, 2 << 20)
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([cache]),
        max_model_len=4096,
        max_in_flight_tokens=4096,
        hash_block_size=256,
        enable_caching=True,
    )
    init_none_hash(sha256)
    return manager


def _blocks(manager, request, spec):
    return next(
        m
        for m in manager.coordinator.single_type_managers
        if isinstance(m.kv_cache_spec, spec)
        and getattr(m.kv_cache_spec, "compress_ratio", 1) == 1
    ).req_to_blocks[request.request_id]


def test_swa_speculative_page_does_not_reuse_the_live_window_block(monkeypatch):
    """Window 512, block 256. At computed 765 a 6-token suffix reaches
    position 770 and allocates the next page. The oldest live token, 254,
    still sits in block 0. An optimistic free (length 771) would null that
    block and reuse_first could hand its storage to the new page. The
    scheduler subtracts in-flight tokens before that free; this is that order.
    """
    manager = _manager(monkeypatch)
    prefix = 765
    hasher = get_request_block_hasher(256, sha256)
    request = Request(
        "swa",
        list(range(prefix)),
        SamplingParams(max_tokens=30),
        None,
        block_hasher=hasher,
    )
    assert manager.allocate_slots(request, prefix) is not None
    request.num_computed_tokens = prefix
    # Scheduler order: allocate the speculative suffix BEFORE adding it to
    # num_computed_tokens / num_in_flight_tokens.
    assert manager.allocate_slots(request, 6) is not None
    window = _blocks(manager, request, SlidingWindowMLASpec)
    live = window[0]
    speculative = window[3]
    assert live is not speculative
    assert live.ref_cnt == 1 and not live.is_null
    assert speculative.block_hash is None
    request.num_computed_tokens += 6
    request.num_in_flight_tokens = 6
    assert manager.allocate_slots(request, 1) is not None
    assert window[0] is live and live.ref_cnt == 1
    # The same manager frees block 0 if the cursor is the optimistic length.
    # That is the overwrite the paged layout does not prevent by itself.
    manager.remove_skipped_blocks(request.request_id, prefix + 6)
    assert window[0].is_null and live.ref_cnt == 0


def test_speculative_block_is_not_prefix_cached_or_kept_across_preemption(monkeypatch):
    manager = _manager(monkeypatch)
    hasher = get_request_block_hasher(256, sha256)
    prefix = 256
    request = Request(
        "main",
        list(range(prefix)),
        SamplingParams(max_tokens=64),
        None,
        block_hasher=hasher,
    )
    assert manager.allocate_slots(request, prefix) is not None
    request.num_computed_tokens = prefix
    assert manager.allocate_slots(request, 256) is not None
    main = _blocks(manager, request, MLAAttentionSpec)
    assert main[0].block_hash is not None
    assert main[1].block_hash is None
    # In flight, then preempted before replay. free() may cache only the
    # committed token ids, which do not cover the speculative page.
    request.num_computed_tokens = prefix + 256
    request.num_in_flight_tokens = 256
    manager.free(request)
    assert main[1].block_hash is None
    # The committed page stays published. The speculative page, whose token
    # ids were never in the request, is not a hit even for a prompt that
    # contains that exact suffix.
    other = Request(
        "other",
        list(range(prefix + 256)),
        SamplingParams(max_tokens=8),
        None,
        block_hasher=hasher,
    )
    _hits, n = manager.get_computed_blocks(other)
    assert n == prefix


def test_main_rejected_row_is_outside_the_committed_window():
    """Absolute slots: position 765 shares a page with committed rows, and
    position 768 starts the next page. The eager window at the committed
    cursor reads neither. The optimistic cursor reads both."""
    block = 256
    table = torch.tensor([[4, 1, 7, 2]], dtype=torch.int32)
    cache = torch.zeros(8, block, dtype=torch.int32)
    length = 765
    for pos in range(length):
        cache[table[0, pos // block], pos % block] = pos + 1
    # Rejected rows of a width-6 suffix, including the page that starts at 768.
    for pos in range(length, length + 6):
        cache[table[0, pos // block], pos % block] = -pos
    committed = cache.clone()
    cache[table[0, 768 // block], 768 % block] = 99999
    # Eager SWA reads positions[q - 511, q]. The query at the committed
    # cursor does not include the rejected row. The optimistic cursor does.
    query = length - 1
    rows = window_indices(torch.tensor([query, query + 6]), WINDOW)
    assert 765 not in rows[0].tolist()
    assert 768 not in rows[0].tolist()
    assert 765 in rows[1].tolist()
    same_page = table[0, 765 // block]
    assert cache[same_page, 764 % block].item() == 765
    assert cache[same_page, 765 % block].item() == -765
    assert (rows[0] >= 0).sum() == WINDOW
    logical = rows[0][rows[0] >= 0].long()
    physical = table[0, logical // block] * block + logical % block
    assert torch.equal(
        cache[physical // block, physical % block],
        committed[physical // block, physical % block],
    )
    optimistic = rows[1][rows[1] >= 0].long()
    assert (cache[table[0, optimistic // block], optimistic % block] == 99999).any()


def test_nested_forward_and_duplicate_writer_raise_then_second_span_reconciles():
    table = torch.tensor([1])
    actual, reference = new_indexer(), new_indexer()
    prepare(actual, table, 0, 5)
    prepare(reference, table, 0, 5)
    actual.verification = VerificationState()
    span = QuerySpan(0, 6, 5, 11)
    batch = [("request", span)]
    scope, metadata = forward_scope(actual, batch, table[None], ("request",))
    with scope:
        with (
            pytest.raises(RuntimeError, match="single active"),
            forward_scope(actual, batch, table[None], ("request",))[0],
        ):
            pytest.fail("nested forward ran")
        k, phase = keys(5, 11)
        FlashCompressedIndexer.prepare_groups(
            actual,
            k,
            phase,
            torch.ones(32),
            span,
            table,
            table,
            metadata.verification,
        )
        with pytest.raises(RuntimeError, match="Duplicate"):
            metadata.verification.stage_tail(span, "tail", k, phase)
    prepare(actual, table, 7, 13, verify=True, offset=20)
    for p in (5, 6):
        prepare(reference, table, p, p + 1)
    prepare(reference, table, 7, 8, offset=20)
    for p in range(8, 17):
        prepare(actual, table, p, p + 1, offset=30)
        prepare(reference, table, p, p + 1, offset=30)
    assert torch.equal(actual.k_cache.kv_cache, reference.k_cache.kv_cache)
    assert torch.equal(actual.tail_cache.kv_cache, reference.tail_cache.kv_cache)


def test_uncovered_and_rewound_starts_raise():
    table = torch.tensor([1])
    indexer = new_indexer()
    prepare(indexer, table, 0, 4)
    with pytest.raises(RuntimeError, match="missed an index tail writer"):
        indexer.verification = VerificationState()
        with forward_scope(
            indexer, [("request", QuerySpan(0, 4, 4, 8))], table[None], ("request",)
        )[0]:
            pass
    with pytest.raises(RuntimeError, match="does not cover forward start"):
        prepare(indexer, table, 5, 6)
    with pytest.raises(RuntimeError, match="rewinds"):
        prepare(indexer, table, 3, 4)
    prepare(indexer, table, 4, 8, verify=True)
    with pytest.raises(RuntimeError, match="does not cover forward start"):
        prepare(indexer, table, 9, 10)


@pytest.mark.parametrize("ending", ["cancel", "finish", "preempt", "prefix_hit"])
def test_released_ring_cannot_publish_into_the_next_owner(monkeypatch, ending):
    manager = _manager(monkeypatch)
    hasher = get_request_block_hasher(256, sha256)
    prompt = list(range(263))
    old = Request(
        "old", prompt.copy(), SamplingParams(max_tokens=16), None, block_hasher=hasher
    )
    assert manager.allocate_slots(old, len(prompt)) is not None
    pool = torch.tensor([1, 2])
    actual, reference = new_indexer(), new_indexer()
    cache = actual.tail_cache
    first_ring = _blocks(manager, old, CircularBufferSpec)[0]
    cache.kv_cache = cache.kv_cache.new_full(
        (manager.block_pool.num_gpu_blocks, RATIO, 9), -7
    )
    ring = torch.tensor([first_ring.block_id])
    prepare(actual, pool, 0, len(prompt), ring_id=ring)
    prepare(
        actual,
        pool,
        len(prompt),
        len(prompt) + 6,
        verify=True,
        ring_id=ring,
        request=old.request_id,
    )
    prior = cache.kv_cache[ring].clone()
    old.num_computed_tokens = len(prompt) + (2 if ending != "cancel" else 0)
    old.append_output_token_ids([100, 101, 102] if ending != "cancel" else [100])
    manager.free(old)
    assert first_ring.ref_cnt == 0
    actual.verification.forget([old.request_id])
    assert not actual.verification.tracks(old.request_id)
    assert torch.equal(cache.kv_cache[ring], prior)

    if ending == "preempt":
        next_request = old
    else:
        next_prompt = prompt if ending == "prefix_hit" else list(range(300, 563))
        next_request = Request(
            "next",
            next_prompt,
            SamplingParams(max_tokens=16),
            None,
            block_hasher=hasher,
        )
    hit_blocks, hit = manager.get_computed_blocks(next_request)
    assert hit == (256 if ending in {"preempt", "prefix_hit"} else 0)
    assert (
        manager.allocate_slots(
            next_request, next_request.num_tokens - hit, hit, hit_blocks
        )
        is not None
    )
    second_ring = _blocks(manager, next_request, CircularBufferSpec)[0]
    assert second_ring.ref_cnt == 1 and second_ring.block_hash is None
    new_ring = torch.tensor([second_ring.block_id])
    # The newly allocated private ring may contain arbitrary bytes. The aligned
    # prefix hit only reuses the pooled cache and needs no prior open members.
    cache.kv_cache[new_ring] = -311
    for pos in range(hit):
        prepare(reference, pool, pos, pos + 1)
    for pos in range(hit, next_request.num_tokens + 5):
        prepare(
            actual,
            pool,
            pos,
            pos + 1,
            ring_id=new_ring,
            offset=13,
            request=next_request.request_id,
        )
        prepare(reference, pool, pos, pos + 1, offset=13)
        groups = torch.arange((pos + 1) // RATIO)
        pages = pool_page_ids(groups, pool, PAGE, POOL_BLOCK // PAGE)
        assert torch.equal(
            actual.k_cache.kv_cache[pages, groups % PAGE],
            reference.k_cache.kv_cache[pages, groups % PAGE],
        )
        live = (pos + 1) % RATIO
        assert torch.equal(
            cache.kv_cache[new_ring, :live], reference.tail_cache.kv_cache[1:2, :live]
        )


def test_exception_after_staging_can_replay_original_start():
    table = torch.tensor([1])
    actual, reference = new_indexer(), new_indexer()
    for indexer in (actual, reference):
        prepare(indexer, table, 0, 5)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        prepare(actual, table, 5, 11, verify=True, fail=True)
    with pytest.raises(RuntimeError, match="does not cover forward start"):
        prepare(actual, table, 6, 7)
    for pos in range(5, 13):
        for indexer in (actual, reference):
            prepare(indexer, table, pos, pos + 1, offset=27)
    assert torch.equal(actual.k_cache.kv_cache, reference.k_cache.kv_cache)
    assert torch.equal(actual.tail_cache.kv_cache, reference.tail_cache.kv_cache)


@pytest.mark.parametrize("boundary", [320, 323])
def test_partial_prefix_restores_private_ring_after_verification(boundary):
    # The partial-cache implementation is admitted only for Mamba hybrids.
    # Use that real snapshot/CoW machinery with Hy4's ring shape. Standalone
    # Hy4's block-aligned allocation is covered by the ownership test above.
    groups = [
        KVCacheGroupSpec(
            ["main"],
            FullAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=8,
                dtype=torch.float16,
            ),
        ),
        KVCacheGroupSpec(["tail"], ring_spec(8)),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=256,
                shapes=((4,),),
                dtypes=(torch.float16,),
                mamba_cache_mode="align",
            ),
        ),
    ]
    manager = KVCacheManager(
        KVCacheConfig(64, [], groups),
        max_model_len=1024,
        hash_block_size=256,
        partial_prefix_cache=True,
    )
    assert manager.partial_cache is not None
    init_none_hash(sha256)
    ring_gid = 1
    hasher = get_request_block_hasher(256, sha256)
    prompt = list(range(boundary + 6))
    source = Request(
        "source",
        prompt.copy(),
        SamplingParams(
            max_tokens=16, extra_args={"prefix_cache_boundaries": [boundary]}
        ),
        None,
        block_hasher=hasher,
    )
    if boundary % 4:
        # Production rounds boundaries to 64. Also test a synthetic snapshot
        # with live members so a missing restored write cannot go unnoticed.
        source.prefix_cache_boundaries = (boundary,)
    assert source.prefix_cache_boundaries == (boundary,)
    actual, reference = new_indexer(blocks=8), new_indexer()
    tail = actual.tail_cache
    tail.kv_cache = tail.kv_cache.new_full(
        (manager.block_pool.num_gpu_blocks, RATIO, 9), -7
    )
    source_table = torch.tensor([1, 2])
    target_table = torch.tensor([2, 1])
    manager.new_step_starts()
    assert manager.allocate_slots(source, boundary) is not None
    source_ring = torch.tensor(
        [_blocks(manager, source, CircularBufferSpec)[0].block_id]
    )
    prepare(actual, source_table, 0, boundary, ring_id=source_ring)
    before_snapshot = tail.kv_cache[source_ring].clone()
    source.num_computed_tokens = boundary
    manager.new_step_starts()

    def run_ring_copies():
        copies, groups = manager.take_block_copies_with_groups()
        touched = []
        for (src, dst), group in zip(copies, groups, strict=True):
            if group == ring_gid:
                tail.kv_cache[dst].copy_(tail.kv_cache[src])
                touched.append((src, dst))
        assert len(touched) == 1
        return touched[0]

    _, saved_ring = run_ring_copies()
    # Mirror the pooled prefix in a distinct synthetic table. The manager
    # controls allocation and both copies of the actual ring storage.
    groups = torch.arange(boundary // RATIO)
    origin = pool_page_ids(groups, source_table, PAGE, POOL_BLOCK // PAGE)
    target = pool_page_ids(groups, target_table, PAGE, POOL_BLOCK // PAGE)
    actual.k_cache.kv_cache[target, groups % PAGE] = actual.k_cache.kv_cache[
        origin, groups % PAGE
    ].clone()
    assert manager.allocate_slots(source, 6) is not None
    prepare(actual, source_table, boundary, len(prompt), ring_id=source_ring, offset=30)
    source.num_computed_tokens = len(prompt)
    source.append_output_token_ids([901])
    assert manager.allocate_slots(source, 6) is not None
    prepare(
        actual,
        source_table,
        len(prompt),
        len(prompt) + 6,
        verify=True,
        ring_id=source_ring,
        request=source.request_id,
        offset=50,
    )
    assert not torch.equal(tail.kv_cache[source_ring], before_snapshot)
    manager.free(source)
    actual.verification.forget([source.request_id])

    manager.new_step_starts()
    follower = Request(
        "follower",
        prompt[:boundary] + [621, 622, 623, 624],
        SamplingParams(max_tokens=16),
        None,
        block_hasher=hasher,
    )
    hit_blocks, hit = manager.get_computed_blocks(follower)
    assert hit == boundary
    assert manager.allocate_slots(follower, 4, hit, hit_blocks) is not None
    restored_src, restored_dst = run_ring_copies()
    assert restored_src == saved_ring
    assert restored_dst != saved_ring
    assert _blocks(manager, follower, CircularBufferSpec)[0].block_id == restored_dst
    restored_ring = torch.tensor([restored_dst])
    assert torch.equal(tail.kv_cache[restored_ring], before_snapshot)

    for pos in range(hit):
        prepare(reference, source_table, pos, pos + 1)
    for pos in range(hit, hit + 8):
        prepare(
            actual,
            target_table,
            pos,
            pos + 1,
            ring_id=restored_ring,
            request=follower.request_id,
            offset=70,
        )
        prepare(reference, source_table, pos, pos + 1, offset=70)
        groups = torch.arange((pos + 1) // RATIO)
        origin = pool_page_ids(groups, source_table, PAGE, POOL_BLOCK // PAGE)
        target = pool_page_ids(groups, target_table, PAGE, POOL_BLOCK // PAGE)
        assert torch.equal(
            actual.k_cache.kv_cache[target, groups % PAGE],
            reference.k_cache.kv_cache[origin, groups % PAGE],
        )
        live = (pos + 1) % RATIO
        assert torch.equal(
            tail.kv_cache[restored_ring, :live],
            reference.tail_cache.kv_cache[1:2, :live],
        )
    assert torch.equal(tail.kv_cache[saved_ring : saved_ring + 1], before_snapshot)
