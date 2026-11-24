# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Image rows contribute nothing. Later n-grams still hash the raw image id."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.hy_v4_flash.nvidia.engram import FlashNgramHashState

pytestmark = pytest.mark.skip_global_cleanup

VISUAL = 17
OTHER = 19


def _state():
    layers = (2, 12, 22, 32)
    config = SimpleNamespace(
        engram_layer_ids=layers,
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=8,
        engram_pad_id=3,
        engram_seed=42,
        vocab_size=32,
        engram_layer_head_vocab_sizes={
            str(layer): [[11, 13], [17, 19]] for layer in layers
        },
    )
    state = FlashNgramHashState(config)
    identity = torch.arange(config.vocab_size, dtype=torch.int32)
    for layer in layers:
        state.load_lookup(identity, layer)
    return state


def _hashes(state, tokens):
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, tokens.numel()], dtype=torch.int32),
        num_actual_tokens=int(tokens.numel()),
    )
    lookback = torch.full((1, state.lookback_depth), 3, dtype=torch.int32)
    return state.prepare_hashes(tokens, lookback, metadata)


def test_visual_rows_are_suppressed_and_later_ngrams_keep_the_raw_id():
    state = _state()
    # Text, image pad, text, engram pad, text. The image id is inside the next n-gram.
    tokens = torch.tensor([4, VISUAL, 8, 3, 9], dtype=torch.int32)
    state.visual_token_ids = ()
    plain = _hashes(state, tokens)
    assert not torch.equal(plain[:, 1, :], torch.full_like(plain[:, 1, :], -1))

    state.visual_token_ids = (VISUAL, OTHER)
    masked = _hashes(state, tokens)
    assert torch.equal(masked[:, :1], plain[:, :1])
    assert torch.equal(masked[:, 1, :], torch.full_like(masked[:, 1, :], -1))
    assert torch.equal(masked[:, 2:], plain[:, 2:])
    assert torch.equal(masked[:, 3, :], torch.full_like(masked[:, 3, :], -1))

    replaced = tokens.clone()
    replaced[1] = 5
    state.visual_token_ids = ()
    other = _hashes(state, replaced)
    assert not torch.equal(plain[:, 2:3], other[:, 2:3])

    fresh = _state()
    assert fresh.visual_token_ids == ()
    assert torch.equal(_hashes(fresh, tokens), plain)
