# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Proposal-only LM head for Hy4 DFly2."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .dense import W8StorageHead

logger = init_logger(__name__)


class DraftHead:
    """Score proposals with a vocab-parallel FP16 or g128 int8 head."""

    def __init__(self, target: ParallelLMHead, mode: str):
        vocab_size = target.org_vocab_size
        if mode == "int8":
            self.head = _int8_head(target)
            self.processor = LogitsProcessor(vocab_size)
            return
        weight = target.weight
        with torch.device(weight.device):
            self.head = ParallelLMHead(
                vocab_size, weight.shape[1], params_dtype=torch.float16,
                prefix="hy4_dfly.lm_head",
            )
        with torch.no_grad():
            self.head.weight.copy_(weight)
        self.processor = LogitsProcessor(vocab_size)
        logger.info_once(
            "Hy4 DFly2 fp16 proposal head: %d vocab rows, %d resident bytes per rank",
            vocab_size, self.head.weight.nbytes,
        )

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.processor.get_top_tokens(self.head, hidden.to(torch.float16))


def _int8_head(target: ParallelLMHead) -> W8StorageHead:
    group_size = 128
    if (
        isinstance(target, W8StorageHead)
        and target.w8_group_size == group_size
        and target.packed_head is not None
    ):
        logger.info_once("Hy4 DFly2 int8 proposal head reuses the target g128 pack.")
        return target

    weight = target.weight
    with torch.device(weight.device):
        head = W8StorageHead(
            target.org_vocab_size,
            weight.shape[1],
            group_size=group_size,
            prefix="hy4_dfly.lm_head",
        )
    with torch.no_grad():
        if weight.dtype == torch.float16:
            head.weight = weight
        else:
            head.weight.copy_(weight)
        head.quant_method.process_weights_after_loading(head)
    logger.info_once(
        "Hy4 DFly2 int8 proposal head: %d packed bytes per rank.",
        head.packed_head.nbytes,
    )
    return head
