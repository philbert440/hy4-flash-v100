# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for 1Cat from Leonccaa/ReVolta v0.2.0rc13 (8b49690a3).

import torch


def window_indices(positions: torch.Tensor, window: int) -> torch.Tensor:
    """Left-pack each causal window for the shared paged reader."""
    start = (positions - window + 1).clamp_min(0)
    indices = start[:, None] + torch.arange(window, device=positions.device)
    return indices.masked_fill(indices > positions[:, None], -1).int()
