# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

"""
MTGR Dynamic Masking Strategy (Simplified Version).

This module implements MTGR's masking strategy without the real-time (RT) sequence,
as described in the paper Section 4.2.

Masking Rules (Simplified for 2 groups):
1. User + Sequence tokens: Full attention (visible to all tokens)
2. Candidate tokens: Diagonal + Static attention
   - Can see all User tokens
   - Can see all Sequence tokens
   - Can see themselves
   - CANNOT see other candidates

This prevents candidates from "cheating" by observing other candidates' features,
ensuring independent ranking.
"""

import torch
from typing import Tuple


def create_mtgr_mask(
    num_user: int,
    num_seq: int,
    num_cand: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create MTGR attention mask (simplified version without RT sequence).

    Args:
        num_user: Number of user tokens per sample
        num_seq: Number of sequence tokens per sample
        num_cand: Number of candidate tokens per sample
        batch_size: Batch size
        device: Device to create mask on

    Returns:
        mask: [total_tokens, total_tokens] additive attention mask
              0.0 = can attend (visible)
              -inf = cannot attend (hidden)

    Example:
        >>> # 2 user, 3 seq, 2 candidates per sample, batch size 1
        >>> mask = create_mtgr_mask(2, 3, 2, 1, 'cpu')
        >>> # Result: [7, 7] mask where:
        >>> #   - User and Seq can attend to all tokens
        >>> #   - Cand1 can attend to [U, S, Cand1] but not Cand2
        >>> #   - Cand2 can attend to [U, S, Cand2] but not Cand1
    """
    total_per_sample = num_user + num_seq + num_cand
    total_tokens = batch_size * total_per_sample

    # Initialize mask with zeros (all positions can attend)
    mask = torch.zeros(total_tokens, total_tokens, device=device)

    # For each sample in the batch
    for batch_idx in range(batch_size):
        # Calculate token ranges for this sample
        sample_start = batch_idx * total_per_sample
        user_start = sample_start
        user_end = user_start + num_user
        seq_start = user_end
        seq_end = seq_start + num_seq
        cand_start = seq_end
        cand_end = cand_start + num_cand

        # Rule 1: User and Sequence tokens - full attention (no masking needed)
        # These tokens can already attend to everything (mask = 0)

        # Rule 2: Candidate tokens - diagonal masking
        # Each candidate can only see:
        #   - All User tokens (already 0)
        #   - All Sequence tokens (already 0)
        #   - Itself (keep 0)
        #   - Other candidates (set to -inf)

        for i in range(num_cand):
            cand_i_idx = cand_start + i

            # Mask out all OTHER candidates
            for j in range(num_cand):
                if i != j:  # Don't mask self
                    cand_j_idx = cand_start + j
                    mask[cand_i_idx, cand_j_idx] = float('-inf')

    return mask


def visualize_mask(
    mask: torch.Tensor,
    num_user: int,
    num_seq: int,
    num_cand: int,
    batch_idx: int = 0,
) -> str:
    """
    Visualize the MTGR mask for debugging (single sample).

    Args:
        mask: [total_tokens, total_tokens] attention mask
        num_user: Number of user tokens per sample
        num_seq: Number of sequence tokens per sample
        num_cand: Number of candidate tokens per sample
        batch_idx: Which sample to visualize

    Returns:
        String representation of the mask
    """
    total_per_sample = num_user + num_seq + num_cand
    sample_start = batch_idx * total_per_sample
    sample_end = sample_start + total_per_sample

    # Extract mask for this sample
    sample_mask = mask[sample_start:sample_end, sample_start:sample_end]

    # Convert to binary (0 = can attend, 1 = masked)
    binary_mask = (sample_mask == float('-inf')).int()

    # Build string representation
    lines = []
    lines.append(f"\nMTGR Mask Visualization (Sample {batch_idx}):")
    lines.append("=" * 80)

    # Header
    header = "       "
    for i in range(num_user):
        header += f" U{i:2d}"
    for i in range(num_seq):
        header += f" S{i:2d}"
    for i in range(num_cand):
        header += f" C{i:2d}"
    lines.append(header)

    # Rows
    token_idx = 0
    for i in range(num_user):
        row = f"U{i:2d} : "
        for j in range(total_per_sample):
            row += "  ✗ " if binary_mask[token_idx, j] == 1 else "  ✓ "
        lines.append(row)
        token_idx += 1

    for i in range(num_seq):
        row = f"S{i:2d} : "
        for j in range(total_per_sample):
            row += "  ✗ " if binary_mask[token_idx, j] == 1 else "  ✓ "
        lines.append(row)
        token_idx += 1

    for i in range(num_cand):
        row = f"C{i:2d} : "
        for j in range(total_per_sample):
            row += "  ✗ " if binary_mask[token_idx, j] == 1 else "  ✓ "
        lines.append(row)
        token_idx += 1

    lines.append("=" * 80)
    lines.append("✓ = can attend (visible)")
    lines.append("✗ = cannot attend (masked)")
    lines.append("")

    return "\n".join(lines)
