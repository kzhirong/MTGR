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
Group Layer Normalization for MTGR.

This module implements the Group Layer Normalization (GLN) technique from the
MTGR paper (Section 4.2). Unlike standard LayerNorm which normalizes all tokens
together, GLN normalizes different token groups separately.

Paper reference: MTGR: Industrial-Scale Generative Recommendation Framework in Meituan
Section 4.2: "the group layer norm ensures that tokens from different domains share
a similar distribution before self-attention and aligns different semantic spaces
of different domains"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GroupLayerNorm(nn.Module):
    """
    Group Layer Normalization for MTGR.

    Applies LayerNorm separately to different token groups (User/Sequence/Candidate)
    instead of normalizing all tokens together. This is critical for MTGR because
    tokens come from heterogeneous semantic spaces:
    - User tokens: profile features (age, gender, etc.)
    - Sequence tokens: historical item interactions
    - Candidate tokens: items to rank (with cross features)

    Standard LayerNorm would compute mean/std across all these mixed domains,
    which is suboptimal. GLN computes separate statistics for each domain.

    Args:
        normalized_shape: Dimension of each token (d_model)
        eps: Epsilon for numerical stability in normalization
        elementwise_affine: Whether to learn affine parameters (weight/bias)

    Example:
        >>> gln = GroupLayerNorm(normalized_shape=128)
        >>> # Batch size 4, 3 user features, 50 seq items, 5 candidates
        >>> # Total: 4 * (3 + 50 + 5) = 232 tokens
        >>> x = torch.randn(232, 128)
        >>> group_boundaries = (3, 50, 5)  # per-sample counts
        >>> output = gln(x, group_boundaries)
        >>> output.shape  # [232, 128]
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # Shared affine parameters across all groups
            # (could also have separate parameters per group for more flexibility)
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(
        self,
        x: torch.Tensor,
        group_boundaries: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Apply LayerNorm separately to each token group.

        Args:
            x: [total_tokens, d_model] - All tokens concatenated
               Order: [user_tokens, seq_tokens, cand_tokens]
            group_boundaries: (num_user_tokens, num_seq_tokens, num_cand_tokens)
                             These are per-sample counts, not total counts.

        Returns:
            Normalized tokens [total_tokens, d_model] with same shape as input

        Example:
            Input: [232, 128] with group_boundaries=(3, 50, 5), batch_size=4
            - User tokens: x[0:12]    (4 samples * 3 features)
            - Seq tokens:  x[12:212]  (4 samples * 50 items)
            - Cand tokens: x[212:232] (4 samples * 5 candidates)
            Each group normalized separately.
        """
        num_user, num_seq, num_cand = group_boundaries

        # Calculate batch size from total tokens
        total_per_sample = num_user + num_seq + num_cand
        if total_per_sample == 0:
            # Empty input, return as-is
            return x

        batch_size = x.shape[0] // total_per_sample

        # Split tokens by group
        user_end = batch_size * num_user
        seq_end = user_end + batch_size * num_seq

        user_tokens = x[:user_end]       # [B*num_user, d_model]
        seq_tokens = x[user_end:seq_end] # [B*num_seq, d_model]
        cand_tokens = x[seq_end:]        # [B*num_cand, d_model]

        # Apply LayerNorm separately to each group
        # This ensures each domain has its own mean≈0, std≈1 distribution
        normalized_groups = []

        if user_tokens.numel() > 0:
            user_normed = F.layer_norm(
                user_tokens,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps
            )
            normalized_groups.append(user_normed)

        if seq_tokens.numel() > 0:
            seq_normed = F.layer_norm(
                seq_tokens,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps
            )
            normalized_groups.append(seq_normed)

        if cand_tokens.numel() > 0:
            cand_normed = F.layer_norm(
                cand_tokens,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps
            )
            normalized_groups.append(cand_normed)

        # Concatenate back in original order
        return torch.cat(normalized_groups, dim=0)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'{self.normalized_shape}, '
            f'eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}'
        )
