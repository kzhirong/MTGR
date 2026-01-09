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
import abc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from generative_recommenders.common import fx_unwrap_optional_tensor, HammerModule
from generative_recommenders.modules.group_layer_norm import GroupLayerNorm
from generative_recommenders.ops.hstu_attention import delta_hstu_mha
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output,
    hstu_compute_uqvk,
    hstu_preprocess_and_attention,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from torch.autograd.profiler import record_function


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


class STU(HammerModule, abc.ABC):
    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


@dataclass
class STULayerConfig:
    embedding_dim: int
    num_heads: int
    hidden_dim: int
    attention_dim: int
    output_dropout_ratio: float = 0.3
    causal: bool = True
    target_aware: bool = True
    max_attn_len: Optional[int] = None
    attn_alpha: Optional[float] = None
    use_group_norm: bool = False
    recompute_normed_x: bool = True
    recompute_uvqk: bool = True
    recompute_y: bool = True
    sort_by_length: bool = True
    contextual_seq_len: int = 0

    # MTGR-specific configuration
    mtgr_mode: bool = False  # Enable MTGR Group Layer Normalization
    mtgr_num_user_tokens: int = 0  # Number of user tokens per sample
    mtgr_num_seq_tokens: int = 0   # Max sequence tokens per sample
    mtgr_num_cand_tokens: int = 0  # Number of candidate tokens per sample


@torch.fx.wrap
def _update_kv_cache(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    max_kv_caching_len: int,
    kv_caching_lengths: Optional[torch.Tensor],
    orig_k_cache: Optional[torch.Tensor],
    orig_v_cache: Optional[torch.Tensor],
    orig_max_kv_caching_len: int,
    orig_kv_caching_offsets: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    if kv_caching_lengths is not None:
        kv_caching_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            kv_caching_lengths
        )
        delta_offsets = seq_offsets - kv_caching_offsets
        k_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(k).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        v_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(v).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        if max_kv_caching_len == 0:
            max_kv_caching_len = int(kv_caching_lengths.max().item())
        return (
            k_cache,
            v_cache,
            max_kv_caching_len,
            kv_caching_offsets,
        )
    else:
        return (
            orig_k_cache,
            orig_v_cache,
            orig_max_kv_caching_len,
            orig_kv_caching_offsets,
        )


@torch.fx.wrap
def _construct_full_kv(
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    max_kv_caching_len: int,
    kv_caching_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    L, _ = delta_k.shape
    B = kv_caching_offsets.shape[0] - 1
    delta_size = L // B
    full_k = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=k_cache,
        values_right=delta_k,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_v = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=v_cache,
        values_right=delta_v,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_kv_caching_offsets = kv_caching_offsets + delta_size * torch.arange(
        B + 1, device=delta_k.device
    )
    return (
        full_k,
        full_v,
        max_kv_caching_len + delta_size,
        full_kv_caching_offsets,
    )


class STULayer(STU):
    max_kv_caching_len: int
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    kv_caching_offsets: Optional[torch.Tensor]

    def __init__(
        self,
        config: STULayerConfig,
        is_inference: bool = False,
    ) -> None:
        super().__init__(
            is_inference=is_inference,
        )
        self.reset_kv_cache()
        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        self._attn_alpha: float = config.attn_alpha or 1.0 / (self._attention_dim**0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len

        self._uvqk_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    self._embedding_dim,
                    (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._uvqk_weight)
        self._uvqk_beta: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(
                (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
            ),
        )
        self._input_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self._embedding_dim,)),
        )
        self._input_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._embedding_dim,)),
        )
        self._output_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self._hidden_dim * self._num_heads * 3,
                    self._embedding_dim,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        output_norm_shape: int = (
            self._hidden_dim * self._num_heads
            if not self._use_group_norm
            else self._num_heads
        )
        self._output_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((output_norm_shape,)),
        )
        self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((output_norm_shape,)),
        )

        # MTGR-specific components
        self._mtgr_mode: bool = config.mtgr_mode
        if self._mtgr_mode:
            # Group Layer Normalization for input (before attention)
            self._mtgr_input_group_norm = GroupLayerNorm(
                normalized_shape=self._embedding_dim,
                eps=1e-6,
                elementwise_affine=True,
            )
            # Group Layer Normalization for output (after attention)
            self._mtgr_output_group_norm = GroupLayerNorm(
                normalized_shape=self._embedding_dim,
                eps=1e-6,
                elementwise_affine=True,
            )
            # Store token boundaries (will be updated during forward pass)
            self._mtgr_group_boundaries = (
                config.mtgr_num_user_tokens,
                config.mtgr_num_seq_tokens,
                config.mtgr_num_cand_tokens,
            )

    def reset_kv_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.kv_caching_offsets = None
        self.max_kv_caching_len = 0

    def update_kv_cache(
        self,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        max_kv_caching_len: int,
        kv_caching_lengths: Optional[torch.Tensor],
    ) -> None:
        self.k_cache, self.v_cache, self.max_kv_caching_len, self.kv_caching_offsets = (
            _update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                k=k,
                v=v,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                orig_k_cache=self.k_cache,
                orig_v_cache=self.v_cache,
                orig_max_kv_caching_len=self.max_kv_caching_len,
                orig_kv_caching_offsets=self.kv_caching_offsets,
            )
        )

    def construct_full_kv(
        self,
        delta_k: torch.Tensor,
        delta_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return _construct_full_kv(
            delta_k=delta_k,
            delta_v=delta_v,
            k_cache=fx_unwrap_optional_tensor(self.k_cache),
            v_cache=fx_unwrap_optional_tensor(self.v_cache),
            max_kv_caching_len=self.max_kv_caching_len,
            kv_caching_offsets=self.kv_caching_offsets,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # MTGR Mode: Decomposed operations with Group Layer Normalization
        if self._mtgr_mode:
            # Step 1: Apply Group LayerNorm to input (normalizes by domain)
            with record_function("## mtgr_input_group_norm ##"):
                x_normed = self._mtgr_input_group_norm(x, self._mtgr_group_boundaries)

            # Step 2: Manual UVQK projection (without normalization)
            # This replicates hstu_compute_uqvk but skips the built-in LayerNorm
            with record_function("## mtgr_uvqk_projection ##"):
                # Linear projection: uvqk = x_normed @ W + b
                uvqk = torch.addmm(
                    self._uvqk_beta.to(x.dtype),
                    x_normed,
                    self._uvqk_weight.to(x.dtype)
                )

                # Split into U, V, Q, K
                u, v, q, k = torch.split(
                    uvqk,
                    [
                        self._hidden_dim * self._num_heads,
                        self._hidden_dim * self._num_heads,
                        self._attention_dim * self._num_heads,
                        self._attention_dim * self._num_heads,
                    ],
                    dim=1,
                )

                # Apply SiLU activation to U
                u = torch.nn.functional.silu(u)

                # Reshape Q, K, V for multi-head attention
                q = q.view(-1, self._num_heads, self._attention_dim)
                k = k.view(-1, self._num_heads, self._attention_dim)
                v = v.view(-1, self._num_heads, self._hidden_dim)

            # Step 3: MTGR Attention with Dynamic Masking
            with record_function("## mtgr_attention ##"):
                # Calculate batch size and tokens per sample
                total_per_sample = sum(self._mtgr_group_boundaries)
                total_tokens = q.shape[0]
                batch_size = total_tokens // total_per_sample

                # Create MTGR mask (diagonal masking for candidates)
                from generative_recommenders.modules.mtgr_mask import create_mtgr_mask

                mask_2d = create_mtgr_mask(
                    num_user=self._mtgr_group_boundaries[0],
                    num_seq=self._mtgr_group_boundaries[1],
                    num_cand=self._mtgr_group_boundaries[2],
                    batch_size=batch_size,
                    device=q.device,
                )  # Shape: [total_tokens, total_tokens]

                # Reshape mask for multi-head attention
                # [total_tokens, total_tokens] → [batch, num_heads, seq_len, seq_len]
                mask_per_sample = mask_2d.view(
                    batch_size,
                    total_per_sample,
                    total_per_sample
                )
                # Expand for multiple heads (broadcast over head dimension)
                mask_4d = mask_per_sample.unsqueeze(1)  # [batch, 1, seq_len, seq_len]

                # Reshape Q, K, V for PyTorch attention
                # From: [total_tokens, num_heads, dim]
                # To:   [batch, num_heads, seq_len, dim]
                q_4d = q.view(batch_size, total_per_sample, self._num_heads, self._attention_dim)
                q_4d = q_4d.transpose(1, 2)  # [batch, num_heads, seq_len, dim]

                k_4d = k.view(batch_size, total_per_sample, self._num_heads, self._attention_dim)
                k_4d = k_4d.transpose(1, 2)

                v_4d = v.view(batch_size, total_per_sample, self._num_heads, self._hidden_dim)
                v_4d = v_4d.transpose(1, 2)

                # Apply PyTorch native attention with MTGR mask
                attn_output_4d = torch.nn.functional.scaled_dot_product_attention(
                    query=q_4d,
                    key=k_4d,
                    value=v_4d,
                    attn_mask=mask_4d,
                    dropout_p=0.0,
                    scale=self._attn_alpha,
                )  # Shape: [batch, num_heads, seq_len, hidden_dim]

                # Reshape back to match expected output format
                # [batch, num_heads, seq_len, hidden_dim] → [total_tokens, num_heads, hidden_dim]
                attn_output = attn_output_4d.transpose(1, 2).contiguous()
                attn_output = attn_output.view(total_tokens, self._num_heads, self._hidden_dim)

            # Update KV cache if needed
            self.update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                k=k.flatten(1, 2),
                v=v.flatten(1, 2),
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )

            # Step 4: Manual output processing (without normalization)
            with record_function("## mtgr_output_processing ##"):
                # attn_output is [B, H, D] from hstu_mha, reshape to [B, H*D]
                attn_flat = attn_output.view(-1, self._num_heads * self._hidden_dim)

                # Element-wise multiply: y = attn ⊙ u
                y = attn_flat * u

                # Concatenate [y, u, x_normed] as per HSTU architecture
                concat = torch.cat([y, u, x_normed], dim=1)

                # Apply dropout if training
                if self.training and self._output_dropout_ratio > 0:
                    concat = torch.nn.functional.dropout(
                        concat,
                        p=self._output_dropout_ratio,
                        training=True
                    )

                # Final linear projection: output = concat @ W
                output_unnormed = torch.matmul(concat, self._output_weight.to(x.dtype))

            # Step 5: Apply Group LayerNorm to output
            with record_function("## mtgr_output_group_norm ##"):
                output_normed = self._mtgr_output_group_norm(
                    output_unnormed,
                    self._mtgr_group_boundaries
                )

            # Step 6: Residual connection
            return output_normed + x

        # Standard HSTU Mode (unchanged)
        else:
            with record_function("## stu_preprocess_and_attention ##"):
                u, attn_output, k, v = hstu_preprocess_and_attention(
                    x=x,
                    norm_weight=self._input_norm_weight.to(x.dtype),
                    norm_bias=self._input_norm_bias.to(x.dtype),
                    norm_eps=1e-6,
                    num_heads=self._num_heads,
                    attn_dim=self._attention_dim,
                    hidden_dim=self._hidden_dim,
                    uvqk_weight=self._uvqk_weight.to(x.dtype),
                    uvqk_bias=self._uvqk_beta.to(x.dtype),
                    max_seq_len=max_seq_len,
                    seq_offsets=x_offsets,
                    attn_alpha=self._attn_alpha,
                    causal=self._causal,
                    num_targets=num_targets if self._target_aware else None,
                    max_attn_len=self._max_attn_len,
                    contextual_seq_len=self._contextual_seq_len,
                    recompute_uvqk_in_backward=self._recompute_uvqk,
                    recompute_normed_x_in_backward=self._recompute_normed_x,
                    sort_by_length=self._sort_by_length,
                    prefill=kv_caching_lengths is not None,
                    kernel=self.hammer_kernel(),
                )

            self.update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                k=k,
                v=v,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )

            with record_function("## stu_compute_output ##"):
                return hstu_compute_output(
                    attn=attn_output,
                    u=u,
                    x=x,
                    norm_weight=self._output_norm_weight.to(x.dtype),
                    norm_bias=self._output_norm_bias.to(x.dtype),
                    norm_eps=1e-6,
                    dropout_ratio=self._output_dropout_ratio,
                    output_weight=self._output_weight.to(x.dtype),
                    group_norm=self._use_group_norm,
                    num_heads=self._num_heads,
                    linear_dim=self._hidden_dim,
                    concat_ux=True,
                    training=self.training,
                    kernel=self.hammer_kernel(),
                    recompute_y_in_backward=self._recompute_y,
                )

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("## stu_compute_uqvk ##"):
            delta_u, delta_q, delta_k, delta_v = hstu_compute_uqvk(
                x=delta_x,
                norm_weight=self._input_norm_weight.to(delta_x.dtype),
                norm_bias=self._input_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(delta_x.dtype),
                uvqk_bias=self._uvqk_beta.to(delta_x.dtype),
                kernel=self.hammer_kernel(),
            )
        k, v, max_seq_len, seq_offsets = self.construct_full_kv(
            delta_k=delta_k.flatten(1, 2),
            delta_v=delta_v.flatten(1, 2),
        )
        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )
        k = k.view(-1, self._num_heads, self._attention_dim)
        v = v.view(-1, self._num_heads, self._hidden_dim)
        with record_function("## delta_hstu_mha ##"):
            delta_attn_output = delta_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                delta_q=delta_q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                kernel=self.hammer_kernel(),
            ).view(-1, self._hidden_dim * self._num_heads)
        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=delta_attn_output,
                u=delta_u,
                x=delta_x,
                norm_weight=self._output_norm_weight.to(delta_x.dtype),
                norm_bias=self._output_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(delta_x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_ux=True,
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )


class STUStack(STU):
    def __init__(
        self,
        stu_list: List[STU],
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return x

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            delta_x = layer.cached_forward(  # pyre-ignore [29]
                delta_x=delta_x,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return delta_x
