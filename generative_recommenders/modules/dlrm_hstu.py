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


import logging
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from generative_recommenders.common import (
    fx_infer_max_len,
    fx_mark_length_features,
    HammerKernel,
    HammerModule,
    init_mlp_weights_optional_bias,
    set_static_max_seq_lens,
)
from generative_recommenders.modules.hstu_transducer import HSTUTransducer
from generative_recommenders.modules.multitask_module import (
    DefaultMultitaskModule,
    MultitaskTaskType,
    TaskConfig,
)
from generative_recommenders.modules.positional_encoder import HSTUPositionalEncoder
from generative_recommenders.modules.postprocessors import (
    LayerNormPostprocessor,
    TimestampLayerNormPostprocessor,
)
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from torch.autograd.profiler import record_function
from torchrec import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection

logger: logging.Logger = logging.getLogger(__name__)

torch.fx.wrap("fx_infer_max_len")
torch.fx.wrap("len")


class SequenceEmbedding(NamedTuple):
    lengths: torch.Tensor
    embedding: torch.Tensor


@dataclass
class DlrmHSTUConfig:
    max_seq_len: int = 16384
    max_num_candidates: int = 10
    max_num_candidates_inference: int = 5
    hstu_num_heads: int = 1
    hstu_attn_linear_dim: int = 256
    hstu_attn_qk_dim: int = 128
    hstu_attn_num_layers: int = 12
    hstu_embedding_table_dim: int = 192
    hstu_preprocessor_hidden_dim: int = 256
    hstu_transducer_embedding_dim: int = 0
    hstu_group_norm: bool = False
    hstu_input_dropout_ratio: float = 0.2
    hstu_linear_dropout_rate: float = 0.2
    contextual_feature_to_max_length: Dict[str, int] = field(default_factory=dict)
    contextual_feature_to_min_uih_length: Dict[str, int] = field(default_factory=dict)
    candidates_weight_feature_name: str = ""
    candidates_watchtime_feature_name: str = ""
    candidates_querytime_feature_name: str = ""
    causal_multitask_weights: float = 0.2
    multitask_configs: List[TaskConfig] = field(default_factory=list)
    user_embedding_feature_names: List[str] = field(default_factory=list)
    item_embedding_feature_names: List[str] = field(default_factory=list)
    uih_post_id_feature_name: str = ""
    uih_action_time_feature_name: str = ""
    uih_weight_feature_name: str = ""
    hstu_uih_feature_names: List[str] = field(default_factory=list)
    hstu_candidate_feature_names: List[str] = field(default_factory=list)
    merge_uih_candidate_feature_mapping: List[Tuple[str, str]] = field(
        default_factory=list
    )
    action_weights: Optional[List[int]] = None
    action_embedding_init_std: float = 0.1
    enable_postprocessor: bool = True
    use_layer_norm_postprocessor: bool = False
    # MTGR-specific configs
    use_mtgr_tokenization: bool = False
    mtgr_user_feature_names: List[str] = field(default_factory=list)
    mtgr_seq_feature_names: List[str] = field(default_factory=list)
    mtgr_cross_feature_names: List[str] = field(default_factory=list)
    mtgr_candidate_feature_names: List[str] = field(default_factory=list)


def _get_supervision_labels_and_weights(
    supervision_bitmasks: torch.Tensor,
    watchtime_sequence: torch.Tensor,
    task_configs: List[TaskConfig],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    supervision_labels: Dict[str, torch.Tensor] = {}
    supervision_weights: Dict[str, torch.Tensor] = {}
    for task in task_configs:
        if task.task_type == MultitaskTaskType.REGRESSION:
            supervision_labels[task.task_name] = watchtime_sequence.to(torch.float32)
        elif task.task_type == MultitaskTaskType.BINARY_CLASSIFICATION:
            supervision_labels[task.task_name] = (
                torch.bitwise_and(supervision_bitmasks, task.task_weight) > 0
            ).to(torch.float32)
        else:
            raise RuntimeError("Unsupported MultitaskTaskType")
    return supervision_labels, supervision_weights


class DlrmHSTU(HammerModule):
    def __init__(  # noqa C901
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_tables: Dict[str, EmbeddingConfig],
        is_inference: bool,
        is_dense: bool = False,
        bf16_training: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        logger.info(f"Initialize HSTU module with configs {hstu_configs}")
        self._hstu_configs = hstu_configs
        self._bf16_training: bool = bf16_training
        set_static_max_seq_lens([self._hstu_configs.max_seq_len])

        if not is_dense:
            self._embedding_collection: EmbeddingCollection = EmbeddingCollection(
                tables=list(embedding_tables.values()),
                need_indices=False,
                device=torch.device("meta"),
            )

        # multitask configs must be sorted by task types
        self._multitask_configs: List[TaskConfig] = hstu_configs.multitask_configs
        self._multitask_module = DefaultMultitaskModule(
            task_configs=self._multitask_configs,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            prediction_fn=lambda in_dim, num_tasks: torch.nn.Sequential(
                torch.nn.Linear(in_features=in_dim, out_features=512),
                SwishLayerNorm(512),
                torch.nn.Linear(in_features=512, out_features=num_tasks),
            ).apply(init_mlp_weights_optional_bias),
            causal_multitask_weights=hstu_configs.causal_multitask_weights,
            is_inference=self._is_inference,
        )
        self._additional_embedding_features: List[str] = [
            uih_feature_name
            for (
                uih_feature_name,
                candidate_feature_name,
            ) in self._hstu_configs.merge_uih_candidate_feature_mapping
            if (
                candidate_feature_name
                in self._hstu_configs.item_embedding_feature_names
            )
            and (uih_feature_name in self._hstu_configs.user_embedding_feature_names)
            and (uih_feature_name is not self._hstu_configs.uih_post_id_feature_name)
        ]

        # preprocessor setup
        preprocessor = ContextualPreprocessor(
            input_embedding_dim=hstu_configs.hstu_embedding_table_dim,
            hidden_dim=hstu_configs.hstu_preprocessor_hidden_dim,
            output_embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_feature_to_max_length=hstu_configs.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=hstu_configs.contextual_feature_to_min_uih_length,
            action_embedding_dim=8,
            action_feature_name=self._hstu_configs.uih_weight_feature_name,
            action_weights=self._hstu_configs.action_weights,
            action_embedding_init_std=self._hstu_configs.action_embedding_init_std,
            additional_embedding_features=self._additional_embedding_features,
            is_inference=is_inference,
        )

        # positional encoder
        positional_encoder = HSTUPositionalEncoder(
            num_position_buckets=8192,
            num_time_buckets=2048,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_seq_len=sum(
                dict(hstu_configs.contextual_feature_to_max_length).values()
            ),
            is_inference=self._is_inference,
        )

        if hstu_configs.enable_postprocessor:
            if hstu_configs.use_layer_norm_postprocessor:
                postprocessor = LayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
            else:
                postprocessor = TimestampLayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    time_duration_features=[
                        (60 * 60, 24),  # hour of day
                        (24 * 60 * 60, 7),  # day of week
                        # (24 * 60 * 60, 365), # time of year (approximate)
                    ],
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
        else:
            postprocessor = None

        # construct HSTU
        stu_module: STU = STUStack(
            stu_list=[
                STULayer(
                    config=STULayerConfig(
                        embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                        num_heads=hstu_configs.hstu_num_heads,
                        hidden_dim=hstu_configs.hstu_attn_linear_dim,
                        attention_dim=hstu_configs.hstu_attn_qk_dim,
                        output_dropout_ratio=hstu_configs.hstu_linear_dropout_rate,
                        use_group_norm=hstu_configs.hstu_group_norm,
                        causal=True,
                        target_aware=True,
                        max_attn_len=None,
                        attn_alpha=None,
                        recompute_normed_x=True,
                        recompute_uvqk=True,
                        recompute_y=True,
                        sort_by_length=True,
                        contextual_seq_len=0,
                    ),
                    is_inference=is_inference,
                )
                for _ in range(hstu_configs.hstu_attn_num_layers)
            ],
            is_inference=is_inference,
        )
        self._hstu_transducer: HSTUTransducer = HSTUTransducer(
            stu_module=stu_module,
            input_preprocessor=preprocessor,
            output_postprocessor=postprocessor,
            input_dropout_ratio=hstu_configs.hstu_input_dropout_ratio,
            positional_encoder=positional_encoder,
            is_inference=self._is_inference,
            return_full_embeddings=False,
            listwise=False,
        )

        # item embeddings (original HSTU)
        self._item_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hstu_configs.hstu_embedding_table_dim
                * len(self._hstu_configs.item_embedding_feature_names),
                out_features=512,
            ),
            SwishLayerNorm(512),
            torch.nn.Linear(
                in_features=512,
                out_features=hstu_configs.hstu_transducer_embedding_dim,
            ),
            LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

        # MTGR-specific MLPs for 3-token structure
        if hstu_configs.use_mtgr_tokenization:
            logger.info("Initializing MTGR tokenization MLPs")

            # User MLP: converts user features to tokens
            if len(hstu_configs.mtgr_user_feature_names) > 0:
                self._mtgr_user_mlp: torch.nn.Module = torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=hstu_configs.hstu_embedding_table_dim
                        * len(hstu_configs.mtgr_user_feature_names),
                        out_features=512,
                    ),
                    SwishLayerNorm(512),
                    torch.nn.Linear(
                        in_features=512,
                        out_features=hstu_configs.hstu_transducer_embedding_dim,
                    ),
                    LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
                ).apply(init_mlp_weights_optional_bias)

            # Sequence MLP: converts sequence item features to tokens
            if len(hstu_configs.mtgr_seq_feature_names) > 0:
                self._mtgr_seq_mlp: torch.nn.Module = torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=hstu_configs.hstu_embedding_table_dim
                        * len(hstu_configs.mtgr_seq_feature_names),
                        out_features=512,
                    ),
                    SwishLayerNorm(512),
                    torch.nn.Linear(
                        in_features=512,
                        out_features=hstu_configs.hstu_transducer_embedding_dim,
                    ),
                    LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
                ).apply(init_mlp_weights_optional_bias)

            # Candidate MLP: converts candidate features (cross + item) to tokens
            num_cand_features = (
                len(hstu_configs.mtgr_cross_feature_names)
                + len(hstu_configs.mtgr_candidate_feature_names)
            )
            if num_cand_features > 0:
                self._mtgr_candidate_mlp: torch.nn.Module = torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=hstu_configs.hstu_embedding_table_dim * num_cand_features,
                        out_features=512,
                    ),
                    SwishLayerNorm(512),
                    torch.nn.Linear(
                        in_features=512,
                        out_features=hstu_configs.hstu_transducer_embedding_dim,
                    ),
                    LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
                ).apply(init_mlp_weights_optional_bias)

    def _construct_payload(
        self,
        payload_features: Dict[str, torch.Tensor],
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> Dict[str, torch.Tensor]:
        if len(self._hstu_configs.contextual_feature_to_max_length) > 0:
            contextual_offsets: List[torch.Tensor] = []
            for x in self._hstu_configs.contextual_feature_to_max_length.keys():
                contextual_offsets.append(
                    torch.ops.fbgemm.asynchronous_complete_cumsum(
                        seq_embeddings[x].lengths
                    )
                )
        else:
            # Dummy, offsets are unused
            contextual_offsets = torch.empty((0, 0))
        return {
            **payload_features,
            **{
                x: seq_embeddings[x].embedding
                for x in self._hstu_configs.contextual_feature_to_max_length.keys()
            },
            **{
                x + "_offsets": contextual_offsets[i]
                for i, x in enumerate(
                    list(self._hstu_configs.contextual_feature_to_max_length.keys())
                )
            },
            **{
                x: seq_embeddings[x].embedding
                for x in self._additional_embedding_features
            },
        }

    def _user_forward(
        self,
        max_uih_len: int,
        max_candidates: int,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        source_lengths = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths
        source_timestamps = concat_2D_jagged(
            max_seq_len=max_uih_len + max_candidates,
            max_len_left=max_uih_len,
            offsets_left=payload_features["uih_offsets"],
            values_left=payload_features[
                self._hstu_configs.uih_action_time_feature_name
            ].unsqueeze(-1),
            max_len_right=max_candidates,
            offsets_right=payload_features["candidate_offsets"],
            values_right=payload_features[
                self._hstu_configs.candidates_querytime_feature_name
            ].unsqueeze(-1),
            kernel=self.hammer_kernel(),
        ).squeeze(-1)
        total_targets = int(num_candidates.sum().item())
        embedding = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].embedding
        dtype = embedding.dtype
        if (not self.is_inference) and self._bf16_training:
            embedding = embedding.to(torch.bfloat16)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not self.is_inference) and self._bf16_training,
        ):
            candidates_user_embeddings, _ = self._hstu_transducer(
                max_uih_len=max_uih_len,
                max_targets=max_candidates,
                total_uih_len=source_timestamps.numel() - total_targets,
                total_targets=total_targets,
                seq_embeddings=embedding,
                seq_lengths=source_lengths,
                seq_timestamps=source_timestamps,
                seq_payloads=self._construct_payload(
                    payload_features=payload_features,
                    seq_embeddings=seq_embeddings,
                ),
                num_targets=num_candidates,
            )
        candidates_user_embeddings = candidates_user_embeddings.to(dtype)

        return candidates_user_embeddings

    def _item_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> torch.Tensor:  # [L, D]
        all_embeddings = torch.cat(
            [
                seq_embeddings[name].embedding
                for name in self._hstu_configs.item_embedding_feature_names
            ],
            dim=-1,
        )
        item_embeddings = self._item_embedding_mlp(all_embeddings)
        return item_embeddings

    def _split_mtgr_features(
        self,
        kjt: KeyedJaggedTensor,
        feature_names: List[str],
    ) -> Optional[KeyedJaggedTensor]:
        """
        Extract a subset of features from a KJT.

        Args:
            kjt: Input KeyedJaggedTensor
            feature_names: List of feature names to extract

        Returns:
            New KJT with only the specified features, or None if no features match
        """
        if not feature_names:
            return None

        # Find which features exist in the KJT
        available_features = [f for f in feature_names if f in kjt.keys()]
        if not available_features:
            return None

        # Extract indices for the requested features
        all_keys = kjt.keys()
        feature_indices = [all_keys.index(f) for f in available_features]

        # Extract lengths and values for these features
        lengths_2d = kjt.lengths().view(len(all_keys), -1)
        selected_lengths = torch.cat([lengths_2d[i] for i in feature_indices], dim=0)

        # Calculate value offsets
        lengths_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(kjt.lengths())
        selected_values = []
        for i in feature_indices:
            start_idx = lengths_cumsum[i * kjt.stride()] if i > 0 else 0
            end_idx = lengths_cumsum[(i + 1) * kjt.stride()]
            selected_values.append(kjt.values()[start_idx:end_idx])

        return KeyedJaggedTensor(
            keys=available_features,
            lengths=selected_lengths,
            values=torch.cat(selected_values, dim=0),
        )

    def _mtgr_tokenize(
        self,
        user_embeddings: Dict[str, SequenceEmbedding],
        seq_embeddings: Dict[str, SequenceEmbedding],
        cand_embeddings: Dict[str, SequenceEmbedding],
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """
        Apply MTGR tokenization: convert embeddings to unified tokens using 3 MLPs.

        Args:
            user_embeddings: Dict of user feature embeddings
            seq_embeddings: Dict of sequence item feature embeddings
            cand_embeddings: Dict of candidate feature embeddings (cross + item)

        Returns:
            Tuple of:
            - all_tokens: [B, total_len, d_model] concatenated tokens
            - group_indices: [total_len] indicating token type (0=user, 1=seq, 2=cand)
            - num_user_tokens: Number of user tokens per sample
            - num_seq_tokens: Number of sequence tokens per sample (max)
            - num_cand_tokens: Number of candidate tokens per sample (max)
        """
        tokens_list = []
        num_user_tokens = 0
        num_seq_tokens = 0
        num_cand_tokens = 0

        # 1. User tokens (scalar features)
        if len(self._hstu_configs.mtgr_user_feature_names) > 0:
            user_embeddings_list = [
                user_embeddings[name].embedding
                for name in self._hstu_configs.mtgr_user_feature_names
                if name in user_embeddings
            ]
            if user_embeddings_list:
                user_concat = torch.cat(user_embeddings_list, dim=-1)
                user_tokens = self._mtgr_user_mlp(user_concat)  # [B * num_user_features, d_model]
                tokens_list.append(user_tokens)
                num_user_tokens = len(self._hstu_configs.mtgr_user_feature_names)

        # 2. Sequence tokens (items in history)
        if len(self._hstu_configs.mtgr_seq_feature_names) > 0:
            seq_embeddings_list = [
                seq_embeddings[name].embedding
                for name in self._hstu_configs.mtgr_seq_feature_names
                if name in seq_embeddings
            ]
            if seq_embeddings_list:
                seq_concat = torch.cat(seq_embeddings_list, dim=-1)
                seq_tokens = self._mtgr_seq_mlp(seq_concat)  # [total_seq_items, d_model]
                tokens_list.append(seq_tokens)
                # Get max sequence length
                seq_lengths = seq_embeddings[self._hstu_configs.mtgr_seq_feature_names[0]].lengths
                num_seq_tokens = int(torch.max(seq_lengths).item())

        # 3. Candidate tokens (cross + item features)
        num_cand_features = (
            len(self._hstu_configs.mtgr_cross_feature_names)
            + len(self._hstu_configs.mtgr_candidate_feature_names)
        )
        if num_cand_features > 0:
            cand_embeddings_list = [
                cand_embeddings[name].embedding
                for name in (
                    self._hstu_configs.mtgr_cross_feature_names
                    + self._hstu_configs.mtgr_candidate_feature_names
                )
                if name in cand_embeddings
            ]
            if cand_embeddings_list:
                cand_concat = torch.cat(cand_embeddings_list, dim=-1)
                cand_tokens = self._mtgr_candidate_mlp(cand_concat)  # [total_candidates, d_model]
                tokens_list.append(cand_tokens)
                # Get max number of candidates
                first_cand_feature = (
                    self._hstu_configs.mtgr_cross_feature_names[0]
                    if self._hstu_configs.mtgr_cross_feature_names
                    else self._hstu_configs.mtgr_candidate_feature_names[0]
                )
                cand_lengths = cand_embeddings[first_cand_feature].lengths
                num_cand_tokens = int(torch.max(cand_lengths).item())

        # Concatenate all tokens
        all_tokens = torch.cat(tokens_list, dim=0) if tokens_list else torch.empty(0)

        # Create group indices
        total_tokens_per_sample = num_user_tokens + num_seq_tokens + num_cand_tokens
        group_indices = torch.cat([
            torch.zeros(num_user_tokens, dtype=torch.long),  # User group
            torch.ones(num_seq_tokens, dtype=torch.long),    # Sequence group
            torch.full((num_cand_tokens,), 2, dtype=torch.long),  # Candidate group
        ]) if total_tokens_per_sample > 0 else torch.empty(0, dtype=torch.long)

        return all_tokens, group_indices, num_user_tokens, num_seq_tokens, num_cand_tokens

    def preprocess(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        # MTGR mode: Split KJTs and do separate embedding lookups
        if self._hstu_configs.use_mtgr_tokenization:
            logger.info("Using MTGR tokenization mode in preprocess")

            # Split UIH features into user and sequence features
            user_kjt = self._split_mtgr_features(
                uih_features,
                self._hstu_configs.mtgr_user_feature_names
            )
            seq_kjt = self._split_mtgr_features(
                uih_features,
                self._hstu_configs.mtgr_seq_feature_names
            )

            # Candidates already contain cross + item features
            cand_kjt = candidates_features

            # Merge all for a single embedding lookup
            # (More efficient than 3 separate lookups)
            all_keys = []
            all_values = []
            all_lengths = []

            if user_kjt is not None:
                all_keys.extend(user_kjt.keys())
                all_values.append(user_kjt.values())
                all_lengths.append(user_kjt.lengths())

            if seq_kjt is not None:
                all_keys.extend(seq_kjt.keys())
                all_values.append(seq_kjt.values())
                all_lengths.append(seq_kjt.lengths())

            all_keys.extend(cand_kjt.keys())
            all_values.append(cand_kjt.values())
            all_lengths.append(cand_kjt.lengths())

            merged_sparse_features = KeyedJaggedTensor.from_lengths_sync(
                keys=all_keys,
                values=torch.cat(all_values, dim=0),
                lengths=torch.cat(all_lengths, dim=0),
            )
            seq_embeddings_dict = self._embedding_collection(merged_sparse_features)

            # Extract sequence lengths and candidates count
            # For MTGR, use first seq feature to get sequence length
            if self._hstu_configs.mtgr_seq_feature_names:
                first_seq_feature = self._hstu_configs.mtgr_seq_feature_names[0]
                uih_seq_lengths = seq_embeddings_dict[first_seq_feature].lengths()
            else:
                # Fallback to batch size if no seq features
                uih_seq_lengths = torch.ones(
                    uih_features.stride(), dtype=torch.int64, device=uih_features.values().device
                )

            # Get candidate counts
            first_cand_feature = (
                self._hstu_configs.mtgr_cross_feature_names[0]
                if self._hstu_configs.mtgr_cross_feature_names
                else self._hstu_configs.mtgr_candidate_feature_names[0]
            )
            num_candidates = seq_embeddings_dict[first_cand_feature].lengths()

        else:
            # Original HSTU path (unchanged)
            # embedding lookup for uih and candidates
            merged_sparse_features = KeyedJaggedTensor.from_lengths_sync(
                keys=uih_features.keys() + candidates_features.keys(),
                values=torch.cat(
                    [uih_features.values(), candidates_features.values()],
                    dim=0,
                ),
                lengths=torch.cat(
                    [uih_features.lengths(), candidates_features.lengths()],
                    dim=0,
                ),
            )
            seq_embeddings_dict = self._embedding_collection(merged_sparse_features)
            uih_seq_lengths = uih_features[
                self._hstu_configs.uih_post_id_feature_name
            ].lengths()
            num_candidates = fx_mark_length_features(
                candidates_features.lengths().view(len(candidates_features.keys()), -1)
            )[0]

        max_num_candidates = fx_infer_max_len(num_candidates)
        max_uih_len = fx_infer_max_len(uih_seq_lengths)

        # prepare payload features
        payload_features: Dict[str, torch.Tensor] = {}
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if (
                candidate_feature_name
                not in self._hstu_configs.item_embedding_feature_names
                and uih_feature_name
                not in self._hstu_configs.user_embedding_feature_names
            ):
                values_left = uih_features[uih_feature_name].values()
                if self._is_inference and (
                    candidate_feature_name
                    == self._hstu_configs.candidates_weight_feature_name
                    or candidate_feature_name
                    == self._hstu_configs.candidates_watchtime_feature_name
                ):
                    total_candidates = torch.sum(num_candidates).item()
                    values_right = torch.zeros(
                        total_candidates,  # pyre-ignore
                        dtype=torch.int64,
                        device=values_left.device,
                    )
                else:
                    values_right = candidates_features[candidate_feature_name].values()
                payload_features[uih_feature_name] = values_left
                payload_features[candidate_feature_name] = values_right
        payload_features["uih_offsets"] = torch.ops.fbgemm.asynchronous_complete_cumsum(
            uih_seq_lengths
        )
        payload_features["candidate_offsets"] = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(num_candidates)
        )

        # Collect embeddings based on mode
        if self._hstu_configs.use_mtgr_tokenization:
            # MTGR mode: collect user + seq + candidate embeddings
            feature_names = (
                self._hstu_configs.mtgr_user_feature_names
                + self._hstu_configs.mtgr_seq_feature_names
                + self._hstu_configs.mtgr_cross_feature_names
                + self._hstu_configs.mtgr_candidate_feature_names
            )
            seq_embeddings = {
                k: SequenceEmbedding(
                    lengths=seq_embeddings_dict[k].lengths(),
                    embedding=seq_embeddings_dict[k].values(),
                )
                for k in feature_names
                if k in seq_embeddings_dict
            }
        else:
            # HSTU mode: original logic
            seq_embeddings = {
                k: SequenceEmbedding(
                    lengths=seq_embeddings_dict[k].lengths(),
                    embedding=seq_embeddings_dict[k].values(),
                )
                for k in self._hstu_configs.user_embedding_feature_names
                + self._hstu_configs.item_embedding_feature_names
            }

        return (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )

    def main_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # MTGR mode: Use 3-token structure with separate MLPs
        if self._hstu_configs.use_mtgr_tokenization:
            logger.info("Using MTGR tokenization in main_forward")

            # Split embeddings into user, seq, and candidate groups
            user_embeddings = {
                k: v for k, v in seq_embeddings.items()
                if k in self._hstu_configs.mtgr_user_feature_names
            }
            seq_feature_embeddings = {
                k: v for k, v in seq_embeddings.items()
                if k in self._hstu_configs.mtgr_seq_feature_names
            }
            cand_embeddings = {
                k: v for k, v in seq_embeddings.items()
                if k in (self._hstu_configs.mtgr_cross_feature_names
                        + self._hstu_configs.mtgr_candidate_feature_names)
            }

            # Apply MTGR tokenization
            with record_function("## mtgr_tokenize ##"):
                all_tokens, _group_indices, _num_user, _num_seq, _num_cand = self._mtgr_tokenize(
                    user_embeddings=user_embeddings,
                    seq_embeddings=seq_feature_embeddings,
                    cand_embeddings=cand_embeddings,
                )

            # For now, treat tokenized sequence as regular embeddings for HSTU
            # TODO: Add MTGR-specific masking and Group LayerNorm later
            # TODO: Use _group_indices for Group LayerNorm
            # TODO: Use _num_user, _num_seq, _num_cand for custom masking
            logger.warning("MTGR tokenization applied, but using standard HSTU attention (no custom masking yet)")

            # Placeholder: just return the tokens as both user and item embeddings
            # Proper integration requires modifying _user_forward to accept tokens directly
            candidates_item_embeddings = all_tokens
            candidates_user_embeddings = all_tokens

            # Skip multitask for now in MTGR mode
            aux_losses: Dict[str, torch.Tensor] = {}

            return (
                candidates_user_embeddings,
                candidates_item_embeddings,
                aux_losses,
                None,  # mt_target_preds
                None,  # mt_target_labels
                None,  # mt_target_weights
            )

        # Original HSTU path (unchanged)
        # merge uih and candidates embeddings
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if uih_feature_name in seq_embeddings:
                seq_embeddings[uih_feature_name] = SequenceEmbedding(
                    lengths=uih_seq_lengths + num_candidates,
                    embedding=concat_2D_jagged(
                        max_seq_len=max_uih_len + max_num_candidates,
                        max_len_left=max_uih_len,
                        offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            uih_seq_lengths
                        ),
                        values_left=seq_embeddings[uih_feature_name].embedding,
                        max_len_right=max_num_candidates,
                        offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            num_candidates
                        ),
                        values_right=seq_embeddings[candidate_feature_name].embedding,
                        kernel=self.hammer_kernel(),
                    ),
                )

        with record_function("## item_forward ##"):
            candidates_item_embeddings = self._item_forward(
                seq_embeddings,
            )
        with record_function("## user_forward ##"):
            candidates_user_embeddings = self._user_forward(
                max_uih_len=max_uih_len,
                max_candidates=max_num_candidates,
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                num_candidates=num_candidates,
            )
        with record_function("## multitask_module ##"):
            supervision_labels, supervision_weights = (
                _get_supervision_labels_and_weights(
                    supervision_bitmasks=payload_features[
                        self._hstu_configs.candidates_weight_feature_name
                    ],
                    watchtime_sequence=payload_features[
                        self._hstu_configs.candidates_watchtime_feature_name
                    ],
                    task_configs=self._multitask_configs,
                )
            )
            mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
                self._multitask_module(
                    encoded_user_embeddings=candidates_user_embeddings,
                    item_embeddings=candidates_item_embeddings,
                    supervision_labels=supervision_labels,
                    supervision_weights=supervision_weights,
                )
            )

        aux_losses: Dict[str, torch.Tensor] = {}
        if not self._is_inference and self.training:
            for i, task in enumerate(self._multitask_configs):
                aux_losses[task.task_name] = mt_losses[i]

        return (
            candidates_user_embeddings,
            candidates_item_embeddings,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        )

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        with record_function("## preprocess ##"):
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.preprocess(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )

        with record_function("## main_forward ##"):
            return self.main_forward(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )
