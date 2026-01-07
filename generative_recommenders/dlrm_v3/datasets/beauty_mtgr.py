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

"""
MTGR-style dataset for Beauty dataset.

This dataset implements dynamic slicing, negative sampling, and cross-feature
computation to adapt the Beauty dataset for MTGR training.
"""

import logging
import pickle
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger = logging.getLogger(__name__)


class MTGRBeautyDataset(Dataset):
    """
    MTGR-style dataset for Beauty data with dynamic slicing and negative sampling.

    Features:
    - Dynamic slicing: Each epoch generates different training samples
    - Negative sampling: Random negatives sampled per batch
    - User features: user_id, avg_rating, avg_review_len
    - Sequence features: item_id, category, brand, price
    - Cross features: brand_count_in_history
    - Candidate features: item_id, category, brand, price

    Args:
        data_file: Path to the Beauty dataset pickle file (e.g., Beauty_train.pkl)
        item_features_file: Path to item metadata (Beauty_item_feature_map.pkl)
        num_negatives: Number of negative samples per positive sample (K-1)
        max_seq_len: Maximum sequence length for UIH
        is_inference: Whether this is for inference (no random slicing/negatives)
        hstu_config: Optional HSTU model configuration (for compatibility)

    Note:
        Beauty dataset already uses integer IDs (as strings) in the training data.
        No datamaps needed - we convert directly with int(id_string).
    """

    def __init__(
        self,
        data_file: str,
        item_features_file: str,
        num_negatives: int = 4,
        max_seq_len: int = 100,
        is_inference: bool = False,
        hstu_config: Optional[object] = None,
    ):
        super().__init__()

        self.hstu_config = hstu_config  # Store for compatibility, not used yet
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len
        self.is_inference = is_inference

        # Load data
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'rb') as f:
            self.users = pickle.load(f)

        logger.info(f"Loading item features from {item_features_file}")
        with open(item_features_file, 'rb') as f:
            self.item_features = pickle.load(f)

        # For negative sampling, use all items that have features
        # Item features keys are already integer IDs (as strings)
        self.all_item_ids = [int(item_id_str) for item_id_str in self.item_features.keys()]

        logger.info(f"Dataset initialized with {len(self.users)} users")
        logger.info(f"Total items with features: {len(self.all_item_ids)}")
        logger.info(f"Num negatives per positive: {self.num_negatives}")

        # Define feature keys for KJTs
        # UIH keys: user features + sequence features
        self._uih_keys = [
            "user_id",           # User profile features
            "avg_rating",
            "avg_review_len",
            "seq_item_id",       # Sequence item features
            "seq_category",
            "seq_brand",
            "seq_price",
        ]

        # Candidate keys: cross features + item features
        self._candidates_keys = [
            "brand_count",       # Cross features
            "cand_item_id",      # Candidate item features
            "cand_category",
            "cand_brand",
            "cand_price",
        ]

        logger.info(f"Dataset initialized with {len(self.users)} users")
        logger.info(f"Total items: {len(self.all_item_ids)}")
        logger.info(f"Num negatives per positive: {self.num_negatives}")

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        """
        Get a single sample with dynamic slicing and negative sampling.

        Returns:
            Tuple of (uih_features_kjt, candidates_features_kjt)
        """
        user = self.users[idx]
        user_seq = user['sequential_features']
        seq_len = len(user_seq['full_item_ids'])

        # Dynamic slicing: choose random slice point t
        # Try to find a valid position where the target item exists in item2id
        max_attempts = 10
        for attempt in range(max_attempts):
            if self.is_inference:
                # For inference, use the last item as target
                t = seq_len - 1
            else:
                # For training, random slice (ensure at least 1 history item)
                t = random.randint(1, seq_len - 1)

            # Check if target item has features
            target_item_str = user_seq['full_item_ids'][t]
            if target_item_str in self.item_features:
                break

            # If inference mode and item not found, try previous items
            if self.is_inference and t > 1:
                continue
        else:
            # If no valid item found after max_attempts, use a default fallback
            # Find first valid item in the sequence
            for i in range(1, seq_len):
                if user_seq['full_item_ids'][i] in self.item_features:
                    t = i
                    break
            else:
                # If still no valid item, skip this sample
                raise ValueError(f"User {idx} has no valid items in item_features")

        # Extract history [0:t]
        history = {
            'item_ids': user_seq['full_item_ids'][:t],
            'categories': user_seq['full_categories'][:t],
            'brands': user_seq['full_brands'][:t],
            'prices': user_seq['full_prices'][:t],
            'ratings': user_seq['full_ratings'][:t],
            'review_lens': user_seq['full_review_lens'][:t],
        }

        # Positive candidate (target at position t) - convert string to int
        pos_item_id = int(user_seq['full_item_ids'][t])

        # Sample negative candidates
        if not self.is_inference:
            # Include full history for negative sampling (up to t)
            neg_item_ids = self.sample_negatives(
                user_history=user_seq['full_item_ids'][:t+1],
                num_samples=self.num_negatives
            )
            candidate_ids = [pos_item_id] + neg_item_ids
        else:
            # For inference, only use positive
            candidate_ids = [pos_item_id]

        # Compute user features
        user_features = self.compute_user_features(history)

        # === Build UIH KJT (User + Sequence) ===
        uih_values = []
        uih_lengths = []

        # User ID (scalar) - convert string to int
        uih_values.append(int(user['user_id']))
        uih_lengths.append(1)

        # Avg rating (scalar, quantized to int)
        uih_values.append(int(user_features['avg_rating'] * 10))  # 4.5 -> 45
        uih_lengths.append(1)

        # Avg review length (scalar)
        uih_values.append(int(user_features['avg_review_len']))
        uih_lengths.append(1)

        # Sequence features (truncate if needed)
        actual_seq_len = min(len(history['item_ids']), self.max_seq_len)
        seq_item_ids = history['item_ids'][-actual_seq_len:] if actual_seq_len < len(history['item_ids']) else history['item_ids']
        seq_categories = history['categories'][-actual_seq_len:] if actual_seq_len < len(history['categories']) else history['categories']
        seq_brands = history['brands'][-actual_seq_len:] if actual_seq_len < len(history['brands']) else history['brands']
        seq_prices = history['prices'][-actual_seq_len:] if actual_seq_len < len(history['prices']) else history['prices']

        # Filter out items that don't have features and convert to integer IDs
        seq_item_ids_int = []
        seq_category_ids = []
        seq_brand_ids = []
        seq_price_ids = []

        for i, item_id in enumerate(seq_item_ids):
            if item_id in self.item_features:
                seq_item_ids_int.append(int(item_id))  # Convert string to int
                seq_category_ids.append(hash(seq_categories[i]) % 10000)
                seq_brand_ids.append(hash(seq_brands[i]) % 10000)
                seq_price_ids.append(int(seq_prices[i] * 100))  # Convert to cents

        # Update actual sequence length after filtering
        actual_seq_len = len(seq_item_ids_int)

        uih_values.extend(seq_item_ids_int)
        uih_lengths.append(actual_seq_len)

        uih_values.extend(seq_category_ids)
        uih_lengths.append(actual_seq_len)

        uih_values.extend(seq_brand_ids)
        uih_lengths.append(actual_seq_len)

        uih_values.extend(seq_price_ids)
        uih_lengths.append(actual_seq_len)

        # Create UIH KJT
        uih_features_kjt = KeyedJaggedTensor(
            keys=self._uih_keys,
            lengths=torch.tensor(uih_lengths, dtype=torch.int64),
            values=torch.tensor(uih_values, dtype=torch.int64),
        )

        # === Build Candidates KJT (Cross + Item features) ===
        num_candidates = len(candidate_ids)
        cand_lengths = []

        # Collect features for all candidates (grouped by feature type)
        brand_counts = []
        item_ids = []
        categories = []
        brands = []
        prices = []

        for cand_id in candidate_ids:
            # Cross features
            cross_features = self.compute_cross_features(history, cand_id)
            brand_counts.append(cross_features['brand_count'])

            # Item features (convert int ID to string for lookup)
            item_feats = self.item_features[str(cand_id)]
            item_ids.append(cand_id)
            categories.append(hash(item_feats['category']) % 10000)
            brands.append(hash(item_feats['brand']) % 10000)
            prices.append(int(item_feats['price'] * 100))

        # Concatenate all features (grouped by type, not interleaved!)
        cand_values = brand_counts + item_ids + categories + brands + prices

        # All features have num_candidates length
        for _ in range(len(self._candidates_keys)):
            cand_lengths.append(num_candidates)

        # Create Candidates KJT
        candidates_features_kjt = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=torch.tensor(cand_lengths, dtype=torch.int64),
            values=torch.tensor(cand_values, dtype=torch.int64),
        )

        return (uih_features_kjt, candidates_features_kjt)

    def sample_negatives(
        self,
        user_history: List[str],
        num_samples: int,
    ) -> List[int]:
        """
        Sample negative items not in user's history.

        Args:
            user_history: List of item IDs (as strings) the user has interacted with
            num_samples: Number of negatives to sample

        Returns:
            List of negative item IDs (as integers)
        """
        # Convert user history to set of integer IDs for fast lookup
        # Filter out items that don't have features
        user_history_ids = {int(item_id) for item_id in user_history if item_id in self.item_features}

        negatives = []
        max_attempts = num_samples * 10  # Prevent infinite loop
        attempts = 0

        while len(negatives) < num_samples and attempts < max_attempts:
            # Sample random item ID
            neg_id = random.choice(self.all_item_ids)

            # Check if it's not in user's history and not already sampled
            if neg_id not in user_history_ids and neg_id not in negatives:
                negatives.append(neg_id)

            attempts += 1

        if len(negatives) < num_samples:
            logger.warning(
                f"Could only sample {len(negatives)} negatives out of {num_samples} requested"
            )

        return negatives

    def compute_user_features(
        self,
        history: Dict[str, List],
    ) -> Dict[str, float]:
        """
        Compute aggregated user features from history.

        Args:
            history: Dictionary containing history sequences

        Returns:
            Dictionary with computed user features
        """
        ratings = history['ratings']
        review_lens = history['review_lens']

        return {
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0.0,
            'avg_review_len': sum(review_lens) / len(review_lens) if review_lens else 0.0,
        }

    def compute_cross_features(
        self,
        history: Dict[str, List],
        candidate_item_id: int,
    ) -> Dict[str, int]:
        """
        Compute cross features between candidate and user history.

        Args:
            history: Dictionary containing history sequences
            candidate_item_id: Item ID of the candidate

        Returns:
            Dictionary with computed cross features
        """
        # Get candidate's brand (convert int ID to string for lookup)
        candidate_brand = self.item_features[str(candidate_item_id)]['brand']

        # Count occurrences of candidate's brand in user history
        brand_count = sum(1 for brand in history['brands'] if brand == candidate_brand)

        return {
            'brand_count': brand_count,
        }
