"""Test script for MTGRBeautyDataset"""

import sys
import logging
from generative_recommenders.dlrm_v3.datasets.beauty_mtgr import MTGRBeautyDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_basic():
    """Test basic dataset functionality"""
    logger.info("=" * 60)
    logger.info("Testing MTGRBeautyDataset - Basic Functionality")
    logger.info("=" * 60)

    # Create dataset
    dataset = MTGRBeautyDataset(
        data_file="data/Beauty_train.pkl",
        datamaps_file="data/Beauty_datamaps.pkl",
        item_features_file="data/Beauty_item_feature_map.pkl",
        num_negatives=4,
        max_seq_len=100,
        is_inference=False,
    )

    logger.info(f"\n✓ Dataset created successfully")
    logger.info(f"  - Total users: {len(dataset)}")
    logger.info(f"  - Total items: {len(dataset.all_item_ids)}")
    logger.info(f"  - UIH keys: {dataset._uih_keys}")
    logger.info(f"  - Candidates keys: {dataset._candidates_keys}")

    # Test __getitem__
    logger.info("\n" + "=" * 60)
    logger.info("Testing __getitem__ method")
    logger.info("=" * 60)

    try:
        uih_kjt, cand_kjt = dataset[0]

        logger.info(f"\n✓ Sample 0 loaded successfully")
        logger.info(f"\nUIH KJT:")
        logger.info(f"  - Keys: {uih_kjt.keys()}")
        logger.info(f"  - Lengths: {uih_kjt.lengths()}")
        logger.info(f"  - Values shape: {uih_kjt.values().shape}")
        logger.info(f"  - Stride (batch size): {uih_kjt.stride()}")

        logger.info(f"\nCandidates KJT:")
        logger.info(f"  - Keys: {cand_kjt.keys()}")
        logger.info(f"  - Lengths: {cand_kjt.lengths()}")
        logger.info(f"  - Values shape: {cand_kjt.values().shape}")
        logger.info(f"  - Stride (batch size): {cand_kjt.stride()}")

        # Check expected number of candidates
        expected_num_cands = 1 + dataset.num_negatives  # 1 positive + K negatives
        actual_num_cands = cand_kjt.lengths()[0].item()
        assert actual_num_cands == expected_num_cands, f"Expected {expected_num_cands} candidates, got {actual_num_cands}"
        logger.info(f"\n✓ Correct number of candidates: {actual_num_cands} (1 pos + {dataset.num_negatives} neg)")

    except Exception as e:
        logger.error(f"\n✗ Error loading sample: {e}")
        raise

    # Test multiple samples
    logger.info("\n" + "=" * 60)
    logger.info("Testing multiple samples")
    logger.info("=" * 60)

    try:
        for i in range(3):
            uih_kjt, cand_kjt = dataset[i]
            logger.info(f"  Sample {i}: UIH values={uih_kjt.values().shape[0]}, Cand values={cand_kjt.values().shape[0]}")

        logger.info(f"\n✓ Multiple samples loaded successfully")

    except Exception as e:
        logger.error(f"\n✗ Error loading multiple samples: {e}")
        raise

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed! ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_dataset_basic()
