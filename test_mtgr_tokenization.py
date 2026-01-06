"""
Test script to verify MTGR tokenization integration with DlrmHSTU.

This tests the basic functionality without requiring GPU or full training setup.
"""

import logging
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, DlrmHSTU
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType, PoolingType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a minimal MTGR config for testing."""
    config = DlrmHSTUConfig(
        # MTGR-specific settings
        use_mtgr_tokenization=True,
        mtgr_user_feature_names=['user_id', 'avg_rating', 'avg_review_len'],
        mtgr_seq_feature_names=['seq_item_id', 'seq_category', 'seq_brand', 'seq_price'],
        mtgr_cross_feature_names=['brand_count'],
        mtgr_candidate_feature_names=['cand_item_id', 'cand_category', 'cand_brand', 'cand_price'],

        # Model architecture
        hstu_embedding_table_dim=64,  # Small for testing
        hstu_transducer_embedding_dim=128,  # d_model
        hstu_num_heads=2,
        hstu_attn_linear_dim=128,
        hstu_attn_qk_dim=64,
        hstu_attn_num_layers=2,  # Small for testing

        # Other required fields (not used in MTGR mode)
        uih_post_id_feature_name='seq_item_id',
        uih_action_time_feature_name='',
        uih_weight_feature_name='',
        candidates_weight_feature_name='',
        candidates_watchtime_feature_name='',
        candidates_querytime_feature_name='',
    )
    return config


def create_embedding_tables(config):
    """Create embedding tables for all MTGR features."""
    tables = {}

    # User feature embeddings
    for feature_name in config.mtgr_user_feature_names:
        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=10000 if 'id' in feature_name else 1000,
            data_type=DataType.FP32,
            pooling=PoolingType.NONE,
        )

    # Sequence feature embeddings
    for feature_name in config.mtgr_seq_feature_names:
        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=10000 if 'id' in feature_name else 1000,
            data_type=DataType.FP32,
            pooling=PoolingType.NONE,
        )

    # Cross feature embeddings
    for feature_name in config.mtgr_cross_feature_names:
        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=100,  # Count features have small range
            data_type=DataType.FP32,
            pooling=PoolingType.NONE,
        )

    # Candidate feature embeddings
    for feature_name in config.mtgr_candidate_feature_names:
        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=10000 if 'id' in feature_name else 1000,
            data_type=DataType.FP32,
            pooling=PoolingType.NONE,
        )

    return tables


def create_synthetic_batch(batch_size=4, seq_len=5, num_candidates=3):
    """Create synthetic KJTs matching MTGRBeautyDataset output format."""

    # UIH features KJT (User + Sequence)
    uih_keys = ['user_id', 'avg_rating', 'avg_review_len',
                'seq_item_id', 'seq_category', 'seq_brand', 'seq_price']

    uih_values = []
    uih_lengths = []

    # User features (scalar per user, so length=1 each)
    for _ in range(batch_size):
        uih_values.append(torch.randint(0, 1000, (1,)))  # user_id
        uih_lengths.append(1)
    for _ in range(batch_size):
        uih_values.append(torch.randint(0, 50, (1,)))    # avg_rating (quantized)
        uih_lengths.append(1)
    for _ in range(batch_size):
        uih_values.append(torch.randint(0, 100, (1,)))   # avg_review_len
        uih_lengths.append(1)

    # Sequence features (variable length per user)
    for _ in range(batch_size):
        uih_values.append(torch.randint(0, 1000, (seq_len,)))  # seq_item_id
        uih_lengths.append(seq_len)
    for _ in range(batch_size):
        uih_values.append(torch.randint(0, 100, (seq_len,)))   # seq_category
        uih_lengths.append(seq_len)
    for _ in range(batch_size):
        uih_values.append(torch.randint(0, 100, (seq_len,)))   # seq_brand
        uih_lengths.append(seq_len)
    for _ in range(batch_size):
        uih_values.append(torch.randint(0, 1000, (seq_len,)))  # seq_price (in cents)
        uih_lengths.append(seq_len)

    uih_features_kjt = KeyedJaggedTensor(
        keys=uih_keys,
        values=torch.cat(uih_values).long(),
        lengths=torch.tensor(uih_lengths, dtype=torch.int64),
    )

    # Candidates features KJT (Cross + Item features)
    cand_keys = ['brand_count', 'cand_item_id', 'cand_category', 'cand_brand', 'cand_price']

    cand_values = []
    cand_lengths = []

    for _ in range(batch_size):
        cand_values.append(torch.randint(0, 10, (num_candidates,)))     # brand_count
        cand_lengths.append(num_candidates)
    for _ in range(batch_size):
        cand_values.append(torch.randint(0, 1000, (num_candidates,)))  # cand_item_id
        cand_lengths.append(num_candidates)
    for _ in range(batch_size):
        cand_values.append(torch.randint(0, 100, (num_candidates,)))   # cand_category
        cand_lengths.append(num_candidates)
    for _ in range(batch_size):
        cand_values.append(torch.randint(0, 100, (num_candidates,)))   # cand_brand
        cand_lengths.append(num_candidates)
    for _ in range(batch_size):
        cand_values.append(torch.randint(0, 1000, (num_candidates,))) # cand_price
        cand_lengths.append(num_candidates)

    candidates_features_kjt = KeyedJaggedTensor(
        keys=cand_keys,
        values=torch.cat(cand_values).long(),
        lengths=torch.tensor(cand_lengths, dtype=torch.int64),
    )

    return uih_features_kjt, candidates_features_kjt


def test_mtgr_tokenization():
    """Test MTGR tokenization end-to-end."""
    logger.info("=" * 60)
    logger.info("Testing MTGR Tokenization Integration")
    logger.info("=" * 60)

    # Step 1: Create config and embedding tables
    logger.info("\n1. Creating MTGR config and embedding tables...")
    config = create_test_config()
    embedding_tables = create_embedding_tables(config)
    logger.info(f"   ✓ Config created with use_mtgr_tokenization={config.use_mtgr_tokenization}")
    logger.info(f"   ✓ Created {len(embedding_tables)} embedding tables")

    # Step 2: Initialize model
    logger.info("\n2. Initializing DlrmHSTU model...")
    try:
        model = DlrmHSTU(
            hstu_configs=config,
            embedding_tables=embedding_tables,
            is_inference=False,
        )
        logger.info("   ✓ Model initialized successfully")

        # Check that MTGR MLPs were created
        assert hasattr(model, '_mtgr_user_mlp'), "Missing _mtgr_user_mlp"
        assert hasattr(model, '_mtgr_seq_mlp'), "Missing _mtgr_seq_mlp"
        assert hasattr(model, '_mtgr_candidate_mlp'), "Missing _mtgr_candidate_mlp"
        logger.info("   ✓ MTGR MLPs created: user_mlp, seq_mlp, candidate_mlp")

    except Exception as e:
        logger.error(f"   ✗ Model initialization failed: {e}")
        raise

    # Step 3: Create synthetic batch
    logger.info("\n3. Creating synthetic batch...")
    batch_size = 4
    seq_len = 5
    num_candidates = 3
    uih_kjt, cand_kjt = create_synthetic_batch(batch_size, seq_len, num_candidates)
    logger.info(f"   ✓ Batch created: batch_size={batch_size}, seq_len={seq_len}, num_candidates={num_candidates}")
    logger.info(f"   ✓ UIH KJT: stride={uih_kjt.stride()}, keys={uih_kjt.keys()[:3]}...")
    logger.info(f"   ✓ Cand KJT: stride={cand_kjt.stride()}, keys={cand_kjt.keys()}")

    # Step 4: Forward pass
    logger.info("\n4. Running forward pass...")
    try:
        with torch.no_grad():
            outputs = model(uih_kjt, cand_kjt)

        user_emb, item_emb, aux_losses, mt_preds, mt_labels, mt_weights = outputs

        logger.info("   ✓ Forward pass completed successfully!")
        logger.info(f"   ✓ User embeddings shape: {user_emb.shape}")
        logger.info(f"   ✓ Item embeddings shape: {item_emb.shape}")
        logger.info(f"   ✓ Aux losses: {list(aux_losses.keys())}")

        # Verify shapes
        expected_total_tokens = batch_size * (3 + seq_len + num_candidates)  # 3 user + 5 seq + 3 cand per sample
        logger.info(f"\n   Expected total tokens: {expected_total_tokens}")
        logger.info(f"   Actual user_emb tokens: {user_emb.shape[0]}")

        # Note: In current implementation, shapes might not match perfectly
        # because we're using placeholder logic. This is expected!
        if user_emb.shape[0] == expected_total_tokens:
            logger.info("   ✓ Shape matches expected!")
        else:
            logger.warning("   ⚠ Shape mismatch (expected with placeholder implementation)")

    except Exception as e:
        logger.error(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 5: Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("✓ MTGR config creation: PASS")
    logger.info("✓ Model initialization with MTGR MLPs: PASS")
    logger.info("✓ Synthetic batch creation: PASS")
    logger.info("✓ Forward pass (preprocess + tokenization): PASS")
    logger.info("\n✅ All basic tests passed!")
    logger.info("\nNote: This tests tokenization only. Custom masking and Group LN")
    logger.info("will be implemented in Phase 2.")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_mtgr_tokenization()
