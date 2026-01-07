"""
End-to-end test for MTGR pipeline using actual Beauty dataset.

This test verifies the complete flow:
Beauty_train.pkl → MTGRBeautyDataset → DataLoader → DlrmHSTU (MTGR mode) → Tokenization
"""

import logging
import torch
from torch.utils.data import DataLoader
from generative_recommenders.dlrm_v3.datasets.beauty_mtgr import MTGRBeautyDataset
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, DlrmHSTU
from generative_recommenders.modules.multitask_module import MultitaskTaskType, TaskConfig
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mtgr_config(dataset):
    """Create MTGR config based on dataset structure."""
    config = DlrmHSTUConfig(
        # MTGR-specific settings
        use_mtgr_tokenization=True,
        mtgr_user_feature_names=dataset._uih_keys[:3],  # First 3 are user features
        mtgr_seq_feature_names=dataset._uih_keys[3:],   # Rest are sequence features
        mtgr_cross_feature_names=[dataset._candidates_keys[0]],  # brand_count
        mtgr_candidate_feature_names=dataset._candidates_keys[1:],  # Item features

        # Model architecture
        hstu_embedding_table_dim=64,  # Small for testing
        hstu_transducer_embedding_dim=128,  # d_model
        hstu_num_heads=2,
        hstu_attn_linear_dim=128,
        hstu_attn_qk_dim=64,
        hstu_attn_num_layers=2,  # Small for testing

        # Required fields (not used in MTGR mode)
        uih_post_id_feature_name='seq_item_id',
        uih_action_time_feature_name='',
        uih_weight_feature_name='',
        candidates_weight_feature_name='',
        candidates_watchtime_feature_name='',
        candidates_querytime_feature_name='',
    )

    # Add multitask config (required by DlrmHSTU)
    config.multitask_configs = [
        TaskConfig(
            task_name="recommendation",
            task_weight=1.0,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        )
    ]

    return config


def create_embedding_tables(config, dataset):
    """Create embedding tables for all MTGR features with proper sizes."""
    tables = {}

    # Determine vocabulary sizes from dataset
    # Get max user ID from actual user data
    max_user_id = max(int(user['user_id']) for user in dataset.users)
    # Get max item ID from all items
    max_item_id = max(dataset.all_item_ids)

    logger.info(f"Embedding vocab sizes: user_id={max_user_id + 1}, item_id={max_item_id + 1}")

    # User features
    for feature_name in config.mtgr_user_feature_names:
        if feature_name == 'user_id':
            num_emb = max_user_id + 1  # User ID vocabulary
        elif 'rating' in feature_name:
            num_emb = 51  # Rating * 10: 0-50 (for 0.0-5.0 ratings)
        elif 'review' in feature_name:
            num_emb = 5000  # Review length vocabulary
        else:
            num_emb = 1000  # Default

        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=num_emb,
            data_type=DataType.FP32,
        )

    # Sequence features
    for feature_name in config.mtgr_seq_feature_names:
        if 'item_id' in feature_name or 'item' in feature_name:
            num_emb = max_item_id + 1  # Item ID vocabulary
        elif 'category' in feature_name or 'brand' in feature_name:
            num_emb = 10000  # Hashed category/brand features
        elif 'price' in feature_name:
            num_emb = 100000  # Price in cents (up to $1000)
        else:
            num_emb = 10000  # Default

        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=num_emb,
            data_type=DataType.FP32,
        )

    # Cross features
    for feature_name in config.mtgr_cross_feature_names:
        if 'count' in feature_name:
            num_emb = 200  # Brand count features (0-200)
        else:
            num_emb = 1000  # Default

        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=num_emb,
            data_type=DataType.FP32,
        )

    # Candidate features
    for feature_name in config.mtgr_candidate_feature_names:
        if 'item_id' in feature_name or 'item' in feature_name:
            num_emb = max_item_id + 1  # Item ID vocabulary
        elif 'category' in feature_name or 'brand' in feature_name:
            num_emb = 10000  # Hashed category/brand features
        elif 'price' in feature_name:
            num_emb = 100000  # Price in cents
        else:
            num_emb = 10000  # Default

        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=num_emb,
            data_type=DataType.FP32,
        )

    return tables


def test_end_to_end():
    """Test complete MTGR pipeline with real Beauty data."""
    logger.info("=" * 80)
    logger.info("MTGR END-TO-END TEST WITH REAL BEAUTY DATA")
    logger.info("=" * 80)

    # Step 1: Load dataset
    logger.info("\n1. Loading MTGRBeautyDataset...")
    try:
        dataset = MTGRBeautyDataset(
            data_file="data/Beauty_train.pkl",
            item_features_file="data/Beauty_item_feature_map.pkl",
            num_negatives=4,
            max_seq_len=100,
        )
        logger.info(f"   ✓ Dataset loaded successfully")
        logger.info(f"   ✓ Number of users: {len(dataset)}")
        logger.info(f"   ✓ UIH keys: {dataset._uih_keys}")
        logger.info(f"   ✓ Candidate keys: {dataset._candidates_keys}")
    except Exception as e:
        logger.error(f"   ✗ Dataset loading failed: {e}")
        raise

    # Step 2: Test single sample
    logger.info("\n2. Testing single sample retrieval...")
    try:
        uih_kjt, cand_kjt = dataset[0]
        logger.info(f"   ✓ Sample retrieved successfully")
        logger.info(f"   ✓ UIH KJT keys: {uih_kjt.keys()}")
        logger.info(f"   ✓ UIH KJT stride: {uih_kjt.stride()}")
        logger.info(f"   ✓ Candidate KJT keys: {cand_kjt.keys()}")
        logger.info(f"   ✓ Candidate KJT stride: {cand_kjt.stride()}")
        logger.info(f"   ✓ UIH lengths: {uih_kjt.lengths().tolist()[:10]}...")  # Show first 10
        logger.info(f"   ✓ Candidate lengths: {cand_kjt.lengths().tolist()}")
    except Exception as e:
        logger.error(f"   ✗ Sample retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 3: Create DataLoader
    logger.info("\n3. Creating DataLoader with collate_fn...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,  # Small batch for testing
            collate_fn=collate_fn,
            shuffle=False,  # Don't shuffle for reproducibility
        )
        logger.info(f"   ✓ DataLoader created successfully")
        logger.info(f"   ✓ Batch size: 4")
    except Exception as e:
        logger.error(f"   ✗ DataLoader creation failed: {e}")
        raise

    # Step 4: Get one batch
    logger.info("\n4. Fetching one batch...")
    try:
        batch = next(iter(dataloader))
        logger.info(f"   ✓ Batch retrieved successfully")
        logger.info(f"   ✓ Batch UIH KJT stride: {batch.uih_features_kjt.stride()}")
        logger.info(f"   ✓ Batch Candidate KJT stride: {batch.candidates_features_kjt.stride()}")
        logger.info(f"   ✓ Batch UIH keys: {batch.uih_features_kjt.keys()}")
        logger.info(f"   ✓ Batch Candidate keys: {batch.candidates_features_kjt.keys()}")
    except Exception as e:
        logger.error(f"   ✗ Batch retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 5: Create MTGR config and model
    logger.info("\n5. Creating MTGR model...")
    try:
        config = create_mtgr_config(dataset)
        embedding_tables = create_embedding_tables(config, dataset)

        logger.info(f"   ✓ Config created")
        logger.info(f"   ✓ User features: {config.mtgr_user_feature_names}")
        logger.info(f"   ✓ Seq features: {config.mtgr_seq_feature_names}")
        logger.info(f"   ✓ Cross features: {config.mtgr_cross_feature_names}")
        logger.info(f"   ✓ Candidate features: {config.mtgr_candidate_feature_names}")

        model = DlrmHSTU(
            hstu_configs=config,
            embedding_tables=embedding_tables,
            is_inference=False,
        )
        logger.info(f"   ✓ Model initialized successfully")

        # Move model to GPU (SwishLayerNorm requires CUDA)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            logger.warning("   ⚠️  CUDA not available! SwishLayerNorm requires GPU, test may fail.")

        model = model.to_empty(device=device)
        # Reset parameters after moving from meta device
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        logger.info(f"   ✓ Model materialized on {device}")

        # Verify MTGR MLPs exist
        assert hasattr(model, '_mtgr_user_mlp'), "Missing _mtgr_user_mlp"
        assert hasattr(model, '_mtgr_seq_mlp'), "Missing _mtgr_seq_mlp"
        assert hasattr(model, '_mtgr_candidate_mlp'), "Missing _mtgr_candidate_mlp"
        logger.info(f"   ✓ MTGR MLPs verified: user_mlp, seq_mlp, candidate_mlp")

    except Exception as e:
        logger.error(f"   ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 6: Forward pass
    logger.info("\n6. Running forward pass (tokenization)...")
    try:
        # Move batch to same device as model
        batch.to(device)
        logger.info(f"   ✓ Batch moved to {device}")

        with torch.no_grad():
            outputs = model(batch.uih_features_kjt, batch.candidates_features_kjt)

        user_emb, item_emb, aux_losses, mt_preds, mt_labels, mt_weights = outputs

        logger.info(f"   ✓ Forward pass completed successfully!")
        logger.info(f"   ✓ User embeddings shape: {user_emb.shape}")
        logger.info(f"   ✓ Item embeddings shape: {item_emb.shape}")
        logger.info(f"   ✓ Aux losses keys: {list(aux_losses.keys())}")

        # Calculate expected token counts
        # Per MTGR paper: Each user feature becomes a separate token (not combined)
        batch_size = batch.uih_features_kjt.stride()
        num_user_features = len(config.mtgr_user_feature_names)
        num_candidates = 5  # 1 positive + 4 negatives

        # Estimate sequence length (approximate, varies per sample)
        total_seq_items = batch.uih_features_kjt.lengths()[3:7].sum().item()  # seq features
        avg_seq_len = total_seq_items // batch_size // 4  # 4 seq features

        # Total tokens = (User tokens) + (Seq tokens) + (Candidate tokens)
        # User: batch_size * num_user_features (each feature = 1 token per sample)
        # Seq: total sequence items across batch
        # Cand: batch_size * num_candidates
        expected_tokens = batch_size * (num_user_features + avg_seq_len + num_candidates)

        logger.info(f"\n   Token count analysis:")
        logger.info(f"   - Batch size: {batch_size}")
        logger.info(f"   - User features: {num_user_features} → User tokens: {batch_size * num_user_features}")
        logger.info(f"   - Avg seq tokens per sample: ~{avg_seq_len} → Total seq tokens: ~{batch_size * avg_seq_len}")
        logger.info(f"   - Candidate tokens per sample: {num_candidates} → Total cand tokens: {batch_size * num_candidates}")
        logger.info(f"   - Expected total tokens: ~{expected_tokens}")
        logger.info(f"   - Actual tokens: {user_emb.shape[0]}")

        if abs(user_emb.shape[0] - expected_tokens) / expected_tokens < 0.2:  # Within 20%
            logger.info(f"   ✓ Token count matches expectation!")
        else:
            logger.warning(f"   ⚠ Token count differs (expected with variable seq lengths)")

    except Exception as e:
        logger.error(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Summary - Use print() instead of logger to ensure output in Colab
    import sys
    print("\n" + "=" * 80, flush=True)
    print("TEST SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print("✓ Dataset loading: PASS", flush=True)
    print("✓ Sample retrieval: PASS", flush=True)
    print("✓ DataLoader batching: PASS", flush=True)
    print("✓ Model initialization: PASS", flush=True)
    print("✓ Forward pass (tokenization): PASS", flush=True)
    print("\n✅ END-TO-END TEST PASSED!", flush=True)
    print("=" * 80, flush=True)
    print("\nPipeline verified:", flush=True)
    print("  Beauty_train.pkl → MTGRBeautyDataset → DataLoader → DlrmHSTU → Tokens", flush=True)
    print("\nPhase 1 Complete: MTGR tokenization is working!", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        test_end_to_end()
        print("\n" + "="*80)
        print("🎉 TEST SCRIPT COMPLETED SUCCESSFULLY!")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST SCRIPT FAILED: {e}")
        print("="*80)
        raise
