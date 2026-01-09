"""
Test MTGR masking strategy (simplified version without RT).
"""

import torch
from generative_recommenders.modules.mtgr_mask import create_mtgr_mask, visualize_mask


def test_mtgr_mask_simple():
    """Test MTGR mask with a simple example."""
    print("=" * 80)
    print("TEST: MTGR Mask (Simplified - No RT)")
    print("=" * 80)

    # Simple example: 2 user, 3 seq, 2 candidates, batch size 1
    num_user = 2
    num_seq = 3
    num_cand = 2
    batch_size = 1

    mask = create_mtgr_mask(num_user, num_seq, num_cand, batch_size, 'cpu')

    print(f"\nMask shape: {mask.shape}")
    print(f"Expected: [{batch_size * (num_user + num_seq + num_cand)}, "
          f"{batch_size * (num_user + num_seq + num_cand)}] = [7, 7]")

    # Visualize
    print(visualize_mask(mask, num_user, num_seq, num_cand, batch_idx=0))

    # Verify specific positions
    print("Verification:")
    print("-" * 80)

    # User tokens should see everything
    assert mask[0, :].eq(0).all(), "User token 0 should see all tokens"
    assert mask[1, :].eq(0).all(), "User token 1 should see all tokens"
    print("✓ User tokens can see all tokens")

    # Sequence tokens should see everything
    assert mask[2, :].eq(0).all(), "Seq token 0 should see all tokens"
    assert mask[3, :].eq(0).all(), "Seq token 1 should see all tokens"
    assert mask[4, :].eq(0).all(), "Seq token 2 should see all tokens"
    print("✓ Sequence tokens can see all tokens")

    # Candidate tokens - check diagonal masking
    # Candidate 0 (index 5):
    assert mask[5, 0:5].eq(0).all(), "Cand 0 should see User+Seq"
    assert mask[5, 5] == 0, "Cand 0 should see itself"
    assert mask[5, 6] == float('-inf'), "Cand 0 should NOT see Cand 1"
    print("✓ Candidate 0: sees User+Seq+Self, NOT other candidates")

    # Candidate 1 (index 6):
    assert mask[6, 0:5].eq(0).all(), "Cand 1 should see User+Seq"
    assert mask[6, 5] == float('-inf'), "Cand 1 should NOT see Cand 0"
    assert mask[6, 6] == 0, "Cand 1 should see itself"
    print("✓ Candidate 1: sees User+Seq+Self, NOT other candidates")

    print("\n" + "=" * 80)
    print("✅ MTGR MASK TEST PASSED!")
    print("=" * 80)


def test_mtgr_mask_batch():
    """Test MTGR mask with batch size > 1."""
    print("\n" + "=" * 80)
    print("TEST: MTGR Mask with Batch Size = 2")
    print("=" * 80)

    num_user = 2
    num_seq = 2
    num_cand = 3
    batch_size = 2

    mask = create_mtgr_mask(num_user, num_seq, num_cand, batch_size, 'cpu')

    print(f"\nMask shape: {mask.shape}")
    total_tokens = batch_size * (num_user + num_seq + num_cand)
    print(f"Expected: [{total_tokens}, {total_tokens}] = [14, 14]")

    # Sample 0 tokens: indices 0-6
    # Sample 1 tokens: indices 7-13

    # Verify candidates in sample 0 don't see each other
    sample0_cand_start = 0 + num_user + num_seq  # 4
    assert mask[4, 5] == float('-inf'), "Sample 0: Cand 0 should not see Cand 1"
    assert mask[4, 6] == float('-inf'), "Sample 0: Cand 0 should not see Cand 2"
    assert mask[5, 4] == float('-inf'), "Sample 0: Cand 1 should not see Cand 0"
    print("✓ Sample 0: Candidates properly masked")

    # Verify candidates in sample 1 don't see each other
    sample1_cand_start = 7 + num_user + num_seq  # 11
    assert mask[11, 12] == float('-inf'), "Sample 1: Cand 0 should not see Cand 1"
    assert mask[11, 13] == float('-inf'), "Sample 1: Cand 0 should not see Cand 2"
    assert mask[12, 11] == float('-inf'), "Sample 1: Cand 1 should not see Cand 0"
    print("✓ Sample 1: Candidates properly masked")

    # Verify samples don't interfere with each other
    # Sample 0 tokens should not be masked by Sample 1's rules
    assert mask[0, 0:7].eq(0).all(), "Sample 0 User token sees all Sample 0 tokens"
    assert mask[7, 7:14].eq(0).all(), "Sample 1 User token sees all Sample 1 tokens"
    print("✓ Samples are independent (no cross-sample masking)")

    print("\n" + "=" * 80)
    print("✅ BATCH MTGR MASK TEST PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    torch.manual_seed(42)
    test_mtgr_mask_simple()
    test_mtgr_mask_batch()
    print("\n" + "=" * 80)
    print("🎉 ALL MTGR MASK TESTS PASSED!")
    print("=" * 80)
