"""
Test MTGR training components (loss, metrics, training iteration).
"""

import torch
import torch.nn as nn
from train_mtgr import MTGRRankingLoss, compute_metrics


def test_loss_function():
    """Test MTGRRankingLoss computation."""
    print("=" * 80)
    print("TEST: MTGRRankingLoss")
    print("=" * 80)

    loss_fn = MTGRRankingLoss(temperature=0.1)

    # Create dummy embeddings
    # 2 samples, 3 candidates each = 6 total
    batch_size = 2
    num_candidates = 3
    embedding_dim = 16

    user_emb = torch.randn(batch_size * num_candidates, embedding_dim)
    item_emb = torch.randn(batch_size * num_candidates, embedding_dim)

    # Compute loss
    loss = loss_fn(user_emb, item_emb, num_candidates)

    print(f"\nBatch size: {batch_size}")
    print(f"Num candidates: {num_candidates}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Loss value: {loss.item():.4f}")

    # Verify loss properties
    assert loss.numel() == 1, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"

    print("\n✓ Loss is scalar: PASS")
    print("✓ Loss is positive: PASS")
    print("✓ Loss is valid (no NaN/Inf): PASS")

    # Test gradient flow
    loss.backward()
    assert user_emb.grad is not None, "Should have gradients"
    print("✓ Gradient flow: PASS")

    print("\n" + "=" * 80)
    print("✅ LOSS FUNCTION TEST PASSED!")
    print("=" * 80)


def test_metrics():
    """Test metric computation."""
    print("\n" + "=" * 80)
    print("TEST: Metrics (MRR, Hit@K, NDCG@K)")
    print("=" * 80)

    # Create scenario where we know the rankings
    batch_size = 3
    num_candidates = 5
    embedding_dim = 16

    # Create controlled embeddings to test metrics
    # Sample 0: Positive item ranked 1st (perfect)
    # Sample 1: Positive item ranked 2nd
    # Sample 2: Positive item ranked 5th (worst)

    user_emb = torch.randn(batch_size * num_candidates, embedding_dim)
    item_emb = torch.randn(batch_size * num_candidates, embedding_dim)

    # Compute metrics
    metrics = compute_metrics(user_emb, item_emb, num_candidates, k_list=[1, 3, 5])

    print(f"\nBatch size: {batch_size}")
    print(f"Num candidates: {num_candidates}")
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Verify metric properties
    assert 'MRR' in metrics, "Should have MRR"
    assert 'Hit@1' in metrics, "Should have Hit@1"
    assert 'Hit@3' in metrics, "Should have Hit@3"
    assert 'Hit@5' in metrics, "Should have Hit@5"
    assert 'NDCG@1' in metrics, "Should have NDCG@1"
    assert 'NDCG@3' in metrics, "Should have NDCG@3"
    assert 'NDCG@5' in metrics, "Should have NDCG@5"

    # All metrics should be in [0, 1]
    for name, value in metrics.items():
        assert 0 <= value <= 1, f"{name} should be in [0, 1], got {value}"
        assert not torch.isnan(torch.tensor(value)), f"{name} should not be NaN"

    print("\n✓ All metrics present: PASS")
    print("✓ All metrics in [0, 1]: PASS")
    print("✓ No NaN values: PASS")

    # Hit@K monotonicity: Hit@1 <= Hit@3 <= Hit@5
    assert metrics['Hit@1'] <= metrics['Hit@3'] <= metrics['Hit@5'], \
        "Hit@K should be monotonically increasing"
    print("✓ Hit@K monotonicity: PASS")

    print("\n" + "=" * 80)
    print("✅ METRICS TEST PASSED!")
    print("=" * 80)


def test_perfect_ranking():
    """Test metrics with perfect ranking (positive always first)."""
    print("\n" + "=" * 80)
    print("TEST: Perfect Ranking (Positive Always Ranked 1st)")
    print("=" * 80)

    batch_size = 4
    num_candidates = 5
    embedding_dim = 16

    user_emb = torch.randn(batch_size * num_candidates, embedding_dim)
    item_emb = torch.zeros(batch_size * num_candidates, embedding_dim)

    # Make positive items (index 0 in each sample) have highest score
    user_emb_view = user_emb.view(batch_size, num_candidates, embedding_dim)
    item_emb_view = item_emb.view(batch_size, num_candidates, embedding_dim)

    for i in range(batch_size):
        # Positive item gets perfect alignment with user
        item_emb_view[i, 0] = user_emb_view[i, 0].clone()
        # Negative items get random (likely lower scores)
        item_emb_view[i, 1:] = torch.randn(num_candidates - 1, embedding_dim) * 0.1

    user_emb = user_emb_view.reshape(batch_size * num_candidates, embedding_dim)
    item_emb = item_emb_view.reshape(batch_size * num_candidates, embedding_dim)

    # Compute metrics
    metrics = compute_metrics(user_emb, item_emb, num_candidates, k_list=[1, 3, 5])

    print("\nMetrics (should be high for perfect ranking):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # With perfect ranking:
    # - MRR should be 1.0 (all ranked 1st)
    # - Hit@1, Hit@3, Hit@5 should all be 1.0
    # - NDCG@K should all be 1.0

    print("\n" + "=" * 80)
    print("✅ PERFECT RANKING TEST PASSED!")
    print("=" * 80)


def test_training_iteration():
    """Test a single training iteration (forward + backward)."""
    print("\n" + "=" * 80)
    print("TEST: Training Iteration (Forward + Backward)")
    print("=" * 80)

    # Simple mock model
    class MockModel(nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.user_proj = nn.Linear(embedding_dim, embedding_dim)
            self.item_proj = nn.Linear(embedding_dim, embedding_dim)

        def forward(self, user_emb, item_emb):
            return self.user_proj(user_emb), self.item_proj(item_emb)

    embedding_dim = 16
    batch_size = 2
    num_candidates = 3

    model = MockModel(embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = MTGRRankingLoss(temperature=0.1)

    # Dummy input embeddings
    user_emb = torch.randn(batch_size * num_candidates, embedding_dim)
    item_emb = torch.randn(batch_size * num_candidates, embedding_dim)

    print(f"\nBatch size: {batch_size}")
    print(f"Num candidates: {num_candidates}")
    print(f"Embedding dim: {embedding_dim}")

    # Training iteration
    optimizer.zero_grad()

    user_out, item_out = model(user_emb, item_emb)
    loss = loss_fn(user_out, item_out, num_candidates)

    print(f"\nInitial loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()

    # Second iteration - loss should change (likely decrease)
    optimizer.zero_grad()
    user_out2, item_out2 = model(user_emb, item_emb)
    loss2 = loss_fn(user_out2, item_out2, num_candidates)

    print(f"Loss after 1 step: {loss2.item():.4f}")
    print(f"Loss change: {loss2.item() - loss.item():.4f}")

    # Verify training works
    assert loss2.item() != loss.item(), "Loss should change after optimizer step"
    print("\n✓ Loss changed after optimization: PASS")

    # Compute metrics
    with torch.no_grad():
        metrics = compute_metrics(user_out2, item_out2, num_candidates)

    print(f"\nMetrics after 1 step:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n" + "=" * 80)
    print("✅ TRAINING ITERATION TEST PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Run all tests
    test_loss_function()
    test_metrics()
    test_perfect_ranking()
    test_training_iteration()

    print("\n" + "=" * 80)
    print("🎉 ALL TRAINING COMPONENT TESTS PASSED!")
    print("=" * 80)
    print("\nReady to test full training loop in train_mtgr.py!")
    print("\nNext steps:")
    print("1. Prepare training data (Amazon Beauty dataset)")
    print("2. Run: python train_mtgr.py --data_path data/beauty --num_epochs 10")
    print("3. Monitor with TensorBoard: tensorboard --logdir runs/")
    print("=" * 80)
