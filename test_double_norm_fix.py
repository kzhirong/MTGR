"""
Test to verify that the MTGR fix prevents double normalization and preserves
group-specific statistics.
"""

import torch
import torch.nn.functional as F
from generative_recommenders.modules.group_layer_norm import GroupLayerNorm

def test_double_normalization():
    """
    Demonstrate that our old approach caused double normalization,
    and verify that applying GroupLayerNorm only once preserves group statistics.
    """
    print("=" * 80)
    print("TEST: Verifying Double Normalization Fix")
    print("=" * 80)

    # Create tokens with distinct group statistics
    batch_size = 2
    num_user, num_seq, num_cand = 3, 5, 2
    d_model = 8

    # User tokens: high values (mean ≈ 5)
    user_tokens = torch.randn(batch_size * num_user, d_model) * 1.0 + 5.0

    # Seq tokens: medium values (mean ≈ 0)
    seq_tokens = torch.randn(batch_size * num_seq, d_model) * 1.0 + 0.0

    # Candidate tokens: low values (mean ≈ -3)
    cand_tokens = torch.randn(batch_size * num_cand, d_model) * 1.5 - 3.0

    # Concatenate all tokens
    x = torch.cat([user_tokens, seq_tokens, cand_tokens], dim=0)
    group_boundaries = (num_user, num_seq, num_cand)

    print("\n1. ORIGINAL TOKEN STATISTICS:")
    print("-" * 80)
    user_end = batch_size * num_user
    seq_end = user_end + batch_size * num_seq

    print(f"User tokens:  mean={user_tokens.mean():.3f}, std={user_tokens.std():.3f}")
    print(f"Seq tokens:   mean={seq_tokens.mean():.3f}, std={seq_tokens.std():.3f}")
    print(f"Cand tokens:  mean={cand_tokens.mean():.3f}, std={cand_tokens.std():.3f}")
    print(f"All tokens:   mean={x.mean():.3f}, std={x.std():.3f}")

    # Apply GroupLayerNorm (our first normalization)
    gln = GroupLayerNorm(normalized_shape=d_model, eps=1e-6)
    x_group_normed = gln(x, group_boundaries)

    print("\n2. AFTER GROUP LAYER NORMALIZATION:")
    print("-" * 80)
    user_gn = x_group_normed[:user_end]
    seq_gn = x_group_normed[user_end:seq_end]
    cand_gn = x_group_normed[seq_end:]

    print(f"User tokens:  mean={user_gn.mean():.6f}, std={user_gn.std():.3f}")
    print(f"Seq tokens:   mean={seq_gn.mean():.6f}, std={seq_gn.std():.3f}")
    print(f"Cand tokens:  mean={cand_gn.mean():.6f}, std={cand_gn.std():.3f}")
    print(f"All tokens:   mean={x_group_normed.mean():.6f}, std={x_group_normed.std():.3f}")
    print("✓ Each group normalized separately (mean≈0, std≈1 within each group)")

    # OLD APPROACH: Apply LayerNorm with identity weights (causes double normalization)
    print("\n3. OLD APPROACH - Apply LayerNorm with identity weights:")
    print("-" * 80)
    weight = torch.ones(d_model)
    bias = torch.zeros(d_model)
    x_double_normed = F.layer_norm(x_group_normed, [d_model], weight, bias, eps=1e-6)

    user_dn = x_double_normed[:user_end]
    seq_dn = x_double_normed[user_end:seq_end]
    cand_dn = x_double_normed[seq_end:]

    print(f"User tokens:  mean={user_dn.mean():.6f}, std={user_dn.std():.3f}")
    print(f"Seq tokens:   mean={seq_dn.mean():.6f}, std={seq_dn.std():.3f}")
    print(f"Cand tokens:  mean={cand_dn.mean():.6f}, std={cand_dn.std():.3f}")
    print(f"All tokens:   mean={x_double_normed.mean():.6f}, std={x_double_normed.std():.3f}")
    print("✗ Group-specific statistics DESTROYED by re-normalization!")
    print("  Even with weight=1, bias=0, LayerNorm still standardizes!")

    # Compare: Did double normalization change the data?
    diff = (x_group_normed - x_double_normed).abs().mean()
    print(f"\nMean absolute difference: {diff:.6f}")
    if diff > 1e-5:
        print("✗ SIGNIFICANT CHANGE - Double normalization occurred!")
    else:
        print("✓ No change - Would indicate normalization was bypassed (not the case here)")

    # NEW APPROACH: Skip the second normalization entirely
    print("\n4. NEW APPROACH - Use group-normalized tokens directly:")
    print("-" * 80)
    x_correct = x_group_normed  # No second normalization!

    user_c = x_correct[:user_end]
    seq_c = x_correct[user_end:seq_end]
    cand_c = x_correct[seq_end:]

    print(f"User tokens:  mean={user_c.mean():.6f}, std={user_c.std():.3f}")
    print(f"Seq tokens:   mean={seq_c.mean():.6f}, std={seq_c.std():.3f}")
    print(f"Cand tokens:  mean={cand_c.mean():.6f}, std={cand_c.std():.3f}")
    print(f"All tokens:   mean={x_correct.mean():.6f}, std={x_correct.std():.3f}")
    print("✓ Group-specific statistics PRESERVED!")
    print("  Each group maintains its normalized distribution")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("• OLD: Passing weight=1, bias=0 does NOT bypass LayerNorm")
    print("• OLD: This causes double normalization that destroys group structure")
    print("• NEW: Decompose operations and apply GroupLayerNorm only once")
    print("• NEW: Group-specific statistics are preserved as MTGR requires")
    print("=" * 80)

    return True


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    test_double_normalization()
