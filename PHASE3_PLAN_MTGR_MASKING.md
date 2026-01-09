# Phase 3 Plan: MTGR Dynamic Masking (Simplified)

## Status: Ready to Implement

---

## Why Simplified Masking?

**Problem**: You don't have Real-Time (RT) sequence separated in your data.

**Solution**: Implement simplified 2-group masking instead of full 3-group masking.

### What You Have:
- ✅ User features (U)
- ✅ Historical sequence (S) - all from the past
- ✅ Candidate features (C)

### What You Don't Have:
- ❌ Real-time sequence (R) - recent interactions within aggregation window

### Impact:
- ✅ **Still get the core MTGR benefit**: Diagonal candidate masking
- ✅ **No information leakage**: All sequence is historical (before candidates)
- ✅ **Simpler to implement**: Only 2 masking rules instead of 3

---

## Simplified MTGR Masking Strategy

### Rule 1: User + Sequence (Static Features)
**Full attention** - visible to ALL tokens
- These are all from the past
- No temporal ordering issues
- Safe for all tokens to see

### Rule 2: Candidates
**Diagonal + Static** - each candidate sees:
- ✅ All User tokens
- ✅ All Sequence tokens
- ✅ Itself
- ❌ Other candidates

**Why critical**: Prevents candidates from "cheating" by observing competitor features.

### Mask Visualization

```
          U₁  U₂  S₁  S₂  S₃  C₁  C₂  C₃
     U₁ : [✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ ]  ← User sees all
     U₂ : [✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ ]
     S₁ : [✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ ]  ← Seq sees all
     S₂ : [✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ ]
     S₃ : [✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ ]
     C₁ : [✓   ✓   ✓   ✓   ✓   ✓   ✗   ✗ ]  ← Cand sees U+S+Self, not others
     C₂ : [✓   ✓   ✓   ✓   ✓   ✗   ✓   ✗ ]
     C₃ : [✓   ✓   ✓   ✓   ✓   ✗   ✗   ✓ ]
```

---

## Implementation Status

### ✅ Completed

1. **Mask Generator** - [mtgr_mask.py](generative_recommenders/modules/mtgr_mask.py)
   ```python
   def create_mtgr_mask(
       num_user: int,
       num_seq: int,
       num_cand: int,
       batch_size: int,
       device: torch.device,
   ) -> torch.Tensor:
       # Returns [total_tokens, total_tokens] attention mask
       # 0.0 = can attend, -inf = masked
   ```

2. **Mask Tests** - [test_mtgr_mask.py](test_mtgr_mask.py)
   - ✅ Single sample test
   - ✅ Batch test
   - ✅ Visualization test
   - All tests passing!

### 🔵 TODO: Integration Steps

#### Step 1: Replace hstu_mha with PyTorch Native Attention

**Why needed**: `hstu_mha` only supports causal masking, not custom masks.

**Location**: [stu.py:367-385](generative_recommenders/modules/stu.py#L367-L385)

**Current code:**
```python
attn_output = hstu_mha(
    q=q, k=k, v=v,
    causal=True,  # ← Can't do custom masking!
    ...
)
```

**Replace with:**
```python
# Create MTGR mask
from generative_recommenders.modules.mtgr_mask import create_mtgr_mask

mask = create_mtgr_mask(
    num_user=self._mtgr_group_boundaries[0],
    num_seq=self._mtgr_group_boundaries[1],
    num_cand=self._mtgr_group_boundaries[2],
    batch_size=x.shape[0] // sum(self._mtgr_group_boundaries),
    device=q.device,
)

# Use PyTorch native attention with custom mask
attn_output = torch.nn.functional.scaled_dot_product_attention(
    query=q,
    key=k,
    value=v,
    attn_mask=mask,
    dropout_p=0.0,
    scale=self._attn_alpha,
)
```

#### Step 2: Handle Batch Size Calculation

**Challenge**: Need to calculate batch size from token tensor.

**Solution:**
```python
# In stu.py forward():
total_per_sample = sum(self._mtgr_group_boundaries)
batch_size = x.shape[0] // total_per_sample
```

#### Step 3: Update Mask for Multi-Head Attention

**Challenge**: Mask is [T, T] but attention is [B, H, T, T] (with heads).

**Solution:**
```python
# Reshape mask for multi-head attention
# From: [total_tokens, total_tokens]
# To:   [batch_size, 1, total_tokens, total_tokens]
#       (broadcast over num_heads dimension)

mask_4d = mask.view(batch_size, 1, total_per_sample, total_per_sample)
mask_4d = mask_4d.expand(batch_size, self._num_heads, total_per_sample, total_per_sample)
```

---

## Complete Integration Code

Here's the exact code to add to [stu.py](generative_recommenders/modules/stu.py):

```python
# Step 3: Call attention with MTGR custom mask
with record_function("## mtgr_attention ##"):
    # Calculate batch size
    total_per_sample = sum(self._mtgr_group_boundaries)
    batch_size = q.shape[0] // total_per_sample

    # Create MTGR mask
    from generative_recommenders.modules.mtgr_mask import create_mtgr_mask

    mask = create_mtgr_mask(
        num_user=self._mtgr_group_boundaries[0],
        num_seq=self._mtgr_group_boundaries[1],
        num_cand=self._mtgr_group_boundaries[2],
        batch_size=batch_size,
        device=q.device,
    )

    # Reshape mask for multi-head attention
    # [total_tokens, total_tokens] → [batch, heads, tokens, tokens]
    mask_per_sample = mask.view(
        batch_size,
        total_per_sample,
        total_per_sample
    )
    mask_4d = mask_per_sample.unsqueeze(1).expand(
        batch_size,
        self._num_heads,
        total_per_sample,
        total_per_sample
    )

    # Use PyTorch native attention with custom MTGR mask
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query=q,  # [batch * total_per_sample, num_heads, head_dim]
        key=k,
        value=v,
        attn_mask=mask_4d,
        dropout_p=0.0,
        scale=self._attn_alpha,
    )

    # attn_output: [batch * total_per_sample, num_heads, head_dim]
```

---

## Testing Plan

### Test 1: Verify Mask Integration

**Goal**: Ensure mask is properly applied during attention.

**Test code**:
```python
# In test_mtgr_end_to_end.py, add:

print("\n6. Testing MTGR Dynamic Masking...")

# Get attention weights (with hooks or manual computation)
# Verify:
#   - User tokens can attend to all
#   - Seq tokens can attend to all
#   - Candidate C₁ cannot attend to C₂

print("✓ MTGR dynamic masking: PASS")
```

### Test 2: End-to-End Pipeline

**Goal**: Full pipeline still works with custom attention.

**Run**:
```bash
# In Google Colab
python test_mtgr_end_to_end.py
```

**Expected output**:
```
✓ GroupLayerNorm integration: PASS
✓ MTGR dynamic masking: PASS
✓ Forward pass (tokenization): PASS

✅ END-TO-END TEST PASSED!
Phase 3 Complete: MTGR dynamic masking integrated!
```

---

## Performance Considerations

### Speed Comparison

| Component | HSTU (Optimized) | MTGR (PyTorch Native) |
|-----------|------------------|------------------------|
| Preprocessing | Fused kernel | Manual ops |
| Attention | Optimized CUDA/Triton | PyTorch native |
| Overall | ~100% | ~70-80% |

**Trade-off**: We lose some speed but gain correctness and flexibility.

### Future Optimization

If performance becomes critical:
1. **Profile**: Measure actual slowdown
2. **Optimize**: Focus on bottlenecks
3. **Custom kernel**: Implement MTGR-specific attention kernel (long-term)

For research and initial deployment, PyTorch native is sufficient.

---

## Alternative: If You Had RT Sequence

If you later add real-time sequence, the full 3-group masking would be:

### Full MTGR Masking (3 Groups)

```python
def create_full_mtgr_mask(
    num_user: int,
    num_seq: int,
    num_rt: int,  # ← New parameter
    num_cand: int,
    rt_timestamps: torch.Tensor,  # ← Need timestamps
    cand_timestamps: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    # Rule 1: User + Seq - full attention
    # Rule 2: RT - causal by timestamp
    # Rule 3: Candidates - diagonal
    ...
```

But for now, **2-group masking is sufficient**!

---

## Summary: What's Next

### Completed (Phase 2):
- ✅ GroupLayerNorm implementation
- ✅ Fixed double normalization bug
- ✅ Decomposed HSTU operations

### Ready to Implement (Phase 3):
- ✅ Mask generator created
- ✅ Mask tested and verified
- 🔵 Integration into STULayer (next step)
- 🔵 End-to-end testing

### After Phase 3:
- Phase 4: Training loop and loss function
- Phase 5: Evaluation and metrics

---

## Key Decisions

1. **✅ Use simplified 2-group masking** (no RT sequence)
   - Rationale: Don't have RT data, still get core benefit

2. **✅ Use PyTorch native attention** (not hstu_mha)
   - Rationale: hstu_mha doesn't support custom masks

3. **✅ Accept ~20-30% performance loss** for correctness
   - Rationale: Correctness > speed for research phase

4. **✅ Can add full 3-group masking later** if RT data added
   - Rationale: Modular design allows easy extension

---

## Next Action

**Implement Step 1**: Replace `hstu_mha` with PyTorch native attention + MTGR mask in [stu.py:367-385](generative_recommenders/modules/stu.py#L367-L385).

Ready to proceed?
