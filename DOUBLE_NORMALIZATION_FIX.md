# MTGR Double Normalization Fix

## Executive Summary

**Issue**: MTGR Group Layer Normalization was being applied twice, destroying the group-specific statistics that MTGR depends on.

**Root Cause**: We misunderstood that passing `weight=ones(), bias=zeros()` to LayerNorm would bypass normalization. It doesn't - LayerNorm still standardizes data (mean=0, std=1) even with identity affine parameters.

**Solution**: Decompose HSTU's fused operations and apply GroupLayerNorm explicitly without any subsequent re-normalization.

**Status**: ✅ Fixed in [stu.py:331-427](generative_recommenders/modules/stu.py#L331-L427)

---

## The Problem Explained

### What We Thought (WRONG)

```python
# We thought identity weights would skip normalization:
output = LayerNorm(x, weight=ones(), bias=zeros())  # We thought: output ≈ x
```

### What Actually Happens (CORRECT)

LayerNorm formula: `y = (x - μ) / σ · γ + β`

With `γ=1, β=0`:
```python
y = (x - μ) / σ · 1 + 0
  = (x - μ) / σ
```

**This is standardization, NOT identity!** The data is still forced to mean=0, std=1.

### Impact on MTGR

**What MTGR needs (from paper Figure 2(b)):**
```
Input X
  ↓
GroupLayerNorm (normalize User/Seq/Cand separately)
  ↓ [tokens preserve group-specific distributions]
Linear Projections (Q, K, V, U)
  ↓
Attention
```

**What our old code was doing (BROKEN):**
```
Input X
  ↓
GroupLayerNorm (normalize User/Seq/Cand separately)
  ↓ [tokens have group-specific distributions]
HSTU kernel's LayerNorm (re-normalize ALL tokens together)
  ↓ [GROUP-SPECIFIC STATS DESTROYED!]
Linear Projections (Q, K, V, U)
  ↓
Attention
```

The kernel's built-in LayerNorm was re-normalizing all tokens globally, erasing the domain-specific normalization that GroupLayerNorm created.

---

## The Fix

### Old Implementation (BROKEN)

```python
# Apply GroupLayerNorm
x_normed = GroupLayerNorm(x, boundaries)

# Pass to fused kernel with "identity" weights
u, attn, k, v = hstu_preprocess_and_attention(
    x=x_normed,
    norm_weight=ones(),  # ← Doesn't actually bypass normalization!
    norm_bias=zeros(),   # ← LayerNorm still standardizes!
    ...
)
```

### New Implementation (FIXED)

```python
# Step 1: Apply GroupLayerNorm ONCE
x_normed = GroupLayerNorm(x, boundaries)

# Step 2: Manual projection (NO normalization)
uvqk = x_normed @ weight + bias
u, v, q, k = split(uvqk)
u = silu(u)

# Step 3: Call attention-only kernel (NO normalization inside)
attn = hstu_mha(q, k, v, ...)

# Step 4: Process output manually (NO normalization)
y = attn * u
output_unnormed = concat([y, u, x_normed]) @ output_weight

# Step 5: Apply GroupLayerNorm to output
output_normed = GroupLayerNorm(output_unnormed, boundaries)

# Step 6: Residual
return output_normed + x
```

---

## Key Changes in stu.py

### What We Decomposed

**Before (used fused kernel `hstu_preprocess_and_attention`):**
- ❌ LayerNorm (built-in, can't disable)
- ✅ Linear projection
- ✅ SiLU activation
- ✅ Attention

**After (manual operations):**
- ✅ GroupLayerNorm (external, explicit)
- ✅ Linear projection (manual `torch.addmm`)
- ✅ SiLU activation (manual `F.silu`)
- ✅ Attention (`hstu_mha` kernel - no normalization)

### What We Preserved

✅ **Attention optimization**: Still uses `hstu_mha` optimized kernel
✅ **Correct architecture**: Matches MTGR paper Figure 2(b)
✅ **Group-specific stats**: GroupLayerNorm applied only once
✅ **Clean code**: Explicit, well-documented operations

### What We Lost

⚠️ **Preprocessing fusion**: LayerNorm + Linear are now separate ops
⚠️ **Slight performance cost**: Extra kernel launches for decomposed ops

But this is necessary for correctness! The attention (most expensive part) is still optimized.

---

## How to Test

### Local Test (verify concept)

```bash
python test_double_norm_fix.py
```

This demonstrates:
1. `weight=1, bias=0` does NOT bypass LayerNorm
2. Double normalization changes the data
3. Our approach preserves group-specific statistics

### End-to-End Test (verify full pipeline)

```bash
# In Google Colab (requires GPU + TorchRec)
python test_mtgr_end_to_end.py
```

Expected output:
```
✓ GroupLayerNorm integration: PASS
✓ Forward pass (tokenization): PASS
Phase 2 Complete: Group Layer Normalization integrated!
```

---

## Mathematical Proof

### LayerNorm Definition

```
μ = mean(x)
σ = std(x)
y = (x - μ) / σ · γ + β
```

### With Identity Parameters (γ=1, β=0)

```
y = (x - μ) / σ · 1 + 0
  = (x - μ) / σ
```

**Result**: Data is still standardized to mean≈0, std≈1

**Proof by example**:
```python
x = [1, 2, 3, 4, 5]  # mean=3, std=1.41
y = LayerNorm(x, weight=1, bias=0)
y = [-1.41, -0.71, 0.00, 0.71, 1.41]  # mean≈0, std≈1
```

**Conclusion**: Identity parameters do NOT create an identity function!

---

## References

1. **MTGR Paper** - Figure 2(b) "Self Attention Block", Section 4.2
   - Shows GroupLayerNorm as separate component before projections
   - No indication of normalization inside attention kernel

2. **HSTU Implementation** - [hstu_compute.py:62-68](generative_recommenders/ops/hstu_compute.py#L62-L68)
   - `hstu_compute_uqvk` has LayerNorm as first operation
   - Cannot be disabled or bypassed with parameters

3. **GroupLayerNorm** - [group_layer_norm.py](generative_recommenders/modules/group_layer_norm.py)
   - Normalizes User/Seq/Cand tokens separately
   - Preserves domain-specific distributions

---

## Timeline

- **Phase 1**: MTGR tokenization ✅ (completed earlier)
- **Phase 2.1-2.4**: GroupLayerNorm implementation ✅ (completed earlier)
- **Phase 2.5**: Bug discovered - double normalization ⚠️
- **Phase 2.6**: Fix implemented - decomposed operations ✅ (this fix)
- **Phase 3**: Next - Dynamic Masking (future work)

---

## Authors

- Initial implementation: Previous work
- Bug identified by: User feedback + screenshot analysis
- Fix implemented by: Claude (with user guidance)

---

## See Also

- [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) - MTGR tokenization
- [PHASE2_COMPLETE.md](PHASE2_COMPLETE.md) - GroupLayerNorm (original)
- [test_double_norm_fix.py](test_double_norm_fix.py) - Verification test
- [test_mtgr_end_to_end.py](test_mtgr_end_to_end.py) - Full pipeline test
