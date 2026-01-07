# Colab Testing Session - Issues & Fixes

**Date**: January 6, 2026
**Task**: End-to-end testing of MTGR pipeline on Google Colab

This document details every issue encountered during Colab testing and the fixes applied.

---

## Issue 1: File Path Mismatch

### Problem
```
FileNotFoundError: [Errno 2] No such file or directory:
'/Users/zhirong/GitHub/generative-recommenders/data/Beauty_train.pkl'
```

**Cause**: Test file hardcoded local macOS paths

### Fix
**File**: `test_mtgr_end_to_end.py` (lines 105-111)

**Before**:
```python
dataset = MTGRBeautyDataset(
    data_file="/Users/zhirong/GitHub/generative-recommenders/data/Beauty_train.pkl",
    ...
)
```

**After**:
```python
dataset = MTGRBeautyDataset(
    data_file="data/Beauty_train.pkl",  # Relative path
    ...
)
```

**Lesson**: Use relative paths for portability across environments

---

## Issue 2: Data Format Mismatch (Item IDs)

### Problem
```
KeyError: '2762'
```
When accessing `self.item2id[user_seq['full_item_ids'][t]]`

**Cause**: Training data uses integer IDs (e.g., `'2762'`), but `datamaps['item2id']` expected Amazon ASINs (e.g., `'B004756YJA'`)

**Analysis**:
```python
# Training data
train[0]['sequential_features']['full_item_ids'][:5]
# Output: ['1917', '10589', '4672', '2762', '2080']  ← Integer IDs

# Datamaps (WRONG!)
datamaps['item2id']
# Output: {'B004756YJA': '1', 'B004ZT0SSG': '2', ...}  ← Amazon ASINs

# Result: 100% mismatch (12,099 items, 0 found)
```

### Root Cause
The Beauty dataset preprocessing happened in stages:
1. **Stage 1**: Raw Amazon data → Convert ASINs to integers → Save to `Beauty_train.pkl`
2. **Stage 2**: Create `Beauty_datamaps.pkl` preserving ASIN → int mappings
3. **Result**: Training data already uses remapped IDs, but datamaps still has original ASINs

### Fix Attempt 1 (Workaround)
**File**: `beauty_mtgr.py` (lines 87-102)

Built identity mapping from data itself:
```python
# Ignore datamaps, build from item_features
self.item2id = {item_id_str: int(item_id_str)
                for item_id_str in self.item_features.keys()}
```

**Problem with this approach**:
- Confusing (loads datamaps but doesn't use it)
- Not the proper abstraction

### Fix Attempt 2 (Proper Solution) ✅
**Created**: `fix_beauty_datamaps.py`

Generated corrected datamaps file:
```python
# Fixed datamaps now has identity mappings
{
    'user2id': {'1': 1, '2': 2, ..., '22363': 22363},
    'item2id': {'1': 1, '2': 2, ..., '12101': 12101},
}
```

**File**: `beauty_mtgr.py` (lines 78-94) - Final clean version

**After**:
```python
# Load datamaps properly
with open(datamaps_file, 'rb') as f:
    datamaps = pickle.load(f)
    self.user2id = datamaps['user2id']  # Now works!
    self.item2id = datamaps['item2id']
```

**Lesson**: Fix the source data format issue, don't work around it in code

---

## Issue 3: Data Format Mismatch (User IDs)

### Problem
```
KeyError: '5715'
```
When accessing `self.user2id[user['user_id']]`

**Cause**: Same issue as Item IDs - training data uses integer user IDs, datamaps expected Amazon user strings

**Analysis**:
```python
# Training data
train[0]['user_id']  # Output: '5715'  ← Integer ID

# Datamaps (WRONG!)
datamaps['user2id']  # Output: {'A1YJEY40YUW4SE': '1', ...}  ← Amazon user IDs

# Result: 100% mismatch (17,890 users, 0 found)
```

### Additional Problem: Val/Test User Coverage
- **Train**: 17,890 users
- **Val**: 2,236 users (100% NOT in train!)
- **Test**: 2,237 users (100% NOT in train!)

If we built `user2id` only from training data, val/test would fail!

### Fix
**Solved by**: `fix_beauty_datamaps.py`

The corrected datamaps now includes ALL users across all splits:
```python
# Load all splits to get complete vocabulary
for split in [train_data, val_data, test_data]:
    all_users.update(user['user_id'] for user in split)

# Create identity mapping for ALL users
user2id = {user_id_str: int(user_id_str) for user_id_str in all_users}
# Result: 22,363 users (covers train + val + test)
```

**Lesson**: Dataset mappings must cover ALL splits, not just training data

---

## Issue 4: Missing EmbeddingConfig Parameter

### Problem
```
TypeError: EmbeddingConfig.__init__() got an unexpected keyword argument 'pooling'
```

**Cause**: Used incorrect TorchRec API - this codebase's `EmbeddingConfig` doesn't support `pooling` parameter

### Fix
**File**: `test_mtgr_end_to_end.py` (line 14, and all EmbeddingConfig calls)

**Before**:
```python
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType, PoolingType

tables[feature_name] = EmbeddingConfig(
    name=feature_name,
    embedding_dim=64,
    num_embeddings=10000,
    data_type=DataType.FP32,
    pooling=PoolingType.NONE,  # ✗ Not supported
)
```

**After**:
```python
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType

tables[feature_name] = EmbeddingConfig(
    name=feature_name,
    embedding_dim=64,
    num_embeddings=10000,
    data_type=DataType.FP32,
    # No pooling parameter
)
```

**Lesson**: Check the codebase's existing usage patterns before copying from external docs

---

## Issue 5: Missing Multitask Config

### Problem
```
AssertionError: task_configs must be non-empty.
```

**Cause**: `DlrmHSTU.__init__()` requires `multitask_configs` but we didn't provide it

### Fix
**File**: `test_mtgr_end_to_end.py` (lines 48-55)

**Added**:
```python
# Add multitask config (required by DlrmHSTU)
config.multitask_configs = [
    TaskConfig(
        task_name="recommendation",
        task_weight=1.0,
        task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
    )
]
```

**Imports added**:
```python
from generative_recommenders.modules.multitask_module import MultitaskTaskType, TaskConfig
```

**Lesson**: Even when testing a subset of functionality, provide all required configs

---

## Issue 6: Device Mismatch (Meta vs CPU)

### Problem
```
RuntimeError: Tensor on device cpu is not on the expected device meta!
```

**Cause**: TorchRec initializes embedding tables on "meta" device (lazy initialization), but input data is on CPU

**Background**:
- PyTorch 2.0+ supports "meta" device for lazy initialization
- Tensors are created without allocating memory
- Must be materialized to actual device before use

### Fix Attempt 1
**File**: `test_mtgr_end_to_end.py` (line 233)

```python
model = model.to(device)  # ✗ Fails with NotImplementedError
```

**Error**:
```
NotImplementedError: Cannot copy out of meta tensor; no data!
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to()
```

### Fix Attempt 2 (Proper Solution) ✅
**File**: `test_mtgr_end_to_end.py` (lines 231-236)

**After**:
```python
# Move model to CPU (TorchRec initializes on meta device)
device = torch.device("cpu")
model = model.to_empty(device=device)  # Materialize structure
# Reset parameters after moving from meta device
model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
logger.info(f"   ✓ Model materialized on {device}")
```

**Lesson**: Use `to_empty()` + `reset_parameters()` for meta → device transitions

---

## Issue 7: Embedding Index Out of Range

### Problem
```
IndexError: index out of range in self
```
During embedding lookup in forward pass

**Cause**: Feature values exceeded embedding table sizes

### Investigation
**Created**: `debug_batch_values.py`

Ran analysis on first 100 samples and found:

```
Feature              Max Value    Our Setting    Status
-------------------  -----------  -------------  ------
brand_count          11,520       200            ✗ FAIL (off by 57x!)
cand_category        12,064       10,000         ✗ FAIL
cand_brand           11,915       10,000         ✗ FAIL
seq_price            28,500       100,000        ✓ OK
```

**The `brand_count` value of 11,520 was suspicious** - that's not a count, it's an item ID!

### Root Cause Analysis
**File**: `beauty_mtgr.py` (lines 256-267) - BEFORE FIX

The candidate features were being appended **interleaved** instead of **grouped**:

```python
# WRONG! (Interleaved)
for cand_id in candidate_ids:  # 5 candidates
    cand_values.append(brand_count)   # ← All 5 features per candidate
    cand_values.append(cand_id)
    cand_values.append(category)
    cand_values.append(brand)
    cand_values.append(price)

# Result: [brand_0, id_0, cat_0, brand_0, price_0,
#          brand_1, id_1, cat_1, brand_1, price_1, ...]
```

But KeyedJaggedTensor expects **grouped** values:
```python
# CORRECT! (Grouped by feature)
# [brand_0, brand_1, brand_2, brand_3, brand_4,
#  id_0, id_1, id_2, id_3, id_4,
#  cat_0, cat_1, cat_2, cat_3, cat_4,
#  ...]
```

**What was happening**:
1. KJT expected first 5 values to be `brand_count` for all candidates
2. But got: `[brand_count_0, item_id_0, category_0, brand_0, price_0]`
3. So `item_id_0` (e.g., 10589) was being read as `brand_count_1`!
4. Embedding table for `brand_count` was size 200
5. Lookup with index 10589 → **IndexError**

### Fix
**File**: `beauty_mtgr.py` (lines 251-279)

**Before**:
```python
cand_values = []
for cand_id in candidate_ids:
    cross_features = self.compute_cross_features(history, cand_id)
    cand_values.append(cross_features['brand_count'])  # ✗ Interleaved!

    item_feats = self.item_features[str(cand_id)]
    cand_values.append(cand_id)
    cand_values.append(hash(item_feats['category']) % 10000)
    cand_values.append(hash(item_feats['brand']) % 10000)
    cand_values.append(int(item_feats['price'] * 100))
```

**After**:
```python
# Collect features for all candidates (grouped by feature type)
brand_counts = []
item_ids = []
categories = []
brands = []
prices = []

for cand_id in candidate_ids:
    cross_features = self.compute_cross_features(history, cand_id)
    brand_counts.append(cross_features['brand_count'])

    item_feats = self.item_features[str(cand_id)]
    item_ids.append(cand_id)
    categories.append(hash(item_feats['category']) % 10000)
    brands.append(hash(item_feats['brand']) % 10000)
    prices.append(int(item_feats['price'] * 100))

# Concatenate all features (grouped by type, not interleaved!)
cand_values = brand_counts + item_ids + categories + brands + prices
```

**Verification** (after fix):
```
Feature              Max Value    Status
-------------------  -----------  ------
brand_count          ≤ 100        ✓ OK (actual count now!)
cand_category        < 10,000     ✓ OK (hash values)
cand_brand           < 10,000     ✓ OK (hash values)
cand_item_id         ≤ 12,101     ✓ OK (within vocab)
```

**Lesson**: KeyedJaggedTensor expects features grouped by type, not interleaved per sample!

---

## Summary of All Changes

### Files Modified

1. **`test_mtgr_end_to_end.py`**
   - ✅ Fixed file paths (absolute → relative)
   - ✅ Removed `pooling` parameter from `EmbeddingConfig`
   - ✅ Added `multitask_configs`
   - ✅ Added proper device materialization (`to_empty()` + `reset_parameters()`)

2. **`beauty_mtgr.py`**
   - ✅ Fixed datamaps usage (now uses corrected datamaps)
   - ✅ Fixed candidate feature ordering (interleaved → grouped)

3. **`Beauty_datamaps_fixed.pkl`** (NEW)
   - ✅ Created corrected datamaps with identity mappings
   - ✅ Covers all users/items across train/val/test splits

4. **`fix_beauty_datamaps.py`** (NEW)
   - ✅ Script to generate corrected datamaps
   - ✅ Analyzes all splits to ensure complete vocabulary coverage

5. **`debug_batch_values.py`** (NEW)
   - ✅ Debug tool to inspect actual feature values
   - ✅ Recommends proper embedding table sizes

### Critical Lessons Learned

1. **Data Format Consistency**: Ensure all preprocessing artifacts (datamaps, features, training data) use consistent ID formats
2. **Split Coverage**: Vocabulary mappings must cover ALL splits, not just training
3. **API Compatibility**: Check existing codebase usage patterns before copying from docs
4. **Device Handling**: Use `to_empty()` for meta → device transitions in PyTorch 2.0+
5. **KJT Format**: Features must be grouped by type, not interleaved per sample
6. **Debug First**: When encountering index errors, inspect actual data values before adjusting config

---

## Current Status

### ✅ Resolved
- [x] File path portability
- [x] User ID mapping
- [x] Item ID mapping
- [x] EmbeddingConfig API
- [x] Multitask configuration
- [x] Device placement
- [x] KJT feature ordering

### ⏳ Next Step
Run `test_mtgr_end_to_end.py` on Colab to verify complete pipeline works!

Expected outcome: All 6 test steps pass ✅

---

**Last Updated**: January 6, 2026
**Environment**: Google Colab (GPU + TorchRec)
