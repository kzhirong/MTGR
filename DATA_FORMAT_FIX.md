# Data Format Mismatch Fix

## Problem Summary

**Error**: `KeyError: '2762'` when accessing `self.item2id[user_seq['full_item_ids'][t]]`

## Root Cause

The Beauty dataset has a **data format mismatch** between different files:

### Data Format Analysis

| File | Item ID Format | Example |
|------|----------------|---------|
| **Beauty_train.pkl** | Integer IDs (as strings) | `'1917', '2762', '10589'` |
| **Beauty_item_feature_map.pkl** | Integer IDs (as strings) | `'1', '2', '3', ..., '12101'` |
| **Beauty_datamaps.pkl** | Amazon ASIN → Integer | `{'B004756YJA': '1', 'B004ZT0SSG': '2', ...}` |

### The Mismatch

```python
# Training data already uses integer IDs
train[0]['sequential_features']['full_item_ids'][:5]
# Output: ['1917', '10589', '4672', '2762', '2080']

# Datamaps maps Amazon ASINs to integers (DIFFERENT!)
datamaps['item2id']
# Output: {'B004756YJA': '1', 'B004ZT0SSG': '2', ...}

# item_features uses integer IDs (MATCHES training data!)
list(item_features.keys())[:5]
# Output: ['1', '2', '3', '4', '5']
```

**Result**: 100% of items in training data (12,099 items) were NOT found in `datamaps['item2id']` because:
- Training data uses integer IDs like `'2762'`
- Datamaps expects Amazon ASINs like `'B004756YJA'`

## Why This Happened

The Beauty dataset preprocessing happened in stages:

1. **Stage 1**: Raw Amazon data → Convert ASINs to integers → Save `Beauty_train.pkl`
2. **Stage 2**: Create `Beauty_datamaps.pkl` to preserve ASIN → integer mapping (for reference)
3. **Stage 3**: Create `Beauty_item_feature_map.pkl` using integer IDs

But `Beauty_train.pkl` already uses **remapped integer IDs**, not the original ASINs!

## The Fix

### Changed in `beauty_mtgr.py` (Lines 78-93)

**Before** (❌ Incorrect):
```python
logger.info(f"Loading datamaps from {datamaps_file}")
with open(datamaps_file, 'rb') as f:
    datamaps = pickle.load(f)
    self.user2id = datamaps['user2id']
    self.item2id = datamaps['item2id']  # ❌ Uses ASIN → int mapping
    self.id2item = datamaps['id2item']

logger.info(f"Loading item features from {item_features_file}")
with open(item_features_file, 'rb') as f:
    self.item_features = pickle.load(f)

self.all_item_ids = list(self.item_features.keys())  # ❌ String IDs
```

**After** (✅ Correct):
```python
logger.info(f"Loading datamaps from {datamaps_file}")
with open(datamaps_file, 'rb') as f:
    datamaps = pickle.load(f)
    self.user2id = datamaps['user2id']  # Only use user mapping

logger.info(f"Loading item features from {item_features_file}")
with open(item_features_file, 'rb') as f:
    self.item_features = pickle.load(f)

# IMPORTANT: Training data already uses integer IDs (as strings)
# item_features keys ARE the integer IDs, not the ASIN strings
# So we create a simple identity mapping: item_id_str -> int(item_id_str)
self.item2id = {item_id_str: int(item_id_str) for item_id_str in self.item_features.keys()}

# Create list of all item IDs for negative sampling (as integers)
self.all_item_ids = list(self.item2id.values())  # ✅ Integer IDs
```

### Additional Fixes

**Lines 262, 358**: Convert integer IDs back to strings when accessing `item_features`

```python
# Before: ❌
item_feats = self.item_features[cand_id]  # cand_id is int, but keys are strings!

# After: ✅
item_feats = self.item_features[str(cand_id)]  # Convert to string for lookup
```

## Verification

### Before Fix
```python
# item2id had ASIN keys that don't exist in training data
'2762' in datamaps['item2id']  # ❌ False - KeyError!
```

### After Fix
```python
# item2id now has integer ID strings matching training data
'2762' in self.item2id  # ✅ True
self.item2id['2762']    # ✅ 2762 (integer)
```

## Files Modified

1. **generative_recommenders/dlrm_v3/datasets/beauty_mtgr.py**
   - Line 78-93: Build `item2id` from `item_features.keys()` instead of datamaps
   - Line 262: Convert `cand_id` to string when accessing item_features
   - Line 358: Convert `candidate_item_id` to string when accessing item_features

## Impact

✅ **Solves**: KeyError when accessing items from training data
✅ **Preserves**: All other functionality unchanged
✅ **Maintains**: User ID mapping from datamaps (still needed)

## Next Steps for Colab

1. Upload the updated `beauty_mtgr.py` to Colab
2. Re-run `python test_mtgr_end_to_end.py`
3. The KeyError should be resolved

---

**Date**: January 6, 2026
**Fix Type**: Data format alignment
**Impact**: Critical - Enables dataset to work with Beauty data
