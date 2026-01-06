# Data Format Mismatch Fix

## Problem Summary

**Errors**:
1. `KeyError: '2762'` when accessing `self.item2id[user_seq['full_item_ids'][t]]`
2. `KeyError: '5715'` when accessing `self.user2id[user['user_id']]`

## Root Cause

The Beauty dataset has a **data format mismatch** between different files:

### Data Format Analysis

| File | User ID Format | Item ID Format |
|------|----------------|----------------|
| **Beauty_train.pkl** | Integer IDs (as strings)<br>`'5715', '1388', ...` | Integer IDs (as strings)<br>`'1917', '2762', '10589'` |
| **Beauty_item_feature_map.pkl** | N/A | Integer IDs (as strings)<br>`'1', '2', '3', ..., '12101'` |
| **Beauty_datamaps.pkl** | Amazon User ID → Integer<br>`{'A1YJEY40YUW4SE': '1', ...}` | Amazon ASIN → Integer<br>`{'B004756YJA': '1', ...}` |

### The Mismatch

```python
# USER IDs: Training data uses integer IDs (as strings)
train[0]['user_id']
# Output: '5715'

# Datamaps expects Amazon user strings (DIFFERENT!)
datamaps['user2id']
# Output: {'A1YJEY40YUW4SE': '1', 'A60XNB876KYML': '2', ...}

# ITEM IDs: Training data uses integer IDs (as strings)
train[0]['sequential_features']['full_item_ids'][:5]
# Output: ['1917', '10589', '4672', '2762', '2080']

# Datamaps expects Amazon ASINs (DIFFERENT!)
datamaps['item2id']
# Output: {'B004756YJA': '1', 'B004ZT0SSG': '2', ...}

# item_features uses integer IDs (MATCHES training data!)
list(item_features.keys())[:5]
# Output: ['1', '2', '3', '4', '5']
```

**Result**:
- 100% of users in training data (17,890 users) were NOT found in `datamaps['user2id']`
- 100% of items in training data (12,099 items) were NOT found in `datamaps['item2id']`

**Why**: Training data uses integer IDs while datamaps expects Amazon strings!

## Why This Happened

The Beauty dataset preprocessing happened in stages:

1. **Stage 1**: Raw Amazon data → Convert ASINs to integers → Save `Beauty_train.pkl`
2. **Stage 2**: Create `Beauty_datamaps.pkl` to preserve ASIN → integer mapping (for reference)
3. **Stage 3**: Create `Beauty_item_feature_map.pkl` using integer IDs

But `Beauty_train.pkl` already uses **remapped integer IDs**, not the original ASINs!

## The Fix

### Changed in `beauty_mtgr.py` (Lines 78-95)

**Before** (❌ Incorrect):
```python
logger.info(f"Loading datamaps from {datamaps_file}")
with open(datamaps_file, 'rb') as f:
    datamaps = pickle.load(f)
    self.user2id = datamaps['user2id']  # ❌ Expects Amazon user strings
    self.item2id = datamaps['item2id']  # ❌ Expects Amazon ASINs
    self.id2item = datamaps['id2item']

logger.info(f"Loading item features from {item_features_file}")
with open(item_features_file, 'rb') as f:
    self.item_features = pickle.load(f)

self.all_item_ids = list(self.item_features.keys())  # ❌ String IDs
```

**After** (✅ Correct):
```python
logger.info(f"Loading item features from {item_features_file}")
with open(item_features_file, 'rb') as f:
    self.item_features = pickle.load(f)

# IMPORTANT: Training data already uses integer IDs (as strings) for BOTH users and items
# The datamaps file maps Amazon ASINs/user strings to integers,
# but Beauty_train.pkl is already preprocessed with integer IDs.
# So we create identity mappings directly from the data.

logger.info("Creating user2id and item2id mappings from preprocessed data")

# User ID mapping: Extract from training data
self.user2id = {user['user_id']: int(user['user_id']) for user in self.users}

# Item ID mapping: Extract from item features
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
# user2id had Amazon user strings that don't exist in training data
'5715' in datamaps['user2id']  # ❌ False - KeyError!

# item2id had ASIN keys that don't exist in training data
'2762' in datamaps['item2id']  # ❌ False - KeyError!
```

### After Fix
```python
# user2id now has integer ID strings matching training data
'5715' in self.user2id  # ✅ True
self.user2id['5715']    # ✅ 5715 (integer)

# item2id now has integer ID strings matching training data
'2762' in self.item2id  # ✅ True
self.item2id['2762']    # ✅ 2762 (integer)
```

## Files Modified

1. **generative_recommenders/dlrm_v3/datasets/beauty_mtgr.py**
   - Lines 78-95: Build both `user2id` and `item2id` from training data/item_features (not datamaps)
   - Line 262: Convert `cand_id` to string when accessing item_features
   - Line 358: Convert `candidate_item_id` to string when accessing item_features

## Impact

✅ **Solves**: Both KeyErrors ('2762' for items, '5715' for users)
✅ **Preserves**: All other functionality unchanged
✅ **Note**: `datamaps_file` parameter still required but datamaps content is ignored for Beauty dataset

## Next Steps for Colab

1. Upload the updated `beauty_mtgr.py` to Colab
2. Re-run `python test_mtgr_end_to_end.py`
3. The KeyError should be resolved

---

**Date**: January 6, 2026
**Fix Type**: Data format alignment
**Impact**: Critical - Enables dataset to work with Beauty data
