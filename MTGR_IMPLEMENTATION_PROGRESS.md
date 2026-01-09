# MTGR Implementation Progress

## ✅ Phase 1: Tokenization Layer (COMPLETED)

### What We've Implemented:

#### 1. Configuration ✓
**File**: `dlrm_hstu.py` (lines 100-105)

Added MTGR-specific configuration flags to `DlrmHSTUConfig`:
```python
use_mtgr_tokenization: bool = False  # Enable MTGR mode
mtgr_user_feature_names: List[str]   # e.g., ['user_id', 'avg_rating', 'avg_review_len']
mtgr_seq_feature_names: List[str]    # e.g., ['seq_item_id', 'seq_category', 'seq_brand', 'seq_price']
mtgr_cross_feature_names: List[str]  # e.g., ['brand_count']
mtgr_candidate_feature_names: List[str]  # e.g., ['cand_item_id', 'cand_category', 'cand_brand', 'cand_price']
```

**Key Design Decision**: Feature flags ensure backward compatibility - existing HSTU code works unchanged when `use_mtgr_tokenization=False`.

---

#### 2. Three Separate MLPs ✓
**File**: `dlrm_hstu.py` (lines 276-329)

Added three tokenization MLPs in `DlrmHSTU.__init__()`:

**User MLP** (lines 280-294):
```python
self._mtgr_user_mlp = Sequential(
    Linear(192 * num_user_features, 512),
    SwishLayerNorm(512),
    Linear(512, d_model),
    LayerNorm(d_model)
)
```
- Converts user profile features → unified tokens
- Input: Concatenated embeddings of user features
- Output: [num_user_features, d_model] tokens

**Sequence MLP** (lines 296-310):
```python
self._mtgr_seq_mlp = Sequential(
    Linear(192 * num_seq_features, 512),
    SwishLayerNorm(512),
    Linear(512, d_model),
    LayerNorm(d_model)
)
```
- Converts sequence item features → unified tokens
- Input: Concatenated embeddings of item features (id, category, brand, price)
- Output: [seq_len, d_model] tokens

**Candidate MLP** (lines 312-329):
```python
self._mtgr_candidate_mlp = Sequential(
    Linear(192 * (num_cross_features + num_item_features), 512),
    SwishLayerNorm(512),
    Linear(512, d_model),
    LayerNorm(d_model)
)
```
- Converts candidate features (cross + item) → unified tokens
- Input: Concatenated embeddings of cross features AND item features
- Output: [num_candidates, d_model] tokens

**Key Design Decision**: All three MLPs output tokens with the same dimension (`d_model = hstu_transducer_embedding_dim`), enabling them to be processed by the same HSTU layers.

---

#### 3. KJT Splitting Helper ✓
**File**: `dlrm_hstu.py` (lines 434-477)

Added `_split_mtgr_features()` method:
```python
def _split_mtgr_features(kjt, feature_names):
    """Extract subset of features from KJT"""
    # Find matching features
    # Extract their lengths and values
    # Return new KJT with only those features
```

**Purpose**: Separates the monolithic KJT into user/seq/candidate groups based on feature names.

**Example**:
```python
# Input KJT with keys: ['user_id', 'avg_rating', 'seq_item_id', 'seq_brand', ...]
user_kjt = _split_mtgr_features(uih_kjt, ['user_id', 'avg_rating', 'avg_review_len'])
seq_kjt = _split_mtgr_features(uih_kjt, ['seq_item_id', 'seq_category', 'seq_brand', 'seq_price'])
# Result: Two separate KJTs, each with only their respective features
```

---

#### 4. MTGR Tokenization Method ✓
**File**: `dlrm_hstu.py` (lines 479-572)

Added `_mtgr_tokenize()` method - **the core tokenization logic**:

**Input**:
- `user_embeddings`: Dict[str, SequenceEmbedding]
- `seq_embeddings`: Dict[str, SequenceEmbedding]
- `cand_embeddings`: Dict[str, SequenceEmbedding]

**Process**:
1. **User Tokenization**:
   ```python
   user_concat = concat([emb(user_id), emb(avg_rating), emb(avg_review_len)])
   user_tokens = _mtgr_user_mlp(user_concat)  # [3, d_model]
   ```

2. **Sequence Tokenization**:
   ```python
   for each item i in sequence:
       seq_concat[i] = concat([emb(item_id), emb(category), emb(brand), emb(price)])
   seq_tokens = _mtgr_seq_mlp(seq_concat)  # [seq_len, d_model]
   ```

3. **Candidate Tokenization**:
   ```python
   for each candidate k:
       cand_concat[k] = concat([emb(brand_count), emb(item_id), emb(category), ...])
   cand_tokens = _mtgr_candidate_mlp(cand_concat)  # [K, d_model]
   ```

4. **Concatenation**:
   ```python
   all_tokens = concat([user_tokens, seq_tokens, cand_tokens])
   # Shape: [3 + seq_len + K, d_model]
   ```

**Output**:
- `all_tokens`: [total_len, d_model] - unified token sequence
- `group_indices`: [total_len] - token type labels (0=user, 1=seq, 2=cand)
- `num_user_tokens`, `num_seq_tokens`, `num_cand_tokens` - counts

**Key Design Decision**: Returns `group_indices` to enable future Group Layer Normalization.

---

## 🔄 What's Next: Integrating Tokenization into Forward Pass

### Remaining Tasks:

#### 1. Modify `preprocess()`
**Goal**: Split KJTs and create separate embeddings for MTGR mode

```python
def preprocess(uih_features_kjt, candidates_features_kjt):
    if self._hstu_configs.use_mtgr_tokenization:
        # Split KJTs
        user_kjt = self._split_mtgr_features(uih_features_kjt, mtgr_user_feature_names)
        seq_kjt = self._split_mtgr_features(uih_features_kjt, mtgr_seq_feature_names)
        cand_kjt = candidates_features_kjt  # Already contains cross + item features

        # Separate embedding lookups
        user_embeddings = self._embedding_collection(user_kjt)
        seq_embeddings = self._embedding_collection(seq_kjt)
        cand_embeddings = self._embedding_collection(cand_kjt)

        return user_embeddings, seq_embeddings, cand_embeddings, ...
    else:
        # Original HSTU path (unchanged)
        ...
```

#### 2. Modify `main_forward()`
**Goal**: Use MTGR tokenization when enabled

```python
def main_forward(...):
    if self._hstu_configs.use_mtgr_tokenization:
        # Apply MTGR tokenization
        all_tokens, group_indices, num_user, num_seq, num_cand = self._mtgr_tokenize(
            user_embeddings, seq_embeddings, cand_embeddings
        )

        # Pass to HSTU (for now, treat as regular sequence)
        # TODO later: Add custom masking and Group LN
        user_output = self._hstu_transducer(
            seq_embeddings=all_tokens,
            ...
        )

        # Extract candidate outputs (last K tokens)
        cand_outputs = user_output[:, -(num_cand):, :]
    else:
        # Original HSTU path (unchanged)
        ...
```

---

## 📊 Current Architecture Flow

```
INPUT: uih_kjt + candidates_kjt
    ↓
[IF use_mtgr_tokenization=True]
    ↓
Split KJTs → user_kjt, seq_kjt, cand_kjt
    ↓
Embedding Lookup → user_emb, seq_emb, cand_emb
    ↓
3 MLPs → user_tokens, seq_tokens, cand_tokens
    ↓
Concatenate → all_tokens [3 + seq_len + K, d_model]
    ↓
HSTU Attention (existing, causal mask)  ← TODO: Add MTGR masking later
    ↓
Extract candidate outputs
    ↓
OUTPUT: predictions
```

---

## 🎯 Next Session Goals

1. **Modify `preprocess()`** to handle MTGR KJT splitting and embedding
2. **Modify `main_forward()`** to use `_mtgr_tokenize()` when enabled
3. **Test with toy data** to verify:
   - Shapes are correct
   - Forward pass completes
   - Gradients flow properly
4. **Verify backward compatibility** - ensure HSTU mode still works

---

## 📝 Notes on Design Decisions

### Why Feature Flags?
- Enables gradual migration
- Allows A/B testing between HSTU and MTGR
- Reduces risk of breaking production code

### Why Separate MLPs?
- Aligns with MTGR paper (Fig 2a)
- Each token type has different semantic meaning
- Enables different learning rates / regularization per group later

### Why Not Modify Masking Yet?
- Masking requires changes to HSTU core (STU layers)
- Want to verify tokenization works first
- Will implement in Phase 2

### Why Group Indices?
- Prepared for Group Layer Normalization (Phase 2)
- Enables different preprocessing per token type
- Useful for debugging/visualization

---

## ⚠️ Known Limitations (To Address Later)

1. **No MTGR Masking**: Currently uses HSTU's causal mask, not MTGR's dynamic mask
2. **No Group Layer Norm**: All tokens normalized together
3. **No Real-Time (R) Tokens**: Beauty dataset lacks timestamps
4. **Padding Not Optimized**: Variable-length sequences not efficiently batched

---

## ✅ Code Quality Checklist

- [x] Added proper docstrings
- [x] Used descriptive variable names
- [x] Followed existing code style (SwishLayerNorm, init_mlp_weights)
- [x] Preserved backward compatibility
- [x] No breaking changes to existing HSTU functionality
