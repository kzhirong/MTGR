# DlrmHSTU Architecture Analysis: How to Adapt for MTGR

## Current HSTU Architecture Flow

### 1. Input: Two KJTs
```python
forward(uih_features: KJT, candidates_features: KJT)
```
- `uih_features`: User interaction history (UIH) features
- `candidates_features`: Candidate item features

### 2. Preprocessing (`preprocess()` method, line 373-457)

**Step 2.1: Merge KJTs**
```python
merged_sparse_features = KeyedJaggedTensor.from_lengths_sync(
    keys=uih_features.keys() + candidates_features.keys(),
    values=torch.cat([uih_features.values(), candidates_features.values()]),
    lengths=torch.cat([uih_features.lengths(), candidates_features.lengths()]),
)
```

**Step 2.2: Embedding Lookup (Line 397)**
```python
seq_embeddings_dict = self._embedding_collection(merged_sparse_features)
```
- Uses `EmbeddingCollection` (line 137-141)
- Returns dictionary: `{feature_name: JaggedTensor(values, lengths)}`
- Embeddings are of dimension `hstu_embedding_table_dim` (default 192)

**Step 2.3: Extract Sequence Embeddings**
```python
seq_embeddings = {
    k: SequenceEmbedding(lengths=..., embedding=...)
    for k in user_embedding_feature_names + item_embedding_feature_names
}
```
- Returns: `Dict[str, SequenceEmbedding]`
- Still raw embeddings, **NOT yet tokens**

---

### 3. Main Forward (`main_forward()` method, line 461-546)

**Step 3.1: Merge UIH and Candidate Embeddings (Line 478-499)**
```python
# Concatenates UIH sequence embeddings with candidate embeddings
seq_embeddings[uih_feature_name] = SequenceEmbedding(
    lengths=uih_seq_lengths + num_candidates,
    embedding=concat_2D_jagged(...)  # Merges UIH + Candidates
)
```

**Step 3.2: Item Forward - Create Item Tokens (Line 359-371)**
```python
def _item_forward(seq_embeddings):
    # Concatenate all item embedding features
    all_embeddings = torch.cat([
        seq_embeddings[name].embedding
        for name in item_embedding_feature_names
    ], dim=-1)

    # MLP: (192*num_features) -> 512 -> transducer_embedding_dim
    item_embeddings = self._item_embedding_mlp(all_embeddings)
    return item_embeddings  # Shape: [L, D]
```

**THIS IS THE TOKENIZATION STEP!**
- Input: Raw embeddings (192 dim × num_item_features)
- Output: Unified tokens (transducer_embedding_dim, e.g., 256)

**Step 3.3: User Forward - Create User Tokens + HSTU (Line 304-357)**
```python
def _user_forward(seq_embeddings, payload_features):
    embedding = seq_embeddings[uih_post_id_feature_name].embedding

    # Pass to HSTU Transducer
    candidates_user_embeddings, _ = self._hstu_transducer(
        max_uih_len=max_uih_len,
        max_targets=max_candidates,
        seq_timestamps=source_timestamps,
        seq_embeddings=embedding,  # Already tokenized by _item_forward
        ...
    )
    return candidates_user_embeddings
```

**HSTU Transducer Pipeline (line 244-253):**
1. `ContextualPreprocessor`: Processes contextual features
2. `HSTUPositionalEncoder`: Adds positional/temporal encoding
3. `STUStack`: Self-attention layers (causal, target-aware)
4. `Postprocessor`: Layer normalization

---

## Key Differences: HSTU vs MTGR

| Component | HSTU (Current) | MTGR (Target) |
|-----------|----------------|---------------|
| **Token Types** | 2 types: UIH items + Candidates | **3 types: User (U), Sequence (S), Candidates** |
| **User Features** | Mixed with sequence | **Separate scalar tokens** (user_id, avg_rating, avg_review_len) |
| **Sequence Features** | UIH items only | **Separate sequence tokens** (item_id, category, brand, price) |
| **Cross Features** | Limited (in payload_features) | **Part of Candidate tokens** (brand_count) |
| **MLP Tokenization** | Single `_item_embedding_mlp` | **3 separate MLPs**: User MLP, Seq MLP, Cand MLP |
| **Masking** | Causal + target-aware | **Dynamic masking**: U visible to all, S causal, Candidates diagonal |
| **Group LayerNorm** | Optional via `hstu_group_norm` | **Mandatory** (different normalization per token type) |

---

## Where the Current HSTU Fails for MTGR

### ❌ Problem 1: No Separate User Tokens
```python
# HSTU: User features are contextual, processed separately
# MTGR: User features should be standalone tokens in the sequence

# Current: Contextual features (line 171-183)
preprocessor = ContextualPreprocessor(
    contextual_feature_to_max_length=...,  # Separate pathway
)

# Needed: User tokens as part of main sequence
# [Token(user_id), Token(avg_rating), Token(seq0), Token(seq1), ...]
```

### ❌ Problem 2: Single Item MLP
```python
# HSTU: One MLP for all items (line 256-268)
self._item_embedding_mlp = Sequential(...)  # Shared for UIH + Candidates

# MTGR: Needs 3 separate MLPs
self._user_mlp = Sequential(...)       # For U tokens
self._seq_mlp = Sequential(...)        # For S tokens
self._candidate_mlp = Sequential(...)  # For Candidate tokens
```

### ❌ Problem 3: No Cross Features in Tokens
```python
# HSTU: Cross features in payload_features (line 408-441)
payload_features[candidate_feature_name] = values_right

# MTGR: Cross features should be embedded and MLP'd into candidate tokens
# Candidate token = MLP([Emb(cross_features), Emb(item_features)])
```

### ❌ Problem 4: Masking Strategy
```python
# HSTU: Causal + target-aware (line 228-229)
STULayerConfig(
    causal=True,        # Standard causal masking
    target_aware=True,  # Candidates aware of targets
)

# MTGR: Dynamic masking (Fig 2c in paper)
# - User tokens (U, S): Visible to ALL
# - Real-time tokens (R): Causal within R, visible to later tokens
# - Candidates: Visible ONLY to itself (diagonal)
```

---

## Adaptation Strategy for MTGR

### Option A: Extend DlrmHSTU (Recommended)

**Modifications Needed:**

1. **Add User Feature Processing** (New method)
```python
def _user_feature_forward(self, user_features_kjt):
    """Convert user profile features to tokens"""
    # Embed user_id, avg_rating, avg_review_len
    user_embeddings = self._embedding_collection(user_features_kjt)
    # MLP: concat embeddings -> d_model
    user_tokens = self._user_mlp(user_embeddings)
    return user_tokens  # Shape: [B, num_user_features, d_model]
```

2. **Split Item MLP into Seq MLP + Cand MLP**
```python
def __init__(...):
    # Replace single _item_embedding_mlp with two:
    self._seq_mlp = Sequential(...)      # For S tokens
    self._candidate_mlp = Sequential(...)  # For Candidate tokens (includes cross)
```

3. **Modify preprocess() to Split KJTs**
```python
def preprocess(self, uih_features, candidates_features):
    # Extract user features from uih_features
    user_feature_keys = ['user_id', 'avg_rating', 'avg_review_len']
    seq_feature_keys = ['seq_item_id', 'seq_category', 'seq_brand', 'seq_price']

    user_kjt = uih_features[user_feature_keys]
    seq_kjt = uih_features[seq_feature_keys]

    # Separate embedding lookups
    user_embeddings = self._embedding_collection(user_kjt)
    seq_embeddings = self._embedding_collection(seq_kjt)
    cand_embeddings = self._embedding_collection(candidates_features)

    return user_embeddings, seq_embeddings, cand_embeddings
```

4. **Modify main_forward() to Create 3-Token Sequence**
```python
def main_forward(self, user_emb, seq_emb, cand_emb):
    # Create user tokens
    user_tokens = self._user_mlp(concat_user_embeddings(user_emb))
    # Shape: [B, 3, d_model]  # 3 user features

    # Create sequence tokens
    seq_tokens = self._seq_mlp(concat_seq_embeddings(seq_emb))
    # Shape: [B, seq_len, d_model]

    # Create candidate tokens (with cross features)
    cand_tokens = self._candidate_mlp(concat([cross_emb, cand_emb]))
    # Shape: [B, K, d_model]  # K candidates

    # Concatenate into unified sequence
    all_tokens = torch.cat([user_tokens, seq_tokens, cand_tokens], dim=1)
    # Shape: [B, 3+seq_len+K, d_model]

    # Pass to HSTU with custom mask
    output = self._hstu_transducer(
        seq_embeddings=all_tokens,
        custom_mask=create_mtgr_mask(...)  # NEW: Dynamic masking
    )
```

5. **Implement Group Layer Normalization**
```python
class GroupLayerNorm(nn.Module):
    """Normalize different token types separately"""
    def __init__(self, num_groups, d_model):
        self.layer_norms = nn.ModuleList([
            LayerNorm(d_model) for _ in range(num_groups)
        ])

    def forward(self, x, group_indices):
        # group_indices: [0,0,0, 1,1,1,..., 2,2,2]
        #                 U tokens, S tokens, Cand tokens
        output = torch.zeros_like(x)
        for i, ln in enumerate(self.layer_norms):
            mask = (group_indices == i)
            output[mask] = ln(x[mask])
        return output
```

6. **Implement Dynamic Masking**
```python
def create_mtgr_mask(num_user_tokens, num_seq_tokens, num_cand_tokens):
    """
    Creates MTGR-style mask (Fig 2c):
    - User tokens: Visible to all (full attention column)
    - Seq tokens: Causal (upper triangular)
    - Candidate tokens: Diagonal (only self-attention)
    """
    total_len = num_user_tokens + num_seq_tokens + num_cand_tokens
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)

    # User tokens visible to all
    mask[:, :num_user_tokens] = True

    # Seq tokens: causal
    seq_start = num_user_tokens
    seq_end = seq_start + num_seq_tokens
    for i in range(seq_start, seq_end):
        mask[i, :i+1] = True  # Can see all previous seq tokens + user tokens

    # Candidate tokens: diagonal only
    cand_start = seq_end
    for i in range(num_cand_tokens):
        mask[cand_start + i, cand_start + i] = True

    return mask
```

---

## Changes Summary

### Files to Modify:
1. **`dlrm_hstu.py`**:
   - Add `_user_mlp`, `_seq_mlp`, rename `_item_embedding_mlp` to `_candidate_mlp`
   - Modify `preprocess()` to split user/seq/cand features
   - Modify `main_forward()` to create 3-token structure
   - Add `create_mtgr_mask()` method

2. **`stu.py` (STU layers)**:
   - Modify `STULayer` to accept custom masks
   - Add Group Layer Norm option

3. **`beauty_mtgr.py` (Dataset)**:
   - Already done! ✓
   - Just ensure KJT keys match expected user/seq/cand splits

---

## Next Steps

1. **Create MTGR-specific config**
```python
@dataclass
class MTGRConfig(DlrmHSTUConfig):
    use_mtgr_tokenization: bool = True
    user_feature_names: List[str] = ['user_id', 'avg_rating', 'avg_review_len']
    seq_feature_names: List[str] = ['seq_item_id', 'seq_category', 'seq_brand', 'seq_price']
    cross_feature_names: List[str] = ['brand_count']
    use_group_layer_norm: bool = True
```

2. **Test with toy example** to verify:
   - Embedding lookup works
   - 3 MLPs produce correct token shapes
   - Masking is correct
   - Forward pass completes

3. **Train on Beauty dataset** with our MTGRBeautyDataset

---

## Questions for You

Before I start modifying the code, please clarify:

1. **Should I extend DlrmHSTU** (Option A) or **create a new MTGRModel** class?
2. **Do we want to skip R (real-time) tokens** for now since Beauty data doesn't have timestamps?
3. **Which components should I implement first?**
   - A) Just tokenization (3 MLPs)
   - B) Tokenization + masking
   - C) Full MTGR (tokenization + masking + Group LN)
