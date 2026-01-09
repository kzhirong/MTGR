# Architecture Flow: HSTU vs MTGR

## Current HSTU Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: 2 KJTs                                │
├─────────────────────────────────────────────────────────────────┤
│  uih_features_kjt              candidates_features_kjt          │
│  ├─ uih_post_id                ├─ cand_item_id                  │
│  ├─ uih_ratings                ├─ cand_ratings                  │
│  ├─ uih_timestamps             ├─ cand_timestamps               │
│  └─ uih_weight                 └─ cand_weight                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: MERGE & EMBEDDING LOOKUP                    │
├─────────────────────────────────────────────────────────────────┤
│  merged_kjt = concat(uih_kjt, cand_kjt)                         │
│  seq_embeddings = EmbeddingCollection(merged_kjt)               │
│                                                                  │
│  Output: Dict[feature_name, JaggedTensor]                       │
│    - uih_post_id: [L_uih, 192]                                 │
│    - cand_item_id: [L_cand, 192]                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        STEP 2: CONCATENATE UIH + CANDIDATES EMBEDDINGS          │
├─────────────────────────────────────────────────────────────────┤
│  merged_embeddings = concat_2D_jagged(                          │
│      uih_embeddings,   # [L_uih, 192]                          │
│      cand_embeddings   # [L_cand, 192]                         │
│  )                                                               │
│  → Shape: [L_uih + L_cand, 192]                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3: TOKENIZATION VIA SINGLE MLP                    │
├─────────────────────────────────────────────────────────────────┤
│  item_tokens = _item_embedding_mlp(merged_embeddings)           │
│                                                                  │
│  MLP: Linear(192*num_features, 512)                             │
│       → SwishLayerNorm(512)                                     │
│       → Linear(512, d_model)                                    │
│       → LayerNorm(d_model)                                      │
│                                                                  │
│  Output: [L_uih + L_cand, d_model]                             │
│          ^^^ ALL tokens treated the same ^^^                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: HSTU SELF-ATTENTION                         │
├─────────────────────────────────────────────────────────────────┤
│  user_embeddings = HSTUTransducer(                              │
│      seq_embeddings=item_tokens,                                │
│      seq_timestamps=timestamps,                                 │
│      contextual_features=...,  # Separate pathway!             │
│  )                                                               │
│                                                                  │
│  Masking: Causal + Target-Aware                                 │
│    - UIH can see previous UIH items                             │
│    - Candidates can see all UIH + itself                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## MTGR Flow (What We Need)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: 2 KJTs                                │
├─────────────────────────────────────────────────────────────────┤
│  uih_features_kjt              candidates_features_kjt          │
│  ├─ USER (U):                  ├─ CROSS (C):                    │
│  │  ├─ user_id                 │  └─ brand_count                │
│  │  ├─ avg_rating              │                                 │
│  │  └─ avg_review_len          ├─ ITEM (I):                     │
│  │                              │  ├─ cand_item_id               │
│  └─ SEQUENCE (S):               │  ├─ cand_category              │
│     ├─ seq_item_id              │  ├─ cand_brand                 │
│     ├─ seq_category             │  └─ cand_price                 │
│     ├─ seq_brand                │                                 │
│     └─ seq_price                │                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 1: SEPARATE EMBEDDING LOOKUP (3 GROUPS)             │
├─────────────────────────────────────────────────────────────────┤
│  user_kjt = extract_keys(uih_kjt, ['user_id', 'avg_rating', ...])│
│  seq_kjt = extract_keys(uih_kjt, ['seq_item_id', 'seq_category',│
│                                   'seq_brand', 'seq_price'])     │
│  cand_kjt = candidates_features_kjt  # Already separate         │
│                                                                  │
│  user_embeddings = EmbeddingCollection(user_kjt)                │
│  seq_embeddings = EmbeddingCollection(seq_kjt)                  │
│  cand_embeddings = EmbeddingCollection(cand_kjt)                │
│                                                                  │
│  Output:                                                         │
│    - user_embeddings: Dict[3 features, [B, 192]]               │
│    - seq_embeddings: Dict[4 features, [B, seq_len, 192]]       │
│    - cand_embeddings: Dict[5 features, [B, K, 192]]            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│      STEP 2: TOKENIZATION VIA 3 SEPARATE MLPs                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ USER MLP                                                 │   │
│  │ user_concat = concat([emb(user_id), emb(avg_rating),    │   │
│  │                       emb(avg_review_len)])              │   │
│  │ user_tokens = UserMLP(user_concat)                      │   │
│  │ Shape: [B, 3, d_model]                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SEQUENCE MLP                                             │   │
│  │ For each item i in sequence:                             │   │
│  │   seq_concat[i] = concat([emb(item_id[i]),              │   │
│  │                           emb(category[i]),              │   │
│  │                           emb(brand[i]),                 │   │
│  │                           emb(price[i])])                │   │
│  │ seq_tokens = SeqMLP(seq_concat)                         │   │
│  │ Shape: [B, seq_len, d_model]                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CANDIDATE MLP                                            │   │
│  │ For each candidate k in [1..K]:                          │   │
│  │   cand_concat[k] = concat([emb(brand_count[k]),         │   │
│  │                            emb(item_id[k]),              │   │
│  │                            emb(category[k]),             │   │
│  │                            emb(brand[k]),                │   │
│  │                            emb(price[k])])               │   │
│  │ cand_tokens = CandMLP(cand_concat)                      │   │
│  │ Shape: [B, K, d_model]                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3: CONCATENATE 3 TOKEN TYPES                      │
├─────────────────────────────────────────────────────────────────┤
│  all_tokens = concat([                                           │
│      user_tokens,    # [B, 3, d_model]                          │
│      seq_tokens,     # [B, seq_len, d_model]                    │
│      cand_tokens     # [B, K, d_model]                          │
│  ], dim=1)                                                       │
│                                                                  │
│  group_indices = [0,0,0, 1,1,1,...,1, 2,2,...,2]               │
│                   └─U─┘ └────S────┘ └───C───┘                  │
│                                                                  │
│  Output: [B, 3 + seq_len + K, d_model]                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│       STEP 4: GROUP LAYER NORM (Before HSTU)                     │
├─────────────────────────────────────────────────────────────────┤
│  normed_tokens = GroupLayerNorm(all_tokens, group_indices)     │
│                                                                  │
│  For each group g in [U, S, C]:                                 │
│      tokens[group==g] = LayerNorm_g(tokens[group==g])          │
│                                                                  │
│  → Separate normalization statistics per token type            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        STEP 5: HSTU SELF-ATTENTION WITH DYNAMIC MASK             │
├─────────────────────────────────────────────────────────────────┤
│  Masking Strategy (Fig 2c in paper):                            │
│                                                                  │
│         age ctr seq1 seq2 target1 target2 target3               │
│  age    [ 1   1    1    1     1       1       1   ]             │
│  ctr    [ 1   1    1    1     1       1       1   ]             │
│  seq1   [ 1   1    1    1     1       1       1   ]             │
│  seq2   [ 1   1    1    1     1       1       1   ]             │
│  target1[ 1   1    1    1     1       0       0   ]  ← Diagonal│
│  target2[ 1   1    1    1     0       1       0   ]  ← Diagonal│
│  target3[ 1   1    1    1     0       0       1   ]  ← Diagonal│
│           └─────U────────┘└────S────┘└──────C──────┘           │
│           Full attention  Full attn  Self-only                  │
│                                                                  │
│  Rules:                                                          │
│  1. U tokens: Visible to ALL (full column of 1s)               │
│  2. S tokens: Visible to ALL (full column of 1s)               │
│  3. C tokens: Visible ONLY to itself (diagonal)                │
│                                                                  │
│  output = HSTUTransducer(                                       │
│      seq_embeddings=normed_tokens,                              │
│      custom_mask=mtgr_mask,                                     │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT                                       │
├─────────────────────────────────────────────────────────────────┤
│  Extract candidate outputs:                                      │
│    cand_outputs = output[:, -(K):, :]  # Last K tokens          │
│                                                                  │
│  Predictions = MLP(cand_outputs)                                │
│    → Shape: [B, K, num_tasks]                                   │
│                                                                  │
│  During training:                                                │
│    - First token (pos) has label = 1                            │
│    - Rest K-1 tokens (neg) have label = 0                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Architecture Differences

### Token Creation

**HSTU:**
```python
# Single path for all items
embeddings = concat([uih_emb, cand_emb])  # [L_total, 192]
tokens = item_mlp(embeddings)              # [L_total, d_model]
```

**MTGR:**
```python
# Three separate paths
user_tokens = user_mlp(user_emb)        # [B, 3, d_model]
seq_tokens = seq_mlp(seq_emb)           # [B, N, d_model]
cand_tokens = cand_mlp(cand_emb)        # [B, K, d_model]

# Then concatenate
all_tokens = concat([user_tokens, seq_tokens, cand_tokens])
```

### Attention Masking

**HSTU:**
```
Causal + Target-Aware:
         uih1 uih2 uih3 cand1 cand2
  uih1   [ 1    0    0     0     0  ]
  uih2   [ 1    1    0     0     0  ]
  uih3   [ 1    1    1     0     0  ]
  cand1  [ 1    1    1     1     0  ]  ← Can see all UIH
  cand2  [ 1    1    1     1     1  ]  ← Can see all UIH + prev cand
```

**MTGR:**
```
Dynamic Masking:
         U    S1   S2   C1   C2   C3
  U      [ 1    1    1    1    1    1  ]  ← User visible to all
  S1     [ 1    1    1    1    1    1  ]  ← Seq visible to all
  S2     [ 1    1    1    1    1    1  ]  ← Seq visible to all
  C1     [ 1    1    1    1    0    0  ]  ← Cand sees U,S,self only
  C2     [ 1    1    1    0    1    0  ]  ← Cand sees U,S,self only
  C3     [ 1    1    1    0    0    1  ]  ← Cand sees U,S,self only
```

---

## Code Modification Checklist

- [ ] Split `preprocess()` to extract U, S, C separately from KJTs
- [ ] Add `_user_mlp` for User tokenization
- [ ] Add `_seq_mlp` for Sequence tokenization
- [ ] Rename `_item_embedding_mlp` to `_candidate_mlp` and handle cross features
- [ ] Implement `create_mtgr_mask()` for dynamic masking
- [ ] Add GroupLayerNorm to HSTU layers
- [ ] Modify `main_forward()` to create 3-token sequence
- [ ] Update dataset to match expected KJT format (✓ Already done!)
