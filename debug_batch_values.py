"""
Debug script to inspect actual values in Beauty dataset batch.
This helps identify which features have out-of-range values.
"""

import pickle
from generative_recommenders.dlrm_v3.datasets.beauty_mtgr import MTGRBeautyDataset
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn
from torch.utils.data import DataLoader

# Load dataset
dataset = MTGRBeautyDataset(
    data_file="data/Beauty_train.pkl",
    item_features_file="data/Beauty_item_feature_map.pkl",
    num_negatives=4,
    max_seq_len=100,
)

print(f"Dataset info:")
print(f"  Users: {len(dataset.users)}")
print(f"  Items: {len(dataset.all_item_ids)}")
print(f"  Max user ID: {max(int(u['user_id']) for u in dataset.users)}")
print(f"  Max item ID: {max(dataset.all_item_ids)}")
print()

# Get first sample
print("Getting first sample...")
uih_kjt, cand_kjt = dataset[0]

print("\nUIH KJT:")
print(f"  Keys: {uih_kjt.keys()}")
print(f"  Lengths: {uih_kjt.lengths()}")
print(f"  Values min: {uih_kjt.values().min().item()}")
print(f"  Values max: {uih_kjt.values().max().item()}")
print(f"  Values shape: {uih_kjt.values().shape}")

print("\nCandidates KJT:")
print(f"  Keys: {cand_kjt.keys()}")
print(f"  Lengths: {cand_kjt.lengths()}")
print(f"  Values min: {cand_kjt.values().min().item()}")
print(f"  Values max: {cand_kjt.values().max().item()}")
print(f"  Values shape: {cand_kjt.values().shape}")

# Check per-feature values
print("\n" + "="*80)
print("PER-FEATURE VALUE RANGES:")
print("="*80)

# UIH features
offset = 0
for i, key in enumerate(uih_kjt.keys()):
    length = uih_kjt.lengths()[i].item()
    values = uih_kjt.values()[offset:offset+length]
    print(f"\n{key}:")
    print(f"  Length: {length}")
    if length > 0:
        print(f"  Min: {values.min().item()}")
        print(f"  Max: {values.max().item()}")
        print(f"  Sample values: {values[:5].tolist()}")
    offset += length

# Candidate features
print("\n" + "-"*80)
print("CANDIDATE FEATURES:")
print("-"*80)

offset = 0
for i, key in enumerate(cand_kjt.keys()):
    length = cand_kjt.lengths()[i].item()
    values = cand_kjt.values()[offset:offset+length]
    print(f"\n{key}:")
    print(f"  Length: {length}")
    if length > 0:
        print(f"  Min: {values.min().item()}")
        print(f"  Max: {values.max().item()}")
        print(f"  Sample values: {values[:min(5, length)].tolist()}")
    offset += length

# Now create a batch
print("\n" + "="*80)
print("BATCH VALUES:")
print("="*80)

dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
batch = next(iter(dataloader))

print(f"\nBatch UIH values:")
print(f"  Min: {batch.uih_features_kjt.values().min().item()}")
print(f"  Max: {batch.uih_features_kjt.values().max().item()}")

print(f"\nBatch Candidate values:")
print(f"  Min: {batch.candidates_features_kjt.values().min().item()}")
print(f"  Max: {batch.candidates_features_kjt.values().max().item()}")

# Check specific features in batch
print("\n" + "="*80)
print("RECOMMENDED EMBEDDING TABLE SIZES:")
print("="*80)

# Analyze all samples to get true max values
print("\nAnalyzing first 100 samples for accurate vocab sizes...")
max_values = {}

for idx in range(min(100, len(dataset))):
    uih, cand = dataset[idx]

    # UIH features
    offset = 0
    for i, key in enumerate(uih.keys()):
        length = uih.lengths()[i].item()
        if length > 0:
            values = uih.values()[offset:offset+length]
            max_val = values.max().item()
            max_values[key] = max(max_values.get(key, 0), max_val)
        offset += length

    # Candidate features
    offset = 0
    for i, key in enumerate(cand.keys()):
        length = cand.lengths()[i].item()
        if length > 0:
            values = cand.values()[offset:offset+length]
            max_val = values.max().item()
            max_values[key] = max(max_values.get(key, 0), max_val)
        offset += length

print("\nMax values observed across 100 samples:")
for key, max_val in sorted(max_values.items()):
    recommended_size = max_val + 1
    print(f"  {key:20s}: max={max_val:10d} → num_embeddings={recommended_size}")
