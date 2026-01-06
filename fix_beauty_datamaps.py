"""
Fix Beauty_datamaps.pkl to match the preprocessed integer IDs in Beauty_train.pkl

The current datamaps file contains ASIN->int mappings, but the training data
already uses integer IDs. This script creates a corrected datamaps file.
"""

import pickle
from pathlib import Path

def fix_beauty_datamaps(
    train_file: str = "data/Beauty_train.pkl",
    val_file: str = "data/Beauty_val.pkl",
    test_file: str = "data/Beauty_test.pkl",
    item_features_file: str = "data/Beauty_item_feature_map.pkl",
    output_file: str = "data/Beauty_datamaps_fixed.pkl",
):
    """
    Create corrected datamaps for Beauty dataset.

    The training data already uses integer IDs (as strings), so datamaps
    should be a simple identity mapping, not ASIN->int.
    """
    print("Loading all splits to determine user/item vocabulary...")

    # Load all splits to get complete user/item sets
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)

    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)

    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)

    # Collect all user IDs across splits
    all_users = set()
    for split in [train_data, val_data, test_data]:
        for user in split:
            all_users.add(user['user_id'])

    # Collect all item IDs across splits
    all_items = set()
    for split in [train_data, val_data, test_data]:
        for user in split:
            all_items.update(user['sequential_features']['full_item_ids'])

    print(f"Found {len(all_users)} unique users")
    print(f"Found {len(all_items)} unique items")

    # Load item features to ensure coverage
    with open(item_features_file, 'rb') as f:
        item_features = pickle.load(f)

    print(f"Item features file has {len(item_features)} items")

    # Create identity mappings (string -> int)
    user2id = {user_id_str: int(user_id_str) for user_id_str in all_users}
    item2id = {item_id_str: int(item_id_str) for item_id_str in item_features.keys()}
    id2item = {v: k for k, v in item2id.items()}

    # Verify coverage
    print(f"\nVerifying coverage...")
    print(f"  Users: {len(user2id)} mappings")
    print(f"  User ID range: [{min(user2id.values())}, {max(user2id.values())}]")
    print(f"  Items: {len(item2id)} mappings")
    print(f"  Item ID range: [{min(item2id.values())}, {max(item2id.values())}]")

    # Check for missing items in training data
    missing_in_features = all_items - set(item_features.keys())
    if missing_in_features:
        print(f"\n⚠️  WARNING: {len(missing_in_features)} items in training data not in item_features!")
        print(f"  First 10: {list(missing_in_features)[:10]}")

    # Save corrected datamaps
    datamaps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2item': id2item,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(datamaps, f)

    print(f"\n✅ Fixed datamaps saved to {output_file}")
    print(f"\nSummary:")
    print(f"  - user2id: {len(user2id)} users (identity mapping)")
    print(f"  - item2id: {len(item2id)} items (identity mapping)")
    print(f"  - id2item: {len(id2item)} reverse mappings")

    return datamaps


if __name__ == "__main__":
    datamaps = fix_beauty_datamaps()

    # Verify the fix
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Test with some sample IDs
    test_user_ids = ['5715', '1388', '11980']
    test_item_ids = ['2762', '1917', '10589']

    print("\nUser ID mappings:")
    for uid in test_user_ids:
        if uid in datamaps['user2id']:
            print(f"  '{uid}' -> {datamaps['user2id'][uid]} ✓")
        else:
            print(f"  '{uid}' -> NOT FOUND ✗")

    print("\nItem ID mappings:")
    for iid in test_item_ids:
        if iid in datamaps['item2id']:
            print(f"  '{iid}' -> {datamaps['item2id'][iid]} ✓")
        else:
            print(f"  '{iid}' -> NOT FOUND ✗")

    print("\n✅ Datamaps fixed successfully!")
    print("\nNext steps:")
    print("  1. Replace data/Beauty_datamaps.pkl with data/Beauty_datamaps_fixed.pkl")
    print("  2. Revert beauty_mtgr.py to use datamaps directly")
    print("  3. Test the pipeline again")
