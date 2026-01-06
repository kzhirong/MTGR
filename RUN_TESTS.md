# Quick Test Guide for MTGR Implementation

## TL;DR - Run This on GPU Machine

```bash
# Install dependencies (if not already installed)
pip install torch torchrec fbgemm_gpu

# Run the main end-to-end test
python test_mtgr_end_to_end.py
```

Expected result: All 6 steps should pass with ✓ marks.

---

## All Available Tests

### Test 1: Dataset Only
**File**: `test_mtgr_dataset.py`
**Tests**: MTGRBeautyDataset in isolation
**Runtime**: ~5 seconds

```bash
python test_mtgr_dataset.py
```

### Test 2: Tokenization (Synthetic Data)
**File**: `test_mtgr_tokenization.py`
**Tests**: MTGR tokenization with fake data
**Runtime**: ~10 seconds

```bash
python test_mtgr_tokenization.py
```

### Test 3: End-to-End (Real Data) ⭐
**File**: `test_mtgr_end_to_end.py`
**Tests**: Complete pipeline with Beauty_train.pkl
**Runtime**: ~30 seconds

```bash
python test_mtgr_end_to_end.py
```

---

## What Each Test Verifies

### Test 1 (Dataset)
- ✓ Beauty data files load correctly
- ✓ MTGRBeautyDataset initialization
- ✓ __getitem__ returns valid KJTs
- ✓ KJT structure matches expected format

### Test 2 (Tokenization)
- ✓ MTGR config creation
- ✓ DlrmHSTU initializes with 3 MLPs
- ✓ Synthetic batch processing
- ✓ Forward pass completes
- ✓ Output shapes are correct

### Test 3 (End-to-End) ⭐
- ✓ Real data loading
- ✓ Single sample retrieval
- ✓ DataLoader batching
- ✓ Model initialization
- ✓ Forward pass with real batch
- ✓ Token count verification

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'fbgemm_gpu'`
**Cause**: Running on CPU/macOS environment
**Solution**: Run on Linux machine with CUDA GPU

```bash
# On GPU machine
pip install fbgemm_gpu
```

### Error: `FileNotFoundError: Beauty_train.pkl`
**Cause**: Data files not in expected location
**Solution**: Update file paths in test files

```python
# In test_mtgr_end_to_end.py, line 88
data_file="/path/to/your/Beauty_train.pkl",
datamaps_file="/path/to/your/Beauty_datamaps.pkl",
item_features_file="/path/to/your/Beauty_item_feature_map.pkl",
```

### Error: Shape mismatch
**Cause**: Config doesn't match dataset structure
**Solution**: Check that feature names in config match dataset keys

```python
# Should match dataset._uih_keys and dataset._candidates_keys
print(f"UIH keys: {dataset._uih_keys}")
print(f"Candidate keys: {dataset._candidates_keys}")
```

---

## Expected Output Snippets

### Successful Dataset Test
```
Loading Beauty dataset...
✓ Dataset loaded with XXXXX users
✓ Sample 0 retrieved successfully
✓ UIH KJT has 7 features
✓ Candidate KJT has 5 features
```

### Successful Tokenization Test
```
Testing MTGR Tokenization Integration
✓ Config created with use_mtgr_tokenization=True
✓ Model initialized successfully
✓ MTGR MLPs created: user_mlp, seq_mlp, candidate_mlp
✓ Forward pass completed successfully!
```

### Successful End-to-End Test
```
MTGR END-TO-END TEST WITH REAL BEAUTY DATA
✓ Dataset loading: PASS
✓ Sample retrieval: PASS
✓ DataLoader batching: PASS
✓ Model initialization: PASS
✓ Forward pass (tokenization): PASS

✅ END-TO-END TEST PASSED!
```

---

## What to Do If Tests Fail

1. **Check dependencies**
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import torchrec; print(torchrec.__version__)"
   python -c "import fbgemm_gpu; print('fbgemm_gpu installed')"
   ```

2. **Check data files exist**
   ```bash
   ls -lh data/Beauty*.pkl
   ```

3. **Check syntax**
   ```bash
   python -m py_compile test_mtgr_end_to_end.py
   ```

4. **Run with verbose logging**
   ```bash
   python test_mtgr_end_to_end.py 2>&1 | tee test_output.log
   ```

5. **Check full error trace**
   - Error messages include full stack trace
   - Look for line numbers in dlrm_hstu.py or beauty_mtgr.py

---

## Quick Validation (No Runtime)

If you don't have GPU access yet, you can still validate:

```bash
# Syntax check all test files
python -m py_compile test_mtgr_dataset.py
python -m py_compile test_mtgr_tokenization.py
python -m py_compile test_mtgr_end_to_end.py

# All should complete without errors
echo "✓ All syntax checks passed"
```

---

## Integration with Main Training

Once tests pass, integrate with main training:

```python
# In your training script
from generative_recommenders.dlrm_v3.datasets.beauty_mtgr import MTGRBeautyDataset
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, DlrmHSTU

# Create dataset
train_dataset = MTGRBeautyDataset(
    data_file="data/Beauty_train.pkl",
    datamaps_file="data/Beauty_datamaps.pkl",
    item_features_file="data/Beauty_item_feature_map.pkl",
    num_negatives=4,
    max_seq_len=100,
)

# Create model
config = DlrmHSTUConfig(
    use_mtgr_tokenization=True,
    mtgr_user_feature_names=train_dataset._uih_keys[:3],
    mtgr_seq_feature_names=train_dataset._uih_keys[3:],
    mtgr_cross_feature_names=[train_dataset._candidates_keys[0]],
    mtgr_candidate_feature_names=train_dataset._candidates_keys[1:],
    # ... other config
)

model = DlrmHSTU(config, embedding_tables, is_inference=False)

# Training loop
for batch in train_dataloader:
    outputs = model(batch.uih_features_kjt, batch.candidates_features_kjt)
    # ... rest of training logic
```

---

## Files Location

```
generative-recommenders/
├── data/
│   ├── Beauty_train.pkl
│   ├── Beauty_datamaps.pkl
│   └── Beauty_item_feature_map.pkl
├── generative_recommenders/
│   ├── modules/
│   │   └── dlrm_hstu.py (modified)
│   └── dlrm_v3/
│       └── datasets/
│           └── beauty_mtgr.py (new)
├── test_mtgr_dataset.py (new)
├── test_mtgr_tokenization.py (new)
├── test_mtgr_end_to_end.py (new)
├── TEST_STATUS.md (new)
└── RUN_TESTS.md (this file)
```

---

**Ready to test!** 🚀

Just run `python test_mtgr_end_to_end.py` on a GPU machine.
