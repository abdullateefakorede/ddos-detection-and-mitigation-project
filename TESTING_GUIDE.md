# Testing Guide: DDoS Detection Ensemble Project

## Overview
Your project is a **cybersecurity deep learning ensemble** for DDoS detection. Good news: **You have trained models ready to test!** You don't need to retrain them.

---

## ✅ What You Have

### Pre-trained Models (in `models/` directory)
1. **DBNNetwork_1758192440.pt** - Deep Belief Network
2. **DDoSTransformer_1758192530.pt** - Transformer-based model
3. **LSTMNetwork_1758192370_minimal.pt** - LSTM Network
4. **RNNClassifier_1758192485.pt** - RNN Classifier
5. **TimeSeriesAutoEncoder_1758192664.pt** - AutoEncoder model

### Test Datasets (in `artifact/` directory)
- **ddos_dataset.csv** - 225,745 samples for DDoS testing
- **github_dataset.csv** - GitHub security dataset (~50MB)

### Prediction Results (ready for comparison)
- `ensemble_data/` - Contains multiple ensemble configurations with predictions
- `github_prediction/` - Predictions on GitHub dataset with different model weights
- `data/` - Individual model predictions and history

---

## 🚀 Quick Start: Test Without Retraining

### Option 1: Run the Ensemble Testing Notebook (Easiest)
```bash
# Open cyber_ensemble.ipynb and run cells to:
1. Load saved models from models/ directory
2. Load test data from artifact/
3. Generate ensemble predictions
4. Calculate metrics (accuracy, F1, precision, recall, etc.)
5. Save results to ensemble_data/
```

**Key cells in the notebook:**
- Lists available models with `ModelEnsemble.list_models("models")`
- Creates ensemble: `ModelEnsemble.from_directory("models", [1,3,5])`
- Evaluates: `ensemble.predict(test_loader)`
- Outputs CSVs with predictions and metrics

---

### Option 2: Use the `ensemble.py` Python Script Directly
```python
from ensemble import ModelEnsemble
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch

# 1. Load test data
df = pd.read_csv("artifact/ddos_dataset.csv")
X = df.drop("Label", axis=1).values.astype(np.float32)
y = df["Label"].values.astype(np.int32)
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
test_loader = DataLoader(dataset, batch_size=32)

# 2. List available models
ModelEnsemble.list_models("models")

# 3. Create ensemble with specific models
# Example: Select models 1, 3, 5 with custom weights
ensemble = ModelEnsemble.from_directory(
    models_dir="models",
    model_indices=[1, 3, 5],
    weights=[0.5, 0.3, 0.2]
)

# 4. Get predictions and metrics
results = ensemble.predict(test_loader, save_csv=True)

# 5. View results
print("Ensemble Accuracy:", results['ensemble_metrics']['accuracy'])
print("F1 Score:", results['ensemble_metrics']['f1_score'])
```

---

### Option 3: Test Individual Models
```python
# Load a single model checkpoint
checkpoint = torch.load("models/DDoSTransformer_1758192530.pt", map_location="cpu")
print("Model saved with:", checkpoint.keys())
print("Architecture:", checkpoint['model_config'])
print("State dict keys:", list(checkpoint['model_state_dict'].keys())[:10])
```

---

## 📊 What Gets Tested

The ensemble testing evaluates:
- **Accuracy** - Overall correctness
- **F1 Score** - Harmonic mean of precision/recall
- **Precision** - True positives / (True positives + False positives)
- **Recall/Sensitivity** - True positives / (True positives + False negatives)
- **Specificity** - True negatives / (True negatives + False positives)
- **Cohen's Kappa** - Agreement accounting for chance
- **ROC-AUC** - Area under the ROC curve
- **PR-AUC** - Precision-Recall AUC
- **Confusion Matrix** - TP, TN, FP, FN breakdown

---

## 📁 Output Structure

When you run testing, it creates:

```
ensemble_data/
└── DBNNetwork-0.333_DDoSTransformer-0.333_LSTMNetwork-0.333/
    ├── predictions.csv       # Sample-level predictions with probabilities
    ├── metrics.csv          # Accuracy, F1, precision, recall, etc.
    ├── config.csv           # Model weights and configurations
    └── summary.csv          # High-level statistics
```

---

## 🎯 Testing Scenarios You Can Try

### Scenario 1: Equal Ensemble
All 5 models with equal weights (20% each)
```python
ensemble = ModelEnsemble.from_directory("models")  # All models
```

### Scenario 2: Weighted Ensemble
Different models contribute different amounts
```python
ensemble = ModelEnsemble.from_directory(
    "models",
    model_indices=[1, 3, 5],
    weights=[0.5, 0.3, 0.2]  # Custom weights
)
```

### Scenario 3: Best Models Only
Select only top-performing models
```python
ensemble = ModelEnsemble.from_directory(
    "models",
    model_indices=[2, 5]  # Just Transformer and AutoEncoder
)
```

### Scenario 4: Cross-Dataset Testing
Test on GitHub dataset instead of DDoS dataset
```python
df = pd.read_csv("artifact/github_dataset.csv")
# ... run same ensemble test
```

---

## 🔧 Key Files to Use

| File | Purpose |
|------|---------|
| `ensemble.py` | Main ensemble testing class |
| `cyber_ensemble.ipynb` | Notebook with full testing workflow |
| `cyber-security-dl.ipynb` | Model training (reference) |
| `models/*.pt` | Pre-trained model weights |
| `artifact/*.csv` | Test datasets |

---

## ⚡ Performance Tips

1. **GPU Usage**: Models will auto-detect CUDA. Testing should be fast (seconds to minutes)
2. **Memory**: Make sure 4GB+ available for batch processing
3. **Data Size**: Use batches of 32-64 for optimal speed
4. **Caching**: Notebook has `%%cache` magic - reuse cached results to skip recomputation

---

## ✨ Special Features in Your Setup

1. **Model Auto-Discovery** - Automatically finds and lists all `.pt` files
2. **Deduplication** - Prefers `ultra_clean` > `minimal` > original versions
3. **Weight Normalization** - Automatically normalizes weights to sum to 1.0
4. **CSV Organization** - Results sorted alphabetically by model name
5. **Error Handling** - Robust checkpoint loading with multiple fallback approaches

---

## 🐛 Troubleshooting

### Issue: "Models not found"
```python
# Check what's in models directory
import os
print(os.listdir("models"))
```

### Issue: "Model loading fails"
The ensemble handles this automatically by:
1. Trying standard loading
2. Trying CPU loading
3. Trying with `weights_only=False`
4. Trying fallback versions (_minimal, _ultra_clean)

### Issue: "Out of memory"
- Reduce batch size: `DataLoader(dataset, batch_size=16)`
- Test on smaller subset: `dataset.subset(range(10000))`

---

## 📈 What's Already Evaluated

Your `ensemble_data/` and `github_prediction/` folders show previous test runs:
- Multiple model combinations tested
- Different weight configurations evaluated
- Results from both datasets available

**You can compare new results against these existing ones!**

---

## 🎓 Next Steps

1. **Run Quick Test**: `cyber_ensemble.ipynb` → Run all cells
2. **Review Results**: Check `ensemble_data/` for outputs
3. **Try Variations**: Test different model combinations
4. **Analyze Performance**: Compare metrics across configurations
5. **Optimize**: Adjust weights to maximize performance

---

## 📝 Notes

- **No training required** - Models are frozen, only used for inference
- **Deterministic results** - Same test data → same predictions
- **Fast testing** - Should complete in seconds to minutes
- **Easy comparison** - CSV outputs make results easy to analyze
