# Trained Models

This directory contains the best-performing models for property address classification.

## Overview

| Model | Accuracy | Macro F1 | Size | Status |
|-------|----------|----------|------|--------|
| SVM Classifier | 91.98% | 0.9075 | 605 KB |  Included |
| IndicBERT | **92.88%** | **0.9195** | 1.1 GB | External |

## Models

### 1. SVM Classifier (Baseline)
- **File**: `svm_classifier_v2.pkl` (605 KB)
- **Performance**: 91.98% accuracy, 0.9075 macro F1
- **Status**: **Included in repository**
- **Loading**: Simple pickle load
- **Testing**: See `testing/svm_classifier_testing.ipynb`

**Quick Load:**
```python
import cloudpickle

with open('best_models/svm_classifier_v2.pkl', 'rb') as f:
    pipeline = cloudpickle.load(f)
```

---

### 2. IndicBERT Classifier (Best Model)
- **Directory**: `indicbert_property_classifier/`
- **Performance**: 92.88% accuracy, 0.9195 macro F1
- **Status**: **Model weights hosted externally (1.1 GB)**
- **Included Files**: Tokenizer, config, labels (all small files)
- **Testing**: See `testing/indicbert_testing.ipynb`

**Download Model Weights:**
See `indicbert_property_classifier/MODEL_INFO.md` for Google Drive link.

**Quick Load:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "best_models/indicbert_property_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
```

---

## Why External Hosting?

The IndicBERT model weights (`model.safetensors`) are **1.1 GB**, which:
- Exceeds Git's recommended file size limits (50 MB warning, 100 MB hard limit)
- Would slow down repository cloning and operations
- Is not suitable for version control

**Solution**: Model weights are hosted on Google Drive, while all supporting files (tokenizer, config, labels) are included in the repository.

---

## Reviewing Models Without Downloading

### For Reviewers: Quick Evaluation

Both models have **fully executed testing notebooks** with all results pre-computed:

1. **SVM Testing**: `testing/svm_classifier_testing.ipynb`
   - All metrics, confusion matrix, error analysis
   - Custom predictions, visualizations
   - No download required

2. **IndicBERT Testing**: `testing/indicbert_testing.ipynb`
   - All metrics, confusion matrix, error analysis
   - Custom predictions, confidence analysis
   - No download required

**You can fully evaluate both models by viewing the executed notebooks - no model download needed!**

---

## Directory Structure

```
best_models/
├── README.md                               # This file
├── svm_classifier_v2.pkl                   # SVM model (605 KB)
└── indicbert_property_classifier/
    ├── MODEL_INFO.md                       # Download instructions
    ├── config.json                         # Model config
    ├── tokenizer.json                      # Tokenizer (7.7 MB)
    ├── tokenizer_config.json               # Tokenizer config
    ├── special_tokens_map.json             # Special tokens
    ├── training_args.bin                   # Training metadata
    └── model.safetensors                   # DOWNLOAD FROM DRIVE ⚠️
```

---

## Performance Comparison

### Overall Metrics
| Metric | SVM | IndicBERT | Winner |
|--------|-----|-----------|--------|
| Accuracy | 91.98% | 92.88% | IndicBERT |
| Macro F1 | 0.9075 | 0.9195 | IndicBERT |
| Weighted F1 | 0.9204 | 0.9289 | IndicBERT |
| Inference Speed | Fast | Slow | SVM |
| Model Size | Tiny | Large | SVM |

### Per-Class F1 Scores
| Class | SVM | IndicBERT | Better |
|-------|-----|-----------|--------|
| commercial unit | 0.9672 | 0.9810 | +0.0138 |
| flat | 0.9581 | 0.9585 | +0.0004 |
| houseorplot | 0.9102 | 0.9167 | +0.0065 |
| landparcel | 0.8413 | 0.8511 | +0.0098 |
| others | 0.8610 | 0.8901 | +0.0291 |

**Key Insight**: IndicBERT improves most on challenging classes (others, landparcel).

---

## Recommendation

**For Production/Deployment:**
- Use **SVM** if speed and size matter (real-time API, edge devices)
- Use **IndicBERT** if accuracy is critical and resources available

**For This Submission:**
- **IndicBERT** is the best-performing model and demonstrates advanced ML capabilities

---

## Quick Start

### Test Both Models
```bash
# Navigate to testing directory
cd testing/

# Run SVM testing
jupyter notebook svm_classifier_testing.ipynb

# Run IndicBERT testing (requires model download)
jupyter notebook indicbert_testing.ipynb
```

### Make Predictions

**SVM:**
```python
import cloudpickle

with open('best_models/svm_classifier_v2.pkl', 'rb') as f:
    svm_model = cloudpickle.load(f)

prediction = svm_model.predict(["Flat No. 301, Tower A"])
print(prediction)
```

**IndicBERT:**
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="best_models/indicbert_property_classifier"
)

result = classifier("Flat No. 301, Tower A")
print(result)
```

---

## Notes
- Both models trained on the same dataset (8,936 training, 2,681 validation)
- Both use the same 5 categories: commercial unit, flat, houseorplot, landparcel, others
- Testing notebooks contain complete evaluation and error analysis
