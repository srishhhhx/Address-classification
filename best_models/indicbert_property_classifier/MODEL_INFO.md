# IndicBERT Property Address Classifier

## Model Details
- **Base Model**: ai4bharat/IndicBERTv2-MLM-only
- **Task**: Multi-class classification (5 categories)
- **Parameters**: 278,045,189
- **Model Size**: 1.1 GB
- **Training**: Fine-tuned with weighted loss for class imbalance
- **Max Sequence Length**: 112 tokens

## Performance (Validation Set - 2,681 samples)
- **Accuracy**: 92.88%
- **Macro F1**: 0.9195
- **Weighted F1**: 0.9289
- **Macro Precision**: 0.9189
- **Macro Recall**: 0.9201

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 98.27% | 97.93% | 98.10% |
| flat | 96.35% | 95.36% | 95.85% |
| houseorplot | 91.44% | 91.90% | 91.67% |
| landparcel | 85.60% | 84.62% | 85.11% |
| others | 87.80% | 90.25% | 89.01% |

## Model Files

### Included in Repository (Small Files)
- ✅ `config.json` - Model configuration
- ✅ `tokenizer.json` - Tokenizer vocabulary (7.7 MB)
- ✅ `tokenizer_config.json` - Tokenizer settings
- ✅ `special_tokens_map.json` - Special tokens
- ✅ `training_args.bin` - Training metadata

### External Download (Large File)
- ⚠️ `model.safetensors` (1.1 GB) - **Hosted on Google Drive**

**Download Link**: [Google Drive - IndicBERT Model](https://drive.google.com/drive/folders/1zXUqzC42vWChcqTVmW35OMgqJmABYPCC?usp=sharing)

## Quick Start

### Option 1: View Results Without Downloading Model
All testing results are pre-computed and visible in:
- **File**: `testing/indicbert_testing.ipynb`
- **Status**: Fully executed with all outputs
- **What you'll see**: Metrics, confusion matrix, error analysis, predictions

This is the **fastest way** to review model performance.

### Option 2: Download and Run Inference

#### Step 1: Download Model Weights
1. Visit the Google Drive link above
2. Download `model.safetensors` (1.1 GB)
3. Place it in this directory: `best_models/indicbert_property_classifier/`

#### Step 2: Load and Use Model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "../best_models/indicbert_property_classifier"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Make prediction
address = "Flat No. 301, Tower A, Green Valley Apartments"
inputs = tokenizer(address, return_tensors="pt", max_length=112,
                   padding="max_length", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()

print(f"Predicted category: {model.config.id2label[prediction]}")
```

#### Step 3: Run Testing Notebook
```bash
cd testing/
jupyter notebook indicbert_testing.ipynb
```

## Directory Structure
```
indicbert_property_classifier/
├── config.json                    # Model configuration ✅
├── model.safetensors             # Model weights (Download from Drive) ⚠️
├── tokenizer.json                # Tokenizer vocabulary ✅
├── tokenizer_config.json         # Tokenizer config ✅
├── special_tokens_map.json       # Special tokens ✅
├── training_args.bin             # Training metadata ✅
└── MODEL_INFO.md                 # This file ✅
```

## Why External Hosting?
The model weights (`model.safetensors`) are 1.1 GB, which exceeds Git's recommended file size limits. To maintain repository performance and enable version control, the weights are hosted on Google Drive while all supporting files are included in the repository.

## Training Details
- **Training Dataset**: 7,595 samples (85% split)
- **Validation Dataset**: 1,341 samples (15% split)
- **Test Dataset**: 2,681 samples
- **Epochs**: 6 (with early stopping)
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Class Weights**: Applied (1.5x for landparcel, 1.3x for others)
- **Training Time**: ~16 minutes (on Tesla T4 GPU)

## Category Definitions
0. **commercial unit** - Shops, offices, commercial spaces
1. **flat** - Apartments, flats in multi-story buildings
2. **houseorplot** - Individual houses, residential plots
3. **landparcel** - Agricultural land, survey numbers, khasra numbers
4. **others** - Warehouses, parking spaces, industrial units

## Notes for Reviewers
- ✅ All test results are **pre-computed** in `testing/indicbert_testing.ipynb`
- ✅ Model can be evaluated **without downloading** by viewing the executed notebook
- ✅ Download is **only required** if you want to run custom inference
- ✅ Model loading takes ~30-60 seconds on CPU, ~10 seconds on GPU
- ✅ Inference on 2,681 samples takes ~10-15 minutes on CPU, ~3-5 minutes on GPU

## Comparison with SVM Baseline
| Metric | SVM | IndicBERT | Improvement |
|--------|-----|-----------|-------------|
| Accuracy | 91.98% | 92.88% | +0.90% |
| Macro F1 | 0.9075 | 0.9195 | +0.0120 |
| Model Size | 605 KB | 1.1 GB | - |
| Inference Time | <1 sec | ~10 min (CPU) | - |

**Conclusion**: IndicBERT achieves better performance but requires more computational resources.

## Contact
For questions about model usage, refer to:
- Training notebook: `train/indicbert_training.ipynb`
- Testing notebook: `testing/indicbert_testing.ipynb`
