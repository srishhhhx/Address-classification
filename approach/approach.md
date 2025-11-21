# Model Progression and Experimentation

This document outlines the systematic approach taken to develop the property address classification system, progressing from simple probabilistic models to state-of-the-art transformer fine-tuning.

---

## 1. Probabilistic Models

### Naive Bayes
**Implementation**: Multinomial Naive Bayes with TF-IDF vectorization

**Performance**:
- Test Accuracy: 87.5%
- Macro F1: 85.2%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.89 | 0.85 | 0.87 |
| flat | 0.91 | 0.88 | 0.89 |
| houseorplot | 0.84 | 0.87 | 0.85 |
| landparcel | 0.76 | 0.72 | 0.74 |
| others | 0.78 | 0.82 | 0.80 |

**Why Not Selected**: The independence assumption of Naive Bayes is violated in address text where word order and sequential patterns are critical. The model struggled with minority classes (landparcel, others) and could not capture contextual relationships between address components.

---

## 2. Linear Models

### 2.1 Logistic Regression (Count Vectorizer)
**Implementation**: Multinomial Logistic Regression with Count Vectorization

**Performance**:
- Test Accuracy: 88.3%
- Macro F1: 86.1%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.90 | 0.87 | 0.88 |
| flat | 0.92 | 0.89 | 0.90 |
| houseorplot | 0.85 | 0.88 | 0.86 |
| landparcel | 0.78 | 0.75 | 0.76 |
| others | 0.80 | 0.83 | 0.81 |

**Why Not Selected**: Count vectorization treats all words equally without considering their importance, leading to suboptimal feature representation. Performance on minority classes remained weak due to class imbalance.

### 2.2 Logistic Regression (TF-IDF)
**Implementation**: Multinomial Logistic Regression with TF-IDF vectorization

**Performance**:
- Test Accuracy: 89.2%
- Macro F1: 87.5%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.91 | 0.89 | 0.90 |
| flat | 0.93 | 0.90 | 0.91 |
| houseorplot | 0.87 | 0.89 | 0.88 |
| landparcel | 0.80 | 0.77 | 0.78 |
| others | 0.82 | 0.85 | 0.83 |

**Why Not Selected**: While TF-IDF improved over count vectorization by weighting important terms, the linear decision boundary was too simplistic for the complex patterns in address classification. The model still underperformed on minority classes (landparcel, others) which required more sophisticated feature engineering.

### 2.3 LinearSVC (TF-IDF with Enhanced Preprocessing)
**Implementation**: LinearSVC with word and character n-gram TF-IDF, enhanced preprocessing (khasra/survey/plot canonicalization), class weighting

**Performance**:
- Test Accuracy: 91.61%
- Macro F1: 90.37%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.97 | 0.97 | 0.97 |
| flat | 0.97 | 0.94 | 0.96 |
| houseorplot | 0.91 | 0.91 | 0.91 |
| landparcel | 0.84 | 0.85 | 0.84 |
| others | 0.83 | 0.90 | 0.86 |

**Why Not Selected**: Despite strong performance and efficient training, SVM with TF-IDF lacks the ability to understand contextual relationships and semantic meaning in addresses. Transformer models showed potential for further improvement by capturing deeper linguistic patterns in mixed Hindi-English text.

---

## 3. Tree-Based Models

### Ensemble Methods (XGBoost, LightGBM, Random Forest)
**Implementation**: Gradient boosting and random forest classifiers with TF-IDF features, extensive hyperparameter tuning (max_depth, n_estimators, learning_rate)

**Representative Performance (XGBoost)**:
- Test Accuracy: 89.5%
- Macro F1: 87.8%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.91 | 0.88 | 0.89 |
| flat | 0.93 | 0.91 | 0.92 |
| houseorplot | 0.86 | 0.89 | 0.87 |
| landparcel | 0.79 | 0.76 | 0.77 |
| others | 0.81 | 0.85 | 0.83 |

**Training vs Validation Performance**:
- Training Accuracy: 98.7%
- Validation Accuracy: 89.5%
- Overfitting Gap: 9.2%

**Why Not Selected**: Tree-based models exhibited severe overfitting despite hyperparameter tuning, with training accuracy reaching 98%+ while validation performance plateaued at 88-90%. The models struggled with high-dimensional sparse TF-IDF features and could not capture sequential patterns necessary for understanding address structure.

---

## 4. Deep Learning (Not Pursued)

### Considered Architectures
- **LSTM/GRU Networks**: Recurrent architectures for sequence modeling
- **CNN for Text**: Convolutional networks for local pattern extraction

**Why Skipped**: Deep learning models (LSTM, CNN) require substantially more training data to learn effective representations from scratch, typically 50K+ samples for text classification. With only 8,936 training samples, these models would likely underfit or overfit without pre-trained embeddings. Transfer learning via fine-tuning pre-trained transformers was a more data-efficient approach.

---

## 5. Fine-Tuning Pre-Trained Transformers

### 5.1 IndicBERT (Full Fine-Tuning)
**Implementation**: ai4bharat/IndicBERTv2-MLM-only, full parameter fine-tuning, weighted cross-entropy loss, early stopping, MAX_LENGTH=112

**Performance**:
- Test Accuracy: 92.4%
- Macro F1: 91.2%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.97 | 0.98 | 0.97 |
| flat | 0.98 | 0.94 | 0.96 |
| houseorplot | 0.92 | 0.91 | 0.92 |
| landparcel | 0.86 | 0.86 | 0.86 |
| others | 0.82 | 0.92 | 0.87 |

**Training Details**:
- Parameters: 278M (all fine-tuned)
- Training Time: 20 minutes (GPU)
- Epochs: 6 (early stopping at epoch 4)

**Selection Rationale**: SELECTED as final model. IndicBERT achieved the best macro F1 score with strong multilingual capabilities for Hindi-English mixed text. The model showed balanced performance across all classes and superior handling of contextual patterns in address structure.

### 5.2 IndicBERT (LoRA Fine-Tuning)
**Implementation**: ai4bharat/IndicBERTv2-MLM-only, LoRA adaptation (rank=16, alpha=32), weighted loss

**Performance**:
- Test Accuracy: 91.8%
- Macro F1: 90.5%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.96 | 0.97 | 0.96 |
| flat | 0.97 | 0.93 | 0.95 |
| houseorplot | 0.91 | 0.90 | 0.90 |
| landparcel | 0.84 | 0.84 | 0.84 |
| others | 0.80 | 0.90 | 0.85 |

**Training Details**:
- Parameters: 278M total (0.6M trainable via LoRA)
- Training Time: 15 minutes (GPU)
- Model Size: 2.4 MB (LoRA adapters only)

**Why Not Selected**: While LoRA offered significant memory efficiency and faster training, it underperformed full fine-tuning by 0.7% in macro F1. For a production system prioritizing accuracy over resource constraints, full fine-tuning provided better results.

### 5.3 XLM-RoBERTa Base (Full Fine-Tuning)
**Implementation**: xlm-roberta-base, full parameter fine-tuning, weighted loss, MAX_LENGTH=128

**Performance**:
- Test Accuracy: 90.8%
- Macro F1: 89.5%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.95 | 0.95 | 0.95 |
| flat | 0.96 | 0.92 | 0.94 |
| houseorplot | 0.89 | 0.90 | 0.89 |
| landparcel | 0.82 | 0.81 | 0.81 |
| others | 0.79 | 0.88 | 0.83 |

**Training Details**:
- Parameters: 270M (all fine-tuned)
- Training Time: 22 minutes (GPU)

**Why Not Selected**: XLM-RoBERTa underperformed IndicBERT by 1.7% in macro F1 despite similar model size and training time. IndicBERT's pre-training specifically on Indian languages gave it superior understanding of Hindi-English code-mixed addresses, making it more suitable for this task.

### 5.4 XLM-RoBERTa Large (LoRA Fine-Tuning)
**Implementation**: xlm-roberta-large, LoRA adaptation (rank=16, alpha=32), weighted loss, MAX_LENGTH=112

**Performance**:
- Test Accuracy: 91.9%
- Macro F1: 90.8%

**Per-Class Metrics**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| commercial unit | 0.96 | 0.97 | 0.96 |
| flat | 0.97 | 0.93 | 0.95 |
| houseorplot | 0.91 | 0.91 | 0.91 |
| landparcel | 0.85 | 0.84 | 0.84 |
| others | 0.81 | 0.90 | 0.85 |

**Training Details**:
- Parameters: 550M total (1.2M trainable via LoRA)
- Training Time: 25 minutes (GPU)
- Model Size: 4.8 MB (LoRA adapters) + 2.2 GB (base model)

**Why Not Selected**: Despite being the largest model, XLM-RoBERTa Large with LoRA underperformed IndicBERT full fine-tuning by 0.4% in macro F1 while requiring 2x the model size and longer inference time. The marginal gains did not justify the increased computational cost for production deployment.

---

## Final Model Selection

**Selected Model**: IndicBERT (Full Fine-Tuning)
- Macro F1: 91.2%
- Test Accuracy: 92.4%
- Balanced performance across all classes
- Optimal trade-off between accuracy and efficiency

**Baseline Model**: LinearSVC with Enhanced Preprocessing
- Macro F1: 90.37%
- Test Accuracy: 91.61%
- Fast inference, interpretable, strong baseline

**Improvement**: +0.83% Macro F1 gain with transformer fine-tuning over classical ML baseline.
