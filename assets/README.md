# Assets

This directory contains visual assets for the project documentation.

## Approach Diagram

**File**: `approach_diagram.png`

**Purpose**: Visual representation of the ML pipeline and methodology

**Recommended Content**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA INPUT                               │
│  Training.xlsx (8,936 samples) + Validation.xlsx (2,681 samples) │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                 │
│  • Domain-specific canonicalization (khasra, survey, plot)       │
│  • Text normalization (lowercase, punctuation handling)          │
│  • Token length analysis (95th percentile: 112 tokens)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
│                                                                 │
│  ┌──────────────────┐          ┌──────────────────┐           │
│  │   SVM Approach   │          │  IndicBERT       │           │
│  │                  │          │  Approach        │           │
│  │  • Word TF-IDF   │          │  • Tokenization  │           │
│  │    (1-3 grams)   │          │  • Embeddings    │           │
│  │  • Char TF-IDF   │          │  • Attention     │           │
│  │    (3-5 grams)   │          │    Mechanism     │           │
│  │  • FeatureUnion  │          │  • Fine-tuning   │           │
│  └──────────────────┘          └──────────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                │
│                                                                  │
│  ┌──────────────────┐          ┌──────────────────┐            │
│  │   LinearSVC      │          │  IndicBERTv2     │            │
│  │                  │          │                  │            │
│  │  • C=1.0         │          │  • 278M params   │            │
│  │  • Balanced      │          │  • Weighted loss │            │
│  │    weights       │          │  • 6 epochs      │            │
│  │  • 5-fold CV     │          │  • Early stop    │            │
│  └──────────────────┘          └──────────────────┘            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                 │
│  • Accuracy, Precision, Recall, F1                              │
│  • Confusion Matrix                                             │
│  • Per-class Analysis                                           │
│  • Error Analysis                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESULTS                                 │
│                                                                  │
│  SVM: 91.98% accuracy           IndicBERT: 92.88% accuracy      │
│       0.9075 macro F1                     0.9195 macro F1       │
└─────────────────────────────────────────────────────────────────┘
```

**Suggested Tools**:
- draw.io (diagrams.net)
- Lucidchart
- PowerPoint/Keynote
- Python (matplotlib/seaborn for flowcharts)

**Dimensions**: 1200x800 px or larger for clarity

**Format**: PNG (preferred) or SVG
