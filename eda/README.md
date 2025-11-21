# Exploratory Data Analysis

This directory contains the comprehensive exploratory data analysis for the property address classification dataset.

## Dataset Overview

**Total Samples**: 11,617
- Training: 8,936 (77%)
- Validation: 2,681 (23%)

**Task**: Multi-class classification of Indian property addresses into 5 categories

## Class Distribution and Imbalance

### Training Set Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| flat | 3,232 | 36.2% |
| houseorplot | 2,673 | 29.9% |
| others | 1,197 | 13.4% |
| commercial unit | 965 | 10.8% |
| landparcel | 869 | 9.7% |

### Class Imbalance Analysis

**Imbalance Ratio**: 3.7:1 (flat to landparcel)

**Key Observations**:
- **Majority classes** (flat, houseorplot): 66% of dataset - sufficient representation
- **Minority classes** (landparcel, commercial unit): 20% of dataset - require special handling
- **Catch-all class** (others): 13% - inherently ambiguous and challenging

**Impact on Modeling**:
1. Standard training leads to bias toward majority classes
2. Minority class recall suffers without intervention
3. Landparcel (9.7%) particularly vulnerable to poor performance

**Mitigation Strategies Employed**:
- Class-weighted loss functions in all models
- Stratified cross-validation for fair evaluation
- Per-class performance monitoring during training
- Enhanced preprocessing to capture minority class patterns (khasra, survey variants)

## Key Findings

### 1. Text Characteristics

**Length Statistics**:
- Mean length: 146 characters
- Median length: 128 characters
- 95th percentile: 236 characters
- Token length (95th percentile): 99 tokens

**Insight**: MAX_LENGTH=112 covers 97% of samples while maintaining efficiency

### 2. Language Composition

**Mixed Hindi-English Text**:
- Predominantly Hindi address components (मकान, खसरा, गाटा)
- English structural elements (flat, plot, near)
- Transliterated Hindi in Latin script
- Code-mixing within single addresses

**Example**:
```
"Flat 301, Sunrise Apartments, Sector 12, near railway station"
"मकान नंबर 45, गाटा 123, ग्राम पंचायत सोहना"
"Khasra No 456, Village Malsisar, Tehsil Alsisar"
```

**Impact**: Multilingual models (IndicBERT) outperform English-only models

### 3. Vocabulary Analysis

**Overall Statistics**:
- Total unique words: 15,544
- Average words per address: 12.3
- High vocabulary overlap across classes

**Class-Specific Keywords**:

| Class | Distinctive Terms |
|-------|-------------------|
| flat | flat, floor, apartment, tower, wing, block |
| houseorplot | plot, house, makan, sector, colony |
| commercial unit | shop, office, commercial, market, mall |
| landparcel | khasra, survey, gata, village, pargana, tehsil |
| others | ward, property, near, area (generic terms) |

**Challenge**: High vocabulary overlap means context matters more than individual keywords

### 4. Address Structure Patterns

**Flat Addresses**:
- Format: "Flat X, Floor Y, Building Name, Society/Complex"
- Hierarchical structure with multiple components
- Often includes block/tower/wing identifiers

**Houseorplot Addresses**:
- Format: "House/Plot No X, Sector Y, Colony/Area"
- Simpler structure than flats
- Strong presence of plot numbers

**Landparcel Addresses**:
- Format: "Khasra/Survey/Gata No X, Village, Tehsil, District"
- Rural addressing system
- Administrative hierarchy (village → tehsil → district)

**Commercial Unit**:
- Format: "Shop/Office No X, Market/Complex Name, Location"
- Business-oriented terminology
- Often includes floor information

**Others**:
- Mixed formats, incomplete addresses, generic references
- Lack clear structural patterns
- High ambiguity

### 5. Preprocessing Requirements

**Critical Observations**:
1. **Canonicalization needed**: kh no, kh.no, khno → khasra
2. **Punctuation matters**: Slashes (/) and hyphens (-) indicate structure
3. **Numbers are features**: Plot numbers, flat numbers are discriminative
4. **No stopword removal**: "near", "opposite" provide location context
5. **Preserve structure**: Comma separation indicates address hierarchy

**Example Variants Requiring Normalization**:
```
kh no, kh.no, khno, k no, k.no → khasra
s no, sy no, survey no → surveyno
p no, plt no, plot no → plotno
```

## Model Selection Insights

### Why Traditional ML Works

**SVM Success Factors**:
- Can capture n-gram patterns effectively
- Character n-grams handle spelling variations
- TF-IDF weights discriminative terms
- Class weighting addresses imbalance

**Limitations**:
- Cannot understand context beyond n-grams
- Struggles with semantic similarity
- Bag-of-words loses word order information

### Why Transformers Excel

**IndicBERT Advantages**:
- Captures long-range dependencies in addresses
- Understands context: "plot" in "plot no" vs "houseorplot"
- Multilingual pre-training handles Hindi-English mixing
- Attention mechanism identifies relevant address components

**Critical for This Task**:
- Hierarchical address structure requires contextual understanding
- Mixed-language text benefits from multilingual models
- Ambiguous cases (landparcel vs houseorplot) need semantic reasoning

## Recommendations from EDA

1. **Use contextual models**: Vocabulary overlap necessitates understanding word relationships
2. **Handle class imbalance**: Weighted loss or oversampling for minority classes
3. **Preserve structure**: Keep punctuation and numbers during preprocessing
4. **Canonicalize keywords**: Normalize variant spellings of domain terms
5. **Monitor per-class metrics**: Overall accuracy masks minority class issues
6. **Set MAX_LENGTH=112**: Balances coverage (97%) and efficiency

## EDA Notebook

The complete analysis with visualizations is available in `property_address_eda.ipynb`, covering:
- Class distribution analysis
- Text length distributions
- Vocabulary statistics
- Class-specific keyword extraction
- Character and word patterns
- Address structure examples
- Correlation analysis
- Feature importance insights
- Preprocessing recommendations
- Model selection guidance

All findings from EDA directly informed preprocessing decisions and model architecture choices documented in the training notebooks.
