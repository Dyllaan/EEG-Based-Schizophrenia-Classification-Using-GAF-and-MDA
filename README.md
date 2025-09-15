# EEG-Based Schizophrenia Classification Using GAF and MDA

A comparative study implementing two machine learning pipelines for schizophrenia classification from EEG data using Gramian Angular Field (GAF) transformations and Multilinear Discriminant Analysis (MDA).

## Overview

This project develops and compares two approaches for classifying schizophrenia from EEG recordings:

1. **MDA Pipeline**: Combines GAF image transformation with tensor-based dimensionality reduction (MDA) followed by traditional machine learning classifiers
2. **CNN Pipeline**: End-to-end deep learning approach using a lightweight 3-layer CNN on GAF images

## Key Results

- **MDA Pipeline**: 96.34% ± 0.09% accuracy
- **CNN Pipeline**: 83.04% ± 0.94% accuracy
- **Statistical Significance**: p = 0.0029 (paired t-test)

The MDA approach demonstrated superior performance with a 13.31 percentage point improvement and significantly lower variance across cross-validation folds.

## Dataset

- **Source**: EEG recordings from 81 subjects (control and schizophrenia groups)
- **Processing**: 4,374 GAF images (224×224) derived from 3072-point EEG segments
- **Class Distribution**: 1,728 controls, 2,646 schizophrenia cases
- **Validation**: 3-fold stratified cross-validation

The EEG data is from a sensory task involving control and schizophrenic groups with 81 total subjects.

The dataset is in two parts:
1. https://www.kaggle.com/datasets/broach/button-tone-sz
2. https://www.kaggle.com/datasets/broach/buttontonesz2

These are combined in the data source folder 'EEG'.
## Methodology

### Data Preprocessing
1. Load EEG data from subject folders (columns 4-12 as channels)
2. Segment into overlapping 3072-point windows (50% overlap, up to 6 per channel)
3. Apply quality filtering (>10 unique values, std >0.1)
4. Transform to GAF images using Gramian Angular Summation Field
5. Resize to 224×224 grayscale images

### MDA Pipeline
- Tests multiple ranks [8, 16, 24, 32, 64] over 5 iterations
- Evaluates 7 classifiers: SVM, Random Forest, k-NN, Logistic Regression, MLP, Decision Tree
- Nested hyperparameter optimization with grid search
- Consistently selected MLPClassifier with ranks 32-64

### CNN Pipeline
- 3-layer architecture with batch normalisation and dropout
- SGD optimiser with plateau learning rate scheduling
- Early stopping (7-epoch patience) and warmup (5 epochs)

### Requires
- Python (built on 3.12.4)
### PIP Libraries
- pandas
- numpy
- torch
- torchvision
- scikit-learn (sklearn)
- scipy
- matplotlib
- pyts
- tensorly
