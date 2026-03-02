# ml_module

> Machine learning fundamentals from scratch — linear regression to regularized logistic classification.

## Overview

A progressive series of machine learning implementations built entirely from scratch using NumPy. Covers the full supervised learning pipeline across 5 modules: from basic matrix operations and statistics, through linear and logistic regression with gradient descent, to regularized models with polynomial feature expansion and cross-validation.

This project was built as part of the 42 school AI curriculum.

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.x |
| Core | NumPy |
| Visualization | Matplotlib |
| Data | Pandas |

## Modules

| Module | Topic | Key Implementations |
|--------|-------|---------------------|
| 00 | Foundations | Matrix/Vector class, TinyStatistician, loss functions (MSE), data plotting |
| 01 | Linear Regression | Gradient descent, vectorized gradient, MyLinearRegression, z-score & min-max normalization |
| 02 | Multivariate Regression | Multi-feature regression, polynomial features, train/test split, model benchmarking |
| 03 | Logistic Classification | Sigmoid, cross-entropy loss, MyLogisticRegression, one-vs-all multi-class, precision/recall/F1, confusion matrix |
| 04 | Regularization | L2 penalty, Ridge regression, regularized logistic regression, polynomial + regularization benchmarks |

## Key Features

- Linear regression with gradient descent (single and multi-variate)
- Logistic regression with sigmoid activation and cross-entropy loss
- One-vs-all multi-class classification on solar system census dataset
- Polynomial feature expansion for non-linear decision boundaries
- L2 regularization (Ridge) for both linear and logistic models
- Train/test data splitting and model benchmarking
- Evaluation metrics: MSE, R², precision, recall, F1 score, confusion matrix
- All implementations from scratch — no scikit-learn, no frameworks

## Architecture

```
ml_module/
├── ML Module 00/          # Matrix ops, statistics, prediction, loss, plotting
│   └── ex00–ex09/
├── ML Module 01/          # Gradient descent, linear regression, normalization
│   ├── ex00–ex06/
│   └── data/              # Blue pills dataset
├── ML Module 02/          # Multivariate regression, polynomial, benchmarking
│   ├── ex00–ex10/
│   └── data/              # Space avocado, spacecraft datasets
├── ML Module 03/          # Logistic regression, multi-class, metrics
│   ├── ex00–ex09/
│   └── Data/              # Solar system census dataset
├── ML Module 04/          # Regularization, Ridge, regularized logistic
│   └── ex00–ex09/
└── Subjects/              # Original project PDFs
```

## Getting Started

### Prerequisites

```bash
Python 3.x
NumPy
Matplotlib
Pandas
```

### Usage

Each module contains numbered exercises. Run individual exercises:

```bash
cd "ML Module 01/ex03"
python my_linear_regression.py

cd "ML Module 03/ex07"
python mono_log.py      # Single-class logistic regression
python multi_log.py     # One-vs-all classification
```

## What This Demonstrates

- **ML Pipeline Mastery**: Built the entire supervised learning stack from scratch — data preprocessing, model training, evaluation, and regularization — without any ML library.
- **Mathematical Foundations**: Implemented gradient descent, cross-entropy loss, sigmoid, softmax, and L2 regularization from their mathematical definitions using only NumPy.
- **Progressive Complexity**: Each module builds on the previous, progressing from univariate regression to regularized multi-class classification with polynomial features.

## License

This project was built as part of the 42 school curriculum.

---

*Part of [sungyongcho](https://github.com/sungyongcho)'s project portfolio.*
