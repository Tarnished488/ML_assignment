# Handwritten Digit Recognition System

A handwritten digit recognition project based on the MNIST dataset, built as a course design project.

## Project Structure

```
ML_assignment/
├── data/
│   ├── raw/                  # Raw MNIST data (auto-downloaded)
│   ├── processed/            # Preprocessed data
│   └── features/             # Extracted feature vectors
├── src/
│   ├── data/                 # Data loading & preprocessing (Team Lead)
│   │   ├── loader.py         # MNIST data loading
│   │   └── preprocess.py     # Denoising / Binarization / Centering
│   ├── features/             # Feature engineering (Feature Engineering)
│   │   └── extractor.py      # HOG/LBP/Shape/PCA feature extraction
│   ├── EDA/                  # Exploratory Data Analysis
│   │   └── explorer.py       # MNISTExplorer: label distribution / sample grid / pixel analysis / PCA / t-SNE
│   ├── models/               # Model training & evaluation
│   │   ├── cnn.py            # CNN + FocalLoss definition
│   │   ├── train.py          # Training pipeline (CNN + KNN/LR/RF baselines)
│   │   └── evaluate.py       # Evaluation metrics (Accuracy/Precision/Recall/F1/Confusion Matrix)
│   └── visualization/        # Visualization
│       └── plot.py           # Training curves / Confusion matrices / Model comparison / Misclassified samples / Grad-CAM
├── results/                  # Auto-generated evaluation results and charts
├── tests/                    # Unit tests
├── main.py                   # One-click entry point (training + evaluation + plotting)
└── requirements.txt          # Dependencies
```

## Quick Start

### One Command to Run Everything

```bash
pip install -r requirements.txt
python main.py
```

This automatically completes: Download MNIST → Preprocess → Feature extraction → Train CNN + 3 baseline models → Evaluate on training/validation/test sets → Generate charts.

Optionally enable EDA (Exploratory Data Analysis):
```bash
python main.py --eda --skip-training    # EDA only, skip training
python main.py --eda                    # EDA + full training
```

For more parameters and FAQs, see the [Model & Evaluation Documentation](模型与评估说明文档.md).

### Run Individual Modules

```bash
# Full pipeline (preprocessing + feature extraction + training + evaluation)
python main.py

# Exploratory Data Analysis only
python main.py --eda --skip-training

# EDA with full training
python main.py --eda
```

## Data Split

70,000 MNIST images are split at **Train:Val:Test = 6:2:2**:

| Set | Count | Purpose |
|-----|-------|---------|
| Training | 42,000 | Model parameter training |
| Validation | 14,000 | Early stopping, hyperparameter selection, overfitting detection |
| Test | 14,000 | Final evaluation, shared across all models |

Each model's Accuracy / Precision / Recall / F1 and confusion matrix are reported **separately** on all three sets.

## Data Interface Convention

Preprocessed data format:

```python
X_train: numpy.ndarray  # shape=(N, 28, 28), float32, range [0, 1]
y_train: numpy.ndarray  # shape=(N,), int64, labels 0-9
```

Load via:

```python
from src.data.loader import load_processed_data
X_train, y_train, X_test, y_test = load_processed_data()
```

## Preprocessing Steps

1. **Denoising** — Median filter to remove isolated noise pixels
2. **Binarization** — Threshold segmentation (threshold=0.3) to highlight digit contours
3. **Centering** — Locate the digit bounding box and shift it to the center of the 28×28 canvas

## Exploratory Data Analysis (EDA)

The EDA module lives in `src/EDA/`, with `MNISTExplorer` as its core class. Enable it via the `--eda` flag in `main.py`.

```bash
python main.py --eda --skip-training
python main.py --eda --eda-output my_eda_results
```

### Analysis Components

| Analysis | Method | Purpose |
|----------|--------|---------|
| Label Distribution | `plot_label_distribution()` | Check class balance |
| Sample Grid | `plot_sample_grid()` | Visually inspect image quality per digit |
| Mean/Std Images | `plot_mean_and_std_images()` | Observe average stroke shape and variation per class |
| Digit Correlation Heatmap | `plot_digit_correlation_heatmap()` | Find which digits are most similar in mean appearance |
| Confusing Pairs | `plot_confusing_pairs()` | Visualize the most confusable digit pairs (e.g., 7↔9, 4↔9) |
| Pixel Intensity Distribution | `analyze_pixel_intensity()` | Histogram analysis + KS test for train/test consistency |
| PCA Scatter Plot | `plot_pca_scatter()` | Dimensionality reduction to observe digit cluster separability |
| t-SNE Visualization | `plot_tsne()` | Non-linear dimensionality reduction showing clearer cluster structure |

### Output

When `--eda` is enabled, charts are saved to `eda_results/` (customizable via `--eda-output`).

## Feature Engineering

The feature engineering module is located in `src/features/`, with `FeatureExtractor` as its core class.

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| Raw Pixels | 784 | Flattened raw pixels |
| HOG | 324 | Histogram of Oriented Gradients, captures stroke direction |
| LBP | 10 | Local Binary Patterns, describes texture |
| Shape | 71 | Shape statistics (aspect ratio, centroid, projections, quadrant density) |

Default combination: **HOG 324 + LBP 10 + Shape 71 = 405 dimensions**

## Models

### CNN (Primary Model)

```
Input: 1 x 28 x 28 (grayscale image)
    ↓
Conv1: Conv2d(1→32) + BatchNorm + ReLU + MaxPool
    ↓
Conv2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool
    ↓
Dropout: 0.25
    ↓
FC1: Linear(3136→128) + ReLU + Dropout
    ↓
FC2: Linear(128→10)
    ↓
Output: 10-class probability distribution
```

- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss function: Focal Loss + Label Smoothing (focuses on hard samples, prevents overconfidence)
- Regularization: Kaiming initialization + BatchNorm + Dropout(0.25+0.5) + Data augmentation + Early stopping

### Baseline Models (for comparison)

| Model | Features | Description |
|-------|----------|-------------|
| KNN (k=5) | HOG+LBP+Shape 405d | Classic nearest-neighbor classification |
| Logistic Regression | HOG+LBP+Shape 405d | Linear model representative |
| Random Forest | HOG+LBP+Shape 405d | Ensemble method representative |

## Module Responsibilities

| Module | Lead | Core Task | Status |
|--------|------|-----------|--------|
| data | Team Lead | Data loading, preprocessing, interface design |  |
| features | Feature Engineering | Feature extraction (HOG/LBP/Shape), PCA |  |
| EDA | | Exploratory Data Analysis: label distribution / sample grid / correlation heatmap / PCA / t-SNE |  |
| models | | CNN + baseline model training and evaluation |  |
| visualization | | Training curves, confusion matrices, model comparison charts |  |

## Experimental Results

See detailed results in `results/evaluation_summary.txt` and charts in the `results/` directory.

3-epoch quick test example (Train:Val:Test = 42000:14000:14000):

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| CNN (Focal Loss) | 95.32% | 95.49% | 95.32% |
| Logistic Regression | 98.30% | 97.01% | 97.01% |
| KNN (k=5) | 97.58% | 96.33% | 96.35% |
| Random Forest | 100.00% | 96.40% | 95.84% |

> Results from a 3-epoch fast test. Focal Loss naturally has higher loss early in training. All models will improve further with a full 30-epoch run. Random Forest reaching 100% on training set indicates overfitting.

## Goals

- [x] Project skeleton
- [x] Data loading module
- [x] Preprocessing module
- [x] Feature engineering
- [x] Feature extraction tests
- [x] Exploratory Data Analysis (label distribution / sample grid / correlation heatmap / PCA / t-SNE)
- [x] Model training (CNN + KNN + Logistic Regression + Random Forest)
- [x] Visualization & evaluation (training curves / confusion matrices / model comparison / misclassified samples / per-class accuracy)
- [ ] GUI interface
- [ ] Course design report

## License

MIT License
