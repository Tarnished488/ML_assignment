"""
One-click Run Entry

This is the main entry point for the entire project. Run this single command to complete the full pipeline:

    python run_all.py

=== Full Pipeline (5 Steps) ===

Step 1: Load MNIST data + preprocessing (denoising / binarization / centering)
Step 2: Merge all 70000 images, re-split at 6:2:2 (42000 + 14000 + 14000)
Step 3: Extract HOG+LBP+Shape features (for baseline models)
Step 4: Train CNN + 3 baseline models, evaluate on all three sets
Step 5: Generate visualization plots + save results summary

=== Data Split ===

Train:Val:Test = 6:2:2 (42000:14000:14000)

Why merge the original 60000+10000 and re-split:
- Original MNIST has no validation set, only training and test sets
- We need a validation set for early stopping (to prevent overfitting)
- So we merge all 70000 images and randomly re-split by ratio

=== Usage ===

# Default 30 epochs (full training)
python run_all.py

# Quick test (3 epochs, done in a few minutes)
python run_all.py --epochs 3 --batch-size 256

# Custom parameters
python run_all.py --epochs 20 --lr 0.0005 --batch-size 64
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np

# Add project root to Python search path so src.xxx can be imported
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from src.data.loader import load_mnist, save_processed_data      # Data loading
from src.data.preprocess import Preprocessor                       # Data preprocessing
from src.features.extractor import FeatureExtractor                # Feature extraction
from src.models.train import run_full_pipeline, set_seed, split_dataset  # Model training
from src.visualization.plot import generate_all_plots              # Visualization


def main():
    # ---- Parse command line arguments ----
    parser = argparse.ArgumentParser(description="MNIST Handwritten Digit Recognition - Full Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="CNN training epochs (default 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--save-dir", type=str, default="results", help="Results save directory")
    args = parser.parse_args()

    set_seed(args.seed)
    start_time = time.time()

    print("=" * 60)
    print("  MNIST Handwritten Digit Recognition - Automated Training & Evaluation")
    print("  Data Split: Train:Val:Test = 6:2:2")
    print("=" * 60)

    # ================================================================
    # Step 1: Load MNIST data and preprocess
    # ================================================================
    print("\n[Step 1/5] Loading MNIST data and preprocessing...")

    # load_mnist: download (first time) and load MNIST data
    # data_dir: data storage directory
    # download: automatically download from network if data does not exist
    # preprocess=False: skip internal preprocessing, we control the pipeline ourselves
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist(
        data_dir="data/raw", download=True, preprocess=False
    )

    # Preprocessing: denoising (median filter) -> binarization (threshold 0.3) -> centering (move to canvas center)
    preprocessor = Preprocessor()
    X_train_raw = preprocessor.preprocess_pipeline(X_train_raw)
    X_test_raw = preprocessor.preprocess_pipeline(X_test_raw)
    print(f"  Raw data: Train set {X_train_raw.shape}, Test set {X_test_raw.shape}")

    # ================================================================
    # Step 2: Merge and re-split at 6:2:2
    # ================================================================
    print("\n[Step 2/5] Merging data and splitting at train:val:test = 6:2:2...")

    # Merge original training set (60000) and test set (10000) into 70000 images
    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
    print(f"  Total after merging: {X_all.shape[0]} images")

    # Split by ratio: training 60% / validation 20% / test 20%
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_all, y_all, train_ratio=6/10, val_ratio=2/10, seed=args.seed
    )
    print(f"  Training set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}, Test set: {X_test.shape[0]}")

    # Save preprocessed data (compatible with team member's interface)
    save_processed_data(X_train, y_train, X_test, y_test)

    # ================================================================
    # Step 3: Extract features (for baseline models)
    # ================================================================
    print("\n[Step 3/5] Extracting HOG+LBP+Shape features...")

    # FeatureExtractor: feature extractor written by team member
    # HOG (Histogram of Oriented Gradients): stroke direction info, 324 dims
    # LBP (Local Binary Pattern): texture info, 10 dims
    # Shape (shape statistics): aspect ratio, centroid, projections, etc., 71 dims
    # Total: 405 dims
    extractor = FeatureExtractor()

    # fit_transform: fit on training set and transform (compute necessary statistics)
    X_train_features = extractor.fit_transform(
        X_train, method=("hog", "lbp", "shape"), pca_components=None
    )
    # transform: use parameters fitted on training set to transform validation and test sets
    X_val_features = extractor.transform(X_val, method=("hog", "lbp", "shape"))
    X_test_features = extractor.transform(X_test, method=("hog", "lbp", "shape"))
    print(f"  Feature dimension: {X_train_features.shape[1]}")

    # ================================================================
    # Step 4: Train models + evaluate on three sets
    # ================================================================
    print("\n[Step 4/5] Training CNN + baseline models and evaluating on all three sets...")

    # run_full_pipeline does:
    # 1. Train CNN (PyTorch, Focal Loss + Label Smoothing)
    # 2. Train KNN / Logistic Regression / Random Forest (sklearn)
    # 3. Evaluate all models on training, validation, and test sets
    results = run_full_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test,
        X_train_features, X_val_features, X_test_features,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed,
    )

    # ================================================================
    # Step 5: Generate visualizations + save summary
    # ================================================================
    print("\n[Step 5/5] Generating visualization plots and saving results summary...")

    # generate_all_plots: generate all plots (training curves, confusion matrices, comparison charts, Grad-CAM, etc.)
    generate_all_plots(results, X_test, y_test, save_dir=args.save_dir)

    # _save_summary: save text format evaluation results summary
    _save_summary(results, args.save_dir)

    # ---- Done ----
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n{'='*60}")
    print(f"  All done! Time elapsed: {int(minutes)} min {int(seconds)} sec")
    print(f"  Results saved in: {args.save_dir}/")
    print(f"{'='*60}")


def _save_summary(results: dict, save_dir: str):
    """
    Save text format results summary to evaluation_summary.txt

    Summary includes:
        1. Detailed metrics for each model on each set (train/val/test)
        2. Three-set Accuracy summary comparison table
        3. Overfitting analysis (train-val accuracy gap)

    Args:
        results: Return value from run_full_pipeline()
        save_dir: Save directory
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    with open(f"{save_dir}/evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write("MNIST Handwritten Digit Recognition - Evaluation Results Summary\n")
        f.write("Data Split: Train:Val:Test = 6:2:2 (42000:14000:14000)\n")
        f.write("Loss Function: Focal Loss (gamma=2.0) + Label Smoothing (0.1)\n")
        f.write("=" * 60 + "\n\n")

        # ---- Part 1: Detailed results for three sets ----
        for split_info in results["all_results"]:
            split_name = split_info["split"]
            split_results = split_info["results"]

            f.write(f"{'='*60}\n")
            f.write(f"  [{split_name}]\n")
            f.write(f"{'='*60}\n\n")

            for r in split_results:
                f.write(f"  Model: {r['model_name']}\n")
                f.write(f"    Accuracy:  {r['accuracy']:.4f}\n")
                f.write(f"    Precision: {r['precision_macro']:.4f} (macro)\n")
                f.write(f"    Recall:    {r['recall_macro']:.4f} (macro)\n")
                f.write(f"    F1-Score:  {r['f1_macro']:.4f} (macro)\n\n")

                f.write("    Per-class metrics:\n")
                f.write(r["report"] + "\n")
                f.write("-" * 50 + "\n\n")

        # ---- Part 2: Three-set Accuracy summary comparison table ----
        f.write(f"\n{'='*60}\n")
        f.write("  [Three-set Accuracy Summary]\n")
        f.write(f"{'='*60}\n\n")

        # Table header
        header = f"{'Model':<25}"
        for split_info in results["all_results"]:
            header += f" {split_info['split']:>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # One row per model
        model_names = [r["model_name"] for r in results["all_results"][0]["results"]]
        for model_name in model_names:
            row = f"{model_name:<25}"
            for split_info in results["all_results"]:
                for r in split_info["results"]:
                    if r["model_name"] == model_name:
                        row += f" {r['accuracy']:>10.4f}"
                        break
            f.write(row + "\n")
        f.write("-" * len(header) + "\n")

        # ---- Part 3: Overfitting analysis ----
        history = results["history"]
        if history["train_acc"] and history["val_acc"]:
            gap = history["train_acc"][-1] - history["val_acc"][-1]
            f.write(f"\n  Overfitting Analysis:\n")
            f.write(f"    Final training accuracy: {history['train_acc'][-1]:.2f}%\n")
            f.write(f"    Final validation accuracy: {history['val_acc'][-1]:.2f}%\n")
            f.write(f"    Train-val gap: {gap:.2f}%\n")
            if gap > 5:
                f.write("    Possible overfitting. Suggestions: increase data augmentation, increase Dropout, reduce parameters\n")
            elif gap < -3:
                f.write("    Training accuracy is lower than validation accuracy (due to data augmentation + regularization). Model generalizes well\n")
            else:
                f.write("    Low overfitting level. Model generalizes well\n")

    print(f"  Results summary saved: {save_dir}/evaluation_summary.txt")


if __name__ == "__main__":
    main()
