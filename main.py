"""
Handwritten Digit Recognition System - Main Program
Integrated workflow: data loading, preprocessing, feature extraction, model training
Following best practices from run_all.py
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_mnist, save_processed_data
from src.data.preprocess import Preprocessor
from src.features.extractor import FeatureExtractor
from src.models.train import run_full_pipeline, set_seed, split_dataset
from src.EDA import MNISTExplorer


def main():
    # ---- Parse command line arguments ----
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognition System - Complete Workflow")
    parser.add_argument("--epochs", type=int, default=30,
                       help="CNN training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Results save directory (default: results)")
    parser.add_argument("--method", default="hog,lbp,shape",
                       help="Feature extraction method: raw, hog, lbp, shape or comma-separated combination")
    parser.add_argument("--pca", type=float, default=None,
                       help="PCA dimension reduction: None for no PCA, integer for component count, float for variance ratio")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training, only perform data preparation and feature extraction")
    parser.add_argument("--eda", action="store_true",
                       help="Enable Exploratory Data Analysis (EDA) after preprocessing")
    parser.add_argument("--eda-output", type=str, default="eda_results",
                       help="Output directory for EDA plots (default: eda_results)")

    args = parser.parse_args()

    set_seed(args.seed)
    start_time = time.time()

    print("=" * 60)
    print("Handwritten Digit Recognition System - Complete Workflow")
    print("Data Split: Train:Val:Test = 6:2:2")
    print("=" * 60)

    # ==================== 1. Load MNIST data and preprocessing ====================
    print("\n[Stage 1] Loading MNIST data and preprocessing...")

    # Load raw data (no internal preprocessing, we control it ourselves)
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist(
        data_dir="data/raw", download=True, preprocess=False
    )

    print(f"  Raw data: Train set {X_train_raw.shape}, Test set {X_test_raw.shape}")

    # 应用预处理流水线
    preprocessor = Preprocessor()
    X_train_raw = preprocessor.preprocess_pipeline(X_train_raw)
    X_test_raw = preprocessor.preprocess_pipeline(X_test_raw)

    print(f"  Preprocessing completed: Train set {X_train_raw.shape}, Test set {X_test_raw.shape}")

    # ==================== EDA Stage ====================
    if args.eda:
        print("\n[EDA] Exploratory Data Analysis...")
        eda_output_dir = Path(args.eda_output)
        eda_output_dir.mkdir(parents=True, exist_ok=True)

        # Create EDA explorer
        explorer = MNISTExplorer(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

        # Run comprehensive analysis and save plots to specified directory
        explorer.run_comprehensive_analysis(output_dir=str(eda_output_dir))
        print(f"  EDA results saved to: {eda_output_dir}")

    # ==================== 2. Merge data and split train:val:test = 6:2:2 ====================
    print("\n[Stage 2] Merging data and splitting train:val:test = 6:2:2...")

    # Merge original training set (60000) and test set (10000) into 70000 images
    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
    print(f"  Total after merging: {X_all.shape[0]} images")

    # Split by ratio: training set 60% / validation set 20% / test set 20%
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_all, y_all, train_ratio=6/10, val_ratio=2/10, seed=args.seed
    )

    print(f"  Training set: {X_train.shape[0]} images")
    print(f"  Validation set: {X_val.shape[0]} images")
    print(f"  Test set: {X_test.shape[0]} images")

    # Save preprocessed data (compatible with team member's interface)
    save_processed_data(X_train, y_train, X_test, y_test)
    print("  Preprocessed data saved to data/processed/")

    # ==================== 3. Extract features (for baseline models) ====================
    print("\n[Stage 3] Extracting features...")

    # Parse feature method
    if "," in args.method:
        method = tuple(m.strip() for m in args.method.split(","))
    else:
        method = args.method

    print(f"  Feature method: {method}")

    # Create feature extractor
    extractor = FeatureExtractor()

    # Extract training set features
    X_train_features = extractor.fit_transform(
        X_train, method=method, pca_components=args.pca
    )

    # Extract validation and test set features
    X_val_features = extractor.transform(X_val, method=method)
    X_test_features = extractor.transform(X_test, method=method)

    print(f"  Feature dimension: {X_train_features.shape[1]}")
    print(f"  Training features: {X_train_features.shape}")
    print(f"  Validation features: {X_val_features.shape}")
    print(f"  Test features: {X_test_features.shape}")

    # 保存特征提取器
    features_dir = "data/features"
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    extractor.save(Path(features_dir) / "feature_extractor.pkl")
    print(f"  Feature extractor saved to {features_dir}/feature_extractor.pkl")

    # ==================== 4. Model training and evaluation ====================
    if not args.skip_training:
        print("\n[Stage 4] Training CNN + baseline models...")

        # Run complete training pipeline
        results = run_full_pipeline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_features, X_val_features, X_test_features,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed,
        )

        # ==================== 5. 保存结果 ====================
        print("\n[Stage 5] Saving results...")

        try:
            import torch
            import joblib

            # Ensure save directory exists
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)

            # Save CNN model
            torch.save(results["cnn_model"].state_dict(), f"{args.save_dir}/cnn_model.pth")
            print(f"  CNN model saved to: {args.save_dir}/cnn_model.pth")

            # Save baseline models
            for name, bl in results["baseline_results"].items():
                model_path = f"{args.save_dir}/{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}.pkl"
                joblib.dump(bl["model"], model_path)
                print(f"  {name} model saved to: {model_path}")

            # Save training results
            joblib.dump(results, f"{args.save_dir}/training_results.pkl")
            print(f"  Complete training results saved to: {args.save_dir}/training_results.pkl")

            # Save text summary
            _save_summary(results, args.save_dir)

        except Exception as e:
            print(f"  Error saving models: {e}")

    else:
        print("\n[Stage 4] Skipping model training")
        results = None

    # ==================== Completion ====================
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)

    print("\n" + "=" * 60)
    print("Handwritten Digit Recognition System Complete!")
    print(f"Time elapsed: {int(minutes)} minutes {int(seconds)} seconds")

    if results:
        # Extract test set accuracy
        for result_set in results["all_results"]:
            if result_set["split"] == "Test Set":
                print("\nFinal Test Set Results:")
                print("-" * 40)

                # Import evaluation functions
                from src.models.evaluate import compare_models, print_evaluation

                test_results = result_set["results"]
                print(compare_models(test_results))

                # Display CNN results
                for result in test_results:
                    if result["model_name"] == "CNN":
                        print(f"\nCNN test accuracy: {result['accuracy']:.4f}")
                        break
                break

    print(f"\nOutput directories:")
    print(f"  data/processed/    - Preprocessed image data")
    print(f"  data/features/     - Extracted features and feature extractor")
    print(f"  {args.save_dir}/      - Trained models and results")

    return results


def _save_summary(results: dict, save_dir: str):
    """
    Save text format results summary
    Reference implementation from run_all.py
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    with open(f"{save_dir}/evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write("Handwritten Digit Recognition System - Evaluation Results Summary\n")
        f.write("Data Split: Train:Val:Test = 6:2:2 (42000:14000:14000)\n")
        f.write("=" * 60 + "\n\n")

        # Detailed results for three sets
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

        # Three-set Accuracy summary comparison table
        f.write(f"\n{'='*60}\n")
        f.write("  [Three-set Accuracy Summary]\n")
        f.write(f"{'='*60}\n\n")

        # Table header
        header = f"{'Model':<25}"
        for split_info in results["all_results"]:
            header += f" {split_info['split']:>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # Each model row
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

    print(f"  Results summary saved: {save_dir}/evaluation_summary.txt")


if __name__ == "__main__":
    main()
