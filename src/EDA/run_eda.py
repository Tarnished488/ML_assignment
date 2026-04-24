#!/usr/bin/env python
"""
Simple script to run EDA analysis
Quick start for exploratory data analysis
"""

import sys
from pathlib import Path
import urllib.error
import numpy as np

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import load_mnist, load_processed_data
from src.EDA import MNISTExplorer


def create_mock_mnist_data(n_train=1000, n_test=200):
    """
    Create mock MNIST data for demonstration
    Used when real MNIST data cannot be downloaded

    Args:
        n_train: Number of training samples
        n_test: Number of test samples

    Returns:
        X_train, y_train, X_test, y_test
    """
    print("\nNote: Using mock data instead of real MNIST data")
    print("Real MNIST data download failed, using generated mock data for demonstration")
    print("Mock data is only for demonstrating EDA features and does not represent real MNIST data characteristics")

    # Image size
    img_size = 28
    n_classes = 10

    # Create training data
    X_train = np.random.rand(n_train, img_size, img_size).astype(np.float32) * 0.3

    # Add different patterns for each class
    for i in range(n_train):
        label = i % n_classes
        # Add "digit" at different positions based on label
        center_x = 10 + (label % 3) * 8
        center_y = 10 + (label // 3) * 8

        # Add Gaussian blobs to simulate digits
        for x in range(img_size):
            for y in range(img_size):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 5:
                    X_train[i, x, y] += np.exp(-dist/3) * 0.7

    # Create training labels
    y_train = np.array([i % n_classes for i in range(n_train)])

    # Create test data (similar approach)
    X_test = np.random.rand(n_test, img_size, img_size).astype(np.float32) * 0.3
    for i in range(n_test):
        label = i % n_classes
        center_x = 10 + (label % 3) * 8
        center_y = 10 + (label // 3) * 8

        for x in range(img_size):
            for y in range(img_size):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 5:
                    X_test[i, x, y] += np.exp(-dist/3) * 0.7

    y_test = np.array([i % n_classes for i in range(n_test)])

    # Clip to [0, 1] range
    X_train = np.clip(X_train, 0, 1)
    X_test = np.clip(X_test, 0, 1)

    print(f"Mock data generated: train={n_train}, test={n_test}, image size={img_size}x{img_size}")

    return X_train, y_train, X_test, y_test


def main(allow_mock=True):
    """
    Main function for running EDA analysis

    Args:
        allow_mock: Whether to allow using mock data when real data download fails
    """
    print("=" * 60)
    print("MNIST Dataset Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # Try loading preprocessed data, fall back to raw data if not available
    try:
        print("\n[1/3] Trying to load preprocessed data...")
        X_train, y_train, X_test, y_test = load_processed_data()
        print("   Using saved preprocessed data")
    except FileNotFoundError:
        print("   Preprocessed data not found, trying to load raw data...")
        try:
            print("\n[1/3] Loading raw MNIST data (may require download)...")
            X_train, y_train, X_test, y_test = load_mnist(
                preprocess=True,
                use_mock_on_fail=allow_mock
            )
            print("   Raw data loaded successfully")
        except Exception as e:
            print(f"\nData loading failed: {e}")
            if not allow_mock:
                print("   Mock data is not allowed, EDA analysis terminated.")
                print("   Please check your network connection or use --allow-mock flag.")
                raise
            else:
                print("   MNIST data download failed, using mock data for demonstration...")
                X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)

    # Create EDA explorer
    print("\n[2/3] Initializing EDA explorer...")
    explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # Run basic analysis
    print("\n[3/3] Running exploratory analysis...")
    print("-" * 40)

    try:
        # 1. Label distribution analysis
        print("\n1. Label Distribution Analysis")
        print("-" * 40)
        explorer.print_summary()
        explorer.plot_label_distribution()

        # 2. Mean and standard deviation images
        print("\n2. Mean and Standard Deviation Images")
        print("-" * 40)
        explorer.plot_mean_and_std_images()

        # 3. Pixel value histogram analysis
        print("\n3. Pixel Value Histogram Analysis")
        print("-" * 40)
        explorer.analyze_pixel_intensity()

        # 4. PCA scatter plot visualization
        print("\n4. PCA Scatter Plot Visualization")
        print("-" * 40)
        explorer.plot_pca_scatter()

    except Exception as e:
        print(f"\nError during EDA analysis: {e}")
        print("Some visualization features may not work properly.")
        print("Please ensure all dependencies are installed: matplotlib, seaborn, scipy")

    print("\n" + "=" * 60)
    print("EDA Analysis Complete!")
    print("=" * 60)

    # Hint for more features
    print("\nFor more advanced features, run:")
    print("  python -m src.EDA.demo --mode=advanced")
    print("\nOr see the demo script:")
    print("  python src/EDA/demo.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MNIST Dataset Exploratory Data Analysis (EDA)")
    parser.add_argument("--allow-mock", action="store_true",
                       help="Allow using mock data when real data download fails")
    parser.add_argument("--no-mock", dest="allow_mock", action="store_false",
                       help="Disallow mock data; terminate when real data download fails")
    parser.set_defaults(allow_mock=True)

    args = parser.parse_args()
    main(allow_mock=args.allow_mock)
