#!/usr/bin/env python
"""
运行EDA分析的简单脚本
快速开始探索性数据分析
"""

import sys
from pathlib import Path
import urllib.error
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import load_mnist, load_processed_data
from src.EDA import MNISTExplorer


def create_mock_mnist_data(n_train=1000, n_test=200):
    """
    创建模拟的MNIST数据用于演示
    当无法下载真实MNIST数据时使用

    Args:
        n_train: 训练样本数
        n_test: 测试样本数

    Returns:
        X_train, y_train, X_test, y_test
    """
    print("\nNote: Using simulated data instead of real MNIST data")
    print("Real MNIST data download failed, using generated simulated data for demonstration")
    print("Simulated data is only for demonstrating EDA functionality, does not represent real MNIST data characteristics")

    # Image size
    img_size = 28
    n_classes = 10

    # Create training data
    X_train = np.random.rand(n_train, img_size, img_size).astype(np.float32) * 0.3

    # Add different patterns for each class
    for i in range(n_train):
        label = i % n_classes
        # Add "digits" at different positions in the image based on label
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

    # Create test data (similar method)
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

    # Limit to [0, 1] range
    X_train = np.clip(X_train, 0, 1)
    X_test = np.clip(X_test, 0, 1)

    print(f"Simulated data generated: training set={n_train}, test set={n_test}, image size={img_size}x{img_size}")

    return X_train, y_train, X_test, y_test


def main(allow_mock=True, output_dir=None):
    """
    Main function to run EDA analysis

    Args:
        allow_mock: Whether to allow using simulated data when real data download fails
    """
    print("=" * 60)
    print("MNIST Dataset Exploratory Analysis (EDA)")
    print("=" * 60)

    # Try to load preprocessed data, if not exist then load from raw data
    try:
        print("\n[1/3] Attempting to load preprocessed data...")
        X_train, y_train, X_test, y_test = load_processed_data()
        print("   Using saved preprocessed data")
    except FileNotFoundError:
        print("   Preprocessed data does not exist, attempting to load raw data...")
        try:
            print("\n[1/3] Loading MNIST raw data (may need to download)...")
            X_train, y_train, X_test, y_test = load_mnist(
                preprocess=True,
                use_mock_on_fail=allow_mock
            )
            print("   Raw data loading completed")
        except Exception as e:
            print(f"\nData loading failed: {e}")
            if not allow_mock:
                print("   Simulated data not allowed, EDA analysis terminated.")
                print("   Please check network connection or use --allow-mock parameter.")
                raise
            else:
                print("   MNIST data download failed, using simulated data for demonstration...")
                X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)

    # Create EDA explorer
    print("\n[2/3] Initializing EDA explorer...")
    explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # Run basic analysis
    print("\n[3/3] Running exploratory analysis...")
    print("-" * 40)

    try:
        # Create output directory if it doesn't exist
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Label distribution analysis
        print("\n1. Label distribution analysis")
        print("-" * 40)
        explorer.print_summary()
        save_path = str(Path(output_dir) / "label_distribution.png") if output_dir else None
        explorer.plot_label_distribution(save_path=save_path)

        # 2. Mean and standard deviation images
        print("\n2. Mean and standard deviation images")
        print("-" * 40)
        save_path = str(Path(output_dir) / "mean_std_images.png") if output_dir else None
        explorer.plot_mean_and_std_images(save_path=save_path)

        # 3. Pixel value histogram analysis
        print("\n3. Pixel value histogram analysis")
        print("-" * 40)
        save_path = str(Path(output_dir) / "pixel_intensity.png") if output_dir else None
        explorer.analyze_pixel_intensity(save_path=save_path)

        # 4. PCA scatter plot visualization
        print("\n4. PCA scatter plot visualization")
        print("-" * 40)
        save_path = str(Path(output_dir) / "pca_scatter.png") if output_dir else None
        explorer.plot_pca_scatter(save_path=save_path)

    except Exception as e:
        print(f"\nError during EDA analysis: {e}")
        print("Some visualization functions may not work properly.")
        print("Please ensure all dependencies are installed: matplotlib, seaborn, scipy")

    print("\n" + "=" * 60)
    print("EDA analysis completed!")
    print("=" * 60)

    # Prompt for more features
    print("\nTo use more advanced features, run:")
    print("  python -m src.EDA.demo --mode=advanced")
    print("\nOr view the demo script:")
    print("  python src/EDA/demo.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MNIST dataset exploratory analysis (EDA)")
    parser.add_argument("--allow-mock", action="store_true",
                       help="Allow using simulated data when real data download fails")
    parser.add_argument("--no-mock", dest="allow_mock", action="store_false",
                       help="Disable using simulated data, terminate program when real data download fails")
    parser.add_argument("--output-dir", type=str, default="eda_results",
                       help="Output directory for saving EDA plots")
    parser.set_defaults(allow_mock=True)

    args = parser.parse_args()
    main(allow_mock=args.allow_mock, output_dir=args.output_dir)