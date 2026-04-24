"""
EDA Module Demo Script
Demonstrates how to use the exploratory data analysis features
"""

import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.EDA.explorer import MNISTExplorer, load_and_analyze
from src.data.loader import load_mnist

import numpy as np
import warnings
import urllib.error


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


def demo_basic_usage():
    """Demonstrate basic usage"""
    print("=" * 60)
    print("MNIST Dataset EDA Analysis Demo")
    print("=" * 60)

    # Method 1: Load data directly and analyze
    print("\nMethod 1: Load preprocessed data and run full analysis")
    print("-" * 40)

    try:
        # Try loading from preprocessed data
        explorer = load_and_analyze(data_dir="data/processed")
    except FileNotFoundError:
        print("Preprocessed data not found, trying to load raw data...")
        try:
            # Method 2: Load from raw data (requires torch)
            X_train, y_train, X_test, y_test = load_mnist(preprocess=True)
            explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

            # Run partial analysis
            explorer.print_summary()
            explorer.plot_label_distribution()
            explorer.plot_sample_images(n_samples=16)
            explorer.analyze_pixel_intensity()
        except ImportError as e:
            print(f"\nError: {e}")
            print("\nSolutions:")
            print("  1. Install PyTorch: pip install torch torchvision")
            print("  2. Or run main.py first to generate preprocessed data:")
            print("     python main.py")
            print("\nThen re-run this demo.")
            return
        except (urllib.error.URLError, ConnectionResetError) as e:
            print(f"\nNetwork error: {e}")
            print("MNIST data download failed, using mock data for demonstration...")
            X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
            explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

            # Run partial analysis
            explorer.print_summary()
            explorer.plot_label_distribution()
            explorer.plot_sample_images(n_samples=16)
            explorer.analyze_pixel_intensity()
        except Exception as e:
            print(f"\nError loading data: {e}")
            print("Using mock data for demonstration...")
            X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
            explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

            # Run partial analysis
            explorer.print_summary()
            explorer.plot_label_distribution()
            explorer.plot_sample_images(n_samples=16)
            explorer.analyze_pixel_intensity()
    except Exception as e:
        print(f"Error loading data: {e}")
        return


def demo_advanced_features():
    """Demonstrate advanced features"""
    print("\n\n" + "=" * 60)
    print("Advanced EDA Features Demo")
    print("=" * 60)

    # Load data
    try:
        X_train, y_train, X_test, y_test = load_mnist(preprocess=True)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nAdvanced features require PyTorch to load raw MNIST data.")
        print("Solutions:")
        print("  1. Install PyTorch: pip install torch torchvision")
        print("  2. Or run main.py first to generate preprocessed data, then use basic demo mode.")
        print("     python main.py")
        print("     python src/EDA/demo.py --mode=basic")
        return
    except (urllib.error.URLError, ConnectionResetError) as e:
        print(f"\nNetwork error: {e}")
        print("MNIST data download failed, using mock data for demonstration...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Using mock data for demonstration...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # 1. Anomaly detection
    print("\n1. Anomaly Sample Detection")
    print("-" * 40)
    anomaly_indices = explorer.detect_anomalies(threshold=3.0)
    print(f"Found {len(anomaly_indices)} anomaly samples")

    # 2. Class imbalance analysis
    print("\n2. Class Imbalance Analysis")
    print("-" * 40)
    explorer.analyze_class_imbalance()

    # 3. Per-class image statistics
    print("\n3. Per-class Image Statistics")
    print("-" * 40)
    explorer.analyze_image_statistics_by_class()

    # 4. Show examples of each digit
    print("\n4. Show Example Images for Each Digit")
    print("-" * 40)
    explorer.plot_digit_examples()


def demo_custom_analysis():
    """Demonstrate custom analysis"""
    print("\n\n" + "=" * 60)
    print("Custom Analysis Demo")
    print("=" * 60)

    # Try loading data
    try:
        X_train, y_train, X_test, y_test = load_mnist(preprocess=True)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nCustom analysis requires PyTorch to load raw MNIST data.")
        print("Solutions:")
        print("  1. Install PyTorch: pip install torch torchvision")
        print("  2. Or run main.py first to generate preprocessed data, then use basic demo mode.")
        print("     python main.py")
        print("     python src/EDA/demo.py --mode=basic")
        return
    except (urllib.error.URLError, ConnectionResetError) as e:
        print(f"\nNetwork error: {e}")
        print("MNIST data download failed, using mock data for demonstration...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Using mock data for demonstration...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # Create custom analysis function
    def analyze_specific_digit(explorer, digit: int, n_samples: int = 5):
        """Analyze features of a specific digit"""
        print(f"\nAnalyzing digit {digit}:")
        mask = explorer.y_train == digit
        digit_images = explorer.X_train[mask]

        if len(digit_images) > 0:
            print(f"  Number of samples: {len(digit_images)}")
            print(f"  Mean pixel intensity: {digit_images.mean():.4f}")
            print(f"  Pixel intensity std: {digit_images.std():.4f}")

            # Show samples
            fig, axes = plt.subplots(1, min(n_samples, len(digit_images)), figsize=(3*n_samples, 3))
            for i in range(min(n_samples, len(digit_images))):
                axes[i].imshow(digit_images[i], cmap='gray')
                axes[i].set_title(f'Digit {digit} - Sample {i+1}')
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()

    # Import matplotlib
    import matplotlib.pyplot as plt

    # Analyze a few specific digits
    for digit in [0, 1, 8, 9]:
        analyze_specific_digit(explorer, digit, n_samples=3)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='MNIST Dataset EDA Analysis Demo')
    parser.add_argument('--mode', type=str, default='basic',
                        choices=['basic', 'advanced', 'custom', 'all'],
                        help='Demo mode: basic, advanced, custom, all')

    args = parser.parse_args()

    if args.mode == 'basic' or args.mode == 'all':
        demo_basic_usage()

    if args.mode == 'advanced' or args.mode == 'all':
        demo_advanced_features()

    if args.mode == 'custom' or args.mode == 'all':
        demo_custom_analysis()

    print("\n" + "=" * 60)
    print("EDA Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
