"""
MNIST dataset loading module
Automatically downloads the dataset and provides a unified interface
"""

import numpy as np
import torch
from torchvision import datasets
from pathlib import Path
import cv2  # Added: for image processing
from scipy.ndimage import center_of_mass, shift  # Added: for centering


def preprocess_images(images, binarize=True, denoise=True, center=True):
    """
    Image preprocessing: binarization, denoising, centering

    Args:
        images: input image array, shape=(N, 28, 28), dtype=float32
        binarize: whether to binarize (threshold 0.5)
        denoise: whether to denoise (Gaussian filter)
        center: whether to center (shift to image center)

    Returns:
        processed image array, shape=(N, 28, 28)
    """
    processed = images.copy()
    for i in range(len(processed)):
        img = processed[i]
        if denoise:
            img = cv2.GaussianBlur(img, (3, 3), 0)  # Denoising
        if binarize:
            _, img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)  # Binarization
        if center:
            # Compute center of mass and shift to center (14,14)
            com = center_of_mass(img)
            shift_y = 14 - com[0]
            shift_x = 14 - com[1]
            img = shift(img, (shift_y, shift_x), mode='constant', cval=0)
        processed[i] = img
    return processed


def load_mnist(data_dir: str = "data/raw", download: bool = True, preprocess: bool = True):
    """
    Load MNIST dataset

    Args:
        data_dir: data storage directory
        download: whether to download automatically
        preprocess: whether to apply image preprocessing (binarization, denoising, centering)

    Returns:
        X_train: training images, shape=(60000, 28, 28), dtype=float32, range [0, 1]
        y_train: training labels, shape=(60000,)
        X_test: test images, shape=(10000, 28, 28), dtype=float32, range [0, 1]
        y_test: test labels, shape=(10000,)
    """
    # Ensure directory exists
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Download and load training set
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=None
    )

    # Download and load test set
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=None
    )

    # Convert to numpy arrays and normalize to [0, 1]
    X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy()

    if preprocess:
        X_train = preprocess_images(X_train)
        X_test = preprocess_images(X_test)

    print(f"[Data loading complete]")
    print(f"  Training set: {X_train.shape[0]} images")
    print(f"  Test set: {X_test.shape[0]} images")
    print(f"  Image size: {X_train.shape[1]} x {X_train.shape[2]}")
    print(f"  Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")

    return X_train, y_train, X_test, y_test


def get_data_info(X_train, y_train, X_test, y_test):
    """
    Get dataset statistics

    Returns:
        dict: statistics including data distribution
    """
    info = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'image_shape': X_train[0].shape,
        'train_label_dist': np.bincount(y_train, minlength=10),
        'test_label_dist': np.bincount(y_test, minlength=10),
    }
    return info


def save_processed_data(X_train, y_train, X_test, y_test, save_dir: str = "data/processed"):
    """
    Save preprocessed data

    Args:
        X_train, y_train, X_test, y_test: data
        save_dir: save directory
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "X_train.npy", X_train)
    np.save(save_path / "y_train.npy", y_train)
    np.save(save_path / "X_test.npy", X_test)
    np.save(save_path / "y_test.npy", y_test)

    print(f"[Data saved to] {save_path}")


def load_processed_data(data_dir: str = "data/processed"):
    """
    Load preprocessed data

    Returns:
        X_train, y_train, X_test, y_test
    """
    data_path = Path(data_dir)

    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")

    print(f"[Preprocessed data loading complete]")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Test data loading
    X_train, y_train, X_test, y_test = load_mnist()
    info = get_data_info(X_train, y_train, X_test, y_test)
    print(f"\nLabel distribution: {info['train_label_dist']}")
