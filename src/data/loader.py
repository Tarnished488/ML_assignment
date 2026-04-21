"""
MNIST 数据集加载模块
自动下载数据集并提供统一的接口
"""

import numpy as np
import torch
from torchvision import datasets
from pathlib import Path
import cv2  # 新增：用于图像处理
from scipy.ndimage import center_of_mass, shift  # 新增：用于中心化


def preprocess_images(images, binarize=True, denoise=True, center=True):
    """
    图像预处理：二值化、降噪、中心化

    Args:
        images: 输入图像数组, shape=(N, 28, 28), dtype=float32
        binarize: 是否二值化 (阈值0.5)
        denoise: 是否降噪 (高斯滤波)
        center: 是否中心化 (移至图像中心)

    Returns:
        处理后的图像数组, shape=(N, 28, 28)
    """
    processed = images.copy()
    for i in range(len(processed)):
        img = processed[i]
        if denoise:
            img = cv2.GaussianBlur(img, (3, 3), 0)  # 降噪
        if binarize:
            _, img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)  # 二值化
        if center:
            # 计算质心并移至中心 (14,14)
            com = center_of_mass(img)
            shift_y = 14 - com[0]
            shift_x = 14 - com[1]
            img = shift(img, (shift_y, shift_x), mode='constant', cval=0)
        processed[i] = img
    return processed


def load_mnist(data_dir: str = "data/raw", download: bool = True, preprocess: bool = True):
    """
    加载 MNIST 数据集

    Args:
        data_dir: 数据存储目录
        download: 是否自动下载
        preprocess: 是否应用图像预处理 (二值化、降噪、中心化)

    Returns:
        X_train: 训练图像, shape=(60000, 28, 28), dtype=float32, 范围[0, 1]
        y_train: 训练标签, shape=(60000,)
        X_test: 测试图像, shape=(10000, 28, 28), dtype=float32, 范围[0, 1]
        y_test: 测试标签, shape=(10000,)
    """
    # 确保目录存在
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # 下载并加载训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=None
    )

    # 下载并加载测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=None
    )

    # 转换为 numpy 数组并归一化到 [0, 1]
    X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy()

    if preprocess:
        X_train = preprocess_images(X_train)
        X_test = preprocess_images(X_test)

    print(f"[数据加载完成]")
    print(f"  训练集: {X_train.shape[0]} 张图像")
    print(f"  测试集: {X_test.shape[0]} 张图像")
    print(f"  图像尺寸: {X_train.shape[1]} x {X_train.shape[2]}")
    print(f"  像素范围: [{X_train.min():.2f}, {X_train.max():.2f}]")

    return X_train, y_train, X_test, y_test


def get_data_info(X_train, y_train, X_test, y_test):
    """
    获取数据集统计信息

    Returns:
        dict: 包含数据分布等统计信息
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
    保存预处理后的数据

    Args:
        X_train, y_train, X_test, y_test: 数据
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "X_train.npy", X_train)
    np.save(save_path / "y_train.npy", y_train)
    np.save(save_path / "X_test.npy", X_test)
    np.save(save_path / "y_test.npy", y_test)

    print(f"[数据已保存至] {save_path}")


def load_processed_data(data_dir: str = "data/processed"):
    """
    加载预处理后的数据

    Returns:
        X_train, y_train, X_test, y_test
    """
    data_path = Path(data_dir)

    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")

    print(f"[预处理数据加载完成]")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 测试数据加载
    X_train, y_train, X_test, y_test = load_mnist()
    info = get_data_info(X_train, y_train, X_test, y_test)
    print(f"\n标签分布: {info['train_label_dist']}")
