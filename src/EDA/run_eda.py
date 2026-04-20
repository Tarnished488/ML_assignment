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
    print("\n注意: 使用模拟数据代替真实MNIST数据")
    print("真实MNIST数据下载失败，使用生成的模拟数据进行演示")
    print("模拟数据仅用于展示EDA功能，不代表真实MNIST数据特征")

    # 图像尺寸
    img_size = 28
    n_classes = 10

    # 创建训练数据
    X_train = np.random.rand(n_train, img_size, img_size).astype(np.float32) * 0.3

    # 为每个类别添加不同的模式
    for i in range(n_train):
        label = i % n_classes
        # 根据标签在图像的不同位置添加"数字"
        center_x = 10 + (label % 3) * 8
        center_y = 10 + (label // 3) * 8

        # 添加高斯斑点模拟数字
        for x in range(img_size):
            for y in range(img_size):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 5:
                    X_train[i, x, y] += np.exp(-dist/3) * 0.7

    # 创建训练标签
    y_train = np.array([i % n_classes for i in range(n_train)])

    # 创建测试数据（类似方式）
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

    # 限制在[0, 1]范围内
    X_train = np.clip(X_train, 0, 1)
    X_test = np.clip(X_test, 0, 1)

    print(f"模拟数据已生成: 训练集={n_train}, 测试集={n_test}, 图像尺寸={img_size}x{img_size}")

    return X_train, y_train, X_test, y_test


def main(allow_mock=True):
    """
    运行EDA分析的主函数

    Args:
        allow_mock: 是否允许在真实数据下载失败时使用模拟数据
    """
    print("=" * 60)
    print("MNIST数据集探索性分析 (EDA)")
    print("=" * 60)

    # 尝试加载预处理数据，如果不存在则从原始数据加载
    try:
        print("\n[1/3] 尝试加载预处理数据...")
        X_train, y_train, X_test, y_test = load_processed_data()
        print("   使用已保存的预处理数据")
    except FileNotFoundError:
        print("   预处理数据不存在，尝试加载原始数据...")
        try:
            print("\n[1/3] 加载MNIST原始数据（可能需要下载）...")
            X_train, y_train, X_test, y_test = load_mnist(
                preprocess=True,
                use_mock_on_fail=allow_mock
            )
            print("   原始数据加载完成")
        except Exception as e:
            print(f"\n数据加载失败: {e}")
            if not allow_mock:
                print("   模拟数据不允许使用，EDA分析终止。")
                print("   请检查网络连接或使用 --allow-mock 参数。")
                raise
            else:
                print("   MNIST数据下载失败，使用模拟数据进行演示...")
                X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)

    # 创建EDA探索器
    print("\n[2/3] 初始化EDA探索器...")
    explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # 运行基本分析
    print("\n[3/3] 运行探索性分析...")
    print("-" * 40)

    try:
        # 1. 标签分布分析
        print("\n1. 标签分布分析")
        print("-" * 40)
        explorer.print_summary()
        explorer.plot_label_distribution()

        # 2. 均值和标准差图像
        print("\n2. 均值和标准差图像")
        print("-" * 40)
        explorer.plot_mean_and_std_images()

        # 3. 像素值直方图分析
        print("\n3. 像素值直方图分析")
        print("-" * 40)
        explorer.analyze_pixel_intensity()

        # 4. PCA散点图可视化
        print("\n4. PCA散点图可视化")
        print("-" * 40)
        explorer.plot_pca_scatter()

    except Exception as e:
        print(f"\nEDA分析过程中出错: {e}")
        print("某些可视化功能可能无法正常工作。")
        print("请确保已安装所有依赖项：matplotlib, seaborn, scipy")

    print("\n" + "=" * 60)
    print("EDA分析完成!")
    print("=" * 60)

    # 提示更多功能
    print("\n要使用更多高级功能，请运行:")
    print("  python -m src.EDA.demo --mode=advanced")
    print("\n或查看演示脚本:")
    print("  python src/EDA/demo.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行MNIST数据集探索性分析 (EDA)")
    parser.add_argument("--allow-mock", action="store_true",
                       help="允许在真实数据下载失败时使用模拟数据")
    parser.add_argument("--no-mock", dest="allow_mock", action="store_false",
                       help="禁止使用模拟数据，真实数据下载失败时终止程序")
    parser.set_defaults(allow_mock=True)

    args = parser.parse_args()
    main(allow_mock=args.allow_mock)