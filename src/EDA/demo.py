"""
EDA模块演示脚本
展示如何使用探索性数据分析功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.EDA.explorer import MNISTExplorer, load_and_analyze
from src.data.loader import load_mnist

import numpy as np
import warnings
import urllib.error


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


def demo_basic_usage():
    """演示基本用法"""
    print("=" * 60)
    print("MNIST数据集EDA分析演示")
    print("=" * 60)

    # 方法1: 直接加载数据并分析
    print("\n方法1: 加载预处理数据并运行全面分析")
    print("-" * 40)

    try:
        # 尝试从预处理数据加载
        explorer = load_and_analyze(data_dir="data/processed")
    except FileNotFoundError:
        print("预处理数据不存在，尝试从原始数据加载...")
        try:
            # 方法2: 从原始数据加载（需要torch）
            X_train, y_train, X_test, y_test = load_mnist(preprocess=True)
            explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

            # 运行部分分析
            explorer.print_summary()
            explorer.plot_label_distribution()
            explorer.plot_sample_images(n_samples=16)
            explorer.analyze_pixel_intensity()
        except ImportError as e:
            print(f"\n错误: {e}")
            print("\n解决方案:")
            print("  1. 安装PyTorch: pip install torch torchvision")
            print("  2. 或先运行 main.py 生成预处理数据:")
            print("     python main.py")
            print("\n然后重新运行此演示。")
            return
        except (urllib.error.URLError, ConnectionResetError) as e:
            print(f"\n网络错误: {e}")
            print("MNIST数据下载失败，使用模拟数据进行演示...")
            X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
            explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

            # 运行部分分析
            explorer.print_summary()
            explorer.plot_label_distribution()
            explorer.plot_sample_images(n_samples=16)
            explorer.analyze_pixel_intensity()
        except Exception as e:
            print(f"\n加载数据时出错: {e}")
            print("使用模拟数据进行演示...")
            X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
            explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

            # 运行部分分析
            explorer.print_summary()
            explorer.plot_label_distribution()
            explorer.plot_sample_images(n_samples=16)
            explorer.analyze_pixel_intensity()
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return


def demo_advanced_features():
    """演示高级功能"""
    print("\n\n" + "=" * 60)
    print("高级EDA功能演示")
    print("=" * 60)

    # 加载数据
    try:
        X_train, y_train, X_test, y_test = load_mnist(preprocess=True)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except ImportError as e:
        print(f"\n错误: {e}")
        print("\n高级功能需要PyTorch来加载MNIST原始数据。")
        print("解决方案:")
        print("  1. 安装PyTorch: pip install torch torchvision")
        print("  2. 或先运行 main.py 生成预处理数据，然后使用基础演示模式。")
        print("     python main.py")
        print("     python src/EDA/demo.py --mode=basic")
        return
    except (urllib.error.URLError, ConnectionResetError) as e:
        print(f"\n网络错误: {e}")
        print("MNIST数据下载失败，使用模拟数据进行演示...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"\n加载数据时出错: {e}")
        print("使用模拟数据进行演示...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # 1. 异常检测
    print("\n1. 异常样本检测")
    print("-" * 40)
    anomaly_indices = explorer.detect_anomalies(threshold=3.0)
    print(f"发现 {len(anomaly_indices)} 个异常样本")

    # 2. 类别不平衡分析
    print("\n2. 类别不平衡分析")
    print("-" * 40)
    explorer.analyze_class_imbalance()

    # 3. 按类别分析图像统计
    print("\n3. 按类别分析图像统计特征")
    print("-" * 40)
    explorer.analyze_image_statistics_by_class()

    # 4. 显示每个数字的示例
    print("\n4. 显示每个数字的示例图像")
    print("-" * 40)
    explorer.plot_digit_examples()


def demo_custom_analysis():
    """演示自定义分析"""
    print("\n\n" + "=" * 60)
    print("自定义分析演示")
    print("=" * 60)

    # 尝试加载数据
    try:
        X_train, y_train, X_test, y_test = load_mnist(preprocess=True)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except ImportError as e:
        print(f"\n错误: {e}")
        print("\n自定义分析需要PyTorch来加载MNIST原始数据。")
        print("解决方案:")
        print("  1. 安装PyTorch: pip install torch torchvision")
        print("  2. 或先运行 main.py 生成预处理数据，然后使用基础演示模式。")
        print("     python main.py")
        print("     python src/EDA/demo.py --mode=basic")
        return
    except (urllib.error.URLError, ConnectionResetError) as e:
        print(f"\n网络错误: {e}")
        print("MNIST数据下载失败，使用模拟数据进行演示...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"\n加载数据时出错: {e}")
        print("使用模拟数据进行演示...")
        X_train, y_train, X_test, y_test = create_mock_mnist_data(n_train=1000, n_test=200)
        explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # 创建自定义分析函数
    def analyze_specific_digit(explorer, digit: int, n_samples: int = 5):
        """分析特定数字的特征"""
        print(f"\n分析数字 {digit}:")
        mask = explorer.y_train == digit
        digit_images = explorer.X_train[mask]

        if len(digit_images) > 0:
            print(f"  样本数量: {len(digit_images)}")
            print(f"  平均像素强度: {digit_images.mean():.4f}")
            print(f"  像素强度标准差: {digit_images.std():.4f}")

            # 显示样本
            fig, axes = plt.subplots(1, min(n_samples, len(digit_images)), figsize=(3*n_samples, 3))
            for i in range(min(n_samples, len(digit_images))):
                axes[i].imshow(digit_images[i], cmap='gray')
                axes[i].set_title(f'数字 {digit} - 样本 {i+1}')
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()

    # 导入matplotlib
    import matplotlib.pyplot as plt

    # 分析几个特定的数字
    for digit in [0, 1, 8, 9]:
        analyze_specific_digit(explorer, digit, n_samples=3)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='MNIST数据集EDA分析演示')
    parser.add_argument('--mode', type=str, default='basic',
                        choices=['basic', 'advanced', 'custom', 'all'],
                        help='演示模式: basic(基本), advanced(高级), custom(自定义), all(全部)')

    args = parser.parse_args()

    if args.mode == 'basic' or args.mode == 'all':
        demo_basic_usage()

    if args.mode == 'advanced' or args.mode == 'all':
        demo_advanced_features()

    if args.mode == 'custom' or args.mode == 'all':
        demo_custom_analysis()

    print("\n" + "=" * 60)
    print("EDA演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()