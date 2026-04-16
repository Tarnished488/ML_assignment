"""
手写数字识别系统 - 主程序
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_mnist, save_processed_data, get_data_info
from src.data.preprocess import Preprocessor


def main():
    print("=" * 50)
    print("手写数字识别系统 - 数据预处理模块")
    print("=" * 50)

    # 1. 加载原始数据
    print("\n[步骤1] 加载 MNIST 数据集...")
    X_train, y_train, X_test, y_test = load_mnist()

    # 2. 数据统计信息
    print("\n[步骤2] 数据统计信息...")
    info = get_data_info(X_train, y_train, X_test, y_test)
    print(f"  训练集标签分布: {info['train_label_dist']}")
    print(f"  测试集标签分布: {info['test_label_dist']}")

    # 3. 预处理
    print("\n[步骤3] 执行预处理...")
    preprocessor = Preprocessor()

    # 应用完整预处理流水线
    X_train_processed = preprocessor.preprocess_pipeline(X_train)
    X_test_processed = preprocessor.preprocess_pipeline(X_test)

    print(f"  预处理完成: 训练集 {X_train_processed.shape}, 测试集 {X_test_processed.shape}")

    # 4. 保存预处理后的数据
    print("\n[步骤4] 保存预处理数据...")
    save_processed_data(X_train_processed, y_train,
                       X_test_processed, y_test)

    print("\n" + "=" * 50)
    print("数据预处理完成！其他组员可以开始工作。")
    print("=" * 50)

    return X_train_processed, y_train, X_test_processed, y_test


if __name__ == "__main__":
    main()
