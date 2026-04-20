"""
手写数字识别系统 - 主程序
集成数据加载、预处理、特征提取、模型训练全流程
"""

import sys
from pathlib import Path
import argparse

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_mnist, save_processed_data, get_data_info, load_processed_data
from src.data.preprocess import Preprocessor
from src.features.pipeline import build_feature_dataset
from src.features.extractor import load_features
from src.models.train import split_dataset, run_full_pipeline


def train_and_evaluate_model(X_train, y_train, X_test, y_test, X_train_features=None, X_test_features=None):
    """
    训练并评估机器学习模型（CNN + 基线模型）

    Args:
        X_train: 训练图像
        y_train: 训练标签
        X_test: 测试图像
        y_test: 测试标签
        X_train_features: 训练特征（用于基线模型）
        X_test_features: 测试特征（用于基线模型）

    Returns:
        dict: 包含所有模型结果
    """
    try:
        import numpy as np
        from src.models.evaluate import print_evaluation, compare_models
    except ImportError as e:
        print(f"错误: 缺少必要的依赖库: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        return None

    print("\n[模型训练] 准备数据...")

    # 合并训练集和测试集，重新划分 train:val:test = 6:2:2
    # 这是因为原始MNIST没有验证集，我们需要验证集来做早停
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    print(f"  合并后数据: {X_all.shape[0]} 张图像")
    print("  重新划分为 train:val:test = 6:2:2")

    # 划分数据集
    X_train_split, y_train_split, X_val_split, y_val_split, X_test_split, y_test_split = split_dataset(
        X_all, y_all, train_ratio=6/10, val_ratio=2/10, seed=42
    )

    print(f"  训练集: {X_train_split.shape[0]} 张图像")
    print(f"  验证集: {X_val_split.shape[0]} 张图像")
    print(f"  测试集: {X_test_split.shape[0]} 张图像")

    # 准备基线模型的特征数据
    if X_train_features is not None and X_test_features is not None:
        print("\n[模型训练] 准备特征数据...")
        # 同样需要合并并重新划分特征数据
        X_all_features = np.concatenate([X_train_features, X_test_features], axis=0)
        y_all_features = np.concatenate([y_train, y_test], axis=0)

        # 使用相同的随机种子划分特征数据
        X_train_features_split, y_train_features_split, X_val_features_split, y_val_features_split, X_test_features_split, y_test_features_split = split_dataset(
            X_all_features, y_all_features, train_ratio=6/10, val_ratio=2/10, seed=42
        )

        print(f"  训练特征: {X_train_features_split.shape}")
        print(f"  验证特征: {X_val_features_split.shape}")
        print(f"  测试特征: {X_test_features_split.shape}")
    else:
        print("\n[模型训练] 没有特征数据，跳过基线模型训练")
        X_train_features_split = X_val_features_split = X_test_features_split = None

    # 运行完整训练流程
    print("\n[模型训练] 开始训练...")
    results = run_full_pipeline(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        X_test_split, y_test_split,
        X_train_features_split, X_val_features_split, X_test_features_split,
        batch_size=128, epochs=30, lr=0.001, seed=42
    )

    # 打印测试集结果汇总
    print("\n" + "=" * 60)
    print("测试集最终结果汇总")
    print("=" * 60)

    # 提取测试集结果
    test_results = None
    for result_set in results["all_results"]:
        if result_set["split"] == "测试集":
            test_results = result_set["results"]
            break

    if test_results:
        print(compare_models(test_results))
        print("\n各模型详细结果:")
        for result in test_results:
            print_evaluation(result)

    # 保存模型
    try:
        import torch
        import joblib
        Path("results").mkdir(parents=True, exist_ok=True)

        # 保存CNN模型
        torch.save(results["cnn_model"].state_dict(), "results/cnn_model.pth")
        print(f"\n[模型保存] CNN模型已保存至: results/cnn_model.pth")

        # 保存基线模型
        for name, bl in results["baseline_results"].items():
            model_path = f"results/{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}.pkl"
            joblib.dump(bl["model"], model_path)
            print(f"[模型保存] {name} 模型已保存至: {model_path}")

        # 保存训练结果
        joblib.dump(results, "results/training_results.pkl")
        print(f"[模型保存] 完整训练结果已保存至: results/training_results.pkl")

    except Exception as e:
        print(f"\n[模型保存] 保存模型时出错: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="手写数字识别系统全流程")
    parser.add_argument("--skip-preprocess", action="store_true",
                       help="跳过预处理，使用已保存的预处理数据")
    parser.add_argument("--skip-features", action="store_true",
                       help="跳过特征提取，使用已保存的特征数据")
    parser.add_argument("--method", default="hog,lbp,shape",
                       help="特征提取方法: raw, hog, lbp, shape 或逗号分隔的组合")
    parser.add_argument("--pca", type=float, default=None,
                       help="PCA降维: None为不降维, 整数为组件数, 浮点数为保留方差比例")
    parser.add_argument("--skip-training", action="store_true",
                       help="跳过模型训练，只执行数据准备")

    args = parser.parse_args()

    print("=" * 60)
    print("手写数字识别系统 - 全流程集成")
    print("=" * 60)

    # ==================== 1. 数据加载 ====================
    print("\n[阶段1] 数据加载")
    print("-" * 40)

    if not args.skip_preprocess:
        print("加载原始MNIST数据集...")
        X_train, y_train, X_test, y_test = load_mnist(preprocess=True)

        # 数据统计信息
        info = get_data_info(X_train, y_train, X_test, y_test)
        print(f"  训练集: {info['n_train']} 张图像")
        print(f"  测试集: {info['n_test']} 张图像")
        print(f"  标签分布: {info['train_label_dist']}")
    else:
        print("加载已预处理的MNIST数据...")
        X_train, y_train, X_test, y_test = load_processed_data()
        print(f"  训练集: {X_train.shape[0]} 张图像")
        print(f"  测试集: {X_test.shape[0]} 张图像")

    # ==================== 2. 数据预处理 ====================
    print("\n[阶段2] 数据预处理")
    print("-" * 40)

    if not args.skip_preprocess:
        print("执行预处理流水线...")
        preprocessor = Preprocessor()
        X_train_processed = preprocessor.preprocess_pipeline(X_train)
        X_test_processed = preprocessor.preprocess_pipeline(X_test)

        print(f"  预处理完成:")
        print(f"    训练集: {X_train.shape} -> {X_train_processed.shape}")
        print(f"    测试集: {X_test.shape} -> {X_test_processed.shape}")

        # 保存预处理数据
        print("保存预处理数据...")
        save_processed_data(X_train_processed, y_train, X_test_processed, y_test)
        print("  数据已保存至 data/processed/")
    else:
        print("跳过预处理，使用已保存的预处理数据")
        X_train_processed, y_train, X_test_processed, y_test = X_train, y_train, X_test, y_test

    # ==================== 3. 特征提取 ====================
    print("\n[阶段3] 特征提取")
    print("-" * 40)

    if not args.skip_features:
        print(f"提取特征 (方法: {args.method})...")

        # 解析特征方法
        if "," in args.method:
            method = tuple(m.strip() for m in args.method.split(","))
        else:
            method = args.method

        # 构建特征数据集
        X_train_features, y_train, X_test_features, y_test = build_feature_dataset(
            processed_dir="data/processed",
            save_dir="data/features",
            method=method,
            pca_components=args.pca,
            save_extractor=True
        )

        print(f"  特征提取完成:")
        print(f"    训练特征: {X_train_features.shape}")
        print(f"    测试特征: {X_test_features.shape}")
        print(f"    特征已保存至 data/features/")
    else:
        print("加载已提取的特征...")
        try:
            X_train_features, y_train, X_test_features, y_test = load_features("data/features")
            print(f"  训练特征: {X_train_features.shape}")
            print(f"  测试特征: {X_test_features.shape}")
        except Exception as e:
            print(f"  警告: 加载特征失败: {e}")
            print("  将跳过特征提取和基线模型训练")
            X_train_features = X_test_features = None

    # ==================== 4. 模型训练与评估 ====================
    print("\n[阶段4] 模型训练与评估")
    print("-" * 40)

    if not args.skip_training:
        results = train_and_evaluate_model(
            X_train_processed, y_train,
            X_test_processed, y_test,
            X_train_features, X_test_features
        )
    else:
        print("跳过模型训练")
        results = None

    # ==================== 结果汇总 ====================
    print("\n" + "=" * 60)
    print("手写数字识别系统全流程完成!")
    print("=" * 60)

    if results:
        # 提取测试集准确率
        test_accuracy = 0.0
        for result_set in results["all_results"]:
            if result_set["split"] == "测试集":
                for result in result_set["results"]:
                    if result["model_name"] == "CNN":
                        test_accuracy = result["accuracy"]
                        break
                break
        print(f"最终测试准确率: {test_accuracy:.4f}")

    print("\n输出目录:")
    print("  data/processed/    - 预处理图像数据")
    print("  data/features/     - 提取的特征数据")
    print("  results/           - 训练好的模型和结果")

    return results


if __name__ == "__main__":
    main()
