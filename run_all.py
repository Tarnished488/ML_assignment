"""
一键运行入口

这是整个项目的总入口, 运行这一条命令就能完成全部流程:

    python run_all.py

=== 完整流程 (5 步) ===

Step 1: 加载 MNIST 数据 + 预处理 (降噪/二值化/中心化)
Step 2: 合并全部 70000 张图, 按 6:2:2 重新划分 (42000 + 14000 + 14000)
Step 3: 提取 HOG+LBP+Shape 特征 (供基线模型使用)
Step 4: 训练 CNN + 3 个基线模型, 在三个集合上分别评估
Step 5: 生成可视化图表 + 保存结果摘要

=== 数据划分 ===

训练集:验证集:测试集 = 6:2:2 (42000:14000:14000)

为什么把原始的 60000+10000 合并再分:
- 原始 MNIST 没有验证集, 只有训练集和测试集
- 我们需要验证集来做早停 (防止过拟合)
- 所以把全部 70000 张合并后, 按比例随机重新划分

=== 用法 ===

# 默认 30 epochs (完整训练)
python run_all.py

# 快速测试 (3 epochs, 几分钟就跑完)
python run_all.py --epochs 3 --batch-size 256

# 自定义参数
python run_all.py --epochs 20 --lr 0.0005 --batch-size 64
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np

# 把项目根目录加到 Python 搜索路径, 这样才能 import src.xxx
sys.path.insert(0, str(Path(__file__).parent))

# 导入各模块
from src.data.loader import load_mnist, save_processed_data      # 数据加载
from src.data.preprocess import Preprocessor                       # 数据预处理
from src.features.extractor import FeatureExtractor                # 特征提取
from src.models.train import run_full_pipeline, set_seed, split_dataset  # 模型训练
from src.visualization.plot import generate_all_plots              # 可视化


def main():
    # ---- 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="MNIST 手写数字识别 - 完整流程")
    parser.add_argument("--epochs", type=int, default=30, help="CNN 训练轮数 (默认 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小 (默认 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率 (默认 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument("--save-dir", type=str, default="results", help="结果保存目录")
    args = parser.parse_args()

    set_seed(args.seed)
    start_time = time.time()

    print("=" * 60)
    print("  MNIST 手写数字识别 - 自动化训练与评估")
    print("  数据划分: 训练集:验证集:测试集 = 6:2:2")
    print("=" * 60)

    # ================================================================
    # Step 1: 加载 MNIST 数据并预处理
    # ================================================================
    print("\n[Step 1/5] 加载 MNIST 数据并预处理...")

    # load_mnist: 下载 (首次) 并加载 MNIST 数据
    # data_dir: 数据存放目录
    # download: 如果数据不存在, 自动从网络下载
    # preprocess=False: 先不做预处理, 我们自己控制预处理流程
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist(
        data_dir="data/raw", download=True, preprocess=False
    )

    # 预处理: 降噪 (中值滤波) → 二值化 (阈值 0.3) → 中心化 (移到画布中央)
    preprocessor = Preprocessor()
    X_train_raw = preprocessor.preprocess_pipeline(X_train_raw)
    X_test_raw = preprocessor.preprocess_pipeline(X_test_raw)
    print(f"  原始数据: 训练集 {X_train_raw.shape}, 测试集 {X_test_raw.shape}")

    # ================================================================
    # Step 2: 合并后按 6:2:2 重新划分
    # ================================================================
    print("\n[Step 2/5] 合并数据并按 train:val:test = 6:2:2 划分...")

    # 把原始的训练集 (60000) 和测试集 (10000) 合并成 70000 张
    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
    print(f"  合并后总量: {X_all.shape[0]} 张图像")

    # 按比例划分: 训练集 60% / 验证集 20% / 测试集 20%
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_all, y_all, train_ratio=6/10, val_ratio=2/10, seed=args.seed
    )
    print(f"  训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

    # 保存预处理数据 (兼容组员的接口, 组员可能需要这些数据)
    save_processed_data(X_train, y_train, X_test, y_test)

    # ================================================================
    # Step 3: 提取特征 (供基线模型使用)
    # ================================================================
    print("\n[Step 3/5] 提取 HOG+LBP+Shape 特征...")

    # FeatureExtractor: 组员写的特征提取器
    # HOG (梯度方向直方图): 提取笔画方向信息, 324 维
    # LBP (局部二值模式): 提取纹理信息, 10 维
    # Shape (形状统计): 宽高比、重心、投影等, 71 维
    # 合计 405 维
    extractor = FeatureExtractor()

    # fit_transform: 在训练集上拟合并转换 (计算必要的统计量)
    X_train_features = extractor.fit_transform(
        X_train, method=("hog", "lbp", "shape"), pca_components=None
    )
    # transform: 用训练集上拟合好的参数, 转换验证集和测试集
    X_val_features = extractor.transform(X_val, method=("hog", "lbp", "shape"))
    X_test_features = extractor.transform(X_test, method=("hog", "lbp", "shape"))
    print(f"  特征维度: {X_train_features.shape[1]}")

    # ================================================================
    # Step 4: 训练模型 + 三集评估
    # ================================================================
    print("\n[Step 4/5] 训练 CNN + 基线模型, 并在三个集合上分别评估...")

    # run_full_pipeline 做的事:
    # 1. 训练 CNN (PyTorch, Focal Loss + Label Smoothing)
    # 2. 训练 KNN / Logistic Regression / Random Forest (sklearn)
    # 3. 在训练集、验证集、测试集上分别评估所有模型
    results = run_full_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test,
        X_train_features, X_val_features, X_test_features,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed,
    )

    # ================================================================
    # Step 5: 生成可视化 + 保存摘要
    # ================================================================
    print("\n[Step 5/5] 生成可视化图表并保存结果摘要...")

    # generate_all_plots: 生成全部图表 (训练曲线、混淆矩阵、对比图、Grad-CAM 等)
    generate_all_plots(results, X_test, y_test, save_dir=args.save_dir)

    # _save_summary: 保存文本格式的评估结果摘要
    _save_summary(results, args.save_dir)

    # ---- 完成 ----
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n{'='*60}")
    print(f"  全部完成! 耗时: {int(minutes)}分 {int(seconds)}秒")
    print(f"  结果保存在: {args.save_dir}/")
    print(f"{'='*60}")


def _save_summary(results: dict, save_dir: str):
    """
    保存文本格式的结果摘要到 evaluation_summary.txt

    摘要包含:
        1. 每个集合 (训练/验证/测试) 的每个模型的详细指标
        2. 三集合 Accuracy 汇总对比表
        3. 过拟合分析 (训练-验证准确率差距)

    参数:
        results: run_full_pipeline() 的返回值
        save_dir: 保存目录
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    with open(f"{save_dir}/evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write("MNIST 手写数字识别 - 评估结果摘要\n")
        f.write("数据划分: 训练集:验证集:测试集 = 6:2:2 (42000:14000:14000)\n")
        f.write("损失函数: Focal Loss (gamma=2.0) + Label Smoothing (0.1)\n")
        f.write("=" * 60 + "\n\n")

        # ---- Part 1: 三个集合的详细结果 ----
        for split_info in results["all_results"]:
            split_name = split_info["split"]
            split_results = split_info["results"]

            f.write(f"{'='*60}\n")
            f.write(f"  【{split_name}】\n")
            f.write(f"{'='*60}\n\n")

            for r in split_results:
                f.write(f"  模型: {r['model_name']}\n")
                f.write(f"    Accuracy:  {r['accuracy']:.4f}\n")
                f.write(f"    Precision: {r['precision_macro']:.4f} (macro)\n")
                f.write(f"    Recall:    {r['recall_macro']:.4f} (macro)\n")
                f.write(f"    F1-Score:  {r['f1_macro']:.4f} (macro)\n\n")

                f.write("    各类别指标:\n")
                f.write(r["report"] + "\n")
                f.write("-" * 50 + "\n\n")

        # ---- Part 2: 三集合 Accuracy 汇总对比表 ----
        f.write(f"\n{'='*60}\n")
        f.write("  【三集合 Accuracy 汇总】\n")
        f.write(f"{'='*60}\n\n")

        # 表头
        header = f"{'模型':<25}"
        for split_info in results["all_results"]:
            header += f" {split_info['split']:>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # 每个模型一行
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

        # ---- Part 3: 过拟合分析 ----
        history = results["history"]
        if history["train_acc"] and history["val_acc"]:
            gap = history["train_acc"][-1] - history["val_acc"][-1]
            f.write(f"\n  过拟合分析:\n")
            f.write(f"    最终训练准确率: {history['train_acc'][-1]:.2f}%\n")
            f.write(f"    最终验证准确率: {history['val_acc'][-1]:.2f}%\n")
            f.write(f"    训练-验证差距: {gap:.2f}%\n")
            if gap > 5:
                f.write("    可能存在过拟合, 建议: 增加数据增强、增大 Dropout、减少参数\n")
            elif gap < -3:
                f.write("    训练准确率低于验证准确率 (数据增强+正则化导致), 模型泛化良好\n")
            else:
                f.write("    过拟合程度较低, 模型泛化良好\n")

    print(f"  结果摘要已保存: {save_dir}/evaluation_summary.txt")


if __name__ == "__main__":
    main()
