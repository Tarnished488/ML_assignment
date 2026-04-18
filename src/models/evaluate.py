"""
模型评估模块

这个文件负责计算和展示模型的各种评估指标。

=== 包含哪些指标 ===

1. Accuracy (准确率): 整体分对了多少
   公式: 正确预测数 / 总样本数
   例子: 14000 张测试集分对了 13500 张 → Accuracy = 96.43%

2. Precision (精确率, 宏平均): 预测为某类的样本中, 真正是该类的比例
   公式: TP / (TP + FP), 然后对 10 个类取平均
   例子: 预测了 1000 张图是 "7", 其中 950 张真的是 "7" → Precision(7) = 95%

3. Recall (召回率, 宏平均): 真正是某类的样本中, 被正确识别的比例
   公式: TP / (TP + FN), 然后对 10 个类取平均
   例子: 测试集里有 1000 张 "7", 模型正确识别了 950 张 → Recall(7) = 95%

4. F1-Score (F1 分数, 宏平均): Precision 和 Recall 的调和平均
   公式: 2 × Precision × Recall / (Precision + Recall)
   用途: 综合衡量精确率和召回率, 适合类别不均衡的场景

5. Confusion Matrix (混淆矩阵): 10×10 矩阵
   行 = 真实标签, 列 = 预测标签
   对角线 = 分对了的数量, 越高越好
   非对角线 = 分错了的数量, 可以看到具体是哪些数字容易混淆

6. Classification Report (分类报告): 每个数字各自的 Precision/Recall/F1

=== 三个函数 ===

- evaluate_model(): 计算一个模型的全部指标
- compare_models(): 生成多个模型的对比表格
- print_evaluation(): 打印一个模型的详细结果
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,          # 准确率
    precision_score,         # 精确率
    recall_score,            # 召回回率
    f1_score,                # F1 分数
    confusion_matrix,        # 混淆矩阵
    classification_report,   # 完整分类报告
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    计算单个模型的全部评估指标

    参数:
        y_true: 真实标签, shape=(N,), 值为 0-9
        y_pred: 模型预测的标签, shape=(N,), 值为 0-9
        model_name: 模型名称 (用于显示, 如 "CNN" 或 "KNN (k=5)")

    返回:
        dict, 包含以下键:
            - model_name: 模型名称
            - accuracy: 整体准确率 (0~1 之间的小数)
            - precision_macro: 宏平均精确率
            - recall_macro: 宏平均召回率
            - f1_macro: 宏平均 F1
            - precision_per_class: 每个类别的精确率, shape=(10,)
            - recall_per_class: 每个类别的召回率, shape=(10,)
            - f1_per_class: 每个类别的 F1, shape=(10,)
            - confusion_matrix: 10×10 混淆矩阵
            - report: sklearn 生成的完整分类报告 (字符串)

    关于 "macro" 平均:
        macro = 先算每个类别的指标, 再取平均
        这样每个类别权重相同, 不会被大类主导
    """
    results = {
        "model_name": model_name,

        # ---- 宏平均指标 ----
        "accuracy": accuracy_score(y_true, y_pred),  # 整体准确率
        # average="macro": 对 10 个类别取简单平均
        # zero_division=0: 如果某类没有预测样本, 返回 0 而不是报错
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),

        # ---- 每个类别的指标 ----
        # average=None: 返回每个类别各自的值, shape=(10,)
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0),

        # ---- 混淆矩阵 ----
        # confusion_matrix(y_true, y_pred) 返回 10×10 矩阵
        # cm[i][j] = 真实标签为 i 但被预测为 j 的样本数
        # 对角线 cm[i][i] = 正确预测数
        "confusion_matrix": confusion_matrix(y_true, y_pred),

        # ---- 完整分类报告 ----
        # 包含每个类别的 precision / recall / f1 / support
        # digits=4: 保留 4 位小数
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }
    return results


def compare_models(all_results: list[dict]) -> str:
    """
    生成多个模型的对比表格 (文本格式)

    把多个模型的 Accuracy / Precision / Recall / F1 放在一张表里, 方便对比。

    参数:
        all_results: 列表, 每个元素是 evaluate_model() 的返回值

    返回:
        str: 格式化的表格字符串, 可以直接 print

    输出效果:
        ---------------------------------------------------------------------
        Model                       Accuracy  Precision     Recall   F1-Score
        ---------------------------------------------------------------------
        CNN                           0.9532     0.9546     0.9527     0.9528
        KNN (k=5)                     0.9635     0.9640     0.9630     0.9634
        Logistic Regression           0.9701     0.9699     0.9699     0.9699
        Random Forest                 0.9584     0.9580     0.9581     0.9580
        ---------------------------------------------------------------------
    """
    # 表头
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}"
    separator = "-" * len(header)

    lines = [separator, header, separator]
    for r in all_results:
        # 每个模型一行, 左对齐模型名, 右对齐数字, 保留 4 位小数
        lines.append(
            f"{r['model_name']:<25} "
            f"{r['accuracy']:>10.4f} "
            f"{r['precision_macro']:>10.4f} "
            f"{r['recall_macro']:>10.4f} "
            f"{r['f1_macro']:>10.4f}"
        )
    lines.append(separator)
    return "\n".join(lines)


def print_evaluation(results: dict):
    """
    打印单个模型的完整评估结果到终端

    会打印:
        - 模型名称
        - Accuracy / Precision / Recall / F1 (宏平均)
        - 每个数字 0-9 的详细指标 (sklearn 的 classification_report)

    参数:
        results: evaluate_model() 的返回值
    """
    print(f"\n{'='*50}")
    print(f"  模型: {results['model_name']}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision_macro']:.4f} (macro avg)")
    print(f"  Recall:    {results['recall_macro']:.4f} (macro avg)")
    print(f"  F1-Score:  {results['f1_macro']:.4f} (macro avg)")
    print(f"\n  各类别详情:")
    print(results["report"])
