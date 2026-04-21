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
    accuracy_score,          # Accuracy
    precision_score,         # Precision
    recall_score,            # Recall
    f1_score,                # F1 Score
    confusion_matrix,        # Confusion matrix
    classification_report,   # Complete classification report
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Compute all evaluation metrics for a single model

    Parameters:
        y_true: 真实标签, shape=(N,), 值为 0-9
        y_pred: 模型预测的标签, shape=(N,), 值为 0-9
        model_name: 模型名称 (用于显示, 如 "CNN" 或 "KNN (k=5)")

    Returns:
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

    About "macro" average:
        macro = first compute metrics for each class, then take average
        This gives equal weight to each class,not dominated by large classes
    """
    results = {
        "model_name": model_name,

        # ---- Macro-average metrics ----
        "accuracy": accuracy_score(y_true, y_pred),  # Overall accuracy
        # average="macro": 对 10 个类别取简单平均
        # zero_division=0: 如果某类没有预测样本, 返回 0 而不是报错
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),

        # ---- Per-class metrics ----
        # average=None: 返回每个类别各自的值, shape=(10,)
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0),

        # ---- Confusion matrix ----
        # confusion_matrix(y_true, y_pred) returns 10×10 matrix
        # cm[i][j] = number of samples with true label i predicted as j
        # diagonal cm[i][i] = number of correct predictions
        "confusion_matrix": confusion_matrix(y_true, y_pred),

        # ---- Complete classification report ----
        # Contains precision / recall / f1 / support for each class
        # digits=4: keep 4 decimal places
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }
    return results


def compare_models(all_results: list[dict]) -> str:
    """
    生成多个模型的对比表格 (文本格式)

    把多个模型的 Accuracy / Precision / Recall / F1 放在一张表里, 方便对比。

    Parameters:
        all_results: 列表, 每个元素是 evaluate_model() 的返回值

    Returns:
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

    Parameters:
        results: evaluate_model() 的返回值
    """
    print(f"\n{'='*50}")
    print(f"  Model: {results['model_name']}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision_macro']:.4f} (macro avg)")
    print(f"  Recall:    {results['recall_macro']:.4f} (macro avg)")
    print(f"  F1-Score:  {results['f1_macro']:.4f} (macro avg)")
    print(f"\n  Per-class details:")
    print(results["report"])
