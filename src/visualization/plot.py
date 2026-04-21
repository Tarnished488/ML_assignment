"""
可视化模块

这个文件负责生成所有图表, 共 7 种:

1. 训练曲线 (training_history.png)    — loss 和 accuracy 随 epoch 变化
2. 混淆矩阵热力图 (confusion_matrix_*.png) — 10×10 混淆矩阵
3. 模型对比柱状图 (model_comparison.png) — 各模型的 Accuracy / F1 对比
4. 错误样本展示 (misclassified_cnn.png) — CNN 分错的图片
5. 各类别准确率 (per_class_accuracy.png) — 0-9 每个数字的识别率
6. Grad-CAM 热力图 (gradcam.png) — CNN 关注区域可视化
7. 每个数字的 Grad-CAM (gradcam_per_digit.png) — 0-9 的注意力分布

=== 使用方式 ===

一般不需要单独调用, run_all.py 会自动调用 generate_all_plots() 生成所有图。
如果想单独生成某张图, 可以直接调用对应的函数。

=== 技术细节 ===

- 使用 matplotlib 画图, Agg 后端 (不弹窗, 直接保存为 PNG)
- dpi=150: 分辨率, 足够清晰但文件不会太大
- bbox_inches="tight": 自动裁剪白边
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端: 保存图片不弹窗, 服务器上也能用
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


def save_fig(fig, save_dir, filename):
    """
    保存图片到指定目录

    参数:
        fig: matplotlib 的 Figure 对象
        save_dir: 保存目录 (不存在会自动创建)
        filename: 文件名 (如 "training_history.png")
    """
    os.makedirs(save_dir, exist_ok=True)  # 目录不存在就创建
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")  # dpi=150 高清, tight 去白边
    plt.close(fig)  # 关闭图片, 释放内存
    print(f"  图片已保存: {path}")


# ============================================================
# 1. 训练曲线: loss 和 accuracy 随 epoch 变化
# ============================================================

def plot_training_history(history: dict, save_dir: str = "results"):
    """
    绘制训练/验证的 loss 和 accuracy 曲线

    生成两张并排的子图:
        左图: Train Loss 和 Val Loss 随 epoch 的变化
        右图: Train Acc 和 Val Acc 随 epoch 的变化

    怎么看这个图:
        - Loss 曲线应该持续下降, 如果不降了说明模型没在学
        - 如果 Train Loss 继续降但 Val Loss 开始升了, 说明过拟合了
        - Acc 曲线同理, 应该持续上升

    参数:
        history: train_cnn() 返回的训练历史字典, 包含四个列表:
                 train_loss, train_acc, val_loss, val_acc
        save_dir: 图片保存目录
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 左图: Loss 曲线
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: Accuracy 曲线
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_dir, "training_history.png")


# ============================================================
# 2. 混淆矩阵热力图
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, model_name: str = "CNN", save_dir: str = "results"):
    """
    绘制混淆矩阵热力图

    混淆矩阵怎么看:
        - 横轴 = 预测标签, 纵轴 = 真实标签
        - 对角线 (左上到右下) = 分对了的, 颜色越深说明分对越多
        - 非对角线 = 分错的, 可以看到具体是谁和谁混淆
        - 例如: 第 7 行第 9 列的数字 = 真实是 7 但被预测成 9 的数量

    参数:
        cm: 10×10 混淆矩阵, cm[i][j] = 真实 i 预测 j 的样本数
        model_name: 模型名称 (用于标题和文件名)
        save_dir: 保存目录
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 用蓝色色阶画热力图: 颜色越深 = 数值越大
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)  # 右侧颜色条

    # 设置坐标轴
    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xticklabels=range(10), yticklabels=range(10),
           ylabel="True Label", xlabel="Predicted Label",
           title=f"Confusion Matrix - {model_name}")

    # 在每个格子里写上具体数字
    thresh = cm.max() / 2.0  # 阈值: 大于此值用白字, 小于此值用黑字 (保证可读)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    # 文件名: 把空格和括号替换掉, 如 "KNN (k=5)" → "knn_(k=5)"
    safe_name = model_name.replace(" ", "_").lower()
    save_fig(fig, save_dir, f"confusion_matrix_{safe_name}.png")


# ============================================================
# 3. 模型对比柱状图
# ============================================================

def plot_model_comparison(all_results: list, save_dir: str = "results"):
    """
    绘制多个模型的 Accuracy / F1 对比柱状图

    每个模型两根柱子: 蓝色 = Accuracy, 橙色 = F1-Score
    柱子上方标注具体数值

    参数:
        all_results: 列表, 每个元素是 evaluate_model() 的返回值
        save_dir: 保存目录
    """
    names = [r["model_name"] for r in all_results]       # 模型名列表
    accuracies = [r["accuracy"] for r in all_results]    # Accuracy 列表
    f1_scores = [r["f1_macro"] for r in all_results]     # F1 列表

    x = np.arange(len(names))
    width = 0.35  # 柱子宽度

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Score (macro)", color="coral")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0.9, 1.0)  # Y 轴从 0.9 开始, 放大差异
    ax.grid(axis="y", alpha=0.3)

    # 在每个柱子顶部标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_fig(fig, save_dir, "model_comparison.png")


# ============================================================
# 4. 错误样本展示
# ============================================================

def plot_misclassified(X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "CNN", n_samples: int = 16, save_dir: str = "results"):
    """
    展示模型分错的样本

    每张图显示: 原始图片 + 标题 "True: X, Pred: Y"
    可以直观看到模型犯的错是什么类型的 (比如 7 被看成 9)

    参数:
        X_test: 测试集图像, shape=(N, 28, 28)
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        n_samples: 最多展示多少个 (默认 16)
        save_dir: 保存目录
    """
    # 找出所有分错的样本的索引
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        print("  没有错误分类的样本!")
        return

    n_show = min(n_samples, len(mis_idx))
    selected = mis_idx[:n_show]  # 取前 n_show 个

    # 布局: 4 列, 行数自动计算
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n_show > 1 else [axes]

    for i, idx in enumerate(selected):
        axes[i].imshow(X_test[idx], cmap="gray")  # 灰度图
        axes[i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}", fontsize=10)
        axes[i].axis("off")

    # 隐藏多余的子图 (如果 n_show 不是 4 的倍数)
    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Misclassified Samples - {model_name}", fontsize=14)
    plt.tight_layout()
    save_fig(fig, save_dir, f"misclassified_{model_name.replace(' ', '_').lower()}.png")

    print(f"  共 {len(mis_idx)} 个错误分类样本, 展示了前 {n_show} 个")


# ============================================================
# 5. 各类别准确率
# ============================================================

def plot_per_class_accuracy(all_results: list, save_dir: str = "results"):
    """
    绘制每个数字 0-9 的各类别准确率 (分组柱状图)

    每个数字对应多根柱子, 每根代表一个模型。
    可以看到哪个模型在哪个数字上表现最好/最差。

    参数:
        all_results: 列表, 每个元素是 evaluate_model() 的返回值
        save_dir: 保存目录
    """
    digits = np.arange(10)
    n_models = len(all_results)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, r in enumerate(all_results):
        # recall_per_class[j] = 数字 j 被正确识别的比例 (即该类的 accuracy)
        recall = r["recall_per_class"]
        offset = (i - n_models / 2 + 0.5) * width  # 让多根柱子并排排列
        ax.bar(digits + offset, recall, width, label=r["model_name"])

    ax.set_xlabel("Digit")
    ax.set_ylabel("Recall (Per-class Accuracy)")
    ax.set_title("Per-class Accuracy by Model")
    ax.set_xticks(digits)
    ax.set_ylim(0.9, 1.0)  # Y 轴从 0.9 开始, 放大差异
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_dir, "per_class_accuracy.png")


# ============================================================
# 6. Grad-CAM 热力图: 可视化 CNN "在看哪"
# ============================================================

class GradCAM:
    """
    Grad-CAM: 用梯度加权类激活图, 可视化 CNN 的注意力

    一句话解释: 这张热力图告诉你, CNN 做分类决策时, 图像的哪个区域最重要。

    原理 (4 步):
        1. 前向传播到目标卷积层 → 记录该层的特征图 (activations)
           特征图的每个通道对应一种"模式检测器"的输出
        2. 对目标类别的输出分数做反向传播 → 计算该层的梯度 (gradients)
           梯度反映了每个通道对分类决策的重要性
        3. 对梯度做全局平均池化 → 得到每个通道的标量权重
           权重越大, 这个通道对分类越重要
        4. 加权求和 + ReLU → 得到一张热力图
           ReLU 过滤掉负面影响, 只保留"对分类有正面贡献"的区域

    怎么看热力图:
        - 红色/黄色 = CNN 认为重要的区域 (决策依据)
        - 蓝色 = CNN 忽略的区域
        - 如果 CNN 关注的区域和你直觉一致 → 模型学对了
        - 如果 CNN 关注了无关区域 → 可能是过拟合或学偏了

    参数:
        model: 训练好的 CNN 模型
        target_layer: 要 hook 的目标层 (通常是最后一个卷积层)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None   # 前向传播时记录的特征图
        self.gradients = None     # 反向传播时记录的梯度
        self._hooks = []
        self._register_hooks()    # 注册 hook, 自动捕获数据

    def _register_hooks(self):
        """
        注册 PyTorch 的 hook 函数

        hook 的作用: 在前向/反向传播时, 自动"拦截"某一层的输入/输出,
        不需要修改模型代码就能获取中间层数据。

        forward_hook:  前向传播时触发, 记录该层的输出 (特征图)
        backward_hook: 反向传播时触发, 记录该层的梯度
        """
        def forward_hook(module, input, output):
            # 前向传播时, 保存这一层的输出 (特征图)
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # 反向传播时, 保存这一层的梯度
            self.gradients = grad_output[0].detach()

        # register_forward_hook: 前向传播时调用
        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        # register_full_backward_hook: 反向传播时调用
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        """
        对单张图像生成 Grad-CAM 热力图

        参数:
            input_tensor: 单张图像, shape=(1, 1, 28, 28)
            target_class: 想看哪个类别的注意力 (None = 自动用预测类别)

        返回:
            cam: 热力图, shape=(H, W), 值 0-255 (uint8), 可直接用于画图
            target_class: 实际使用的目标类别
        """
        self.model.eval()

        # 第 1 步: 前向传播 (hook 会自动捕获特征图)
        output = self.model(input_tensor)

        # 确定目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()  # 取预测类别

        # 第 2 步: 反向传播 (hook 会自动捕获梯度)
        self.model.zero_grad()
        # 构造一个 one-hot 向量, 只对目标类别做反向传播
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 第 3 步: 计算权重 (对梯度的空间维度做全局平均)
        # gradients shape: (1, 64, 7, 7) → weights shape: (1, 64, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # 第 4 步: 加权求和 + ReLU
        # (1, 64, 1, 1) × (1, 64, 7, 7) → (1, 1, 7, 7)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # 只保留正面贡献

        # 归一化到 0-255
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(cam * 255)

        return cam, target_class

    def remove_hooks(self):
        """移除 hook, 释放资源 (用完后必须调用)"""
        for hook in self._hooks:
            hook.remove()


def plot_gradcam(model, X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                 device, n_samples: int = 16, save_dir: str = "results"):
    """
    生成 Grad-CAM 热力图: 正确分类样本 + 7/9 混淆样本对比

    展示两部分:
        - 前半部分: 正确分类的样本 (标题绿色), 看模型"看对了"哪些区域
        - 后半部分: 7→9 或 9→7 的混淆样本 (标题红色), 看模型"看错了"哪里

    每个样本展示两列: 左=原图, 右=叠加了热力图的图

    参数:
        model: 训练好的 CNN 模型
        X_test: 测试集图像, shape=(N, 28, 28)
        y_true: 真实标签
        y_pred: 预测标签
        device: 计算设备
        n_samples: 总共展示多少个样本
        save_dir: 保存目录
    """
    # hook 到 features 的倒数第 2 层 (最后一个 Conv2d 后的 BatchNorm)
    # 这是 Grad-CAM 的标准做法: hook 最后一个卷积层附近的层
    target_layer = model.features[-2]
    grad_cam = GradCAM(model, target_layer)

    # ---- Part 1: 正确分类的样本 ----
    correct_idx = np.where(y_true == y_pred)[0]
    n_correct = min(n_samples // 2, len(correct_idx))
    selected_correct = np.random.choice(correct_idx, n_correct, replace=False)

    # ---- Part 2: 7/9 混淆的样本 ----
    # 找出所有 7 被认成 9, 或 9 被认成 7 的样本
    confused_79 = np.where(((y_true == 7) & (y_pred == 9)) |
                            ((y_true == 9) & (y_pred == 7)))[0]
    n_confused = min(n_samples // 2, len(confused_79))
    if n_confused > 0:
        selected_confused = confused_79[:n_confused]
    else:
        selected_confused = np.array([], dtype=int)

    # 合并两类样本
    all_indices = np.concatenate([selected_correct, selected_confused])
    labels = (["correct"] * n_correct + ["7→9/9→7"] * n_confused)

    n_total = len(all_indices)
    if n_total == 0:
        grad_cam.remove_hooks()
        return

    # 布局: 每个样本占 2 列 (原图 + 热力图)
    cols = 4
    rows = (n_total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    jet_cmap = plt.cm.jet  # jet 色阶: 蓝→青→绿→黄→红

    for i, (idx, label_type) in enumerate(zip(all_indices, labels)):
        img = X_test[idx]
        # 准备输入: numpy → tensor, 加 batch 和 channel 维
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)

        # 生成 Grad-CAM 热力图
        cam, pred_class = grad_cam.generate(input_tensor, target_class=int(y_pred[idx]))

        row, col_pair = i // cols, (i % cols) * 2

        # 左: 原始图像
        axes[row, col_pair].imshow(img, cmap="gray")
        color = "green" if label_type == "correct" else "red"
        axes[row, col_pair].set_title(
            f"True:{y_true[idx]} Pred:{y_pred[idx]}", fontsize=9, color=color)
        axes[row, col_pair].axis("off")

        # 右: 叠加热力图
        axes[row, col_pair + 1].imshow(img, cmap="gray", alpha=0.5)  # 底层灰度图
        heatmap = jet_cmap(cam / 255.0)[:, :, :3]  # 转成 RGB 颜色
        axes[row, col_pair + 1].imshow(heatmap, alpha=0.5)  # 叠加热力图
        axes[row, col_pair + 1].set_title("Grad-CAM", fontsize=9)
        axes[row, col_pair + 1].axis("off")

    # 隐藏空白格
    for j in range(n_total, rows * cols):
        row, col_pair = j // cols, (j % cols) * 2
        axes[row, col_pair].axis("off")
        axes[row, col_pair + 1].axis("off")

    fig.suptitle("Grad-CAM: What the CNN is looking at", fontsize=14)
    plt.tight_layout()
    save_fig(fig, save_dir, "gradcam.png")

    grad_cam.remove_hooks()  # 清理 hook
    print(f"  Grad-CAM: {n_correct} correct + {n_confused} confused 7/9 samples")


def plot_gradcam_per_digit(model, X_test: np.ndarray, y_test: np.ndarray,
                           device, save_dir: str = "results"):
    """
    为每个数字 0-9 各选一张样本, 展示 Grad-CAM 热力图

    上排: 10 个数字的原始图像
    下排: 对应的 Grad-CAM 热力图

    用途: 直观展示 CNN 识别不同数字时分别关注图像的哪些区域。
    比如 7 的热力图应该集中在竖笔和横笔的交汇处,
    而 9 的热力图应该集中在顶部的圆圈上。

    参数:
        model: 训练好的 CNN 模型
        X_test: 测试集图像
        y_test: 测试集标签
        device: 计算设备
        save_dir: 保存目录
    """
    target_layer = model.features[-2]
    grad_cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    for digit in range(10):
        # 找到该数字的第一个样本
        idx_arr = np.where(y_test == digit)[0]
        idx = idx_arr[0]
        img = X_test[idx]

        # 生成热力图
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        cam, _ = grad_cam.generate(input_tensor, target_class=digit)

        # 上排: 原图
        axes[0, digit].imshow(img, cmap="gray")
        axes[0, digit].set_title(f"Digit {digit}", fontsize=10)
        axes[0, digit].axis("off")

        # 下排: 热力图
        axes[1, digit].imshow(img, cmap="gray", alpha=0.4)
        axes[1, digit].imshow(plt.cm.jet(cam / 255.0)[:, :, :3], alpha=0.6)
        axes[1, digit].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=11)
    fig.suptitle("Grad-CAM per Digit: CNN Attention Heatmap", fontsize=14)
    plt.tight_layout()
    save_fig(fig, save_dir, "gradcam_per_digit.png")

    grad_cam.remove_hooks()


# ============================================================
# 一键生成所有图
# ============================================================

def generate_all_plots(pipeline_results: dict, X_test: np.ndarray,
                       y_test: np.ndarray, save_dir: str = "results"):
    """
    生成全部可视化图表

    一般由 run_all.py 调用, 不需要手动调用。

    参数:
        pipeline_results: run_full_pipeline() 的返回值, 包含模型、历史记录、评估结果等
        X_test: 测试集图像 (N, 28, 28), 用于展示错误样本和 Grad-CAM
        y_test: 测试集标签
        save_dir: 图片保存目录
    """
    print("\n" + "=" * 50)
    print("  生成可视化图表")
    print("=" * 50)

    history = pipeline_results["history"]
    all_results = pipeline_results["all_results"]

    # 取测试集的结果 (all_results 中最后一个 split)
    test_results = all_results[-1]["results"]

    # 1. 训练曲线
    plot_training_history(history, save_dir)

    # 2. 混淆矩阵 (测试集上各模型)
    for r in test_results:
        cm = r["confusion_matrix"]
        plot_confusion_matrix(cm, model_name=r["model_name"], save_dir=save_dir)

    # 3. 模型对比 (测试集)
    plot_model_comparison(test_results, save_dir)

    # 4. 错误样本 (CNN, 测试集)
    y_pred_cnn_test = pipeline_results["y_pred_cnn_test"]
    plot_misclassified(X_test, y_test, y_pred_cnn_test, model_name="CNN", save_dir=save_dir)

    # 5. 各类别准确率 (测试集)
    plot_per_class_accuracy(test_results, save_dir)

    # 6. Grad-CAM 热力图
    cnn_model = pipeline_results["cnn_model"]
    device = pipeline_results["device"]
    y_pred_cnn = pipeline_results["y_pred_cnn_test"]
    plot_gradcam(cnn_model, X_test, y_test, y_pred_cnn, device, save_dir=save_dir)
    plot_gradcam_per_digit(cnn_model, X_test, y_test, device, save_dir=save_dir)

    print(f"\n  所有图片已保存到 {save_dir}/")
