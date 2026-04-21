"""
模型训练模块

这个文件是整个训练流程的核心, 包含三个部分:
1. CNN 训练流程 (PyTorch)
2. 基线模型训练 (sklearn: KNN, Logistic Regression, Random Forest)
3. 数据划分与加载

=== 数据划分策略 ===

训练集:验证集:测试集 = 6:2:2 (42000:14000:14000)

为什么这样划分:
- 训练集 (42000): 训练模型参数, 占大头, 数据越多模型学得越好
- 验证集 (14000): 训练过程中监控是否过拟合, 用来决定什么时候停 (早停)
- 测试集 (14000): 最终评估, 所有模型共用, 保证对比公平

为什么不用原始的 60000+10000:
- 原始 MNIST 没有验证集, 只有训练集(60000)和测试集(10000)
- 我们需要验证集来做早停, 所以把全部 70000 张合并后重新划分

=== 训练策略 ===

CNN 训练策略:
- 优化器: Adam (自适应学习率, 收敛快, 适合 MNIST)
- 损失函数: Focal Loss + Label Smoothing (关注难分样本, 防止过度自信)
- 学习率调度: StepLR (每 10 个 epoch 衰减为原来的 0.1)
- 早停: 验证集准确率连续 5 轮不提升则停止
- 数据增强: 训练时随机旋转/平移/缩放, 防止过拟合

基线模型训练策略:
- 使用 sklearn 的默认参数 (KNN k=5, LR 1000轮, RF 100棵树)
- 使用组员提取的 HOG+LBP+Shape 405 维特征
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.cnn import MNISTCNN, FocalLoss, count_parameters
from src.models.evaluate import evaluate_model, print_evaluation, compare_models


# ============================================================
# 工具函数: 固定随机种子、选择设备、划分数据集
# ============================================================

def set_seed(seed: int = 42):
    """
    固定所有随机种子, 保证每次运行结果一样

    为什么需要固定种子:
        训练过程中有很多随机操作: 数据打乱顺序、Dropout 随机丢弃、权重随机初始化等。
        不固定种子的话, 每次跑结果都不一样, 没法对比实验。

    覆盖的随机性来源:
        - np.random.seed: NumPy 的随机操作 (数据打乱)
        - torch.manual_seed: PyTorch CPU 上的随机操作
        - torch.cuda.manual_seed_all: PyTorch GPU 上的随机操作
        - cudnn.deterministic: cuDNN 卷积的确定性模式
        - cudnn.benchmark: 关闭自动调优 (自动调优会引入随机性)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    自动选择计算设备: 有 GPU 用 GPU, 没有 GPU 用 CPU

    GPU (CUDA) 训练速度比 CPU 快 5-10 倍, 但需要 NVIDIA 显卡。
    这个函数会自动检测, 不需要手动改代码。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_dataset(X_all, y_all, train_ratio=6/10, val_ratio=2/10, seed=42):
    """
    将合并后的全部数据按比例划分成训练集、验证集、测试集

    默认比例: 训练集:验证集:测试集 = 6:2:2
    70000 张 MNIST → 42000 + 14000 + 14000

    参数:
        X_all: 全部图像, shape=(70000, 28, 28)
        y_all: 全部标签, shape=(70000,)
        train_ratio: 训练集比例, 默认 6/10
        val_ratio:   验证集比例, 默认 2/10
        seed: 随机种子, 固定后划分结果可复现

    返回:
        X_train, y_train, X_val, y_val, X_test, y_test (六个数组)

    划分逻辑:
        1. 先把 70000 个样本的索引随机打乱
        2. 前 20% 做测试集, 接下来 20% 做验证集, 剩下 60% 做训练集
        3. 用索引从 X_all 和 y_all 中取出对应的数据
    """
    set_seed(seed)
    n = len(X_all)
    indices = np.random.permutation(n)  # 随机打乱 0~69999 的索引

    # 计算每个集合的大小
    n_test = int(n * (1 - train_ratio - val_ratio))   # 70000 × 0.2 = 14000
    n_val = int(n * val_ratio)                          # 70000 × 0.2 = 14000

    # 按索引切分
    test_idx = indices[:n_test]                         # 前 14000 个 → 测试集
    val_idx = indices[n_test:n_test + n_val]            # 中间 14000 个 → 验证集
    train_idx = indices[n_test + n_val:]                # 剩下 42000 个 → 训练集

    return (
        X_all[train_idx], y_all[train_idx],   # 训练集
        X_all[val_idx], y_all[val_idx],       # 验证集
        X_all[test_idx], y_all[test_idx],     # 测试集
    )


# ============================================================
# 数据增强: 训练时对图像做随机变换, 等效于扩大训练集
# ============================================================

class _AugmentedDataset(Dataset):
    """
    带数据增强的 PyTorch 数据集

    为什么需要数据增强:
        训练集只有 42000 张图, 模型可能会"记住"这些图而不是学到真正的特征。
        数据增强在每次取图时随机变换一下 (旋转、平移、缩放), 让模型每次看到不同的版本,
        相当于把训练集扩大了几十倍。

    关键点:
        - 只有训练集做增强, 验证集和测试集不做 (保证评估公平)
        - 增强是"在线"的, 即每次取图时实时变换, 不是预先生成一堆新图
        - 每个 epoch 看到的都是不同的变形版本
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        """
        参数:
            images: 图像数据, shape=(N, 1, 28, 28)
            labels: 标签数据, shape=(N,)
            augment: 是否做数据增强 (训练集=True, 验证/测试集=False)
        """
        self.images = images
        self.labels = labels
        self.augment = augment

        # 定义增强操作组合
        self.aug_transform = transforms.Compose([
            # 随机旋转 ±10°: 模拟不同人写字的倾斜角度
            transforms.RandomRotation(10),
            # 随机仿射变换: 同时做平移和缩放
            # translate=(0.1, 0.1): 上下左右最多平移 10%
            # scale=(0.9, 1.1): 缩放范围 90%~110%, 模拟字大字小
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # 随机裁剪: 先在四周补 2 像素, 再随机裁回 28×28, 模拟微小位置偏移
            transforms.RandomCrop(28, padding=2, padding_mode="edge"),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        取第 idx 个样本

        如果 augment=True, 会对图像做随机变换;
        如果 augment=False, 直接返回原图。
        """
        img = self.images[idx]
        label = self.labels[idx]
        if self.augment:
            img = self.aug_transform(img)  # 随机变换
        return img, label


def prepare_cnn_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    """
    把 numpy 数组转成 PyTorch 的 DataLoader

    为什么需要 DataLoader:
        训练时不能一次把 42000 张图全塞进 GPU (内存不够), 需要分批处理。
        DataLoader 自动帮你把数据切成小批次 (batch), 还能自动打乱顺序。

    参数:
        X_train, y_train: 训练集图像和标签 (numpy)
        X_val, y_val:     验证集图像和标签 (numpy)
        X_test, y_test:   测试集图像和标签 (numpy)
        batch_size:       每批处理多少张图 (默认 128)

    返回:
        train_loader, val_loader, test_loader (三个 DataLoader)

    关键区别:
        - 训练集 DataLoader: 带数据增强 + 打乱顺序
        - 验证/测试集 DataLoader: 不做增强 + 不打乱顺序
    """
    # numpy → PyTorch tensor, 加一个通道维: (N, 28, 28) → (N, 1, 28, 28)
    def to_tensor(X):
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    # 训练集: 带数据增强 + 打乱顺序 (shuffle=True)
    train_ds = _AugmentedDataset(
        to_tensor(X_train), torch.tensor(y_train, dtype=torch.long),
        augment=True  # 训练时做增强
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 验证集: 不做增强, 不打乱
    val_loader = DataLoader(
        TensorDataset(to_tensor(X_val), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
    )

    # 测试集: 不做增强, 不打乱
    test_loader = DataLoader(
        TensorDataset(to_tensor(X_test), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# CNN 训练
# ============================================================

def train_cnn(train_loader, val_loader, device, epochs=30, lr=0.001, patience=5):
    """
    训练 CNN 模型

    这是整个训练的核心函数, 流程如下:
        每个 epoch:
        1. 遍历训练集的所有 batch, 更新模型参数
        2. 在验证集上评估, 计算 loss 和 accuracy
        3. 如果验证 accuracy 提升了, 保存当前模型
        4. 如果连续 patience 轮没提升, 提前停止 (早停)

    参数:
        train_loader: 训练集 DataLoader
        val_loader:   验证集 DataLoader
        device:       计算设备 (cpu 或 cuda)
        epochs:       最多训练多少轮 (默认 30)
        lr:           学习率 (默认 0.001)
        patience:     早停耐心值 (默认 5, 即连续 5 轮不提升就停)

    返回:
        model: 训练好的模型 (加载了最佳权重)
        history: 训练历史记录, 包含每个 epoch 的 loss 和 accuracy
    """

    # ---- 初始化模型、损失函数、优化器、学习率调度器 ----
    model = MNISTCNN().to(device)  # 创建模型并放到 GPU/CPU 上
    print(f"\n  CNN 参数量: {count_parameters(model):,}")
    print(f"  设备: {device}")

    # 损失函数: Focal Loss + Label Smoothing
    # - Focal Loss: 自动关注难分样本 (如 7/9 混淆)
    # - Label Smoothing: 防止模型过度自信
    # 详细原理见 cnn.py 中 FocalLoss 类的注释
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

    # 优化器: Adam
    # - Adam 是自适应学习率优化器, 每个参数有独立的学习率
    # - lr=0.001 是 Adam 的推荐默认值
    # - weight_decay=1e-4 是 L2 正则化, 防止参数过大
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 学习率调度器: StepLR
    # 每 10 个 epoch, 学习率乘以 0.1 (变成原来的 1/10)
    # 前期大学习率快速学习, 后期小学习率精细调参
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ---- 训练记录 ----
    # history 字典记录每个 epoch 的训练/验证 loss 和 accuracy
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0          # 历史最佳验证 accuracy
    epochs_no_improve = 0       # 已经连续几轮没有提升了

    # ---- 主训练循环 ----
    for epoch in range(1, epochs + 1):

        # ===== 训练阶段 =====
        model.train()  # 切换到训练模式 (开启 Dropout 和 BatchNorm 的训练行为)
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            # 把数据搬到 GPU/CPU 上
            images, labels = images.to(device), labels.to(device)

            # 1. 前向传播: 图像 → 模型 → 预测
            optimizer.zero_grad()      # 清空上一步的梯度 (必须!)
            outputs = model(images)     # 前向传播, 得到每个类别的得分

            # 2. 计算损失: 预测 vs 真实标签
            loss = criterion(outputs, labels)

            # 3. 反向传播: 计算 gradients
            loss.backward()

            # 4. 更新参数: 用 gradients 调整模型权重
            optimizer.step()

            # 统计这个 batch 的 loss 和正确数
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)  # 取得分最高的类别作为预测
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 一个 epoch 训练完, 计算平均 loss 和 accuracy
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ===== 验证阶段 =====
        model.eval()  # 切换到评估模式 (关闭 Dropout, BatchNorm 用运行均值)
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():  # 不计算梯度, 节省内存和计算
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        # 更新学习率
        scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印这一轮的结果
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ===== 早停判断 =====
        if val_acc > best_val_acc:
            # 验证 accuracy 提升了 → 保存模型, 重置计数器
            best_val_acc = val_acc
            best_state = model.state_dict().copy()  # 保存当前最佳权重
            epochs_no_improve = 0
        else:
            # 没提升 → 计数器 +1
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  早停: 验证集准确率连续 {patience} 轮未提升")
                break  # 提前结束训练

    # 训练结束, 加载最佳权重 (不一定是最后一轮的, 而是验证 accuracy 最高的那轮)
    model.load_state_dict(best_state)
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    return model, history


def predict_cnn(model, data_loader, device):
    """
    用训练好的 CNN 对数据进行预测

    参数:
        model: 训练好的 CNN 模型
        data_loader: 要预测的数据 DataLoader
        device: 计算设备

    返回:
        numpy 数组, shape=(N,), 每个元素是预测的数字 (0-9)
    """
    model.eval()  # 评估模式: 关闭 Dropout
    all_preds = []
    with torch.no_grad():  # 不计算梯度, 纯预测
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)  # 取得分最高的类别
            all_preds.append(predicted.cpu().numpy())
    return np.concatenate(all_preds)


# ============================================================
# 基线模型: 用 sklearn 训练传统 ML 模型做对比
# ============================================================

def train_baselines(X_train_features, y_train, X_val_features, y_val, X_test_features, y_test):
    """
    训练 sklearn 基线模型, 并在训练集、验证集和测试集上分别预测

    为什么需要基线模型:
        只看 CNN 准确率 99% 没有意义, 需要对比才知道 CNN 是不是真的比简单方法好。
        KNN、LR、RF 是三种不同思路的经典方法, 覆盖了近邻、线性、集成三个类型。

    为什么基线用手工特征而非原始像素:
        KNN/LR/RF 不能直接处理 2D 图像, 需要先把图像转成特征向量。
        用组员做的 HOG+LBP+Shape 405 维特征是合理的——这些特征已经精心设计过。

    参数:
        X_train_features: 训练集特征, shape=(42000, 405)
        y_train: 训练集标签, shape=(42000,)
        X_val_features: 验证集特征, shape=(14000, 405)
        y_val: 验证集标签
        X_test_features: 测试集特征, shape=(14000, 405)
        y_test: 测试集标签

    返回:
        dict: {模型名: {"model": 分类器对象,
                        "train_pred": 训练集预测,
                        "val_pred": 验证集预测,
                        "test_pred": 测试集预测}}
    """
    # 定义三个基线模型
    baselines = {
        # KNN (K-近邻): 找训练集中最近的 5 个样本, 投票决定分类
        # n_neighbors=5: 看最近的 5 个邻居
        # n_jobs=-1: 用所有 CPU 核心并行计算
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),

        # Logistic Regression (逻辑回归): 线性分类器
        # max_iter=1000: 最多迭代 1000 次 (默认 100 可能不够收敛)
        # solver="lbfgs": 优化算法, 适合多分类
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1),

        # Random Forest (随机森林): 多棵决策树投票
        # n_estimators=100: 100 棵树
        # random_state=42: 固定随机种子
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, clf in baselines.items():
        print(f"\n  训练 {name}...")
        # 训练: 用训练集的特征和标签拟合模型
        clf.fit(X_train_features, y_train)

        # 在三个集合上分别预测 (用于后续分别评估)
        results[name] = {
            "model": clf,
            "train_pred": clf.predict(X_train_features),  # 训练集预测
            "val_pred": clf.predict(X_val_features),      # 验证集预测
            "test_pred": clf.predict(X_test_features),    # 测试集预测
        }
        print(f"  {name} 完成")

    return results


# ============================================================
# 主流程: 一键跑完 CNN + 基线模型 的训练和评估
# ============================================================

def run_full_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                      X_train_features, X_val_features, X_test_features,
                      batch_size=128, epochs=30, lr=0.001, seed=42):
    """
    完整训练+评估流程

    做三件事:
        1. 训练 CNN (用 PyTorch)
        2. 训练 3 个基线模型 (用 sklearn)
        3. 在训练集、验证集、测试集上分别评估所有模型

    参数:
        X_train, y_train:     训练集图像 (42000, 28, 28) 和标签
        X_val, y_val:         验证集图像 (14000, 28, 28) 和标签
        X_test, y_test:       测试集图像 (14000, 28, 28) 和标签
        X_train_features:     训练集特征 (42000, 405), 基线模型用
        X_val_features:       验证集特征 (14000, 405)
        X_test_features:      测试集特征 (14000, 405)
        batch_size:           CNN 每批处理多少张图
        epochs:               CNN 训练多少轮
        lr:                   CNN 学习率
        seed:                 随机种子

    返回:
        dict: 包含所有结果, 供可视化和保存摘要使用
    """
    set_seed(seed)
    device = get_device()

    print("=" * 60)
    print("  开始模型训练与评估")
    print(f"  训练集: {len(y_train)}  验证集: {len(y_val)}  测试集: {len(y_test)}")
    print(f"  比例 train:val:test = {len(y_train)}:{len(y_val)}:{len(y_test)}")
    print("=" * 60)

    # ---- 第 1 步: 训练 CNN ----
    print("\n[1/3] 训练 CNN...")
    train_loader, val_loader, test_loader = prepare_cnn_data(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    cnn_model, history = train_cnn(train_loader, val_loader, device, epochs=epochs, lr=lr)

    # CNN 在三个集合上分别预测
    # 注意: 训练集要用不带数据增强的 DataLoader (否则预测不准)
    def to_loader(X, y):
        """把 numpy 数据快速转成 DataLoader (不带增强)"""
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size)

    y_pred_cnn_train = predict_cnn(cnn_model, to_loader(X_train, y_train), device)
    y_pred_cnn_val = predict_cnn(cnn_model, val_loader, device)
    y_pred_cnn_test = predict_cnn(cnn_model, test_loader, device)

    # ---- 第 2 步: 训练基线模型 ----
    print("\n[2/3] 训练基线模型 (使用 HOG+LBP+Shape 特征)...")
    baseline_results = train_baselines(
        X_train_features, y_train,
        X_val_features, y_val,
        X_test_features, y_test,
    )

    # ---- 第 3 步: 在三个集合上分别评估所有模型 ----
    print("\n[3/3] 评估所有模型...")
    all_results = []

    # 遍历三个集合: 训练集、验证集、测试集
    for split_name, y_true, y_pred_cnn, feat_key in [
        ("训练集", y_train, y_pred_cnn_train, "train_pred"),
        ("验证集", y_val, y_pred_cnn_val, "val_pred"),
        ("测试集", y_test, y_pred_cnn_test, "test_pred"),
    ]:
        print(f"\n{'='*60}")
        print(f"  【{split_name}】 ({len(y_true)} 样本)")
        print(f"{'='*60}")

        split_results = []

        # 评估 CNN
        cnn_eval = evaluate_model(y_true, y_pred_cnn, model_name="CNN")
        split_results.append(cnn_eval)

        # 评估三个基线模型
        for name, bl in baseline_results.items():
            y_pred_bl = bl[feat_key]  # 取出对应集合的预测结果
            eval_res = evaluate_model(y_true, y_pred_bl, model_name=name)
            split_results.append(eval_res)

        # 打印所有模型的评估结果
        for r in split_results:
            print_evaluation(r)
        print("\n" + compare_models(split_results))

        all_results.append({"split": split_name, "results": split_results})

    # ---- 返回所有结果 ----
    return {
        "cnn_model": cnn_model,            # 训练好的 CNN 模型 (Grad-CAM 会用到)
        "history": history,                # 训练历史 (画训练曲线用)
        "all_results": all_results,         # 三个集合的评估结果
        "y_pred_cnn_test": y_pred_cnn_test, # CNN 在测试集上的预测
        "y_pred_cnn_val": y_pred_cnn_val,   # CNN 在验证集上的预测
        "y_pred_cnn_train": y_pred_cnn_train, # CNN 在训练集上的预测
        "baseline_results": baseline_results,  # 基线模型的结果
        "device": device,                  # 计算设备 (Grad-CAM 需要)
        "splits": {                        # 原始数据 (Grad-CAM 画图用)
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        },
    }
