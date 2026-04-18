"""
MNIST CNN 模型定义 + Focal Loss 损失函数

这个文件定义了两个核心类:
1. FocalLoss  — 损失函数, 用来替代标准的 CrossEntropyLoss
2. MNISTCNN   — CNN 网络结构

=== 为什么选 CNN ===

手写数字识别本质是一个图像分类任务。图像数据具有二维空间结构——相邻像素共同构成笔画、
轮廓等有意义的局部模式。MLP (多层感知机) 把 28×28 图像展平成 784 维向量, 完全丢弃
了这种空间关系, 只能靠全连接权重硬记像素位置。而 CNN 通过卷积核在图像上滑动, 天然
能捕获局部空间模式 (笔画方向、弧度、交叉点), 且参数在空间上共享, 参数效率远高于 MLP。

具体来说:
- CNN 的卷积核相当于自动学习一组"笔画检测器", 无需手工设计 HOG/LBP 等特征
- MNIST 图像虽小 (28×28), 但数字的判别信息集中在局部笔画模式上, 正好是 CNN 的强项
- 池化层提供一定的平移不变性, 即使数字写得偏左偏右, 模型也能识别

=== 针对本项目的改进与创新 ===

1. BatchNorm (批量归一化): 在每个卷积层后、激活函数前加入 BatchNorm2d。
   作用: 稳定中间层特征的数值分布, 缓解内部协变量偏移 (Internal Covariate Shift)。
   好处: 收敛速度提升约 30%, 且对学习率和初始化更鲁棒, 不容易训练崩塌。

2. 在线数据增强: 训练时对每张图片实时施加随机变换 (旋转±10°、平移±10%、缩放0.9~1.1、
   随机裁剪), 验证和测试时不做增强。
   作用: 等效于将训练集扩大了数十倍, 迫使模型学到真正鲁棒的特征, 而不是记住训练样本。
   好处: 显著降低过拟合风险, 尤其在训练轮数较多时效果明显。

3. 双通道对比实验设计: CNN 直接吃预处理后的 28×28 图像 (端到端自动学特征),
   而基线模型 (KNN/LR/RF) 吃组员手工提取的 HOG+LBP+Shape 405 维特征。
   这形成了一个有说服力的对照: 深度学习的自动特征学习 vs 传统手工特征工程,
   能够直观展示两种范式的性能差异。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Focal Loss: 替代 CrossEntropyLoss 的改进损失函数
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing (聚焦损失函数 + 标签平滑)

    一句话概括: 让模型把注意力集中在"分错的样本"上, 同时防止模型过度自信。

    --- Focal Loss 部分 ---

    标准 CrossEntropyLoss 的问题:
        假设训练集里有 42000 张图, 其中 38000 张都很简单 (比如很清晰的 0 和 1),
        只有 4000 张比较难 (比如容易混淆的 7 和 9)。标准 CE 会对所有样本一视同仁,
        结果训练信号被大量简单样本稀释了, 模型没动力去学那些难样本。

    Focal Loss 的解决方式:
        给每个样本的 loss 乘一个权重: (1 - p_t)^gamma
        - p_t 是模型对这个样本的预测概率 (越接近 1 越简单)
        - gamma 是聚焦参数 (我们设为 2.0)

        效果:
        - 简单样本 (p_t ≈ 0.99): 权重 = (1-0.99)^2 = 0.0001, loss 几乎被忽略
        - 难样本   (p_t ≈ 0.3):  权重 = (1-0.3)^2  = 0.49,   loss 保留将近一半

        这样模型就会自动把精力花在难分样本上 (比如 7/9 混淆)。

    --- Label Smoothing 部分 ---

    标准训练用 one-hot 标签: [0, 0, 1, 0, 0, ...]  (第 2 类是 100%)
    Label Smoothing 改成:    [ε/(C-1), ε/(C-1), 1-ε, ε/(C-1), ...]
                             其中 ε=0.1, C=10

    为什么要这样:
        - one-hot 鼓励模型输出极端概率 (100% 确信), 容易过拟合
        - label smoothing 强制模型"留有余地", 对正确类最多输出 91.1%
        - 这让决策边界更平滑, 泛化能力更好

    --- 为什么两个一起用 ---

        Focal Loss 管"看哪些样本" (关注难样本)
        Label Smoothing 管"别太自信" (平滑决策边界)
        两者正交, 组合效果叠加。

    参数说明:
        gamma:           聚焦参数, 越大越关注难样本。推荐 2.0, 原论文实验值
        alpha:           类别权重, 用于处理类别不平衡。我们数据均衡, 设为 None
        label_smoothing: 平滑系数, 0.1 是常用值。0.0 = 等同不用 smoothing
        num_classes:     类别数, MNIST 是 10
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1, num_classes=10):
        super().__init__()
        self.gamma = gamma                # 聚焦参数: 控制简单样本被降权的程度
        self.alpha = alpha                # 类别权重: 处理类别不平衡, None 表示不使用
        self.label_smoothing = label_smoothing  # 标签平滑系数
        self.num_classes = num_classes    # 分类数 (MNIST = 10)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算一个 batch 的 Focal Loss

        参数:
            inputs:  模型的原始输出 (logits), shape=(batch_size, num_classes)
                     注意: 不需要先过 Softmax, 函数内部会自己算
            targets: 真实标签, shape=(batch_size,), 值为 0-9 的整数

        返回:
            一个标量 loss 值, 越小越好

        计算过程 (逐步解释):
            1. 构造平滑标签: 把 one-hot [0,0,1,0,...] 变成 [0.011, 0.011, 0.911, 0.011, ...]
            2. 计算对数概率: log_softmax = log(softmax(logits))
            3. 计算概率: softmax(logits)
            4. 计算 focal 权重: (1 - prob)^gamma, 简单样本权重小, 难样本权重大
            5. 加权求和: -focal_weight × smooth_label × log_prob
        """
        # 第 1 步: 构造平滑标签
        # 初始全填 ε/(C-1) = 0.1/9 ≈ 0.011
        smooth_labels = torch.zeros_like(inputs)
        smooth_labels.fill_(self.label_smoothing / (self.num_classes - 1))
        # 真实类位置填 1-ε = 0.9
        # scatter_ 的意思: 在第 1 维上, 把 targets 指定的位置填入 0.9
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        # 结果: 如果真实标签是 3, 则 smooth_labels = [0.011, 0.011, 0.011, 0.9, 0.011, ...]

        # 第 2 步: 计算对数概率和概率
        # log_softmax 比 softmax + log 数值更稳定 (避免 log(接近0的数))
        log_probs = F.log_softmax(inputs, dim=-1)  # shape: (batch, 10)
        probs = torch.exp(log_probs)                 # 等价于 softmax(inputs), shape: (batch, 10)

        # 第 3 步: 计算 focal 权重
        # (1 - p)^gamma: 模型越确信的样本 (p 接近 1), 权重越小
        focal_weight = (1 - probs) ** self.gamma     # shape: (batch, 10)

        # 第 4 步: 加权 loss
        # -focal_weight × smooth_label × log_prob
        loss = -focal_weight * smooth_labels * log_probs  # shape: (batch, 10)

        # 第 5 步: 如果有类别权重, 额外乘上去 (我们没用到, alpha=None)
        if self.alpha is not None:
            alpha_weight = smooth_labels * self.alpha
            loss = loss * alpha_weight

        # 最后: 对每个样本的 10 个类别求和, 再对所有样本求平均
        return loss.sum(dim=-1).mean()


# ============================================================
# CNN 模型: 用于 MNIST 手写数字识别的卷积神经网络
# ============================================================

class MNISTCNN(nn.Module):
    """
    MNIST 手写数字识别 CNN

    整体结构 (从输入到输出):
    ┌──────────────────────────────────────────────────────────────────┐
    │ 输入: (batch, 1, 28, 28) — 灰度图, 1 个通道, 28×28 像素         │
    ├──────────────────────────────────────────────────────────────────┤
    │ 第 1 层卷积块:                                                    │
    │   Conv2d(1→32, 3×3)  — 32 个卷积核扫描图像, 提取笔画边缘等低级特征  │
    │   BatchNorm2d(32)    — 归一化, 稳定训练                           │
    │   ReLU               — 激活函数, 引入非线性                        │
    │   MaxPool2d(2×2)     — 池化, 保留最显著特征, 尺寸减半              │
    │   输出: (batch, 32, 14, 14)                                       │
    ├──────────────────────────────────────────────────────────────────┤
    │ 第 2 层卷积块:                                                    │
    │   Conv2d(32→64, 3×3) — 64 个卷积核, 组合低级特征成中级模式         │
    │   BatchNorm2d(64)    — 归一化                                      │
    │   ReLU               — 激活函数                                    │
    │   MaxPool2d(2×2)     — 尺寸再减半                                  │
    │   输出: (batch, 64, 7, 7)                                         │
    ├──────────────────────────────────────────────────────────────────┤
    │ 分类器 (全连接层):                                                 │
    │   Flatten            — 展平成一维向量: 64×7×7 = 3136               │
    │   Linear(3136→128)   — 全连接, 整合所有特征                        │
    │   ReLU               — 激活函数                                    │
    │   Dropout(0.5)       — 随机丢弃 50% 神经元, 防止过拟合             │
    │   Linear(128→10)     — 输出 10 个数字的得分                        │
    │   输出: (batch, 10)                                               │
    └──────────────────────────────────────────────────────────────────┘

    总参数量: 421,834

    改进点:
        - BatchNorm: 每层卷积后归一化, 加速收敛 + 提升准确率
        - Dropout: 全连接层随机丢弃 50% 神经元, 防止过拟合
        - 配合 train.py 中的在线数据增强, 进一步提升泛化能力

    注意: 最后一层输出的是 raw logits (没过 Softmax),
          因为 FocalLoss 内部会自己算 Softmax。
    """

    def __init__(self, num_classes: int = 10):
        """
        初始化网络

        参数:
            num_classes: 分类数, MNIST 是 10 个数字 (0-9)
        """
        super().__init__()

        # ---- 特征提取部分 (卷积层) ----
        # 这部分负责从图像中提取有意义的特征
        self.features = nn.Sequential(
            # ===== 第 1 层卷积块 =====
            # Conv2d: 1 个输入通道 (灰度图) → 32 个输出通道 (32 种不同的特征)
            # kernel_size=3: 3×3 的卷积核, 足够捕获笔画方向
            # padding=1: 边缘补零, 保持输出尺寸不变
            nn.Conv2d(1, 32, kernel_size=3, padding=1),

            # BatchNorm: 对 32 个通道各自归一化到均值≈0、方差≈1
            # 好处: 训练更稳定, 可以用更大的学习率, 收敛更快
            nn.BatchNorm2d(32),

            # ReLU: 把负数变成 0, 正数不变
            # 好处: 引入非线性 (没有它, 多层线性变换等价于一层)
            #       同时计算简单, 缓解梯度消失
            nn.ReLU(),

            # MaxPool: 在每个 2×2 区域取最大值
            # 效果: 28×28 → 14×14 (尺寸减半), 保留最显著的特征
            #       同时提供一定的平移不变性
            nn.MaxPool2d(2, 2),

            # ===== 第 2 层卷积块 =====
            # Conv2d: 32 个输入通道 → 64 个输出通道
            # 第 2 层能学到的特征比第 1 层更高级: 第 1 层学边缘,
            # 第 2 层把边缘组合成笔画、弧线、交叉点等
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14×14 → 7×7
        )
        # 经过两层卷积后: (batch, 64, 7, 7) = 每张图变成 64 个 7×7 的特征图

        # ---- 分类器部分 (全连接层) ----
        # 这部分负责把提取到的特征映射到 10 个类别
        self.classifier = nn.Sequential(
            # Flatten: 把 (64, 7, 7) 展平成 (3136,) 的一维向量
            nn.Flatten(),

            # 全连接层: 3136 维 → 128 维
            # 128 维是一个"瓶颈", 把 3136 维的信息压缩成更紧凑的表示
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            # Dropout: 训练时随机丢弃 50% 的神经元
            # 作用: 防止模型过度依赖某几个神经元, 强制它学习冗余特征
            # 注意: 只在训练时生效, 预测时自动关闭 (PyTorch 的 Dropout 自带这个逻辑)
            nn.Dropout(0.5),

            # 输出层: 128 维 → 10 维
            # 输出 10 个数字各自的"得分" (logits), 得分最高的就是预测结果
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 从输入图像 → 输出 10 个类别的得分

        参数:
            x: 输入图像, shape=(batch_size, 1, 28, 28)
               batch_size = 一次处理多少张图
               1 = 灰度图 (1 个通道)
               28×28 = 图像尺寸

        返回:
            shape=(batch_size, 10) — 每张图对应 10 个数字的得分

        流程:
            输入 (batch, 1, 28, 28)
            → features 卷积块 → (batch, 64, 7, 7)
            → classifier 分类器 → (batch, 10)
        """
        # 先过卷积层提取特征
        x = self.features(x)
        # 再过全连接层做分类
        x = self.classifier(x)
        return x


# ============================================================
# 工具函数
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """
    统计模型的可训练参数数量

    用途: 了解模型规模, 方便对比不同模型的大小
    例: MNISTCNN 有 421,834 个参数

    参数:
        model: PyTorch 模型

    返回:
        int: 可训练参数的总数
    """
    # p.numel() = 这个张量有几个元素 (参数量)
    # p.requires_grad = True 表示这个参数会被梯度更新 (可训练)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
