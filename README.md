# 手写数字识别系统

基于 MNIST 数据集的手写数字识别项目，课程设计作业。

## 项目结构

```
ML_assignment/
├── data/
│   ├── raw/                  # 原始 MNIST 数据（自动下载）
│   ├── processed/            # 预处理后数据
│   └── features/             # 提取的特征向量
├── src/
│   ├── data/                 # 数据加载与预处理（组长）
│   │   ├── loader.py         # MNIST 数据加载
│   │   └── preprocess.py     # 降噪/二值化/中心化
│   ├── features/             # 特征工程（特征工程组员）
│   │   ├── extractor.py      # HOG/LBP/Shape/PCA 特征提取
│   │   └── pipeline.py       # 特征提取流水线
│   ├── models/               # 模型训练与评估（烷基化三烃氧二梨）
│   │   ├── cnn.py            # CNN + FocalLoss 定义
│   │   ├── train.py          # 训练流程（CNN + KNN/LR/RF 基线）
│   │   └── evaluate.py       # 评估指标（Accuracy/Precision/Recall/F1/混淆矩阵）
│   └── visualization/        # 可视化（烷基化三烃氧二梨）
│       └── plot.py           # 训练曲线/混淆矩阵/模型对比/错误样本/Grad-CAM
├── results/                  # 运行后自动生成的评估结果和图表
├── tests/                    # 单元测试
├── main.py                   # 一键运行入口（训练 + 评估 + 画图）
└── requirements.txt          # 依赖
```

## 快速开始

### 一条命令跑全部

```bash
pip install -r requirements.txt
python main.py
```

这会自动完成：下载 MNIST → 预处理 → 特征提取 → 训练 CNN + 3 个基线模型 → 在训练集/验证集/测试集上分别评估 → 生成图表。

更多参数和常见问题见 [模型与评估说明文档](模型与评估说明文档.md)。

### 单独运行各模块

```bash
# 仅数据预处理
python main.py

# 仅特征提取
python -m src.features.pipeline

# 特征提取测试
python tests/test_feature_extract.py
```

## 数据划分

70000 张 MNIST 图像按 **训练集:验证集:测试集 = 6:2:2** 划分：

| 集合 | 数量 | 用途 |
|------|------|------|
| 训练集 | 42,000 | 训练模型参数 |
| 验证集 | 14,000 | 早停判断、超参选择、过拟合检测 |
| 测试集 | 14,000 | 最终评估，所有模型共用 |

运行后会在训练集、验证集、测试集上**分别输出**每个模型的 Accuracy / Precision / Recall / F1 和混淆矩阵。

## 数据接口约定

预处理后数据格式：

```python
X_train: numpy.ndarray  # shape=(N, 28, 28), float32, 范围 [0, 1]
y_train: numpy.ndarray  # shape=(N,), int64, 标签 0-9
```

加载方式：

```python
from src.data.loader import load_processed_data
X_train, y_train, X_test, y_test = load_processed_data()
```

## 预处理步骤

1. **降噪** — 中值滤波去除孤立噪点
2. **二值化** — 阈值分割（threshold=0.3），突出数字轮廓
3. **中心化** — 找到数字边界框，移到 28×28 画布正中央

## 特征工程

特征工程模块位于 `src/features/`，核心类为 `FeatureExtractor`。

| 特征 | 维度 | 描述 |
|------|------|------|
| Raw Pixels | 784 | 原始像素展平 |
| HOG | 324 | 梯度方向直方图，提取笔画方向 |
| LBP | 10 | 局部二值模式，描述纹理 |
| Shape | 71 | 形状统计（宽高比、重心、投影、象限密度） |

默认组合：**HOG 324 + LBP 10 + Shape 71 = 405 维**

## 模型

### CNN（主模型）

```
输入层: 1 x 28 x 28 (灰度图像)
    ↓
卷积层1: Conv2d(1→32) + BatchNorm + ReLU + MaxPool
    ↓
卷积层2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool
    ↓
Dropout: 0.25
    ↓
全连接层1: Linear(3136→128) + ReLU + Dropout
    ↓
全连接层2: Linear(128→10)
    ↓
输出层: 10类概率分布
```

- 优化器：Adam (lr=0.001, weight_decay=1e-4)
- 损失函数：Focal Loss + Label Smoothing (关注难分样本，防止过度自信)
- 正则化：Kaiming 初始化 + BatchNorm + Dropout(0.25+0.5) + 数据增强 + 早停

### 基线模型（用于对比）

| 模型 | 特征 | 说明 |
|------|------|------|
| KNN (k=5) | HOG+LBP+Shape 405维 | 经典近邻分类 |
| Logistic Regression | HOG+LBP+Shape 405维 | 线性模型代表 |
| Random Forest | HOG+LBP+Shape 405维 | 集成方法代表 |

## 各模块职责

| 模块 | 负责人 | 核心任务 | 状态 |
|-----|--------|---------|------|
| data | 组长 | 数据加载、预处理、接口定义 |  |
| features | 特征工程 | 特征提取(HOG/LBP/Shape)、PCA降维 |  |
| models | 烷基化三烃氧二梨 | CNN + 基线模型训练与评估 |  |
| visualization | 烷基化三烃氧二梨 | 训练曲线、混淆矩阵、模型对比图 |  |

## 实验结果

详细结果见 `results/evaluation_summary.txt`，图表见 `results/` 目录。

3 epochs 快速测试示例（训练集:验证集:测试集 = 42000:14000:14000）：

| 模型 | 训练集 Acc | 验证集 Acc | 测试集 Acc |
|------|-----------|-----------|-----------|
| CNN (Focal Loss) | 95.32% | 95.49% | 95.32% |
| Logistic Regression | 98.30% | 97.01% | 97.01% |
| KNN (k=5) | 97.58% | 96.33% | 96.35% |
| Random Forest | 100.00% | 96.40% | 95.84% |

> 3 epochs 快速测试结果。Focal Loss 训练初期 loss 较高属正常现象。30 epochs 完整训练后各模型准确率会进一步提升。Random Forest 训练集 100% 说明存在过拟合。

## 目标

- [x] 项目骨架搭建
- [x] 数据加载模块
- [x] 预处理模块
- [x] 特征工程
- [x] 特征提取测试
- [x] 模型训练（CNN + KNN + Logistic Regression + Random Forest）
- [x] 可视化与评估（训练曲线/混淆矩阵/模型对比/错误样本/各类别准确率）
- [ ] GUI 界面
- [ ] 课程设计报告

## 许可证

MIT License
