# 手写数字识别系统

基于 MNIST 数据集的手写数字识别项目，课程设计作业。

## 项目结构

```
ML_assignment/
├── data/
│   ├── raw/                # 原始数据（自动下载）
│   └── processed/          # 预处理后数据
├── src/
│   ├── data/               # 数据加载与预处理（组长）
│   ├── features/           # 特征工程（特征工程专家）
│   ├── models/             # 模型训练（算法工程师）
│   └── visualization/      # 可视化与测试（测试与可视化）
├── notebooks/              # Jupyter 实验笔记
├── tests/                  # 单元测试
├── main.py                 # 主入口
└── requirements.txt        # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
cd D:\ML_assignment
pip install -r requirements.txt
```

### 2. 运行数据预处理

```bash
python main.py
```

这会自动下载 MNIST 数据集并进行预处理。

## 数据接口约定

所有预处理后的数据格式：

```python
X_train: numpy.ndarray  # shape=(60000, 28, 28), float32, 范围[0, 1]
y_train: numpy.ndarray  # shape=(60000,), int64, 标签 0-9
X_test: numpy.ndarray   # shape=(10000, 28, 28)
y_test: numpy.ndarray   # shape=(10000,)
```

其他组员可以直接加载：

```python
from src.data.loader import load_processed_data

X_train, y_train, X_test, y_test = load_processed_data()
```

## 预处理步骤

1. **降噪** - 中值滤波去除孤立噪点
2. **二值化** - 阈值分割，突出数字轮廓
3. **中心化** - 让数字居中，提高识别准确率

## 各模块职责

| 模块 | 负责人 | 核心任务 |
|-----|-------|---------|
| data | 组长 | 数据加载、预处理、接口定义 |
| features | 特征工程 | 特征提取(HOG/LBP)、降维(PCA) |
| models | 算法工程师 | SVM/随机森林/CNN、调参 |
| visualization | 测试可视化 | EDA、混淆矩阵、GUI |

## 目标

- [x] 项目骨架搭建
- [x] 数据加载模块
- [x] 预处理模块
- [ ] 特征工程
- [ ] 模型训练
- [ ] 可视化与评估
- [ ] GUI 界面
- [ ] 课程设计报告
