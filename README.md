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

### 3. 提取特征

数据预处理完成后，可以运行特征工程流水线：

```bash
python -m src.features.pipeline
```

默认会从 `data/processed/` 读取预处理后的 MNIST 数据，提取 `HOG + LBP + shape` 组合特征，并保存到 `data/features/`。

也可以手动指定特征类型和 PCA 降维维度：

```bash
python -m src.features.pipeline --method hog,lbp,shape --pca 100
```

### 4. 测试特征提取

```bash
python tests/test_feature_extract.py
```

该测试会读取已经下载好的 MNIST 数据，先调用 `src.data.preprocess.Preprocessor` 完成预处理，再进行特征提取、PCA 转换以及特征保存/读取测试。

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

## 特征工程

特征工程模块位于 `src/features/`，核心类为 `FeatureExtractor`。

目前支持的特征：

1. **Raw Pixels 原始像素特征** - 将 `28 x 28` 图像展平成 `784` 维向量，保留完整灰度信息。
2. **HOG 梯度方向直方图特征** - 提取数字笔画的边缘方向和局部形状信息，默认输出 `324` 维。
3. **LBP 局部二值模式特征** - 描述局部纹理和邻域灰度变化，默认输出 `10` 维。
4. **Shape 形状统计特征** - 针对 MNIST 提取笔画像素占比、外接框宽高、宽高比、重心位置、水平/垂直投影和四象限密度，默认输出 `71` 维。
5. **PCA 降维** - 可对组合特征进行降维，减少冗余并加快传统机器学习模型训练。

默认组合特征为：

```python
method=("hog", "lbp", "shape")
```

默认组合后维度为：

```text
HOG 324 + LBP 10 + Shape 71 = 405
```

代码使用示例：

```python
from src.features import FeatureExtractor

extractor = FeatureExtractor()

X_train_features = extractor.fit_transform(
    X_train,
    method=("hog", "lbp", "shape"),
    pca_components=100
)

X_test_features = extractor.transform(
    X_test,
    method=("hog", "lbp", "shape")
)
```

## 各模块职责

| 模块 | 负责人 | 核心任务 |
|-----|-------|---------|
| data | 组长 | 数据加载、预处理、接口定义 |
| features | 特征工程 | 特征提取(Raw/HOG/LBP/Shape)、降维(PCA)、特征保存与测试 |
| models | 算法工程师 | SVM/随机森林/CNN、调参 |
| visualization | 测试可视化 | EDA、混淆矩阵、GUI |

## 目标

- [x] 项目骨架搭建
- [x] 数据加载模块
- [x] 预处理模块
- [x] 特征工程
- [x] 特征提取测试
- [ ] 模型训练
- [ ] 可视化与评估
- [ ] GUI 界面
- [ ] 课程设计报告
