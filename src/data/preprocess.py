"""
图像预处理模块
包含：二值化、降噪、中心化等功能
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class Preprocessor:
    """图像预处理类"""

    def __init__(self):
        self.params = {}

    def binarize(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        二值化处理

        Args:
            X: 输入图像, 范围 [0, 1]
            threshold: 二值化阈值

        Returns:
            二值化后的图像, 值为 0 或 1
        """
        return (X >= threshold).astype(np.float32)

    def denoise(self, X: np.ndarray, method: str = 'median', kernel_size: int = 3) -> np.ndarray:
        """
        降噪处理

        Args:
            X: 输入图像, 范围 [0, 1]
            method: 降噪方法 ('median' 或 'morphological')
            kernel_size: 核大小

        Returns:
            降噪后的图像
        """
        # 转换为 uint8 格式用于 OpenCV
        X_uint8 = (X * 255).astype(np.uint8)

        if method == 'median':
            # 中值滤波降噪
            denoised = np.array([cv2.medianBlur(img, kernel_size) for img in X_uint8])
        elif method == 'morphological':
            # 形态学操作去噪
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            denoised = np.array([cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) for img in X_uint8])
        else:
            raise ValueError(f"未知的降噪方法: {method}")

        return denoised.astype(np.float32) / 255.0

    def center_image(self, img: np.ndarray) -> np.ndarray:
        """
        将单个图像中的数字居中

        Args:
            img: 单张图像 (28, 28)

        Returns:
            居中后的图像
        """
        # 找到非零像素的边界框
        coords = np.where(img > 0)
        if len(coords[0]) == 0:
            return img  # 空图像直接返回

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # 提取数字区域
        digit = img[y_min:y_max+1, x_min:x_max+1]

        # 计算居中偏移
        h, w = digit.shape
        target_h, target_w = img.shape

        y_offset = (target_h - h) // 2
        x_offset = (target_w - w) // 2

        # 创建居中后的图像
        centered = np.zeros_like(img)
        centered[y_offset:y_offset+h, x_offset:x_offset+w] = digit

        return centered

    def center(self, X: np.ndarray) -> np.ndarray:
        """
        批量居中处理

        Args:
            X: 输入图像 (N, 28, 28)

        Returns:
            居中后的图像
        """
        return np.array([self.center_image(img) for img in X])

    def resize_to_center(self, img: np.ndarray, target_size: Tuple[int, int] = (20, 20),
                         canvas_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        将数字缩放并居中到画布中心
        参考 MNIST 官方预处理方式

        Args:
            img: 输入图像
            target_size: 数字目标尺寸
            canvas_size: 画布尺寸

        Returns:
            处理后的图像
        """
        # 找到非零像素边界
        coords = np.where(img > 0)
        if len(coords[0]) == 0:
            return img

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # 提取数字
        digit = img[y_min:y_max+1, x_min:x_max+1]

        # 保持宽高比缩放
        h, w = digit.shape
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 缩放
        digit_resized = cv2.resize(digit.astype(np.float32), (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)

        # 居中放置
        canvas = np.zeros(canvas_size, dtype=np.float32)
        y_offset = (canvas_size[0] - new_h) // 2
        x_offset = (canvas_size[1] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

        return canvas

    def preprocess_pipeline(self, X: np.ndarray,
                           binarize: bool = True,
                           denoise_method: Optional[str] = 'median',
                           center: bool = True,
                           threshold: float = 0.3) -> np.ndarray:
        """
        完整预处理流水线

        Args:
            X: 输入图像
            binarize: 是否二值化
            denoise_method: 降噪方法 (None, 'median', 'morphological')
            center: 是否居中
            threshold: 二值化阈值

        Returns:
            预处理后的图像
        """
        result = X.copy()

        # 步骤1: 降噪（在二值化之前效果更好）
        if denoise_method:
            result = self.denoise(result, method=denoise_method)

        # 步骤2: 二值化
        if binarize:
            result = self.binarize(result, threshold=threshold)

        # 步骤3: 居中
        if center:
            result = self.center(result)

        return result

    def compare_preprocessing(self, X: np.ndarray, n_samples: int = 5) -> dict:
        """
        对比不同预处理方式的效果
        用于实验分析和报告撰写

        Args:
            X: 输入图像
            n_samples: 对比样本数

        Returns:
            包含不同预处理结果的字典
        """
        samples = X[:n_samples]

        results = {
            'original': samples,
            'binarize_only': self.binarize(samples),
            'denoise_only': self.denoise(samples),
            'center_only': self.center(samples),
            'full_pipeline': self.preprocess_pipeline(samples),
        }

        return results


if __name__ == "__main__":
    from loader import load_mnist

    # 加载数据并测试预处理
    X_train, y_train, _, _ = load_mnist()

    preprocessor = Preprocessor()

    # 测试完整流水线
    X_processed = preprocessor.preprocess_pipeline(X_train[:100])
    print(f"预处理完成: {X_processed.shape}")

    # 对比不同预处理方式
    comparison = preprocessor.compare_preprocessing(X_train)
    print(f"对比实验结果: {list(comparison.keys())}")
