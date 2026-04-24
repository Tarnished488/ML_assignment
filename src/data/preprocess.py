"""
Image preprocessing module
Includes: binarization, denoising, centering, and other functions
"""

import numpy as np
from typing import Optional, Tuple

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


class Preprocessor:
    """Image preprocessing class"""

    def __init__(self):
        self.params = {}

    def binarize(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binarization

        Args:
            X: input images, range [0, 1]
            threshold: binarization threshold

        Returns:
            binarized images, values are 0 or 1
        """
        return (X >= threshold).astype(np.float32)

    def denoise(self, X: np.ndarray, method: str = 'median', kernel_size: int = 3) -> np.ndarray:
        """
        Denoising

        Args:
            X: input images, range [0, 1]
            method: denoising method ('median' or 'morphological')
            kernel_size: kernel size

        Returns:
            denoised images
        """
        # Convert to uint8 format for OpenCV
        X_uint8 = (X * 255).astype(np.uint8)

        if method == 'median':
            # Median filter denoising
            if cv2 is not None:
                denoised = np.array([cv2.medianBlur(img, kernel_size) for img in X_uint8])
            else:
                denoised = self._median_filter(X_uint8, kernel_size)
        elif method == 'morphological':
            # Morphological denoising
            if cv2 is None:
                raise ImportError("OpenCV is required for morphological denoising.")
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            denoised = np.array([cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) for img in X_uint8])
        else:
            raise ValueError(f"Unknown denoising method: {method}")

        return denoised.astype(np.float32) / 255.0

    def _median_filter(self, X_uint8: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Pure NumPy fallback for median denoising when OpenCV is unavailable."""
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd integer")

        pad = kernel_size // 2
        padded = np.pad(X_uint8, ((0, 0), (pad, pad), (pad, pad)), mode='edge')
        denoised = np.empty_like(X_uint8)

        for y in range(X_uint8.shape[1]):
            for x in range(X_uint8.shape[2]):
                window = padded[:, y:y + kernel_size, x:x + kernel_size]
                denoised[:, y, x] = np.median(window, axis=(1, 2))

        return denoised

    def center_image(self, img: np.ndarray) -> np.ndarray:
        """
        Center the digit in a single image

        Args:
            img: single image (28, 28)

        Returns:
            centered image
        """
        # Find bounding box of non-zero pixels
        coords = np.where(img > 0)
        if len(coords[0]) == 0:
            return img  # Return empty image as-is

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Extract digit region
        digit = img[y_min:y_max+1, x_min:x_max+1]

        # Calculate centering offset
        h, w = digit.shape
        target_h, target_w = img.shape

        y_offset = (target_h - h) // 2
        x_offset = (target_w - w) // 2

        # Create centered image
        centered = np.zeros_like(img)
        centered[y_offset:y_offset+h, x_offset:x_offset+w] = digit

        return centered

    def center(self, X: np.ndarray) -> np.ndarray:
        """
        Batch centering

        Args:
            X: input images (N, 28, 28)

        Returns:
            centered images
        """
        return np.array([self.center_image(img) for img in X])

    def resize_to_center(self, img: np.ndarray, target_size: Tuple[int, int] = (20, 20),
                         canvas_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Scale and center the digit onto the canvas center
        Following MNIST official preprocessing approach

        Args:
            img: input image
            target_size: target digit size
            canvas_size: canvas size

        Returns:
            processed image
        """
        # Find bounding box of non-zero pixels
        coords = np.where(img > 0)
        if len(coords[0]) == 0:
            return img

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Extract digit
        digit = img[y_min:y_max+1, x_min:x_max+1]

        # Scale while preserving aspect ratio
        h, w = digit.shape
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        if cv2 is None:
            raise ImportError("OpenCV is required for resize_to_center.")
        digit_resized = cv2.resize(digit.astype(np.float32), (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)

        # Place at center
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
        Full preprocessing pipeline

        Args:
            X: input images
            binarize: whether to binarize
            denoise_method: denoising method (None, 'median', 'morphological')
            center: whether to center
            threshold: binarization threshold

        Returns:
            preprocessed images
        """
        result = X.copy()

        # Step 1: Denoising (works better before binarization)
        if denoise_method:
            result = self.denoise(result, method=denoise_method)

        # Step 2: Binarization
        if binarize:
            result = self.binarize(result, threshold=threshold)

        # Step 3: Centering
        if center:
            result = self.center(result)

        return result

    def compare_preprocessing(self, X: np.ndarray, n_samples: int = 5) -> dict:
        """
        Compare effects of different preprocessing methods
        Used for experimental analysis and report writing

        Args:
            X: input images
            n_samples: number of samples to compare

        Returns:
            dictionary containing different preprocessing results
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

    # Load data and test preprocessing
    X_train, y_train, _, _ = load_mnist()

    preprocessor = Preprocessor()

    # Test full pipeline
    X_processed = preprocessor.preprocess_pipeline(X_train[:100])
    print(f"Preprocessing complete: {X_processed.shape}")

    # Compare different preprocessing methods
    comparison = preprocessor.compare_preprocessing(X_train)
    print(f"Comparison results: {list(comparison.keys())}")
