"""
Feature extraction module for MNIST images.

The public entry point is FeatureExtractor. It converts preprocessed
28x28 images into model-ready 2D feature matrices and optionally applies
PCA dimensionality reduction.
"""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FeatureMethod = Union[str, Sequence[str]]


class FeatureExtractor:
    """
    Extract HOG, LBP, shape-statistic and raw-pixel features from MNIST images.

    Parameters
    ----------
    hog_cell_size:
        Size of each HOG cell. For 28x28 MNIST, (7, 7) produces a compact
        descriptor while preserving local stroke direction.
    hog_block_size:
        HOG block size measured in cells.
    hog_bins:
        Number of orientation bins used by HOG.
    lbp_radius:
        Radius of the circular LBP neighborhood.
    lbp_points:
        Number of sampling points used by LBP. The output histogram has
        lbp_points + 2 bins when uniform LBP is enabled.
    shape_threshold:
        Pixel threshold used when calculating binary shape statistics.
    use_scaler:
        Whether to standardize features before PCA / model training.
    """

    def __init__(
        self,
        hog_cell_size: Tuple[int, int] = (7, 7),
        hog_block_size: Tuple[int, int] = (2, 2),
        hog_bins: int = 9,
        lbp_radius: int = 1,
        lbp_points: int = 8,
        shape_threshold: float = 0.1,
        use_scaler: bool = True,
    ) -> None:
        self.hog_cell_size = hog_cell_size
        self.hog_block_size = hog_block_size
        self.hog_bins = hog_bins
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.shape_threshold = shape_threshold
        self.use_scaler = use_scaler

        self.scaler: Optional[StandardScaler] = StandardScaler() if use_scaler else None
        self.pca: Optional[PCA] = None

    @staticmethod
    def _ensure_image_batch(X: np.ndarray) -> np.ndarray:
        """Validate and normalize input to shape (N, H, W)."""
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        if X.ndim != 3:
            raise ValueError(f"Expected image array with shape (N, H, W), got {X.shape}")
        return X.astype(np.float32, copy=False)

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert a [0, 1] float image or uint8 image to uint8."""
        if img.dtype == np.uint8:
            return img
        clipped = np.clip(img, 0.0, 1.0)
        return (clipped * 255).astype(np.uint8)

    def extract_raw(self, X: np.ndarray) -> np.ndarray:
        """Flatten each image into a 784-dimensional raw-pixel feature."""
        X = self._ensure_image_batch(X)
        return X.reshape(X.shape[0], -1).astype(np.float32)

    def extract_hog(self, X: np.ndarray) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients features.

        This implementation uses NumPy gradients and block normalization so
        it does not add a new project dependency.
        """
        X = self._ensure_image_batch(X)
        features = [self._hog_single(img) for img in X]
        return np.asarray(features, dtype=np.float32)

    def _hog_single(self, img: np.ndarray) -> np.ndarray:
        img_u8 = self._to_uint8(img)
        img_f = img_u8.astype(np.float32) / 255.0

        gy, gx = np.gradient(img_f)
        magnitude = np.sqrt(gx * gx + gy * gy)
        angle = np.degrees(np.arctan2(gy, gx))
        angle = angle % 180.0

        h, w = img_f.shape
        cell_h, cell_w = self.hog_cell_size
        n_cells_y = h // cell_h
        n_cells_x = w // cell_w
        hist = np.zeros((n_cells_y, n_cells_x, self.hog_bins), dtype=np.float32)
        bin_width = 180.0 / self.hog_bins

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0, y1 = cy * cell_h, (cy + 1) * cell_h
                x0, x1 = cx * cell_w, (cx + 1) * cell_w
                cell_mag = magnitude[y0:y1, x0:x1].ravel()
                cell_angle = angle[y0:y1, x0:x1].ravel()
                bins = np.floor(cell_angle / bin_width).astype(np.int32)
                bins = np.clip(bins, 0, self.hog_bins - 1)
                hist[cy, cx] = np.bincount(bins, weights=cell_mag, minlength=self.hog_bins)

        block_h, block_w = self.hog_block_size
        blocks = []
        eps = 1e-6
        for y in range(n_cells_y - block_h + 1):
            for x in range(n_cells_x - block_w + 1):
                block = hist[y:y + block_h, x:x + block_w].ravel()
                block = block / np.sqrt(np.sum(block * block) + eps)
                blocks.append(block)

        if not blocks:
            return hist.ravel()
        return np.concatenate(blocks).astype(np.float32)

    def extract_lbp(self, X: np.ndarray, uniform: bool = True) -> np.ndarray:
        """
        Extract Local Binary Pattern histogram features.

        Uniform LBP maps patterns with at most two bit transitions to their
        number of active bits and maps all other patterns to the last bin.
        """
        X = self._ensure_image_batch(X)
        features = [self._lbp_single(img, uniform=uniform) for img in X]
        return np.asarray(features, dtype=np.float32)

    def _lbp_single(self, img: np.ndarray, uniform: bool = True) -> np.ndarray:
        img_f = np.clip(img.astype(np.float32), 0.0, 1.0)
        radius = self.lbp_radius
        points = self.lbp_points
        h, w = img_f.shape
        lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint16)

        center = img_f[radius:h - radius, radius:w - radius]
        for p in range(points):
            theta = 2.0 * np.pi * p / points
            dy = int(round(radius * np.sin(theta)))
            dx = int(round(radius * np.cos(theta)))
            neighbor = img_f[radius + dy:h - radius + dy, radius + dx:w - radius + dx]
            lbp |= ((neighbor >= center).astype(np.uint16) << p)

        if uniform:
            mapped = np.vectorize(self._uniform_lbp_code)(lbp)
            hist_bins = points + 2
            hist = np.bincount(mapped.ravel(), minlength=hist_bins).astype(np.float32)
        else:
            hist_bins = 2 ** points
            hist = np.bincount(lbp.ravel(), minlength=hist_bins).astype(np.float32)

        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        return hist

    def _uniform_lbp_code(self, code: int) -> int:
        bits = [(code >> i) & 1 for i in range(self.lbp_points)]
        transitions = sum(bits[i] != bits[(i + 1) % self.lbp_points] for i in range(self.lbp_points))
        if transitions <= 2:
            return sum(bits)
        return self.lbp_points + 1

    def extract_shape(self, X: np.ndarray) -> np.ndarray:
        """
        Extract MNIST-specific shape statistics.

        The descriptor contains global stroke density, bounding-box geometry,
        center of mass, horizontal/vertical projections and quadrant density.
        These features are useful for separating narrow digits such as 1 from
        round or wide digits such as 0, 6, 8 and 9.
        """
        X = self._ensure_image_batch(X)
        features = [self._shape_single(img) for img in X]
        return np.asarray(features, dtype=np.float32)

    def _shape_single(self, img: np.ndarray) -> np.ndarray:
        img_f = np.clip(img.astype(np.float32), 0.0, 1.0)
        h, w = img_f.shape
        mask = img_f > self.shape_threshold

        ink_ratio = float(mask.mean())
        mean_intensity = float(img_f.mean())
        std_intensity = float(img_f.std())

        if not mask.any():
            bbox_features = np.zeros(8, dtype=np.float32)
        else:
            ys, xs = np.where(mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            bbox_h = float(y_max - y_min + 1)
            bbox_w = float(x_max - x_min + 1)
            aspect_ratio = bbox_w / max(bbox_h, 1.0)
            bbox_area_ratio = (bbox_h * bbox_w) / float(h * w)

            weights = img_f[mask]
            weight_sum = max(float(weights.sum()), 1e-6)
            center_y = float((ys * weights).sum() / weight_sum)
            center_x = float((xs * weights).sum() / weight_sum)

            bbox_features = np.asarray(
                [
                    bbox_h / h,
                    bbox_w / w,
                    aspect_ratio,
                    bbox_area_ratio,
                    center_y / max(h - 1, 1),
                    center_x / max(w - 1, 1),
                    (center_y - (h - 1) / 2.0) / h,
                    (center_x - (w - 1) / 2.0) / w,
                ],
                dtype=np.float32,
            )

        horizontal_projection = mask.sum(axis=1).astype(np.float32) / w
        vertical_projection = mask.sum(axis=0).astype(np.float32) / h
        quadrant_density = np.asarray(
            [
                mask[: h // 2, : w // 2].mean(),
                mask[: h // 2, w // 2 :].mean(),
                mask[h // 2 :, : w // 2].mean(),
                mask[h // 2 :, w // 2 :].mean(),
            ],
            dtype=np.float32,
        )

        global_features = np.asarray(
            [ink_ratio, mean_intensity, std_intensity],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                global_features,
                bbox_features,
                horizontal_projection,
                vertical_projection,
                quadrant_density,
            ]
        ).astype(np.float32)

    def extract(self, X: np.ndarray, method: FeatureMethod = "hog") -> np.ndarray:
        """
        Extract features by method.

        method can be "raw", "hog", "lbp", "shape", or a sequence such as
        ("hog", "shape") to concatenate multiple feature families.
        """
        if isinstance(method, str):
            method_names: Iterable[str] = (method,)
        else:
            method_names = method

        feature_parts = []
        for name in method_names:
            normalized = name.lower()
            if normalized == "raw":
                feature_parts.append(self.extract_raw(X))
            elif normalized == "hog":
                feature_parts.append(self.extract_hog(X))
            elif normalized == "lbp":
                feature_parts.append(self.extract_lbp(X))
            elif normalized in {"shape", "stats", "stat"}:
                feature_parts.append(self.extract_shape(X))
            else:
                raise ValueError(f"Unknown feature extraction method: {name}")

        if len(feature_parts) == 1:
            return feature_parts[0]
        return np.concatenate(feature_parts, axis=1).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        method: FeatureMethod = ("hog", "lbp", "shape"),
        pca_components: Optional[Union[int, float]] = None,
    ) -> np.ndarray:
        """Extract features, fit scaler/PCA on training data, and transform it."""
        features = self.extract(X, method=method)

        if self.scaler is not None:
            features = self.scaler.fit_transform(features)

        if pca_components is not None:
            self.pca = PCA(n_components=pca_components, random_state=42)
            features = self.pca.fit_transform(features)

        return features.astype(np.float32)

    def transform(self, X: np.ndarray, method: FeatureMethod = ("hog", "lbp", "shape")) -> np.ndarray:
        """Extract features and apply fitted scaler/PCA to validation or test data."""
        features = self.extract(X, method=method)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        if self.pca is not None:
            features = self.pca.transform(features)

        return features.astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Save extractor parameters, scaler and PCA state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "FeatureExtractor":
        """Load a previously saved FeatureExtractor."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, FeatureExtractor):
            raise TypeError(f"Expected FeatureExtractor object in {path}")
        return obj


def save_features(
    X_train_features: np.ndarray,
    y_train: np.ndarray,
    X_test_features: np.ndarray,
    y_test: np.ndarray,
    save_dir: Union[str, Path] = "data/features",
) -> None:
    """Save extracted feature matrices and labels as .npy files."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "X_train_features.npy", X_train_features)
    np.save(save_path / "y_train.npy", y_train)
    np.save(save_path / "X_test_features.npy", X_test_features)
    np.save(save_path / "y_test.npy", y_test)


def load_features(data_dir: Union[str, Path] = "data/features"):
    """Load feature matrices saved by save_features."""
    data_path = Path(data_dir)
    return (
        np.load(data_path / "X_train_features.npy"),
        np.load(data_path / "y_train.npy"),
        np.load(data_path / "X_test_features.npy"),
        np.load(data_path / "y_test.npy"),
    )
