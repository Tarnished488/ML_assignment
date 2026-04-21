"""
End-to-end feature generation helpers.

Use this module after running the data preprocessing step. It loads
processed MNIST arrays, extracts selected features, optionally applies PCA,
and saves model-ready feature matrices.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from src.features.extractor import FeatureExtractor, FeatureMethod, save_features


def build_feature_dataset(
    processed_dir: Union[str, Path] = "data/processed",
    save_dir: Union[str, Path] = "data/features",
    method: FeatureMethod = ("hog", "lbp", "shape"),
    pca_components: Optional[Union[int, float]] = None,
    save_extractor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build and save train/test feature matrices from preprocessed images.

    Parameters
    ----------
    processed_dir:
        Directory containing X_train.npy, y_train.npy, X_test.npy, y_test.npy.
    save_dir:
        Directory where extracted feature matrices are written.
    method:
        "raw", "hog", "lbp", "shape", or a sequence such as
        ("hog", "lbp", "shape").
    pca_components:
        None for no PCA, an int number of components, or a float variance
        target accepted by sklearn PCA, for example 0.95.
    save_extractor:
        Whether to save the fitted scaler/PCA object for later inference.

    Returns
    -------
    X_train_features, y_train, X_test_features, y_test
    """
    from src.data.loader import load_processed_data

    X_train, y_train, X_test, y_test = load_processed_data(str(processed_dir))

    extractor = FeatureExtractor()
    X_train_features = extractor.fit_transform(
        X_train,
        method=method,
        pca_components=pca_components,
    )
    X_test_features = extractor.transform(X_test, method=method)

    save_features(X_train_features, y_train, X_test_features, y_test, save_dir=save_dir)

    if save_extractor:
        extractor.save(Path(save_dir) / "feature_extractor.pkl")

    return X_train_features, y_train, X_test_features, y_test


def _parse_method(method_text: str) -> Union[str, Sequence[str]]:
    methods = [item.strip() for item in method_text.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one feature method is required.")
    if len(methods) == 1:
        return methods[0]
    return tuple(methods)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MNIST features.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--save-dir", default="data/features")
    parser.add_argument(
        "--method",
        default="hog,lbp,shape",
        help="raw, hog, lbp, shape, or comma-separated methods",
    )
    parser.add_argument("--pca", type=float, default=None, help="PCA components count or variance ratio")
    args = parser.parse_args()

    pca_components: Optional[Union[int, float]]
    if args.pca is not None and args.pca >= 1 and float(args.pca).is_integer():
        pca_components = int(args.pca)
    else:
        pca_components = args.pca

    train_X, train_y, test_X, test_y = build_feature_dataset(
        processed_dir=args.processed_dir,
        save_dir=args.save_dir,
        method=_parse_method(args.method),
        pca_components=pca_components,
    )

    print(f"Saved train features: {train_X.shape}, labels: {train_y.shape}")
    print(f"Saved test features: {test_X.shape}, labels: {test_y.shape}")
