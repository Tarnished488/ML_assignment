"""Tests for MNIST feature extraction."""

import sys
import shutil
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FeatureExtractor, load_features, save_features
from src.data.loader import load_mnist
from src.data.preprocess import Preprocessor


class TestFeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        X_train, y_train, X_test, y_test = load_mnist(
            data_dir=str(PROJECT_ROOT / "data" / "raw"),
            download=False,
            preprocess=False,
        )

        cls.X_train_raw = X_train[:24]
        cls.y_train = y_train[:24]
        cls.X_test_raw = X_test[:8]
        cls.y_test = y_test[:8]

        cls.preprocessor = Preprocessor()
        cls.X_train = cls.preprocessor.preprocess_pipeline(
            cls.X_train_raw,
            binarize=True,
            denoise_method="median",
            center=True,
            threshold=0.3,
        )
        cls.X_test = cls.preprocessor.preprocess_pipeline(
            cls.X_test_raw,
            binarize=True,
            denoise_method="median",
            center=True,
            threshold=0.3,
        )

    def setUp(self) -> None:
        self.extractor = FeatureExtractor()

    def test_load_mnist_then_preprocess_before_feature_extract(self) -> None:
        self.assertEqual(self.X_train_raw.shape, (24, 28, 28))
        self.assertEqual(self.X_test_raw.shape, (8, 28, 28))
        self.assertEqual(self.X_train.shape, self.X_train_raw.shape)
        self.assertEqual(self.X_test.shape, self.X_test_raw.shape)
        self.assertEqual(self.X_train.dtype, np.float32)
        self.assertTrue(np.isin(self.X_train, [0.0, 1.0]).all())
        self.assertTrue(np.isin(self.X_test, [0.0, 1.0]).all())
        self.assertGreater(self.X_train.sum(), 0)
        self.assertGreater(self.X_test.sum(), 0)

    def test_extract_single_feature_types(self) -> None:
        raw = self.extractor.extract(self.X_train, method="raw")
        hog = self.extractor.extract(self.X_train, method="hog")
        lbp = self.extractor.extract(self.X_train, method="lbp")
        shape = self.extractor.extract(self.X_train, method="shape")

        self.assertEqual(raw.shape, (24, 784))
        self.assertEqual(hog.shape, (24, 324))
        self.assertEqual(lbp.shape, (24, 10))
        self.assertEqual(shape.shape, (24, 71))

        self.assertTrue(np.isfinite(raw).all())
        self.assertTrue(np.isfinite(hog).all())
        self.assertTrue(np.isfinite(lbp).all())
        self.assertTrue(np.isfinite(shape).all())

    def test_extract_combined_mnist_features(self) -> None:
        features = self.extractor.extract(self.X_train, method=("hog", "lbp", "shape"))

        self.assertEqual(features.shape, (24, 405))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_fit_transform_and_transform_with_pca(self) -> None:
        train_features = self.extractor.fit_transform(
            self.X_train,
            method=("hog", "lbp", "shape"),
            pca_components=5,
        )
        test_features = self.extractor.transform(
            self.X_test,
            method=("hog", "lbp", "shape"),
        )

        self.assertEqual(train_features.shape, (24, 5))
        self.assertEqual(test_features.shape, (8, 5))
        self.assertTrue(np.isfinite(train_features).all())
        self.assertTrue(np.isfinite(test_features).all())

    def test_save_and_load_features(self) -> None:
        X_train_features = self.extractor.extract(self.X_train, method=("hog", "shape"))
        X_test_features = self.extractor.extract(self.X_test, method=("hog", "shape"))

        tmp_dir = PROJECT_ROOT / "tests" / "tmp_feature_extract"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        try:
            save_features(
                X_train_features,
                self.y_train,
                X_test_features,
                self.y_test,
                save_dir=tmp_dir,
            )
            loaded = load_features(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        np.testing.assert_array_equal(loaded[0], X_train_features)
        np.testing.assert_array_equal(loaded[1], self.y_train)
        np.testing.assert_array_equal(loaded[2], X_test_features)
        np.testing.assert_array_equal(loaded[3], self.y_test)


if __name__ == "__main__":
    unittest.main()
