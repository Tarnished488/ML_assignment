"""
Exploratory Data Analysis (EDA) Module
Comprehensive exploratory data analysis for MNIST dataset, including:
1. Dataset overview and statistical information
2. Label distribution analysis
3. Image feature analysis
4. Anomaly detection
5. Hypothesis testing
6. Visualization analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 尝试导入scipy用于统计检验
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, some statistical test functions are unavailable")


class MNISTExplorer:
    """MNIST Dataset Explorer"""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray):
        """
        Initialize the explorer

        Args:
            X_train: Training images, shape=(N, 28, 28)
            y_train: Training labels, shape=(N,)
            X_test: Test images, shape=(M, 28, 28)
            y_test: Test labels, shape=(M,)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # 基本统计信息
        self.n_train = len(X_train)
        self.n_test = len(X_test)
        self.image_shape = X_train[0].shape

        # 缓存计算结果
        self._stats_cache = {}

        # 设置可视化风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def get_basic_stats(self) -> Dict[str, Any]:
        """获取数据集基本统计信息"""
        if 'basic_stats' in self._stats_cache:
            return self._stats_cache['basic_stats']

        stats = {
            'n_train': self.n_train,
            'n_test': self.n_test,
            'image_shape': self.image_shape,
            'train_labels': np.unique(self.y_train),
            'test_labels': np.unique(self.y_test),
            'train_label_dist': np.bincount(self.y_train, minlength=10),
            'test_label_dist': np.bincount(self.y_test, minlength=10),
            'train_pixel_stats': {
                'min': self.X_train.min(),
                'max': self.X_train.max(),
                'mean': self.X_train.mean(),
                'std': self.X_train.std(),
                'median': np.median(self.X_train)
            },
            'test_pixel_stats': {
                'min': self.X_test.min(),
                'max': self.X_test.max(),
                'mean': self.X_test.mean(),
                'std': self.X_test.std(),
                'median': np.median(self.X_test)
            }
        }

        self._stats_cache['basic_stats'] = stats
        return stats

    def print_summary(self):
        """打印数据集概览"""
        stats = self.get_basic_stats()

        print("=" * 60)
        print("MNIST Dataset Exploratory Analysis Overview")
        print("=" * 60)
        print(f"\n1. Dataset Size:")
        print(f"   Training Set: {stats['n_train']:,} images")
        print(f"   Test Set: {stats['n_test']:,} images")
        print(f"   Image Size: {stats['image_shape']}")

        print(f"\n2. Label Distribution:")
        print(f"   Training Set Labels: {stats['train_labels']}")
        print(f"   Test Set Labels: {stats['test_labels']}")

        print(f"\n3. Pixel Statistics (Training Set):")
        pixel_stats = stats['train_pixel_stats']
        print(f"   Min: {pixel_stats['min']:.4f}")
        print(f"   Max: {pixel_stats['max']:.4f}")
        print(f"   Mean: {pixel_stats['mean']:.4f}")
        print(f"   Std: {pixel_stats['std']:.4f}")
        print(f"   Median: {pixel_stats['median']:.4f}")

        print(f"\n4. Number of Samples per Class (Training Set):")
        for i in range(10):
            count = stats['train_label_dist'][i]
            percentage = count / stats['n_train'] * 100
            print(f"   Digit {i}: {count:6,d} ({percentage:5.1f}%)")

        print("=" * 60)

    def plot_label_distribution(self, save_path: Optional[str] = None):
        """Plot label distribution"""
        stats = self.get_basic_stats()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Training set label distribution
        axes[0].bar(range(10), stats['train_label_dist'], color='skyblue', alpha=0.7)
        axes[0].set_title('Training Set Label Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Digit Label', fontsize=12)
        axes[0].set_ylabel('Number of Samples', fontsize=12)
        axes[0].set_xticks(range(10))
        axes[0].grid(True, alpha=0.3)

        # 添加数量标签
        for i, count in enumerate(stats['train_label_dist']):
            axes[0].text(i, count + 50, f'{count:,}', ha='center', fontsize=9)

        # Test set label distribution
        axes[1].bar(range(10), stats['test_label_dist'], color='lightcoral', alpha=0.7)
        axes[1].set_title('Test Set Label Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Digit Label', fontsize=12)
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_xticks(range(10))
        axes[1].grid(True, alpha=0.3)

        # 添加数量标签
        for i, count in enumerate(stats['test_label_dist']):
            axes[1].text(i, count + 20, f'{count:,}', ha='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Label distribution plot saved to: {save_path}")

        plt.show()



    def analyze_pixel_intensity(self, save_path: Optional[str] = None):
        """Analyze pixel intensity distribution"""
        # 将图像展平
        train_pixels = self.X_train.reshape(-1)
        test_pixels = self.X_test.reshape(-1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Training set pixel histogram
        axes[0].hist(train_pixels, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Training Set Pixel Intensity Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Pixel Intensity', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Test set pixel histogram
        axes[1].hist(test_pixels, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_title('Test Set Pixel Intensity Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Pixel Intensity', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Boxplot comparison
        box_data = [train_pixels[::100], test_pixels[::100]]  # Downsample for speed
        axes[2].boxplot(box_data, labels=['Training Set', 'Test Set'])
        axes[2].set_title('Pixel Intensity Distribution Comparison', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Pixel Intensity', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pixel intensity analysis plot saved to: {save_path}")

        plt.show()

        # Print statistics
        print("\nPixel Intensity Statistics:")
        print(f"Training Set: Mean={train_pixels.mean():.4f}, Std={train_pixels.std():.4f}")
        print(f"Test Set: Mean={test_pixels.mean():.4f}, Std={test_pixels.std():.4f}")

        # 执行统计检验（如果有scipy）
        if HAS_SCIPY:
            _, p_value = stats.ks_2samp(train_pixels[::100], test_pixels[::100])
            print(f"\nKolmogorov-Smirnov Test: p-value={p_value:.6f}")
            if p_value < 0.05:
                print("Warning: Significant difference between training and test set pixel distributions!")
            else:
                print("No significant difference between training and test set pixel distributions.")




    def plot_mean_and_std_images(self, save_path: Optional[str] = None):
        """
        Plot mean and standard deviation images for each digit class

        Shows the average image and standard deviation image for each digit 0-9
        """
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        fig.suptitle('Mean and Standard Deviation Images by Digit Class', fontsize=16, fontweight='bold')

        for digit in range(10):
            # Get all images for this digit
            mask = self.y_train == digit
            if mask.sum() > 0:
                digit_images = self.X_train[mask]

                # Calculate mean and std images
                mean_image = digit_images.mean(axis=0)
                std_image = digit_images.std(axis=0)

                # Plot mean image
                ax_mean = axes[0, digit]
                ax_mean.imshow(mean_image, cmap='gray')
                ax_mean.set_title(f'Digit {digit} - Mean', fontsize=10)
                ax_mean.axis('off')

                # Plot std image
                ax_std = axes[1, digit]
                im = ax_std.imshow(std_image, cmap='hot')
                ax_std.set_title(f'Digit {digit} - Std', fontsize=10)
                ax_std.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mean and std images plot saved to: {save_path}")

        plt.show()

    def plot_pca_scatter(self, n_components: int = 2, save_path: Optional[str] = None):
        """
        Perform PCA and plot scatter plot of first two components

        Args:
            n_components: Number of PCA components (default: 2)
            save_path: Optional path to save the plot
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("Warning: scikit-learn not installed, PCA scatter plot unavailable.")
            return

        # Flatten images for PCA
        X_flat = self.X_train.reshape(self.X_train.shape[0], -1)

        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_flat)

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot each digit with different color
        colors = plt.cm.tab10(np.arange(10) / 10)

        for digit in range(10):
            mask = self.y_train == digit
            if mask.sum() > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=[colors[digit]], label=f'Digit {digit}',
                          alpha=0.6, s=10)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title('PCA Scatter Plot of MNIST Data', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA scatter plot saved to: {save_path}")

        plt.show()

        # Print PCA information
        print(f"\nPCA Analysis Results:")
        print(f"  Total variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        print(f"  PC1 variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"  PC2 variance: {pca.explained_variance_ratio_[1]*100:.1f}%")

    def run_comprehensive_analysis(self, output_dir: Optional[str] = None):
        """
        Run comprehensive EDA analysis with 4 main steps:
        1. Label distribution analysis
        2. Mean and standard deviation images
        3. Pixel value histogram analysis
        4. PCA scatter plot visualization

        Args:
            output_dir: Output directory for saving plots
        """
        print("Starting comprehensive exploratory data analysis...")

        # 1. Label distribution analysis
        print("\n1. Label Distribution Analysis")
        print("-" * 40)
        self.print_summary()
        if output_dir:
            self.plot_label_distribution(f"{output_dir}/label_distribution.png")
        else:
            self.plot_label_distribution()

        # 2. Mean and standard deviation images
        print("\n2. Mean and Standard Deviation Images")
        print("-" * 40)
        if output_dir:
            self.plot_mean_and_std_images(f"{output_dir}/mean_std_images.png")
        else:
            self.plot_mean_and_std_images()

        # 3. Pixel value histogram analysis
        print("\n3. Pixel Value Histogram Analysis")
        print("-" * 40)
        if output_dir:
            self.analyze_pixel_intensity(save_path=f"{output_dir}/pixel_intensity.png")
        else:
            self.analyze_pixel_intensity()

        # 4. PCA scatter plot
        print("\n4. PCA Scatter Plot Visualization")
        print("-" * 40)
        if output_dir:
            self.plot_pca_scatter(save_path=f"{output_dir}/pca_scatter.png")
        else:
            self.plot_pca_scatter()

        print("\nExploratory Data Analysis Complete!")


# 便捷函数
def load_and_analyze(data_dir: str = "data/processed", output_dir: Optional[str] = None):
    """
    Convenience function to load data and run EDA analysis

    Args:
        data_dir: Preprocessed data directory
        output_dir: Output directory for saving plots
    """
    from src.data.loader import load_processed_data

    print("Loading preprocessed data...")
    X_train, y_train, X_test, y_test = load_processed_data(data_dir)

    print("Initializing EDA explorer...")
    explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # 运行全面分析
    explorer.run_comprehensive_analysis(output_dir)

    return explorer


if __name__ == "__main__":
    # 测试代码
    from src.data.loader import load_mnist

    print("=" * 60)
    print("MNISTExplorer 测试代码")
    print("=" * 60)

    print("\n加载MNIST数据...")
    try:
        X_train, y_train, X_test, y_test = load_mnist(
            preprocess=True,
            use_mock_on_fail=True
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("测试终止。")
        exit(1)

    print("初始化EDA探索器...")
    explorer = MNISTExplorer(X_train, y_train, X_test, y_test)

    # 运行四个EDA步骤
    explorer.run_comprehensive_analysis()