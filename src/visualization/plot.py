"""
Visualization Module

This file is responsible for generating all plots, with 7 types in total:

1. Training curves (training_history.png)    — loss and accuracy vs. epoch
2. Confusion matrix heatmap (confusion_matrix_*.png) — 10x10 confusion matrix
3. Model comparison bar chart (model_comparison.png) — Accuracy / F1 comparison across models
4. Misclassified samples (misclassified_cnn.png) — images incorrectly classified by CNN
5. Per-class accuracy (per_class_accuracy.png) — recognition rate for each digit 0-9
6. Grad-CAM heatmap (gradcam.png) — CNN attention region visualization
7. Grad-CAM per digit (gradcam_per_digit.png) — attention distribution for digits 0-9

=== Usage ===

Typically there is no need to call this module directly; run_all.py will
automatically call generate_all_plots() to produce all plots.
If you want to generate a specific plot individually, you can call the
corresponding function directly.

=== Technical Details ===

- Uses matplotlib for plotting with the Agg backend (non-interactive, saves directly to PNG)
- dpi=150: sufficient resolution while keeping file size manageable
- bbox_inches="tight": automatically trims white margins
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: saves plots without opening windows, works on servers too
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


def save_fig(fig, save_dir, filename):
    """
    Save a plot to the specified directory

    Args:
        fig: matplotlib Figure object
        save_dir: directory to save to (created automatically if it does not exist)
        filename: file name (e.g. "training_history.png")
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it does not exist
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")  # dpi=150 for high resolution, tight to trim margins
    plt.close(fig)  # Close the figure to free memory
    print(f"  Plot saved: {path}")


# ============================================================
# 1. Training curves: loss and accuracy vs. epoch
# ============================================================

def plot_training_history(history: dict, save_dir: str = "results"):
    """
    Plot training/validation loss and accuracy curves

    Generates two side-by-side subplots:
        Left:  Train Loss and Val Loss vs. epoch
        Right: Train Acc and Val Acc vs. epoch

    How to read this plot:
        - Loss curves should decrease steadily; if they stall, the model is not learning
        - If Train Loss keeps decreasing but Val Loss starts rising, the model is overfitting
        - Accuracy curves should increase steadily

    Args:
        history: training history dictionary returned by train_cnn(), containing four lists:
                 train_loss, train_acc, val_loss, val_acc
        save_dir: directory to save the plot
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left subplot: Loss curves
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right subplot: Accuracy curves
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_dir, "training_history.png")


# ============================================================
# 2. Confusion matrix heatmap
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, model_name: str = "CNN", save_dir: str = "results"):
    """
    Plot a confusion matrix heatmap

    How to read the confusion matrix:
        - X-axis = predicted labels, Y-axis = true labels
        - Diagonal (top-left to bottom-right) = correct predictions; darker color means more correct
        - Off-diagonal = misclassifications; shows which classes are confused with each other
        - Example: the value at row 7, column 9 = number of samples truly labeled 7 but predicted as 9

    Args:
        cm: 10x10 confusion matrix, cm[i][j] = number of samples with true label i predicted as j
        model_name: model name (used in title and filename)
        save_dir: directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw heatmap using blue colormap: darker color = larger value
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)  # Color bar on the right

    # Configure axes
    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xticklabels=range(10), yticklabels=range(10),
           ylabel="True Label", xlabel="Predicted Label",
           title=f"Confusion Matrix - {model_name}")

    # Write the numeric value in each cell
    thresh = cm.max() / 2.0  # Threshold: use white text above this value, black below (for readability)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    # Filename: replace spaces and brackets, e.g. "KNN (k=5)" -> "knn_(k=5)"
    safe_name = model_name.replace(" ", "_").lower()
    save_fig(fig, save_dir, f"confusion_matrix_{safe_name}.png")


# ============================================================
# 3. Model comparison bar chart
# ============================================================

def plot_model_comparison(all_results: list, save_dir: str = "results"):
    """
    Plot a grouped bar chart comparing Accuracy / F1 across multiple models

    Each model has two bars: blue = Accuracy, orange = F1-Score
    Exact values are annotated above each bar

    Args:
        all_results: list where each element is the return value of evaluate_model()
        save_dir: directory to save the plot
    """
    names = [r["model_name"] for r in all_results]       # Model name list
    accuracies = [r["accuracy"] for r in all_results]    # Accuracy list
    f1_scores = [r["f1_macro"] for r in all_results]     # F1 list

    x = np.arange(len(names))
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Score (macro)", color="coral")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0.9, 1.0)  # Y-axis starts at 0.9 to magnify differences
    ax.grid(axis="y", alpha=0.3)

    # Annotate exact values above each bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_fig(fig, save_dir, "model_comparison.png")


# ============================================================
# 4. Misclassified samples display
# ============================================================

def plot_misclassified(X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "CNN", n_samples: int = 16, save_dir: str = "results"):
    """
    Display samples that the model misclassified

    Each image shows: the original image + title "True: X, Pred: Y"
    Provides an intuitive view of what types of errors the model makes
    (e.g. 7 being mistaken for 9)

    Args:
        X_test: test set images, shape=(N, 28, 28)
        y_true: true labels
        y_pred: predicted labels
        model_name: model name
        n_samples: maximum number of samples to display (default 16)
        save_dir: directory to save the plot
    """
    # Find indices of all misclassified samples
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        print("  No misclassified samples found!")
        return

    n_show = min(n_samples, len(mis_idx))
    selected = mis_idx[:n_show]  # Take the first n_show samples

    # Layout: 4 columns, rows calculated automatically
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n_show > 1 else [axes]

    for i, idx in enumerate(selected):
        axes[i].imshow(X_test[idx], cmap="gray")  # Grayscale image
        axes[i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}", fontsize=10)
        axes[i].axis("off")

    # Hide unused subplots (if n_show is not a multiple of 4)
    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Misclassified Samples - {model_name}", fontsize=14)
    plt.tight_layout()
    save_fig(fig, save_dir, f"misclassified_{model_name.replace(' ', '_').lower()}.png")

    print(f"  Total {len(mis_idx)} misclassified samples, showing first {n_show}")


# ============================================================
# 5. Per-class accuracy
# ============================================================

def plot_per_class_accuracy(all_results: list, save_dir: str = "results"):
    """
    Plot per-digit (0-9) accuracy as a grouped bar chart

    Each digit has multiple bars, one per model.
    Shows which model performs best/worst on each digit.

    Args:
        all_results: list where each element is the return value of evaluate_model()
        save_dir: directory to save the plot
    """
    digits = np.arange(10)
    n_models = len(all_results)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, r in enumerate(all_results):
        # recall_per_class[j] = proportion of digit j correctly recognized (i.e. per-class accuracy)
        recall = r["recall_per_class"]
        offset = (i - n_models / 2 + 0.5) * width  # Arrange multiple bars side by side
        ax.bar(digits + offset, recall, width, label=r["model_name"])

    ax.set_xlabel("Digit")
    ax.set_ylabel("Recall (Per-class Accuracy)")
    ax.set_title("Per-class Accuracy by Model")
    ax.set_xticks(digits)
    ax.set_ylim(0.9, 1.0)  # Y-axis starts at 0.9 to magnify differences
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_dir, "per_class_accuracy.png")


# ============================================================
# 6. Grad-CAM heatmap: visualizing "where the CNN is looking"
# ============================================================

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping for CNN attention visualization

    In short: this heatmap shows which regions of the image are most important
    for the CNN's classification decision.

    Algorithm (4 steps):
        1. Forward pass to the target conv layer -> record feature maps (activations)
           Each channel of the feature map corresponds to a "pattern detector" output
        2. Backpropagate from the target class score -> compute gradients at that layer
           Gradients reflect how important each channel is for the classification decision
        3. Global average pool the gradients -> obtain a scalar weight per channel
           Larger weight means that channel is more important for classification
        4. Weighted sum + ReLU -> produce a heatmap
           ReLU filters out negative contributions, keeping only regions that
           positively contribute to the classification

    How to read the heatmap:
        - Red/Yellow = regions the CNN considers important (basis for its decision)
        - Blue = regions the CNN ignores
        - If the CNN's focus matches your intuition -> the model learned correctly
        - If the CNN focuses on irrelevant regions -> possible overfitting or biased learning

    Args:
        model: trained CNN model
        target_layer: target layer to hook (typically the last convolutional layer)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None   # Feature maps captured during forward pass
        self.gradients = None     # Gradients captured during backward pass
        self._hooks = []
        self._register_hooks()    # Register hooks to automatically capture data

    def _register_hooks(self):
        """
        Register PyTorch hook functions

        Hooks automatically "intercept" a layer's input/output during forward/backward
        passes, allowing access to intermediate layer data without modifying model code.

        forward_hook:  triggered during forward pass, records the layer's output (feature maps)
        backward_hook: triggered during backward pass, records the layer's gradients
        """
        def forward_hook(module, input, output):
            # During forward pass, save this layer's output (feature maps)
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # During backward pass, save this layer's gradients
            self.gradients = grad_output[0].detach()

        # register_forward_hook: called during forward pass
        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        # register_full_backward_hook: called during backward pass
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap for a single image

        Args:
            input_tensor: single image, shape=(1, 1, 28, 28)
            target_class: which class's attention to visualize (None = use predicted class)

        Returns:
            cam: heatmap, shape=(H, W), values 0-255 (uint8), ready for plotting
            target_class: the actual target class used
        """
        self.model.eval()

        # Step 1: forward pass (hook automatically captures feature maps)
        output = self.model(input_tensor)

        # Determine the target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()  # Use predicted class

        # Step 2: backward pass (hook automatically captures gradients)
        self.model.zero_grad()
        # Construct a one-hot vector to backpropagate only for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Step 3: compute weights (global average over spatial dimensions of gradients)
        # gradients shape: (1, 64, 7, 7) -> weights shape: (1, 64, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Step 4: weighted sum + ReLU
        # (1, 64, 1, 1) * (1, 64, 7, 7) -> (1, 1, 7, 7)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Keep only positive contributions

        # Normalize to 0-255
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(cam * 255)

        return cam, target_class

    def remove_hooks(self):
        """Remove hooks and release resources (must be called when done)"""
        for hook in self._hooks:
            hook.remove()


def plot_gradcam(model, X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                 device, n_samples: int = 16, save_dir: str = "results"):
    """
    Generate Grad-CAM heatmaps: correctly classified samples + 7/9 confusion samples

    Displays two parts:
        - First half: correctly classified samples (green title), showing regions
          the model "got right"
        - Second half: 7->9 or 9->7 confused samples (red title), showing where
          the model "got it wrong"

    Each sample is shown in two columns: left = original image, right = image with
    heatmap overlay

    Args:
        model: trained CNN model
        X_test: test set images, shape=(N, 28, 28)
        y_true: true labels
        y_pred: predicted labels
        device: compute device
        n_samples: total number of samples to display
        save_dir: directory to save the plot
    """
    # Hook into the second-to-last layer of features (BatchNorm after the last Conv2d)
    # This is standard Grad-CAM practice: hook near the last convolutional layer
    target_layer = model.features[-2]
    grad_cam = GradCAM(model, target_layer)

    # ---- Part 1: correctly classified samples ----
    correct_idx = np.where(y_true == y_pred)[0]
    n_correct = min(n_samples // 2, len(correct_idx))
    selected_correct = np.random.choice(correct_idx, n_correct, replace=False)

    # ---- Part 2: 7/9 confusion samples ----
    # Find all samples where 7 was predicted as 9, or 9 was predicted as 7
    confused_79 = np.where(((y_true == 7) & (y_pred == 9)) |
                            ((y_true == 9) & (y_pred == 7)))[0]
    n_confused = min(n_samples // 2, len(confused_79))
    if n_confused > 0:
        selected_confused = confused_79[:n_confused]
    else:
        selected_confused = np.array([], dtype=int)

    # Merge both sets of samples
    all_indices = np.concatenate([selected_correct, selected_confused])
    labels = (["correct"] * n_correct + ["7→9/9→7"] * n_confused)

    n_total = len(all_indices)
    if n_total == 0:
        grad_cam.remove_hooks()
        return

    # Layout: each sample occupies 2 columns (original + heatmap)
    cols = 4
    rows = (n_total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    jet_cmap = plt.cm.jet  # jet colormap: blue -> cyan -> green -> yellow -> red

    for i, (idx, label_type) in enumerate(zip(all_indices, labels)):
        img = X_test[idx]
        # Prepare input: numpy -> tensor, add batch and channel dimensions
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)

        # Generate Grad-CAM heatmap
        cam, pred_class = grad_cam.generate(input_tensor, target_class=int(y_pred[idx]))

        row, col_pair = i // cols, (i % cols) * 2

        # Left: original image
        axes[row, col_pair].imshow(img, cmap="gray")
        color = "green" if label_type == "correct" else "red"
        axes[row, col_pair].set_title(
            f"True:{y_true[idx]} Pred:{y_pred[idx]}", fontsize=9, color=color)
        axes[row, col_pair].axis("off")

        # Right: overlay heatmap
        axes[row, col_pair + 1].imshow(img, cmap="gray", alpha=0.5)  # Base grayscale image
        heatmap = jet_cmap(cam / 255.0)[:, :, :3]  # Convert to RGB colors
        axes[row, col_pair + 1].imshow(heatmap, alpha=0.5)  # Overlay heatmap
        axes[row, col_pair + 1].set_title("Grad-CAM", fontsize=9)
        axes[row, col_pair + 1].axis("off")

    # Hide empty cells
    for j in range(n_total, rows * cols):
        row, col_pair = j // cols, (j % cols) * 2
        axes[row, col_pair].axis("off")
        axes[row, col_pair + 1].axis("off")

    fig.suptitle("Grad-CAM: What the CNN is looking at", fontsize=14)
    plt.tight_layout()
    save_fig(fig, save_dir, "gradcam.png")

    grad_cam.remove_hooks()  # Clean up hooks
    print(f"  Grad-CAM: {n_correct} correct + {n_confused} confused 7/9 samples")


def plot_gradcam_per_digit(model, X_test: np.ndarray, y_test: np.ndarray,
                           device, save_dir: str = "results"):
    """
    Select one sample per digit 0-9 and display Grad-CAM heatmaps

    Top row:    original images of digits 0-9
    Bottom row: corresponding Grad-CAM heatmaps

    Purpose: intuitively show which image regions the CNN focuses on when
    recognizing different digits. For example, the heatmap for 7 should
    concentrate on the intersection of the vertical and horizontal strokes,
    while the heatmap for 9 should focus on the top loop.

    Args:
        model: trained CNN model
        X_test: test set images
        y_test: test set labels
        device: compute device
        save_dir: directory to save the plot
    """
    target_layer = model.features[-2]
    grad_cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    for digit in range(10):
        # Find the first sample for this digit
        idx_arr = np.where(y_test == digit)[0]
        idx = idx_arr[0]
        img = X_test[idx]

        # Generate heatmap
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        cam, _ = grad_cam.generate(input_tensor, target_class=digit)

        # Top row: original image
        axes[0, digit].imshow(img, cmap="gray")
        axes[0, digit].set_title(f"Digit {digit}", fontsize=10)
        axes[0, digit].axis("off")

        # Bottom row: heatmap
        axes[1, digit].imshow(img, cmap="gray", alpha=0.4)
        axes[1, digit].imshow(plt.cm.jet(cam / 255.0)[:, :, :3], alpha=0.6)
        axes[1, digit].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=11)
    fig.suptitle("Grad-CAM per Digit: CNN Attention Heatmap", fontsize=14)
    plt.tight_layout()
    save_fig(fig, save_dir, "gradcam_per_digit.png")

    grad_cam.remove_hooks()


# ============================================================
# Generate all plots in one call
# ============================================================

def generate_all_plots(pipeline_results: dict, X_test: np.ndarray,
                       y_test: np.ndarray, save_dir: str = "results"):
    """
    Generate all visualization plots

    Typically called by run_all.py; no need to call manually.

    Args:
        pipeline_results: return value of run_full_pipeline(), containing models,
                          training history, evaluation results, etc.
        X_test: test set images (N, 28, 28), used for displaying misclassified samples and Grad-CAM
        y_test: test set labels
        save_dir: directory to save plots
    """
    print("\n" + "=" * 50)
    print("  Generating visualization plots")
    print("=" * 50)

    history = pipeline_results["history"]
    all_results = pipeline_results["all_results"]

    # Get test set results (last split in all_results)
    test_results = all_results[-1]["results"]

    # 1. Training curves
    plot_training_history(history, save_dir)

    # 2. Confusion matrices (all models on test set)
    for r in test_results:
        cm = r["confusion_matrix"]
        plot_confusion_matrix(cm, model_name=r["model_name"], save_dir=save_dir)

    # 3. Model comparison (test set)
    plot_model_comparison(test_results, save_dir)

    # 4. Misclassified samples (CNN, test set)
    y_pred_cnn_test = pipeline_results["y_pred_cnn_test"]
    plot_misclassified(X_test, y_test, y_pred_cnn_test, model_name="CNN", save_dir=save_dir)

    # 5. Per-class accuracy (test set)
    plot_per_class_accuracy(test_results, save_dir)

    # 6. Grad-CAM heatmaps
    cnn_model = pipeline_results["cnn_model"]
    device = pipeline_results["device"]
    y_pred_cnn = pipeline_results["y_pred_cnn_test"]
    plot_gradcam(cnn_model, X_test, y_test, y_pred_cnn, device, save_dir=save_dir)
    plot_gradcam_per_digit(cnn_model, X_test, y_test, device, save_dir=save_dir)

    print(f"\n  All plots saved to {save_dir}/")
