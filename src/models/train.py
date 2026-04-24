"""
Model Training Module

This file is the core of the entire training pipeline, containing three parts:
1. CNN training pipeline (PyTorch)
2. Baseline model training (sklearn: KNN, Logistic Regression, Random Forest)
3. Data splitting and loading

=== Data Splitting Strategy ===

Train:Val:Test = 6:2:2 (42000:14000:14000)

Why this split:
- Training set (42000): Trains model parameters, the largest portion; more data leads to better learning
- Validation set (14000): Monitors overfitting during training, determines when to stop (early stopping)
- Test set (14000): Final evaluation, shared by all models to ensure fair comparison

Why not use the original 60000+10000:
- The original MNIST has no validation set, only training (60000) and test (10000) sets
- We need a validation set for early stopping, so we merge all 70000 images and re-split

=== Training Strategy ===

CNN training strategy:
- Optimizer: Adam (adaptive learning rate, fast convergence, suitable for MNIST)
- Loss function: Focal Loss + Label Smoothing (focuses on hard samples, prevents overconfidence)
- Learning rate schedule: StepLR (decay to 0.1x every 10 epochs)
- Early stopping: Stop if validation accuracy does not improve for 5 consecutive epochs
- Data augmentation: Random rotation/translation/scaling during training to prevent overfitting

Baseline model training strategy:
- Use sklearn default parameters (KNN k=5, LR 1000 iterations, RF 100 trees)
- Use the HOG+LBP+Shape 405-dimensional features extracted by team members
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.cnn import MNISTCNN, FocalLoss, count_parameters
from src.models.evaluate import evaluate_model, print_evaluation, compare_models


# ============================================================
# Utility functions: set random seed, select device, split dataset
# ============================================================

def set_seed(seed: int = 42):
    """
    Fix all random seeds to ensure reproducible results

    Why we need to fix seeds:
        There are many random operations during training: data shuffling, Dropout,
        weight initialization, etc. Without fixed seeds, results vary each run,
        making it impossible to compare experiments.

    Sources of randomness covered:
        - np.random.seed: NumPy random operations (data shuffling)
        - torch.manual_seed: PyTorch CPU random operations
        - torch.cuda.manual_seed_all: PyTorch GPU random operations
        - cudnn.deterministic: cuDNN convolution deterministic mode
        - cudnn.benchmark: Disable auto-tuning (auto-tuning introduces randomness)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically select the computing device: use GPU if available, otherwise CPU

    GPU (CUDA) training is 5-10x faster than CPU, but requires an NVIDIA GPU.
    This function auto-detects the hardware, no manual code changes needed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_dataset(X_all, y_all, train_ratio=6/10, val_ratio=2/10, seed=42):
    """
    Split the merged dataset into train, validation, and test sets by ratio

    Default ratio: Train:Val:Test = 6:2:2
    70000 MNIST images -> 42000 + 14000 + 14000

    Args:
        X_all: All images, shape=(70000, 28, 28)
        y_all: All labels, shape=(70000,)
        train_ratio: Training set ratio, default 6/10
        val_ratio:   Validation set ratio, default 2/10
        seed: Random seed, fixing it makes the split reproducible

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (six arrays)

    Splitting logic:
        1. Randomly shuffle the indices of all 70000 samples
        2. First 20% -> test set, next 20% -> validation set, remaining 60% -> training set
        3. Use indices to extract corresponding data from X_all and y_all
    """
    set_seed(seed)
    n = len(X_all)
    indices = np.random.permutation(n)  # Randomly shuffle indices 0~69999

    # Calculate the size of each set
    n_test = int(n * (1 - train_ratio - val_ratio))   # 70000 x 0.2 = 14000
    n_val = int(n * val_ratio)                          # 70000 x 0.2 = 14000

    # Split by indices
    test_idx = indices[:n_test]                         # First 14000 -> test set
    val_idx = indices[n_test:n_test + n_val]            # Middle 14000 -> validation set
    train_idx = indices[n_test + n_val:]                # Remaining 42000 -> training set

    return (
        X_all[train_idx], y_all[train_idx],   # Training set
        X_all[val_idx], y_all[val_idx],       # Validation set
        X_all[test_idx], y_all[test_idx],     # Test set
    )


# ============================================================
# Data Augmentation: Apply random transforms to images during training,
# effectively enlarging the training set
# ============================================================

class _AugmentedDataset(Dataset):
    """
    PyTorch Dataset with data augmentation

    Why data augmentation is needed:
        With only 42000 training images, the model might "memorize" them instead of
        learning real features. Data augmentation applies random transforms (rotation,
        translation, scaling) each time an image is fetched, so the model sees a
        different version each time, effectively multiplying the training set size.

    Key points:
        - Only the training set is augmented; validation and test sets are not (ensures fair evaluation)
        - Augmentation is "online", i.e., transforms are applied on-the-fly, not pre-generated
        - Each epoch sees different augmented versions of the images
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        """
        Args:
            images: Image data, shape=(N, 1, 28, 28)
            labels: Label data, shape=(N,)
            augment: Whether to apply data augmentation (True for training set, False for val/test)
        """
        self.images = images
        self.labels = labels
        self.augment = augment

        # Define the augmentation pipeline
        self.aug_transform = transforms.Compose([
            # Random rotation +/-10 degrees: simulate different writing tilt angles
            transforms.RandomRotation(10),
            # Random affine transform: simultaneous translation and scaling
            # translate=(0.1, 0.1): up/down/left/right translation up to 10%
            # scale=(0.9, 1.1): scale range 90%~110%, simulating varying character sizes
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # Random crop: pad 2 pixels on all sides, then randomly crop back to 28x28,
            # simulating slight positional shifts
            transforms.RandomCrop(28, padding=2, padding_mode="edge"),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get the idx-th sample

        If augment=True, applies random transforms to the image;
        if augment=False, returns the original image directly.
        """
        img = self.images[idx]
        label = self.labels[idx]
        if self.augment:
            img = self.aug_transform(img)  # Apply random transforms
        return img, label


def prepare_cnn_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    """
    Convert numpy arrays to PyTorch DataLoaders

    Why we need DataLoaders:
        During training, we cannot load all 42000 images into GPU memory at once,
        so we need to process them in batches. DataLoader automatically splits data
        into mini-batches and can shuffle the order.

    Args:
        X_train, y_train: Training set images and labels (numpy)
        X_val, y_val:     Validation set images and labels (numpy)
        X_test, y_test:   Test set images and labels (numpy)
        batch_size:       Number of images per batch (default 128)

    Returns:
        train_loader, val_loader, test_loader (three DataLoaders)

    Key differences:
        - Training DataLoader: with data augmentation + shuffling
        - Validation/Test DataLoader: no augmentation + no shuffling
    """
    # numpy -> PyTorch tensor, add a channel dimension: (N, 28, 28) -> (N, 1, 28, 28)
    def to_tensor(X):
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    # Training set: with data augmentation + shuffling (shuffle=True)
    train_ds = _AugmentedDataset(
        to_tensor(X_train), torch.tensor(y_train, dtype=torch.long),
        augment=True  # Apply augmentation during training
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation set: no augmentation, no shuffling
    val_loader = DataLoader(
        TensorDataset(to_tensor(X_val), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
    )

    # Test set: no augmentation, no shuffling
    test_loader = DataLoader(
        TensorDataset(to_tensor(X_test), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# CNN Training
# ============================================================

def train_cnn(train_loader, val_loader, device, epochs=30, lr=0.001, patience=5):
    """
    Train the CNN model

    This is the core training function. The process is as follows:
        Each epoch:
        1. Iterate over all training batches and update model parameters
        2. Evaluate on the validation set, computing loss and accuracy
        3. If validation accuracy improved, save the current model
        4. If no improvement for `patience` consecutive epochs, stop early

    Args:
        train_loader: Training set DataLoader
        val_loader:   Validation set DataLoader
        device:       Computing device (cpu or cuda)
        epochs:       Maximum number of training epochs (default 30)
        lr:           Learning rate (default 0.001)
        patience:     Early stopping patience (default 5, stop if no improvement for 5 epochs)

    Returns:
        model: Trained model (loaded with the best weights)
        history: Training history, containing loss and accuracy for each epoch
    """

    # ---- Initialize model, loss function, optimizer, learning rate scheduler ----
    model = MNISTCNN().to(device)  # Create model and move to GPU/CPU
    print(f"\n  CNN parameter count: {count_parameters(model):,}")
    print(f"  Device: {device}")

    # Loss function: Focal Loss + Label Smoothing
    # - Focal Loss: Automatically focuses on hard-to-classify samples (e.g., 7/9 confusion)
    # - Label Smoothing: Prevents the model from becoming overconfident
    # See the FocalLoss class comments in cnn.py for detailed principles
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

    # Optimizer: Adam
    # - Adam is an adaptive learning rate optimizer, each parameter has its own learning rate
    # - lr=0.001 is the recommended default for Adam
    # - weight_decay=1e-4 is L2 regularization, prevents parameters from growing too large
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler: StepLR
    # Every 10 epochs, multiply the learning rate by 0.1 (reduce to 1/10)
    # Large learning rate early for fast learning, small learning rate later for fine-tuning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ---- Training records ----
    # history dict records train/val loss and accuracy for each epoch
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0          # Best validation accuracy so far
    epochs_no_improve = 0       # Number of consecutive epochs with no improvement

    # ---- Main training loop ----
    for epoch in range(1, epochs + 1):

        # ===== Training phase =====
        model.train()  # Switch to training mode (enable Dropout and BatchNorm training behavior)
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            # Move data to GPU/CPU
            images, labels = images.to(device), labels.to(device)

            # 1. Forward pass: images -> model -> predictions
            optimizer.zero_grad()      # Clear gradients from the previous step (must do!)
            outputs = model(images)     # Forward pass, get class scores

            # 2. Compute loss: predictions vs ground truth labels
            loss = criterion(outputs, labels)

            # 3. Backward pass: compute gradients
            loss.backward()

            # 4. Update parameters: adjust model weights using gradients
            optimizer.step()

            # Accumulate loss and correct predictions for this batch
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)  # Take the class with the highest score as prediction
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # One epoch of training complete, compute average loss and accuracy
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ===== Validation phase =====
        model.eval()  # Switch to evaluation mode (disable Dropout, BatchNorm uses running mean)
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():  # Disable gradient computation to save memory and computation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        # Update learning rate
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print this epoch's results
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ===== Early stopping check =====
        if val_acc > best_val_acc:
            # Validation accuracy improved -> save model, reset counter
            best_val_acc = val_acc
            best_state = model.state_dict().copy()  # Save current best weights
            epochs_no_improve = 0
        else:
            # No improvement -> increment counter
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping: validation accuracy has not improved for {patience} epochs")
                break  # Stop training early

    # Training complete, load the best weights (not necessarily the last epoch, but the one with highest val accuracy)
    model.load_state_dict(best_state)
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    return model, history


def predict_cnn(model, data_loader, device):
    """
    Make predictions using the trained CNN

    Args:
        model: Trained CNN model
        data_loader: DataLoader for the data to predict
        device: Computing device

    Returns:
        numpy array, shape=(N,), each element is the predicted digit (0-9)
    """
    model.eval()  # Evaluation mode: disable Dropout
    all_preds = []
    with torch.no_grad():  # Disable gradient computation, pure inference
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)  # Take the class with the highest score
            all_preds.append(predicted.cpu().numpy())
    return np.concatenate(all_preds)


# ============================================================
# Baseline Models: Train traditional ML models using sklearn for comparison
# ============================================================

def train_baselines(X_train_features, y_train, X_val_features, y_val, X_test_features, y_test):
    """
    Train sklearn baseline models and predict on train, validation, and test sets

    Why baseline models are needed:
        A CNN accuracy of 99% is meaningless without comparison; we need baselines to
        verify that CNN is genuinely better than simpler methods. KNN, LR, and RF are
        classic methods covering three paradigms: nearest-neighbor, linear, and ensemble.

    Why baselines use hand-crafted features instead of raw pixels:
        KNN/LR/RF cannot directly process 2D images; images must be converted to feature
        vectors first. Using the HOG+LBP+Shape 405-dimensional features from team members
        is reasonable since these features have been carefully designed.

    Args:
        X_train_features: Training set features, shape=(42000, 405)
        y_train: Training set labels, shape=(42000,)
        X_val_features: Validation set features, shape=(14000, 405)
        y_val: Validation set labels
        X_test_features: Test set features, shape=(14000, 405)
        y_test: Test set labels

    Returns:
        dict: {model_name: {"model": classifier object,
                            "train_pred": training set predictions,
                            "val_pred": validation set predictions,
                            "test_pred": test set predictions}}
    """
    # Define three baseline models
    baselines = {
        # KNN (K-Nearest Neighbors): Find the 5 nearest samples in training set, vote for classification
        # n_neighbors=5: Look at the 5 nearest neighbors
        # n_jobs=-1: Use all CPU cores for parallel computation
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),

        # Logistic Regression: Linear classifier
        # max_iter=1000: Maximum 1000 iterations (default 100 may not converge)
        # solver="lbfgs": Optimization algorithm, suitable for multi-class
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1),

        # Random Forest: Multiple decision trees voting
        # n_estimators=100: 100 trees
        # random_state=42: Fixed random seed
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, clf in baselines.items():
        print(f"\n  Training {name}...")
        # Train: fit the model using training set features and labels
        clf.fit(X_train_features, y_train)

        # Predict on three sets separately (for subsequent evaluation)
        results[name] = {
            "model": clf,
            "train_pred": clf.predict(X_train_features),  # Training set predictions
            "val_pred": clf.predict(X_val_features),      # Validation set predictions
            "test_pred": clf.predict(X_test_features),    # Test set predictions
        }
        print(f"  {name} done")

    return results


# ============================================================
# Main Pipeline: One-click CNN + baseline model training and evaluation
# ============================================================

def run_full_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                      X_train_features, X_val_features, X_test_features,
                      batch_size=128, epochs=30, lr=0.001, seed=42):
    """
    Full training + evaluation pipeline

    Does three things:
        1. Train CNN (using PyTorch)
        2. Train 3 baseline models (using sklearn)
        3. Evaluate all models on train, validation, and test sets

    Args:
        X_train, y_train:     Training set images (42000, 28, 28) and labels
        X_val, y_val:         Validation set images (14000, 28, 28) and labels
        X_test, y_test:       Test set images (14000, 28, 28) and labels
        X_train_features:     Training set features (42000, 405), used by baseline models
        X_val_features:       Validation set features (14000, 405)
        X_test_features:      Test set features (14000, 405)
        batch_size:           Number of images per CNN batch
        epochs:               Number of CNN training epochs
        lr:                   CNN learning rate
        seed:                 Random seed

    Returns:
        dict: Contains all results for visualization and summary saving
    """
    set_seed(seed)
    device = get_device()

    print("=" * 60)
    print("  Starting model training and evaluation")
    print(f"  Training set: {len(y_train)}  Validation set: {len(y_val)}  Test set: {len(y_test)}")
    print(f"  Ratio train:val:test = {len(y_train)}:{len(y_val)}:{len(y_test)}")
    print("=" * 60)

    # ---- Step 1: Train CNN ----
    print("\n[1/3] Training CNN...")
    train_loader, val_loader, test_loader = prepare_cnn_data(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    cnn_model, history = train_cnn(train_loader, val_loader, device, epochs=epochs, lr=lr)

    # CNN predictions on three sets
    # Note: Training set must use a DataLoader without data augmentation (otherwise predictions are inaccurate)
    def to_loader(X, y):
        """Quickly convert numpy data to a DataLoader (no augmentation)"""
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size)

    y_pred_cnn_train = predict_cnn(cnn_model, to_loader(X_train, y_train), device)
    y_pred_cnn_val = predict_cnn(cnn_model, val_loader, device)
    y_pred_cnn_test = predict_cnn(cnn_model, test_loader, device)

    # ---- Step 2: Train baseline models ----
    print("\n[2/3] Training baseline models (using HOG+LBP+Shape features)...")
    baseline_results = train_baselines(
        X_train_features, y_train,
        X_val_features, y_val,
        X_test_features, y_test,
    )

    # ---- Step 3: Evaluate all models on three sets ----
    print("\n[3/3] Evaluating all models...")
    all_results = []

    # Iterate over three sets: training, validation, test
    for split_name, y_true, y_pred_cnn, feat_key in [
        ("Training Set", y_train, y_pred_cnn_train, "train_pred"),
        ("Validation Set", y_val, y_pred_cnn_val, "val_pred"),
        ("Test Set", y_test, y_pred_cnn_test, "test_pred"),
    ]:
        print(f"\n{'='*60}")
        print(f"  [{split_name}] ({len(y_true)} samples)")
        print(f"{'='*60}")

        split_results = []

        # Evaluate CNN
        cnn_eval = evaluate_model(y_true, y_pred_cnn, model_name="CNN")
        split_results.append(cnn_eval)

        # Evaluate three baseline models
        for name, bl in baseline_results.items():
            y_pred_bl = bl[feat_key]  # Get predictions for the corresponding set
            eval_res = evaluate_model(y_true, y_pred_bl, model_name=name)
            split_results.append(eval_res)

        # Print evaluation results for all models
        for r in split_results:
            print_evaluation(r)
        print("\n" + compare_models(split_results))

        all_results.append({"split": split_name, "results": split_results})

    # ---- Return all results ----
    return {
        "cnn_model": cnn_model,            # Trained CNN model (needed for Grad-CAM)
        "history": history,                # Training history (for plotting training curves)
        "all_results": all_results,         # Evaluation results on three sets
        "y_pred_cnn_test": y_pred_cnn_test, # CNN predictions on test set
        "y_pred_cnn_val": y_pred_cnn_val,   # CNN predictions on validation set
        "y_pred_cnn_train": y_pred_cnn_train, # CNN predictions on training set
        "baseline_results": baseline_results,  # Baseline model results
        "device": device,                  # Computing device (needed for Grad-CAM)
        "splits": {                        # Raw data (for Grad-CAM visualization)
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        },
    }
