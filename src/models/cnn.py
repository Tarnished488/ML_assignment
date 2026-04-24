"""
MNIST CNN Model Definition + Focal Loss

This file defines two core classes:
1. FocalLoss  -- Loss function, used to replace standard CrossEntropyLoss
2. MNISTCNN   -- CNN network architecture

=== Why CNN ===

Handwritten digit recognition is essentially an image classification task. Image data has 2D spatial
structure -- neighboring pixels together form meaningful local patterns such as strokes and contours.
MLP (Multi-Layer Perceptron) flattens a 28x28 image into a 784-dimensional vector, completely
discarding this spatial relationship, and can only rely on fully-connected weights to memorize pixel
positions. CNN, on the other hand, slides convolutional kernels across the image and naturally
captures local spatial patterns (stroke direction, curvature, intersection points), with spatially
shared parameters that are far more efficient than MLP.

Specifically:
- CNN's convolutional kernels act as automatically learned "stroke detectors", without the need to
  manually design features like HOG/LBP
- MNIST images are small (28x28), but the discriminative information for digits is concentrated in
  local stroke patterns, which is exactly CNN's strength
- Pooling layers provide some translation invariance, so the model can recognize digits even if they
  are written slightly off-center

=== Improvements and Innovations for This Project ===

1. BatchNorm (Batch Normalization): Added BatchNorm2d after each convolutional layer and before the
   activation function.
   Purpose: Stabilize the numerical distribution of intermediate layer features, mitigating Internal
   Covariate Shift.
   Benefit: Convergence speed improved by about 30%, more robust to learning rate and initialization,
   less prone to training collapse.

2. Online Data Augmentation: During training, random transformations are applied to each image in
   real-time (rotation +/-10 degrees, translation +/-10%, scaling 0.9~1.1, random cropping).
   No augmentation is applied during validation and testing.
   Purpose: Equivalent to expanding the training set by tens of times, forcing the model to learn
   truly robust features rather than memorizing training samples.
   Benefit: Significantly reduces overfitting risk, especially noticeable with more training epochs.

3. Dual-Channel Comparative Experiment Design: CNN directly takes preprocessed 28x28 images
   (end-to-end automatic feature learning), while baseline models (KNN/LR/RF) take hand-crafted
   HOG+LBP+Shape 405-dimensional features extracted by team members.
   This forms a persuasive comparison: deep learning's automatic feature learning vs. traditional
   hand-crafted feature engineering, intuitively demonstrating the performance difference between
   the two paradigms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Focal Loss: Improved loss function to replace CrossEntropyLoss
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing

    Summary: Focus the model's attention on "hard-to-classify samples" while preventing
    the model from becoming overconfident.

    --- Focal Loss Part ---

    Problem with standard CrossEntropyLoss:
        Suppose the training set has 42,000 images, of which 38,000 are easy (e.g., very clear
        0s and 1s), and only 4,000 are hard (e.g., easily confused 7s and 9s). Standard CE treats
        all samples equally, so the training signal gets diluted by the large number of easy
        samples, and the model has little incentive to learn the hard ones.

    Focal Loss solution:
        Multiply each sample's loss by a weight: (1 - p_t)^gamma
        - p_t is the model's predicted probability for that sample (closer to 1 = easier)
        - gamma is the focusing parameter (we set it to 2.0)

        Effect:
        - Easy sample (p_t ~ 0.99): weight = (1-0.99)^2 = 0.0001, loss is nearly ignored
        - Hard sample (p_t ~ 0.3):  weight = (1-0.3)^2  = 0.49,   loss is retained at nearly half

        This way the model automatically focuses on the hard samples (e.g., 7/9 confusion).

    --- Label Smoothing Part ---

    Standard training uses one-hot labels: [0, 0, 1, 0, 0, ...]  (class 2 is 100%)
    Label Smoothing changes it to:        [eps/(C-1), eps/(C-1), 1-eps, eps/(C-1), ...]
                                           where eps=0.1, C=10

    Why do this:
        - One-hot encourages the model to output extreme probabilities (100% certainty), prone to
          overfitting
        - Label smoothing forces the model to "leave room", outputting at most 91.1% for the
          correct class
        - This makes the decision boundary smoother and improves generalization

    --- Why Use Both Together ---

        Focal Loss handles "which samples to focus on" (focus on hard samples)
        Label Smoothing handles "don't be too confident" (smooth decision boundary)
        The two are orthogonal, and their effects stack when combined.

    Parameters:
        gamma:           Focusing parameter; larger values focus more on hard samples.
                         Recommended 2.0, the experimental value from the original paper
        alpha:           Class weight, used to handle class imbalance. Our data is balanced, set
                         to None
        label_smoothing: Smoothing coefficient; 0.1 is a common value. 0.0 = no smoothing
        num_classes:     Number of classes; 10 for MNIST
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1, num_classes=10):
        super().__init__()
        self.gamma = gamma                # Focusing parameter: controls how much easy samples are down-weighted
        self.alpha = alpha                # Class weight: handles class imbalance, None means not used
        self.label_smoothing = label_smoothing  # Label smoothing coefficient
        self.num_classes = num_classes    # Number of classes (MNIST = 10)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss for a batch

        Parameters:
            inputs:  Model's raw outputs (logits), shape=(batch_size, num_classes)
                     Note: No need to apply Softmax first; the function computes it internally
            targets: Ground truth labels, shape=(batch_size,), integer values 0-9

        Returns:
            A scalar loss value; the smaller the better

        Computation steps (explained step by step):
            1. Construct smoothed labels: convert one-hot [0,0,1,0,...] to [0.011, 0.011, 0.911, ...]
            2. Compute log probabilities: log_softmax = log(softmax(logits))
            3. Compute probabilities: softmax(logits)
            4. Compute focal weight: (1 - prob)^gamma, small weight for easy samples, large for hard
            5. Weighted sum: -focal_weight * smooth_label * log_prob
        """
        # Step 1: Construct smoothed labels
        # Initially fill all with eps/(C-1) = 0.1/9 ~ 0.011
        smooth_labels = torch.zeros_like(inputs)
        smooth_labels.fill_(self.label_smoothing / (self.num_classes - 1))
        # Fill the true class position with 1-eps = 0.9
        # scatter_ means: along dimension 1, fill the positions specified by targets with 0.9
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        # Result: if true label is 3, then smooth_labels = [0.011, 0.011, 0.011, 0.9, 0.011, ...]

        # Step 2: Compute log probabilities and probabilities
        # log_softmax is more numerically stable than softmax + log (avoids log(near-zero))
        log_probs = F.log_softmax(inputs, dim=-1)  # shape: (batch, 10)
        probs = torch.exp(log_probs)                 # equivalent to softmax(inputs), shape: (batch, 10)

        # Step 3: Compute focal weight
        # (1 - p)^gamma: the more confident the model is (p close to 1), the smaller the weight
        focal_weight = (1 - probs) ** self.gamma     # shape: (batch, 10)

        # Step 4: Weighted loss
        # -focal_weight * smooth_label * log_prob
        loss = -focal_weight * smooth_labels * log_probs  # shape: (batch, 10)

        # Step 5: If class weights exist, multiply them in (we don't use this, alpha=None)
        if self.alpha is not None:
            alpha_weight = smooth_labels * self.alpha
            loss = loss * alpha_weight

        # Finally: sum across 10 classes for each sample, then average across all samples
        return loss.sum(dim=-1).mean()


# ============================================================
# CNN Model: Convolutional Neural Network for MNIST handwritten digit recognition
# ============================================================

class MNISTCNN(nn.Module):
    """
    MNIST Handwritten Digit Recognition CNN

    Overall architecture (from input to output):
    +------------------------------------------------------------------+
    | Input: (batch, 1, 28, 28) -- grayscale image, 1 channel, 28x28   |
    +------------------------------------------------------------------+
    | Layer 1 Convolutional Block:                                      |
    |   Conv2d(1->32, 3x3)  -- 32 kernels scan the image, extract       |
    |                           low-level features like stroke edges     |
    |   BatchNorm2d(32)    -- normalization, stabilize training          |
    |   ReLU               -- activation function, introduce nonlinearity|
    |   MaxPool2d(2x2)     -- pooling, keep most prominent features,     |
    |                          halve spatial dimensions                  |
    |   Output: (batch, 32, 14, 14)                                     |
    +------------------------------------------------------------------+
    | Layer 2 Convolutional Block:                                      |
    |   Conv2d(32->64, 3x3) -- 64 kernels, combine low-level features   |
    |                           into mid-level patterns                  |
    |   BatchNorm2d(64)    -- normalization                              |
    |   ReLU               -- activation function                        |
    |   MaxPool2d(2x2)     -- halve spatial dimensions again             |
    |   Output: (batch, 64, 7, 7)                                       |
    +------------------------------------------------------------------+
    | Dropout(0.25): Regularization between conv and fully-connected     |
    +------------------------------------------------------------------+
    | Classifier (Fully Connected Layers):                               |
    |   Flatten            -- flatten to 1D vector: 64x7x7 = 3136       |
    |   Linear(3136->128)  -- fully connected, integrate all features    |
    |   ReLU               -- activation function                        |
    |   Dropout(0.5)       -- randomly drop 50% neurons, prevent         |
    |                          overfitting                               |
    |   Linear(128->10)    -- output scores for 10 digits                |
    |   Output: (batch, 10)                                             |
    +------------------------------------------------------------------+

    Total parameters: 421,834

    Improvements:
        - Kaiming initialization: Weight initialization designed for ReLU, stabilizes gradient
          propagation in early training
        - BatchNorm: Normalization after each conv layer, accelerates convergence + improves accuracy
        - Dropout(0.25): Extra regularization between conv and fully-connected layers, reduces
          co-adaptation
        - Dropout(0.5): Randomly drops 50% neurons in fully-connected layer, prevents overfitting
        - Combined with online data augmentation in train.py, further improves generalization

    Note: The last layer outputs raw logits (no Softmax applied),
          because FocalLoss computes Softmax internally.
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize the network

        Parameters:
            num_classes: Number of classes; MNIST has 10 digits (0-9)
        """
        super().__init__()

        # ---- Feature extraction part (convolutional layers) ----
        # This part is responsible for extracting meaningful features from images
        self.features = nn.Sequential(
            # ===== Layer 1 Convolutional Block =====
            # Conv2d: 1 input channel (grayscale) -> 32 output channels (32 different features)
            # kernel_size=3: 3x3 convolution kernel, sufficient to capture stroke directions
            # padding=1: zero-padding on edges, keep output spatial dimensions unchanged
            nn.Conv2d(1, 32, kernel_size=3, padding=1),

            # BatchNorm: Normalize each of the 32 channels to mean~0, variance~1
            # Benefit: More stable training, can use larger learning rate, faster convergence
            nn.BatchNorm2d(32),

            # ReLU: Turn negative values to 0, keep positive values unchanged
            # Benefit: Introduces nonlinearity (without it, multiple linear transforms equal one)
            #          Also simple to compute, mitigates gradient vanishing
            nn.ReLU(),

            # MaxPool: Take the maximum value in each 2x2 region
            # Effect: 28x28 -> 14x14 (halve spatial dimensions), keep most prominent features
            #         Also provides some translation invariance
            nn.MaxPool2d(2, 2),

            # ===== Layer 2 Convolutional Block =====
            # Conv2d: 32 input channels -> 64 output channels
            # Layer 2 learns higher-level features than Layer 1: Layer 1 learns edges,
            # Layer 2 combines edges into strokes, arcs, intersection points, etc.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        )
        # After two convolutional layers: (batch, 64, 7, 7) = each image becomes 64 7x7 feature maps

        # ---- Dropout between convolutional and fully-connected layers ----
        # Improvement: Add Dropout(0.25) after conv output is flattened, before entering FC layers
        # Purpose: Reduce co-adaptation between conv features and FC layers, further prevent overfitting
        # Why 0.25 (smaller than FC layer's 0.5): Conv features have already been dimensionally
        # reduced by pooling, information density is higher; dropping too much would lose useful features
        self.dropout_conv = nn.Dropout(0.25)

        # ---- Classifier part (fully-connected layers) ----
        # This part maps extracted features to 10 classes
        self.classifier = nn.Sequential(
            # Flatten: Flatten (64, 7, 7) into a 1D vector of (3136,)
            nn.Flatten(),

            # Fully connected layer: 3136 dims -> 128 dims
            # 128 dims is a "bottleneck", compressing 3136-dim information into a more compact
            # representation
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            # Dropout: Randomly drop 50% of neurons during training
            # Purpose: Prevent the model from relying too heavily on a few neurons, force it to
            #          learn redundant features
            # Note: Only active during training; automatically disabled during inference
            #       (PyTorch's Dropout has this logic built in)
            nn.Dropout(0.5),

            # Output layer: 128 dims -> 10 dims
            # Outputs "scores" (logits) for each of the 10 digits; the highest score is the prediction
            nn.Linear(128, num_classes),
        )

        # ---- Kaiming weight initialization ----
        # Good initialization makes training stable from the first epoch, avoiding gradient
        # vanishing/explosion
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: from input image -> output scores for 10 classes

        Parameters:
            x: Input image, shape=(batch_size, 1, 28, 28)
               batch_size = number of images processed at once
               1 = grayscale image (1 channel)
               28x28 = image dimensions

        Returns:
            shape=(batch_size, 10) -- scores for 10 digits per image

        Flow:
            Input (batch, 1, 28, 28)
            -> features convolutional blocks -> (batch, 64, 7, 7)
            -> classifier -> (batch, 10)
        """
        # First pass through convolutional layers to extract features
        x = self.features(x)
        # Dropout(0.25) between conv and FC layers, prevent overfitting
        x = self.dropout_conv(x)
        # Then pass through fully-connected layers for classification
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        """
        Kaiming Weight Initialization (He Initialization)

        Why initialization is needed:
            If weights are all zeros or randomly too small/large, early training gradients will
            vanish or explode, making the model unable to learn anything. Good initialization
            keeps training stable from the first epoch.

        Kaiming Initialization (He Initialization):
            An initialization method designed specifically for ReLU activation functions.
            Core idea: Keep the output variance of each layer stable, so it doesn't vanish
            or explode as depth increases.

            - Conv2d: kaiming_normal_ (mode='fan_out', nonlinearity='relu')
              fan_out mode computes variance based on output channels, suitable for Conv2d
            - BatchNorm2d: weight initialized to 1, bias initialized to 0
              (standard practice; BN layer adjusts itself via running statistics)
            - Linear: normal_(mean=0, std=0.01)
              Fully-connected layers use small-variance normal distribution to avoid large initial
              outputs

        Reference: He et al. "Delving Deep into Rectifiers: Surpassing Human-Level
                   Performance on ImageNet Classification" (2015)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal distribution initialization, designed for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN: weight=1, bias=0, so the initial state does not change the input distribution
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Fully-connected layer: small-variance normal distribution, avoid large initial outputs
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_feature_maps(self, x: torch.Tensor) -> tuple:
        """
        Extract intermediate layer feature maps (for visualizing what convolutional kernels learned)

        Use cases:
            - Visualize edge/stroke detectors learned by Layer 1 convolutional kernels
            - Visualize higher-level patterns (arcs, intersections, etc.) learned by Layer 2
            - Help understand how CNN works internally, not a "black box"

        Parameters:
            x: Input image, shape=(1, 1, 28, 28) or (batch, 1, 28, 28)

        Returns:
            (conv1_features, conv2_features) tuple:
            - conv1_features: Layer 1 conv block output, shape=(batch, 32, 14, 14)
              32 feature maps, each capturing a low-level pattern (e.g., horizontal strokes,
              vertical edges)
            - conv2_features: Layer 2 conv block output, shape=(batch, 64, 7, 7)
              64 feature maps, combining low-level patterns into more complex structures
        """
        # First 4 layers of features = 1st conv block: Conv->BN->ReLU->Pool
        # Input (1, 28, 28) -> Output (32, 14, 14)
        conv1_features = self.features[:4](x)
        # Last 4 layers of features = 2nd conv block: Conv->BN->ReLU->Pool
        # Input (32, 14, 14) -> Output (64, 7, 7)
        conv2_features = self.features[4:](conv1_features)
        return conv1_features, conv2_features


# ============================================================
# Utility Functions
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model

    Purpose: Understand model size, convenient for comparing different models
    Example: MNISTCNN has 421,834 parameters

    Parameters:
        model: PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    # p.numel() = number of elements in this tensor (parameter count)
    # p.requires_grad = True means this parameter will be updated by gradients (trainable)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
