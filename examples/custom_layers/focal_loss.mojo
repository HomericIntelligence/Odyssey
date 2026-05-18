"""Example: Custom Layers - Focal Loss.

This example implements a custom loss function for imbalanced datasets.

Usage:
    pixi run mojo run examples/custom_layers/focal_loss.mojo

See documentation: docs/advanced/custom-layers.md
"""

from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones_like,
    full_like,
)
from projectodyssey.core.arithmetic import subtract, multiply, add, power
from projectodyssey.core.elementwise import log, clip
from projectodyssey.core.reduction import mean


struct FocalLoss:
    """Focal loss for addressing class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    var alpha: Float64  # Weighting factor
    var gamma: Float64  # Focusing parameter

    def __init__(out self, alpha: Float64 = 0.25, gamma: Float64 = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(
        self, predictions: AnyTensor, targets: AnyTensor
    ) raises -> AnyTensor:
        """Compute focal loss for binary classification.

        Args:
            predictions: Model predictions (probabilities after sigmoid), shape [batch].
            targets: Ground truth labels (0 or 1), shape [batch].

        Returns:
            Scalar loss value.
        """
        var epsilon = 1e-7

        # Clip predictions to prevent log(0)
        var clipped = clip(predictions, epsilon, 1.0 - epsilon)

        # Compute log(p) and log(1-p)
        var log_pred = log(clipped)
        var one = ones_like(clipped)
        var one_minus_pred = subtract(one, clipped)
        var log_one_minus_pred = log(one_minus_pred)

        # Create alpha and gamma tensors
        var alpha_tensor = full_like(clipped, self.alpha)
        var gamma_tensor = full_like(clipped, self.gamma)
        var one_minus_alpha = subtract(one, alpha_tensor)

        # Compute (1-p)^gamma and p^gamma
        var one_minus_p_pow = power(one_minus_pred, gamma_tensor)
        var p_pow = power(clipped, gamma_tensor)

        # Focal loss: -alpha * (1-p)^gamma * target * log(p) - (1-alpha) * p^gamma * (1-target) * log(1-p)
        var one_minus_targets = subtract(one, targets)

        # First term: -alpha * (1-p)^gamma * target * log(p)
        var term1 = multiply(alpha_tensor, one_minus_p_pow)
        term1 = multiply(term1, targets)
        term1 = multiply(term1, log_pred)

        # Second term: -(1-alpha) * p^gamma * (1-target) * log(1-p)
        var term2 = multiply(one_minus_alpha, p_pow)
        term2 = multiply(term2, one_minus_targets)
        term2 = multiply(term2, log_one_minus_pred)

        # Combine and negate
        var sum_terms = add(term1, term2)
        var zero = zeros([len(predictions.shape())], predictions.dtype())
        for i in range(len(predictions.shape())):
            zero._set_float64(i, 0.0)
        var loss_per_sample = subtract(zero, sum_terms)

        # Return mean loss
        return mean(loss_per_sample, axis=-1, keepdims=False)


def main() raises:
    """Demonstrate focal loss."""

    # Create focal loss
    var loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Test with sample binary classification predictions and targets
    var predictions = zeros([4], DType.float32)
    predictions._set_float64(0, 0.9)  # High confidence, correct (target=1)
    predictions._set_float64(1, 0.3)  # Low confidence, correct (target=0)
    predictions._set_float64(2, 0.7)  # Medium confidence, correct (target=1)
    predictions._set_float64(3, 0.1)  # High confidence, correct (target=0)

    var targets = zeros([4], DType.float32)
    targets._set_float64(0, 1.0)  # Positive class
    targets._set_float64(1, 0.0)  # Negative class
    targets._set_float64(2, 1.0)  # Positive class
    targets._set_float64(3, 0.0)  # Negative class

    print("Binary classification focal loss example")
    print("Predictions: [0.9, 0.3, 0.7, 0.1]")
    print("Targets: [1, 0, 1, 0]")
    print()

    var loss = loss_fn(predictions, targets)
    print("Loss value:", loss._get_float64(0))

    print("\nFocal loss example complete!")
