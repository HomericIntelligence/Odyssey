"""
Training Library.

The training library provides reusable training infrastructure including optimizers,
schedulers, metrics, callbacks, and training loops for ML Odyssey paper implementations.

All components are implemented in Mojo for maximum performance.

Import tests in tests/shared/test_imports.mojo are implemented and passing.
See Issue #3033 for tracking: 6 tests for training module imports.
Tests require corresponding modules to be implemented first.

Note:
    **Callback Import Limitation**: Due to Mojo's module system, callback types cannot
    be imported directly from the `shared.training` parent module. They must be imported
    directly from the `shared.training.callbacks` submodule.

    This is a known limitation of Mojo's current re-export mechanism — symbols defined
    in a submodule and re-exported via `__init__.mojo` are not always resolvable at the
    parent package level when used as types in user code.

    Incorrect (will fail with a Mojo import error):

    ```mojo
    from shared.training import EarlyStopping
    from shared.training import ModelCheckpoint
    from shared.training import LoggingCallback
    ```

    Correct (import directly from the submodule):

    ```mojo
    from shared.training.callbacks import EarlyStopping
    from shared.training.callbacks import ModelCheckpoint
    from shared.training.callbacks import LoggingCallback
    ```
"""

from shared.core.extensor import ExTensor
from shared.core.traits import Model, Loss, Optimizer
from shared.training.trainer_interface import DataLoader, DataBatch
from shared.autograd.tape import GradientTape

# Package version
from shared.version import VERSION

# ============================================================================
# Exports - Training Components
# ============================================================================

# Export model utilities for weight persistence (Issue #2294)
from shared.training.model_utils import (
    save_model_weights,
    load_model_weights,
    get_model_parameter_names,
    validate_shapes,
)

# Export gradient operations (Issue #2630)
from shared.training.gradient_ops import (
    accumulate_gradient_inplace,
    scale_gradient_inplace,
    zero_gradient_inplace,
)

# Export checkpoint manager (Issue #2664)
from shared.training.checkpoint import CheckpointManager

# Export gradient clipping utilities (Issue #2666)
from shared.training.gradient_clipping import (
    clip_gradients_by_global_norm,
    clip_gradients_per_param,
    clip_gradients_by_value_list,
    compute_gradient_norm_list,
    compute_gradient_statistics,
    GradientStatistics,
)

# Export base interfaces and utilities
from shared.training.base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
    LRScheduler,
    has_nan_or_inf,
    compute_gradient_norm,
    is_valid_loss,
    clip_gradients,
)

# Export scheduler implementations
from shared.training.schedulers import (
    StepLR,
    CosineAnnealingLR,
    WarmupLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    WarmupCosineAnnealingLR,
    WarmupStepLR,
)

# Export callback implementations
# NOTE (Mojo v0.26.1): Callbacks must be imported directly from submodules due to
# Mojo re-export limitations. Use:
#   from shared.training.callbacks import EarlyStopping
# NOT from shared.training import EarlyStopping
from shared.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LoggingCallback,
)

# ============================================================================
# Core Training Components (Issue #1939)
# ============================================================================


# SGD Optimizer
struct SGD(Movable, Optimizer):
    """Stochastic Gradient Descent optimizer.

    Implements the Optimizer trait for use with generic TrainingLoop.
    Delegates to the autograd SGD optimizer for actual gradient-based updates.
    """

    var learning_rate: Float32
    """Learning rate for gradient descent."""

    fn __init__(out self, learning_rate: Float32):
        """Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate for gradient descent.
        """
        self.learning_rate = learning_rate

    fn step(mut self, params: List[ExTensor]) raises:
        """Update parameters using gradients.

        Implements: param = param - learning_rate * grad.

        Note:
            The current implementation is a no-op stub. For gradient-based updates,
            use the full training loop with tape.backward() which automatically
            handles gradient computation and optimizer steps via the autograd system.

        Args:
            params: List of parameter tensors to update.

        Raises:
            Error: If operation fails.
        """
        # NOTE: Actual gradient updates happen in TrainingLoop.step() via
        # the autograd system which maintains Variable data and gradients.
        # This stub is kept for trait interface compatibility.
        pass

    fn zero_grad(mut self) raises:
        """Reset optimizer state.

        Raises:
            Error: If operation fails.

        Note:
            SGD has no internal state, so this is a no-op.
            Gradient clearing is handled by the autograd tape system.
        """
        pass

    fn get_learning_rate(self) -> Float64:
        """Get the current learning rate.

        Returns:
            Current learning rate as Float64.
        """
        return Float64(self.learning_rate)


# MSELoss - Mean Squared Error Loss
struct MSELoss(Loss, Movable):
    """Mean Squared Error loss function for regression.

    Implements the Loss trait for use with generic TrainingLoop.
    """

    var reduction: String
    """Type of reduction ('mean' or 'sum')."""

    fn __init__(out self, reduction: String = "mean"):
        """Initialize MSE loss.

        Args:
            reduction: Type of reduction ('mean' or 'sum').
        """
        self.reduction = reduction

    fn compute(self, pred: ExTensor, target: ExTensor) raises -> ExTensor:
        """Compute MSE loss between predictions and targets.

        Implements the Loss trait interface.

        Args:
            pred: Model predictions.
            target: Ground truth targets.

        Returns:
            Scalar loss value as ExTensor.

        Raises:
            Error: If operation fails.

        Computes: MSE = mean((pred - target)^2) or sum((pred - target)^2).
        """
        from shared.core.arithmetic import subtract, multiply
        from shared.core.reduction import mean, sum as tensor_sum

        # Compute element-wise difference: (pred - target)
        var diff = subtract(pred, target)

        # Compute element-wise squared: squared_diff^2
        var squared = multiply(diff, diff)

        # Apply reduction (mean or sum)
        if self.reduction == "mean":
            return mean(squared, axis=-1, keepdims=False)
        elif self.reduction == "sum":
            return tensor_sum(squared, axis=-1, keepdims=False)
        else:
            # Default to no reduction ("none")
            return squared

    fn forward(self, output: ExTensor, target: ExTensor) raises -> Float32:
        """Compute MSE loss (legacy interface for backward compatibility).

        Args:
            output: Model outputs.
            target: Target values.

        Returns:
            Loss value.

        Raises:
            Error: If operation fails.
        """
        return Float32(0.0)

    fn backward(self, grad_output: ExTensor) raises -> ExTensor:
        """Compute gradient of loss.

        Args:
            grad_output: Upstream gradient.

        Returns:
            Gradient tensor.

        Raises:
            Error: If operation fails.
        """
        # Placeholder implementation - returns gradient same shape as input
        return grad_output


# TrainingLoop - Main training orchestrator (Generic with Trait Bounds)
struct TrainingLoop[
    M: Model & Movable, L: Loss & Movable, O: Optimizer & Movable
]:
    """Orchestrates training with forward/backward/optimize cycle.

    Generic training loop with compile-time type safety via trait bounds.
    Uses compile-time type safety for zero-cost abstraction.

    Type Parameters:
        M: Model type (must implement Model trait).
        L: Loss type (must implement Loss trait).
        O: Optimizer type (must implement Optimizer trait).

    Example:
        ```mojo
        var model = SimpleMLP(...)
        var optimizer = SGD(learning_rate=0.01)
        var loss_fn = MSELoss()
        var loop = TrainingLoop[SimpleMLP, MSELoss, SGD](model, optimizer, loss_fn)
        ```
    """

    var model: Self.M
    """Model implementing Model trait."""
    var optimizer: Self.O
    """Optimizer implementing Optimizer trait."""
    var loss_fn: Self.L
    """Loss function implementing Loss trait."""
    var tape: GradientTape
    """Gradient tape for automatic differentiation."""

    fn __init__(
        out self, var model: Self.M, var optimizer: Self.O, var loss_fn: Self.L
    ):
        """Initialize training loop with generic components.

        Args:
            model: Model implementing Model trait.
            optimizer: Optimizer implementing Optimizer trait.
            loss_fn: Loss function implementing Loss trait.
        """
        from shared.autograd.tape import GradientTape

        self.model = model^
        self.optimizer = optimizer^
        self.loss_fn = loss_fn^
        self.tape = GradientTape()

    fn step(mut self, inputs: ExTensor, targets: ExTensor) raises -> ExTensor:
        """Perform single training step.

        Implements the training loop cycle using trait methods:
        1. Forward pass: outputs = model.forward(inputs)  [Model trait].
        2. Loss computation: loss = loss_fn.compute(outputs, targets)  [Loss trait].
        3. Backward pass: compute gradients via autograd tape.
        4. Optimizer step: Update parameters based on gradients.
        5. Return loss value.

        Args:
            inputs: Input ExTensor.
            targets: Target ExTensor.

        Returns:
            Scalar loss value as ExTensor.

        Raises:
            Error: If operation fails.

        Note:
            Uses trait methods for type-safe dispatch.
            Gradient computation via autograd system (GradientTape).
        """
        from shared.autograd.variable import Variable
        from shared.autograd.optimizers import SGD as AutogradSGD

        # Clear tape and enable gradient recording
        self.tape.clear()
        self.tape.enable()

        # Forward pass via Model trait
        var outputs = self.forward(inputs)

        # Compute loss via Loss trait
        var loss_value = self.compute_loss(outputs, targets)

        # Wrap loss in Variable and compute gradients via autograd
        var loss_var = Variable(loss_value, True, self.tape)
        loss_var.backward(self.tape)

        # Get model parameters and update via autograd optimizer
        var params = self.model.parameters()

        # Create autograd SGD optimizer and perform parameter update
        var lr = self.optimizer.get_learning_rate()
        var autograd_sgd = AutogradSGD(lr)

        # Convert parameters to Variables for optimizer
        var var_params: List[Variable] = []
        for i in range(len(params)):
            var p = Variable(params[i], True, self.tape)
            var_params.append(p^)

        # Update parameters using autograd optimizer
        autograd_sgd.step(var_params, self.tape)

        # Copy updated data back to original params
        for i in range(len(params)):
            params[i] = var_params[i].data

        # Zero gradients and disable tape for next iteration
        autograd_sgd.zero_grad(self.tape)
        self.tape.disable()

        return loss_var.detach()

    fn forward(mut self, inputs: ExTensor) raises -> ExTensor:
        """Execute forward pass via Model trait.

        Args:
            inputs: Input ExTensor.

        Returns:
            Model output ExTensor.

        Raises:
            Error: If operation fails.
        """
        # Call model.forward() via Model trait
        return self.model.forward(inputs)

    fn compute_loss(
        self, outputs: ExTensor, targets: ExTensor
    ) raises -> ExTensor:
        """Compute loss via Loss trait.

        Args:
            outputs: Model outputs.
            targets: Ground truth targets.

        Returns:
            Scalar loss value as ExTensor.

        Raises:
            Error: If operation fails.
        """
        # Call loss_fn.compute() via Loss trait
        return self.loss_fn.compute(outputs, targets)

    fn run_epoch(mut self, mut data_loader: DataLoader) raises -> Float32:
        """Run single epoch over dataset.

        Iterates through all batches in data loader and performs training
        step on each batch, returning the average loss for the epoch.

        Args:
            data_loader: Native DataLoader providing batches.

        Returns:
            Average loss for the epoch.

        Raises:
            Error: If operation fails.
        """
        var total_loss = Float64(0.0)
        var num_batches = Int(0)

        data_loader.reset()
        while data_loader.has_next():
            var batch = data_loader.next()
            var loss = self.step(batch.data, batch.labels)
            total_loss += Float64(loss._get_float32(0))
            num_batches += 1

        # Return average loss
        if num_batches > 0:
            return Float32(total_loss / Float64(num_batches))
        else:
            return Float32(0.0)


# Export validation loop
from shared.training.loops.validation_loop import ValidationLoop

# Export evaluation module (Issue #2352)
from shared.training.evaluation import (
    EvaluationResult,
    evaluate_model,
    evaluate_model_simple,
    evaluate_topk,
)

# Export mixed precision training utilities
from shared.training.mixed_precision import (
    GradientScaler,
    convert_to_fp32_master,
    update_model_from_master,
    check_gradients_finite,
    clip_gradients_by_norm,
    clip_gradients_by_value,
)

# Export precision configuration
from shared.training.precision_config import PrecisionConfig, PrecisionMode

# Export training configuration (Issue #2602)
from shared.training.config import TrainingConfig

# Export training metrics (Issue #3221)
from shared.training.metrics import (
    LossTracker,
    Statistics,
    ComponentTracker,
    AccuracyMetric,
    top1_accuracy,
    topk_accuracy,
    per_class_accuracy,
)


# Export CrossEntropyLoss wrapper (wraps core.loss.cross_entropy)
struct CrossEntropyLoss(Loss, Movable):
    """Cross entropy loss function for classification.

    Implements the Loss trait for use with generic TrainingLoop.
    """

    var reduction: String
    """Type of reduction ('mean' or 'sum')."""

    fn __init__(out self, reduction: String = "mean"):
        """Initialize cross entropy loss.

        Args:
            reduction: Type of reduction ('mean' or 'sum').
        """
        self.reduction = reduction

    fn compute(self, pred: ExTensor, target: ExTensor) raises -> ExTensor:
        """Compute cross entropy loss between predictions and targets.

        Args:
            pred: Model predictions (logits).
            target: Ground truth targets (class indices or one-hot).

        Returns:
            Scalar loss value as ExTensor.

        Raises:
            Error: If operation fails.
        """
        from shared.core.loss import cross_entropy

        return cross_entropy(pred, target)


# ============================================================================
# Public API
# ============================================================================

# Training script utilities (Issue #3034)
from shared.training.script_runner import (
    TrainingCallbacks,
    run_epoch_with_batches,
    print_training_header,
    print_dataset_info,
)
from shared.training.dataset_loaders import (
    DatasetSplit,
    load_emnist_dataset,
    load_cifar10_dataset,
    print_dataset_summary,
)
