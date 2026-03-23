"""Base training interfaces and contracts.

This module defines the core traits and types for the training subsystem:
- Callback: Lifecycle hooks for training events
- TrainingState: Shared state passed to callbacks
- LRScheduler: Learning rate scheduling interface

These contracts establish clear APIs for gradient utilities (#2393) and training callbacks (#2392)
"""

from collections import Dict, List
from shared.tensor.any_tensor import AnyTensor
from shared.core import has_nan, has_inf
from math import sqrt


# ============================================================================
# Callback System
# ============================================================================


struct CallbackSignal(Copyable, ImplicitlyCopyable, Movable):
    """Signal returned by callbacks to control training flow.

    Values:
        CONTINUE (0): Continue training normally.
        STOP (1): Stop training immediately.
    """

    var value: Int
    """Signal value (0=CONTINUE, 1=STOP)."""

    fn __init__(out self, value: Int):
        """Initialize callback signal.

        Args:
            value: Signal value (0=CONTINUE, 1=STOP).
        """
        self.value = value


# Callback signal constants
comptime CONTINUE = CallbackSignal(0)
comptime STOP = CallbackSignal(1)


struct TrainingState(Copyable, Movable):
    """Training state passed to callbacks.

    This struct provides callbacks with access to training metrics and control.
    It uses borrowed references to avoid ownership issues.

    Lifecycle:
        Created at training start, updated each epoch/batch, destroyed at end.

    Fields:
        epoch: Current epoch number (0-indexed).
        batch: Current batch number within epoch (0-indexed).
        metrics: Dictionary of metric name -> value.
        learning_rate: Current learning rate.
        should_stop: Flag set by callbacks to request training stop.

    Example:
        ```mojo
        var state = TrainingState(epoch=0, batch=0, metrics={}, lr=0.1)
        state.metrics["train_loss"] = 0.5
        state.metrics["val_loss"] = 0.6
        ```
    """

    var epoch: Int
    """Current epoch number (0-indexed)."""
    var batch: Int
    """Current batch number within epoch (0-indexed)."""
    var metrics: Dict[String, Float64]
    """Dictionary of metric name -> value."""
    var learning_rate: Float64
    """Current learning rate."""
    var should_stop: Bool
    """Flag set by callbacks to request training stop."""

    fn __init__(
        out self,
        epoch: Int = 0,
        batch: Int = 0,
        learning_rate: Float64 = 0.0,
    ):
        """Initialize training state.

        Args:
            epoch: Current epoch number.
            batch: Current batch number.
            learning_rate: Current learning rate.
        """
        self.epoch = epoch
        self.batch = batch
        self.metrics = Dict[String, Float64]()
        self.learning_rate = learning_rate
        self.should_stop = False


trait Callback:
    """Callback interface for training lifecycle hooks.

    Callbacks receive training events at specific points in the training loop.
    They can monitor metrics, save checkpoints, or request early stopping.

    Lifecycle Event Order:
        1. on_train_begin()        # Once at training start.
        2. For each epoch:
            a. on_epoch_begin()    # Start of epoch.
            b. For each batch:
                i.  on_batch_begin()   # Before forward pass.
                ii. on_batch_end()     # After backward pass and optimizer step.
            c. on_epoch_end()      # After validation (if any).
        3. on_train_end()          # Once at training end.

    State Modification:
        - Callbacks can READ all fields of TrainingState.
        - Callbacks can WRITE metrics dictionary (add new metrics).
        - Callbacks can SET should_stop flag (request early stopping).
        - Callbacks CANNOT modify model/optimizer directly (no references provided).

    Return Values:
        - Most hooks return CallbackSignal (CONTINUE or STOP).
        - STOP signal triggers graceful training shutdown.
        - Multiple callbacks: first STOP signal takes precedence.

    Example:
        ```mojo
        truct MyCallback(Callback):
            fn on_epoch_end(mut self, mut state: TrainingState) raises -> CallbackSignal:
                print("Epoch", state.epoch, "loss:", state.metrics["train_loss"])
                return CONTINUE
        ```
    """

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called once at the start of training.

        Args:
            state: Training state (epoch=0, batch=0, empty metrics).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called once at the end of training.

        Args:
            state: Final training state with complete metrics history.

        Returns:
            CallbackSignal (ignored, training already ending).
        """
        ...

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called at the start of each epoch.

        Args:
            state: Training state (epoch set, batch=0, previous epoch metrics).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...

    fn on_epoch_end(
        mut self, mut state: TrainingState
    ) raises -> CallbackSignal:
        """Called at the end of each epoch (after validation).

        Args:
            state: Training state with current epoch metrics (train_loss, val_loss, etc.).

        Returns:
            CallbackSignal (CONTINUE or STOP).

        Raises:
            Error: If operation fails.
        """
        ...

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called at the start of each batch.

        Args:
            state: Training state (epoch, batch set).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Called at the end of each batch (after optimizer step).

        Args:
            state: Training state (may include batch metrics like batch_loss).

        Returns:
            CallbackSignal (CONTINUE or STOP).
        """
        ...


# ============================================================================
# Learning Rate Scheduler Interface
# ============================================================================


trait LRScheduler:
    """Learning rate scheduler interface.

    Schedulers compute learning rates based on training progress.
    They do NOT modify the optimizer directly - the training loop
    is responsible for calling get_lr() and updating the optimizer.

    Integration Pattern:
        1. Training loop calls scheduler.get_lr(epoch, batch).
        2. Training loop sets optimizer.learning_rate = new_lr.
        3. Scheduler is stateless (pure function of epoch/batch).

    Scheduler Types:
        - Step decay: Reduce LR at fixed intervals.
        - Cosine annealing: Smooth cosine decay.
        - Warmup: Linear increase then constant.

    Example:
        ```mojo
        truct StepLR(LRScheduler):
            var base_lr: Float64
            var step_size: Int
            var gamma: Float64

            fn get_lr(self, epoch: Int, batch: Int) -> Float64:
                var steps = epoch // self.step_size
                return self.base_lr * (self.gamma ** steps)
        ```
    """

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate for given epoch and batch.

        Args:
            epoch: Current epoch number (0-indexed).
            batch: Current batch number within epoch (0-indexed).

        Returns:
            Learning rate to use for this step.

        Notes:
            - Schedulers should be deterministic (same inputs -> same output).
            - epoch and batch are 0-indexed.
            - batch parameter is optional (defaults to 0 for epoch-based schedulers).
        """
        ...


# ============================================================================
# Numerical Safety
# ============================================================================


fn has_nan_or_inf(tensor: AnyTensor) -> Bool:
    """Check if tensor contains NaN or Inf values (numerical instability detection).

    Detects numerical instability in gradients during training by checking for:
    - NaN (Not a Number) values indicating undefined operations.
    - Inf (positive or negative infinity) indicating overflow.

    Args:
        tensor: Tensor to check for numerical instability.

    Returns:
        True if tensor contains any NaN or Inf values, False otherwise.

    Example:
        var gradients = ...  # Computed gradients
        if has_nan_or_inf(gradients):
            print("Numerical instability detected! Stopping training.")
            break

    Note:
        - Works with all tensor dtypes (float32, float64, float16, integer types).
        - Integer tensors cannot have NaN/Inf and will always return False.
        - Used for gradient validation during training.
    """
    return has_nan(tensor) or has_inf(tensor)


fn is_valid_loss(loss: Float64) raises -> Bool:
    """Check if loss value is valid (not NaN or inf).

    Args:
        loss: Loss value to check.

    Returns:
        True if loss is finite (not NaN, not inf), False otherwise.

    Raises:
        Error: If operation fails.

    Example:
        if not is_valid_loss(loss):
            print("Training diverged! Loss is", loss)
            break

    Note:
        Uses has_nan_or_inf internally for consistency with gradient validation.
    """
    # Create a single-element tensor for loss value
    var shape = List[Int]()
    shape.append(1)
    var loss_tensor = AnyTensor(shape, DType.float64)

    # Access the tensor's data pointer and set the loss value
    var ptr = loss_tensor._data.bitcast[Float64]()
    ptr[0] = loss

    # Use has_nan_or_inf for consistency
    return not has_nan_or_inf(loss_tensor)


fn compute_gradient_norm(
    parameters: List[AnyTensor], norm_type: String = "L2"
) -> Float64:
    """Compute gradient norm for training diagnostics and exploding gradient detection.

    Computes the global norm of all gradients in a parameter list using either
    L1 or L2 norm. Used for:
    - Gradient clipping (by computing norm to clip by).
    - Training diagnostics (monitoring gradient magnitude).
    - Exploding gradient detection (norm > threshold).

    Args:
        parameters: List of gradient tensors to compute norm over.
        norm_type: Type of norm to compute ("L2" or "L1"). Defaults to "L2".

    Returns:
        Global norm of all gradients as Float64.

    Example:
        var grad_norm = compute_gradient_norm(gradients, "L2")
        if grad_norm > max_grad_norm:
            # Clip gradients
            ...

    Notes:
        - L2 norm: sqrt(sum of all gradient elements squared).
        - L1 norm: sum of absolute values of all gradient elements.
        - Returns 0.0 for empty parameter list.
        - Aggregates norms across all tensors in the list.

    Reference:
        Used in Gradient Clipping by Global Norm (arXiv:1308.0850).
    """
    var total_norm_sq = Float64(0.0)
    var total_abs_norm = Float64(0.0)

    # Aggregate norm over all parameter tensors
    for i in range(len(parameters)):
        var tensor = parameters[i]
        var size = tensor.numel()

        # Handle each dtype separately for efficiency
        if tensor.dtype() == DType.float32:
            var ptr = tensor._data.bitcast[Float32]()
            for j in range(size):
                var val = Float64(ptr[j])
                if norm_type == "L2":
                    total_norm_sq += val * val
                elif norm_type == "L1":
                    total_abs_norm += abs(val)
        elif tensor.dtype() == DType.float64:
            var ptr = tensor._data.bitcast[Float64]()
            for j in range(size):
                var val = ptr[j]
                if norm_type == "L2":
                    total_norm_sq += val * val
                elif norm_type == "L1":
                    total_abs_norm += abs(val)
        elif tensor.dtype() == DType.float16:
            var ptr = tensor._data.bitcast[Float16]()
            for j in range(size):
                var val = Float64(Float32(ptr[j]))
                if norm_type == "L2":
                    total_norm_sq += val * val
                elif norm_type == "L1":
                    total_abs_norm += abs(val)
        # For integer types, skip (gradients are typically float)

    # Return appropriate norm
    if norm_type == "L2":
        return sqrt(total_norm_sq)
    elif norm_type == "L1":
        return total_abs_norm
    else:
        # Default to L2 if unknown norm type
        return sqrt(total_norm_sq)


fn clip_gradients(
    var gradients: List[Float64], max_norm: Float64
) -> List[Float64]:
    """Clip gradients by global norm to prevent exploding gradients.

    Args:
        gradients: List of gradient values.
        max_norm: Maximum allowed gradient norm.

    Returns:
        Clipped gradients with norm <= max_norm.

    Example:
        clipped_grads = clip_gradients(grads, max_norm=1.0)

    Note:
        This is a legacy function that works with lists of Float64.
        For tensor-based gradient clipping, use compute_gradient_norm()
        with AnyTensor parameters instead.
    """
    # Compute L2 norm of the gradient list
    var norm_sq = Float64(0.0)
    for i in range(len(gradients)):
        var val = gradients[i]
        norm_sq += val * val

    var norm = sqrt(norm_sq)

    # If norm exceeds max_norm, scale down all gradients
    if norm > max_norm and norm > 0.0:
        var scale = max_norm / norm
        for i in range(len(gradients)):
            gradients[i] = gradients[i] * scale

    return gradients^
