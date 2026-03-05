"""Learning rate scheduler implementations.

This module provides implementations of common learning rate scheduling strategies:
- StepLR: Decay learning rate at fixed intervals.
- CosineAnnealingLR: Smooth cosine decay.
- WarmupLR: Linear warmup phase.

All schedulers implement the LRScheduler trait from base.mojo.
"""

from math import pi, cos
from shared.training.base import LRScheduler


# ============================================================================
# Step Learning Rate Scheduler
# ============================================================================


@register_passable("trivial")
struct StepLR(LRScheduler):
    """Step decay: reduce learning rate at fixed intervals.

    Reduces the learning rate by a factor of gamma every step_size epochs.

    Formula:
        lr = base_lr * gamma^(epoch // step_size).

    Attributes:
        base_lr: Initial learning rate.
        step_size: Number of epochs between LR reductions.
        gamma: Multiplicative factor for LR reduction.

    Example:
        ```mojo
        var scheduler = StepLR(
            base_lr=0.1,
            step_size=10,
            gamma=0.1
        )
        # After 10 epochs: LR = 0.1 * 0.1 = 0.01
        # After 20 epochs: LR = 0.1 * 0.01 = 0.001
        ```
    """

    var base_lr: Float64
    """Initial learning rate."""
    var step_size: Int
    """Number of epochs between LR reductions."""
    var gamma: Float64
    """Multiplicative factor for LR reduction."""

    fn __init__(out self, base_lr: Float64, step_size: Int, gamma: Float64):
        """Initialize StepLR scheduler.

        Args:
            base_lr: Initial learning rate.
            step_size: Number of epochs between LR reductions.
            gamma: Multiplicative factor for LR reduction.
        """
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate using step decay formula.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused for epoch-based scheduler).

        Returns:
            Learning rate for this epoch.
        """
        if self.step_size <= 0:
            return self.base_lr

        var num_steps = epoch // self.step_size
        var decay_factor = self.gamma**num_steps
        return self.base_lr * decay_factor


# ============================================================================
# Cosine Annealing Learning Rate Scheduler
# ============================================================================


@register_passable("trivial")
struct CosineAnnealingLR(LRScheduler):
    """Cosine annealing: smooth cosine decay from base_lr to eta_min.

    The learning rate follows a cosine curve, starting at base_lr and
    smoothly decaying to eta_min over T_max epochs.

    Formula:
        lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2.

    Attributes:
        base_lr: Initial learning rate.
        T_max: Maximum number of epochs (period).
        eta_min: Minimum learning rate.

    Example:
        ```mojo
        var scheduler = CosineAnnealingLR(
            base_lr=0.1,
            T_max=100,
            eta_min=0.0
        )
        # At epoch 0: LR = 0.1 (maximum)
        # At epoch 50: LR = 0.05 (halfway)
        # At epoch 100: LR = 0.0 (minimum)
        ```
    """

    var base_lr: Float64
    """Initial learning rate."""
    var T_max: Int
    """Maximum number of epochs (period)."""
    var eta_min: Float64
    """Minimum learning rate."""

    fn __init__(out self, base_lr: Float64, T_max: Int, eta_min: Float64 = 0.0):
        """Initialize Cosine Annealing scheduler.

        Args:
            base_lr: Initial learning rate.
            T_max: Maximum number of epochs (period).
            eta_min: Minimum learning rate.
        """
        self.base_lr = base_lr
        self.T_max = T_max
        self.eta_min = eta_min

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate using cosine annealing formula.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        if self.T_max <= 0:
            return self.base_lr

        # Clamp epoch to T_max range
        var clamped_epoch = epoch
        if clamped_epoch > self.T_max:
            clamped_epoch = self.T_max

        # Cosine annealing formula
        var progress = Float64(clamped_epoch) / Float64(self.T_max)
        var cosine_factor = (1.0 + cos(pi * progress)) / 2.0
        return self.eta_min + (self.base_lr - self.eta_min) * cosine_factor


# ============================================================================
# Warmup Learning Rate Scheduler
# ============================================================================


@register_passable("trivial")
struct WarmupLR(LRScheduler):
    """Linear warmup: gradually increase learning rate during initial epochs.

    The learning rate increases linearly from 0 to base_lr over warmup_epochs,
    then remains constant at base_lr.

    Formula:
        lr = base_lr * (epoch / warmup_epochs)  for epoch < warmup_epochs.
        lr = base_lr                            for epoch >= warmup_epochs.

    Attributes:
        base_lr: Target learning rate after warmup.
        warmup_epochs: Number of epochs for warmup phase.

    Example:
        ```mojo
        var scheduler = WarmupLR(
            base_lr=0.1,
            warmup_epochs=10
        )
        # Epochs 0-9: LR increases from 0 to 0.1
        # Epochs 10+: LR = 0.1
        ```
    """

    var base_lr: Float64
    """Target learning rate after warmup."""
    var warmup_epochs: Int
    """Number of epochs for warmup phase."""

    fn __init__(out self, base_lr: Float64, warmup_epochs: Int):
        """Initialize Warmup scheduler.

        Args:
            base_lr: Target learning rate after warmup.
            warmup_epochs: Number of epochs for warmup.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate with linear warmup.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        if self.warmup_epochs <= 0:
            return self.base_lr

        if epoch >= self.warmup_epochs:
            return self.base_lr

        # Linear warmup
        var progress = Float64(epoch) / Float64(self.warmup_epochs)
        return self.base_lr * progress


# ============================================================================
# ReduceLROnPlateau Learning Rate Scheduler
# ============================================================================

# Mode constants for ReduceLROnPlateau
comptime MODE_MIN: Int = 0  # Minimize metric (for loss)
comptime MODE_MAX: Int = 1  # Maximize metric (for accuracy)


@register_passable("trivial")
struct ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when metric stops improving.

    Monitors a metric (e.g., validation loss) and reduces the learning rate
    by a factor when the metric has not improved for patience epochs.

    Attributes:
        base_lr: Initial learning rate.
        mode: Optimization mode (MODE_MIN=0 for loss, MODE_MAX=1 for accuracy).
        factor: Multiplicative factor for LR reduction.
        patience: Number of epochs without improvement before reducing LR.
        best_metric: Best metric value seen so far.
        epochs_without_improvement: Counter for epochs without improvement.
        current_lr: Current learning rate (updated by step()).

    Example:
        ```mojo
        var scheduler = ReduceLROnPlateau(
            base_lr=0.1,
            mode="min",
            factor=0.1,
            patience=10
        )
        # If validation loss doesn't improve for 10 epochs:
        # LR = 0.1 * 0.1 = 0.01
        ```
    """

    var base_lr: Float64
    """Initial learning rate."""
    var mode: Int
    """Optimization mode (MODE_MIN=0 for loss, MODE_MAX=1 for accuracy)."""
    var factor: Float64
    """Multiplicative factor for LR reduction."""
    var patience: Int
    """Number of epochs without improvement before reducing LR."""
    var best_metric: Float64
    """Best metric value seen so far."""
    var epochs_without_improvement: Int
    """Counter for epochs without improvement."""
    var current_lr: Float64
    """Current learning rate (updated by step())."""

    fn __init__(
        out self,
        base_lr: Float64,
        mode: String = "min",
        factor: Float64 = 0.1,
        patience: Int = 10,
    ):
        """Initialize ReduceLROnPlateau scheduler.

        Args:
            base_lr: Initial learning rate.
            mode: Optimization mode ("min" or "max").
            factor: Multiplicative factor for LR reduction.
            patience: Epochs without improvement before reducing LR.
        """
        self.base_lr = base_lr
        self.factor = factor
        self.patience = patience
        self.current_lr = base_lr
        self.epochs_without_improvement = 0

        # Convert string mode to int and initialize best_metric
        if mode == "min":
            self.mode = MODE_MIN
            self.best_metric = Float64(1e10)
        else:
            self.mode = MODE_MAX
            self.best_metric = Float64(-1e10)

    fn step(mut self, metric: Float64) -> Float64:
        """Update scheduler based on metric value.

        Args:
            metric: Current metric value (e.g., validation loss).

        Returns:
            New learning rate.
        """
        var improved = False

        if self.mode == MODE_MIN:
            # For loss, improvement means metric decreased
            if metric < self.best_metric:
                improved = True
                self.best_metric = metric
        else:
            # For accuracy, improvement means metric increased
            if metric > self.best_metric:
                improved = True
                self.best_metric = metric

        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            # Reduce LR if no improvement for patience epochs
            # NOTE: Check is inside else block to avoid reducing on improving steps
            if self.epochs_without_improvement >= self.patience:
                self.current_lr = self.current_lr * self.factor
                self.epochs_without_improvement = 0

        return self.current_lr

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Get current learning rate.

        Args:
            epoch: Current epoch (unused in ReduceLROnPlateau).
            batch: Current batch (unused).

        Returns:
            Current learning rate.
        """
        return self.current_lr


# ============================================================================
# Exponential Learning Rate Scheduler
# ============================================================================


struct ExponentialLR(Copyable, LRScheduler, Movable):
    """Exponential decay: reduce learning rate by gamma every epoch.

    The learning rate decays exponentially at each epoch following the formula:
        lr = base_lr * gamma^epoch.

    This provides smooth, continuous decay throughout training.

    Attributes:
        base_lr: Initial learning rate.
        gamma: Multiplicative decay factor per epoch (typically 0.9-0.99).

    Example:
        ```mojo
        var scheduler = ExponentialLR(
            base_lr=0.1,
            gamma=0.95
        )
        # Epoch 0: LR = 0.1
        # Epoch 1: LR = 0.095
        # Epoch 10: LR = 0.0599
        ```
    """

    var base_lr: Float64
    """Initial learning rate."""
    var gamma: Float64
    """Multiplicative decay factor per epoch."""

    fn __init__(out self, base_lr: Float64, gamma: Float64):
        """Initialize ExponentialLR scheduler.

        Args:
            base_lr: Initial learning rate.
            gamma: Multiplicative decay factor per epoch (0 < gamma < 1).
                   Typical values: 0.9-0.99 for gradual decay.
        """
        self.base_lr = base_lr
        self.gamma = gamma

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate using exponential decay formula.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        if self.gamma <= 0.0:
            return self.base_lr

        var decay_factor = self.gamma ** Float64(epoch)
        return self.base_lr * decay_factor


# ============================================================================
# MultiStep Learning Rate Scheduler
# ============================================================================


struct MultiStepLR(Copyable, LRScheduler, Movable):
    """Multi-step decay: reduce learning rate at specified epochs.

    Reduces the learning rate by a factor of gamma at specified milestone epochs.

    Formula:
        lr = base_lr * gamma^(number of milestones passed).

    Attributes:
        base_lr: Initial learning rate.
        milestones: List of epoch numbers where learning rate is multiplied by gamma.
        gamma: Multiplicative factor for LR reduction.

    Example:
        ```mojo
        var milestones = List[Int]()
        milestones.append(30)
        milestones.append(60)
        milestones.append(90)
        var scheduler = MultiStepLR(
            base_lr=0.1,
            milestones=milestones,
            gamma=0.1
        )
        # Epochs 0-29: LR = 0.1
        # Epochs 30-59: LR = 0.01
        # Epochs 60-89: LR = 0.001
        # Epochs 90+: LR = 0.0001
        ```
    """

    var base_lr: Float64
    """Initial learning rate."""
    var milestones: List[Int]
    """Epoch numbers where learning rate is multiplied by gamma."""
    var gamma: Float64
    """Multiplicative factor for LR reduction."""

    fn __init__(
        out self, base_lr: Float64, milestones: List[Int], gamma: Float64
    ):
        """Initialize MultiStepLR scheduler.

        Args:
            base_lr: Initial learning rate.
            milestones: List of epoch numbers where to decay learning rate.
            gamma: Multiplicative factor for LR reduction.
        """
        self.base_lr = base_lr
        self.milestones = milestones.copy()
        self.gamma = gamma

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate based on milestone crossings.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        var num_milestones_passed = 0

        for i in range(len(self.milestones)):
            if epoch >= self.milestones[i]:
                num_milestones_passed += 1

        var decay_factor = self.gamma ** Float64(num_milestones_passed)
        return self.base_lr * decay_factor


# ============================================================================
# Composite Warmup + Decay Schedulers
# ============================================================================


struct WarmupCosineAnnealingLR(Copyable, LRScheduler, Movable):
    """Combined warmup and cosine annealing scheduler.

    Combines linear warmup phase with cosine annealing decay:
    1. Warmup phase (0 to warmup_epochs): Linear increase from 0 to base_lr.
    2. Annealing phase (warmup_epochs to total_epochs): Cosine decay from base_lr to eta_min.

    Attributes:
        base_lr: Maximum learning rate (after warmup).
        warmup_epochs: Number of epochs for warmup phase.
        T_max: Total number of epochs (for cosine calculation).
        eta_min: Minimum learning rate in annealing phase.

    Example:
        ```mojo
        var scheduler = WarmupCosineAnnealingLR(
            base_lr=0.1,
            warmup_epochs=10,
            T_max=100,
            eta_min=0.0
        )
        # Epochs 0-9: LR increases from 0 to 0.1
        # Epochs 10-100: LR decays from 0.1 to 0.0 following cosine curve
        ```
    """

    var base_lr: Float64
    """Maximum learning rate."""
    var warmup_epochs: Int
    """Number of epochs for warmup phase."""
    var T_max: Int
    """Total number of epochs."""
    var eta_min: Float64
    """Minimum learning rate."""

    fn __init__(
        out self,
        base_lr: Float64,
        warmup_epochs: Int,
        T_max: Int,
        eta_min: Float64 = 0.0,
    ):
        """Initialize WarmupCosineAnnealingLR scheduler.

        Args:
            base_lr: Maximum learning rate.
            warmup_epochs: Number of epochs for warmup.
            T_max: Total number of epochs.
            eta_min: Minimum learning rate during annealing.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate with warmup then cosine annealing.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup phase
            var progress = Float64(epoch) / Float64(self.warmup_epochs)
            return self.base_lr * progress
        else:
            # Cosine annealing phase
            var annealing_epoch = epoch - self.warmup_epochs
            var annealing_T_max = self.T_max - self.warmup_epochs

            if annealing_T_max <= 0:
                return self.base_lr

            var clamped_epoch = annealing_epoch
            if clamped_epoch > annealing_T_max:
                clamped_epoch = annealing_T_max

            var progress = Float64(clamped_epoch) / Float64(annealing_T_max)
            var cosine_factor = (1.0 + cos(pi * progress)) / 2.0
            return self.eta_min + (self.base_lr - self.eta_min) * cosine_factor


struct WarmupStepLR(Copyable, LRScheduler, Movable):
    """Combined warmup and step decay scheduler.

    Combines linear warmup phase with step decay:
    1. Warmup phase (0 to warmup_epochs): Linear increase from 0 to base_lr.
    2. Step decay phase: Reduce LR by gamma every step_size epochs.

    Attributes:
        base_lr: Maximum learning rate (after warmup).
        warmup_epochs: Number of epochs for warmup phase.
        step_size: Number of epochs between LR reductions in decay phase.
        gamma: Multiplicative factor for LR reduction.

    Example:
        ```mojo
        var scheduler = WarmupStepLR(
            base_lr=0.1,
            warmup_epochs=10,
            step_size=30,
            gamma=0.1
        )
        # Epochs 0-9: LR increases from 0 to 0.1
        # Epochs 10-39: LR = 0.1
        # Epochs 40-69: LR = 0.01
        # Epochs 70+: LR = 0.001
        ```
    """

    var base_lr: Float64
    """Maximum learning rate."""
    var warmup_epochs: Int
    """Number of epochs for warmup phase."""
    var step_size: Int
    """Number of epochs between LR reductions."""
    var gamma: Float64
    """Multiplicative factor for LR reduction."""

    fn __init__(
        out self,
        base_lr: Float64,
        warmup_epochs: Int,
        step_size: Int,
        gamma: Float64,
    ):
        """Initialize WarmupStepLR scheduler.

        Args:
            base_lr: Maximum learning rate.
            warmup_epochs: Number of epochs for warmup.
            step_size: Number of epochs between LR reductions.
            gamma: Multiplicative factor for LR reduction.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma

    fn get_lr(self, epoch: Int, batch: Int = 0) -> Float64:
        """Compute learning rate with warmup then step decay.

        Args:
            epoch: Current epoch (0-indexed).
            batch: Current batch (unused).

        Returns:
            Learning rate for this epoch.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup phase
            var progress = Float64(epoch) / Float64(self.warmup_epochs)
            return self.base_lr * progress
        else:
            # Step decay phase
            if self.step_size <= 0:
                return self.base_lr

            var decay_epoch = epoch - self.warmup_epochs
            var num_steps = decay_epoch // self.step_size
            var decay_factor = self.gamma ** Float64(num_steps)
            return self.base_lr * decay_factor
