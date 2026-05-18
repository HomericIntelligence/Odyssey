"""Learning rate schedulers for adaptive learning rate decay during training.

Implements learning rate scheduling strategies to adjust the learning rate during training
This helps improve convergence and model performance by reducing learning rate over time.

Implemented schedulers:
- StepLR: Decay learning rate by gamma every step_size epochs.
- ExponentialLR: Decay learning rate by gamma every epoch.

Usage Pattern:
    ```
    # Create scheduler with base learning rate
    var scheduler = StepLR(base_lr=0.01, step_size=10, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        # Get current learning rate
        var lr = scheduler.step(epoch)
        optimizer.learning_rate = lr

        # Training step...

        # Alternative: query current learning rate without stepping
        var current_lr = scheduler.get_lr()
    ```

Design Note:
    Schedulers track base_lr (initial learning rate), current_lr (computed LR),
    and last_epoch (for state management). The step() method updates state and
    returns the new learning rate, while get_lr() returns current without changing state.
"""

# pow is not needed - use ** operator instead


struct StepLR:
    """Step decay learning rate scheduler.

        Decays the learning rate by gamma every step_size epochs:
            `lr = base_lr * gamma^(epoch // step_size)`

        This is useful for:
        - Reducing learning rate at fixed intervals (e.g., every 10 epochs).
        - Implementing common schedules like dividing LR by 10 every 30 epochs.
        - Coarse-grained learning rate adjustment.

        Attributes:
            base_lr: Initial learning rate (unchanged).
            current_lr: Current learning rate (updated by step()).
            last_epoch: Last epoch for which step() was called.
            step_size: Number of epochs before decay.
            gamma: Multiplicative factor (e.g., 0.1 means 10x reduction).

    Examples:
    ```
            # Divide learning rate by 10 every 30 epochs
            var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

            # Halve learning rate every 5 epochs
            var scheduler = StepLR(base_lr=0.01, step_size=5, gamma=0.5)

            # Training loop
            for epoch in range(100):
                var lr = scheduler.step(epoch)
                optimizer.learning_rate = lr
    ```
    """

    var base_lr: Float64
    """Initial learning rate (unchanged)."""
    var current_lr: Float64
    """Current learning rate (updated by step())."""
    var last_epoch: Int
    """Last epoch for which step() was called."""
    var step_size: Int
    """Number of epochs before decay."""
    var gamma: Float64
    """Multiplicative decay factor."""

    def __init__(
        out self,
        base_lr: Float64,
        step_size: Int,
        gamma: Float64 = 0.1,
    ):
        """Initialize StepLR scheduler.

        Args:
            base_lr: Initial learning rate.
            step_size: Period of learning rate decay (in epochs).
            gamma: Multiplicative decay factor, range (0, 1]
                    0.1 = 10x reduction per step
                    0.5 = 2x reduction per step
                    Values >1 increase LR (not recommended).

        Examples:
        ```
                # Standard: divide by 10 every 30 epochs
                var sched = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

                # Custom: halve every 10 epochs
                var sched = StepLR(base_lr=0.01, step_size=10, gamma=0.5)
        ```
        """
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = -1
        self.current_lr = base_lr

    def step(mut self, epoch: Int) -> Float64:
        """Update learning rate for the given epoch and return it.

        Computes: `lr = base_lr * gamma^(epoch // step_size)`

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            The learning rate for this epoch.

        Examples:
        ```
                var sched = StepLR(base_lr=0.1, step_size=10, gamma=0.1)
                var lr0 = sched.step(0)   # 0.1
                var lr10 = sched.step(10) # 0.01 (0.1 * 0.1^1)
                var lr20 = sched.step(20) # 0.001 (0.1 * 0.1^2)
        ```
        """
        self.last_epoch = epoch
        var decay_factor = self.gamma ** Float64(epoch // self.step_size)
        self.current_lr = self.base_lr * decay_factor
        return self.current_lr

    def get_lr(self) -> Float64:
        """Get the current learning rate without updating state.

        Returns:
            The current learning rate (last value computed by step()).

        Examples:
        ```
                var sched = StepLR(base_lr=0.1, step_size=10, gamma=0.1)
                sched.step(0)
                var lr = sched.get_lr()  # Returns 0.1
        ```
        """
        return self.current_lr


struct ExponentialLR:
    """Exponential decay learning rate scheduler.

        Decays the learning rate exponentially by gamma every epoch:
            `lr = base_lr * gamma^epoch`

        This provides smooth, continuous decay throughout training and is useful for:
        - Gradual, continuous learning rate reduction.
        - Fine-grained control over decay rate.
        - Exponential schedules (e.g., lr decays by 5% per epoch).

        Attributes:
            base_lr: Initial learning rate (unchanged).
            current_lr: Current learning rate (updated by step()).
            last_epoch: Last epoch for which step() was called.
            gamma: Multiplicative decay per epoch, range (0, 1)
                   0.95 = 5% decay per epoch
                   0.9 = 10% decay per epoch
                   0.5 = 50% decay per epoch.

    Examples:
    ```
            # 5% decay per epoch
            var scheduler = ExponentialLR(base_lr=0.1, gamma=0.95)

            # 10% decay per epoch
            var scheduler = ExponentialLR(base_lr=0.01, gamma=0.9)

            # Training loop
            for epoch in range(100):
                var lr = scheduler.step(epoch)
                optimizer.learning_rate = lr
    ```
    """

    var base_lr: Float64
    """Initial learning rate (unchanged)."""
    var current_lr: Float64
    """Current learning rate (updated by step())."""
    var last_epoch: Int
    """Last epoch for which step() was called."""
    var gamma: Float64
    """Multiplicative decay per epoch."""

    def __init__(out self, base_lr: Float64, gamma: Float64):
        """Initialize ExponentialLR scheduler.

        Args:
            base_lr: Initial learning rate.
            gamma: Multiplicative decay per epoch, range (0, 1)
                    Typical values: 0.9-0.99
                    0.95 = 5% reduction per epoch
                    0.9 = 10% reduction per epoch
                    0.5 = 50% reduction per epoch (aggressive).

        Examples:
        ```
                # Moderate decay: 5% per epoch
                var sched = ExponentialLR(base_lr=0.1, gamma=0.95)

                # Aggressive decay: 10% per epoch
                var sched = ExponentialLR(base_lr=0.01, gamma=0.9)
        ```
        """
        self.base_lr = base_lr
        self.gamma = gamma
        self.last_epoch = -1
        self.current_lr = base_lr

    def step(mut self, epoch: Int) -> Float64:
        """Update learning rate for the given epoch and return it.

        Computes: `lr = base_lr * gamma^epoch`

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            The learning rate for this epoch.

        Examples:
        ```
                var sched = ExponentialLR(base_lr=0.1, gamma=0.95)
                var lr0 = sched.step(0)   # 0.1
                var lr1 = sched.step(1)   # 0.095 (0.1 * 0.95)
                var lr10 = sched.step(10) # 0.0599 (0.1 * 0.95^10)
        ```
        """
        self.last_epoch = epoch
        var decay_factor = self.gamma ** Float64(epoch)
        self.current_lr = self.base_lr * decay_factor
        return self.current_lr

    def get_lr(self) -> Float64:
        """Get the current learning rate without updating state.

        Returns:
            The current learning rate (last value computed by step()).

        Examples:
        ```
                var sched = ExponentialLR(base_lr=0.1, gamma=0.95)
                sched.step(0)
                var lr = sched.get_lr()  # Returns 0.1
        ```
        """
        return self.current_lr
