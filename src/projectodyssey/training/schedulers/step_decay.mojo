"""Step Learning Rate Decay Scheduler.

Decays the learning rate by a factor (gamma) every step_size epochs
This is a simple but effective learning rate schedule used in many classic papers.

Formula:
    lr = initial_lr * (gamma ** (epoch // step_size))

Example:
    ```mojo
    initial_lr = 0.1, step_size = 30, gamma = 0.1
    - Epochs 0-29: lr = 0.1
    - Epochs 30-59: lr = 0.01
    - Epochs 60-89: lr = 0.001
    - Epochs 90+: lr = 0.0001
    ```

This scheduler is commonly used in:
- AlexNet (decay by 10x every 30 epochs)
- ResNet (decay at epochs 30, 60, 90)
- VGG (similar step decay strategy)
"""


def step_lr(
    initial_lr: Float32, epoch: Int, step_size: Int = 30, gamma: Float32 = 0.1
) raises -> Float32:
    """Compute learning rate with step decay.

    Args:
            initial_lr: Initial learning rate at epoch 0.
            epoch: Current epoch number (0-indexed).
            step_size: Number of epochs between each decay step (default: 30).
            gamma: Multiplicative decay factor (default: 0.1).

    Returns:
            Decayed learning rate for current epoch.

    Raises:
            Error: If step_size is not positive or gamma is not in range (0, 1).

        Example:
            ```mojo
            from projectodyssey.training.schedulers import step_lr

            var initial_lr = Float32(0.01)

            for epoch in range(100):
                var lr = step_lr(initial_lr, epoch, step_size=30, gamma=0.1)
                print("Epoch", epoch, "LR:", lr)

                # ... training with current learning rate
            ```

    Note:
            - Pure function (no state management).
            - Caller passes current epoch and gets back learning rate.
            - Gamma typically 0.1 (10x reduction) or 0.5 (2x reduction).
            - Step size depends on dataset and model.
              - CIFAR-10: typically 30-50 epochs.
              - ImageNet: typically 30 epochs.
    """
    if step_size <= 0:
        raise Error("step_size must be positive")

    if gamma <= 0.0 or gamma >= 1.0:
        raise Error("gamma must be in range (0, 1)")

    # Compute number of decay steps
    var num_steps = epoch // step_size

    # Compute decay factor: gamma ^ num_steps
    var decay_factor = Float32(1.0)
    for _ in range(num_steps):
        decay_factor *= gamma

    return initial_lr * decay_factor


def multistep_lr(
    initial_lr: Float32, epoch: Int, milestones: List[Int], gamma: Float32 = 0.1
) raises -> Float32:
    """Compute learning rate with decay at specific milestone epochs.

        Decays learning rate by gamma at each milestone epoch.
        More flexible than step_lr - allows arbitrary decay schedule.

    Args:
            initial_lr: Initial learning rate at epoch 0.
            epoch: Current epoch number (0-indexed).
            milestones: List of epoch numbers where lr should be decayed.
            gamma: Multiplicative decay factor at each milestone (default: 0.1).

    Returns:
            Decayed learning rate for current epoch

        Example:
            ```mojo
            from projectodyssey.training.schedulers import multistep_lr

            var initial_lr = Float32(0.1)
            var milestones = [30, 60, 90]  # Decay at these epochs

            for epoch in range(100):
                var lr = multistep_lr(initial_lr, epoch, milestones, gamma=0.1)
                # Epoch 0-29: lr = 0.1
                # Epoch 30-59: lr = 0.01
                # Epoch 60-89: lr = 0.001
                # Epoch 90+: lr = 0.0001
            ```

    Raises:
        Error: If operation fails.

    Note:
            - Used in ResNet paper: decay at epochs [30, 60, 90].
            - Allows fine-grained control over LR schedule.
            - Milestones should be sorted in ascending order.
    """
    if gamma <= 0.0 or gamma >= 1.0:
        raise Error("gamma must be in range (0, 1)")

    # Count how many milestones have been passed
    var num_decays = 0
    for i in range(len(milestones)):
        if epoch >= milestones[i]:
            num_decays += 1

    # Apply decay for each milestone passed
    var decay_factor = Float32(1.0)
    for _ in range(num_decays):
        decay_factor *= gamma

    return initial_lr * decay_factor


def exponential_lr(
    initial_lr: Float32, epoch: Int, gamma: Float32 = 0.95
) raises -> Float32:
    """Compute learning rate with exponential decay.

        Decays learning rate by gamma every epoch (exponential decay).

        Formula:
            `lr = initial_lr * (gamma ** epoch)`

    Args:
            initial_lr: Initial learning rate at epoch 0.
            epoch: Current epoch number (0-indexed).
            gamma: Decay factor per epoch (default: 0.95 for ~5% decay per epoch).

    Returns:
            Decayed learning rate for current epoch.

        Example:
            ```mojo
            from projectodyssey.training.schedulers import exponential_lr

            var initial_lr = Float32(0.1)

            for epoch in range(100):
                var lr = exponential_lr(initial_lr, epoch, gamma=0.95)
                # Smooth exponential decay
            ```

    Raises:
        Error: If operation fails.

    Note:
            - Smoother decay than step_lr.
            - Gamma typically 0.9-0.99 (1-10% decay per epoch).
            - Less aggressive than step decay.
            - Good for fine-tuning.
    """
    if gamma <= 0.0 or gamma >= 1.0:
        raise Error("gamma must be in range (0, 1)")

    # Compute gamma^epoch
    var decay_factor = Float32(1.0)
    for _ in range(epoch):
        decay_factor *= gamma

    return initial_lr * decay_factor


def constant_lr(initial_lr: Float32, epoch: Int) -> Float32:
    """Constant learning rate (no decay).

        Returns the same learning rate for all epochs.
        Useful as baseline or for simple experiments.

    Args:
            initial_lr: Learning rate to use.
            epoch: Current epoch (ignored).

    Returns:
            Constant learning rate.

        Example:
            ```mojo
            from projectodyssey.training.schedulers import constant_lr

            var initial_lr = Float32(0.01)
            var lr = constant_lr(initial_lr, epoch)  # Always returns 0.01
            ```
    """
    return initial_lr
