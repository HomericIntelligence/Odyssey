"""Learning Rate Schedulers

Scheduler implementations for adjusting learning rates during training

Includes:
- StepLR: Step decay scheduler (decay every N epochs)
- CosineAnnealingLR: Cosine annealing scheduler
- WarmupLR: Linear warmup scheduler
- ExponentialLR: Exponential decay scheduler
- MultiStepLR: Multi-step decay scheduler
- ReduceLROnPlateau: Metric-based decay scheduler
- WarmupCosineAnnealingLR: Combined warmup and cosine annealing
- WarmupStepLR: Combined warmup and step decay

All schedulers are struct-based implementations of the LRScheduler trait

Note:
    All symbols in this module are re-exported cleanly through the parent
    `shared.training` package. You may import directly from either location:

    ```mojo
    from shared.training.schedulers import StepLR
    from shared.training import StepLR  # also works
    ```

    No Mojo re-export limitation applies here (unlike `shared.training.callbacks`).
"""

# Export scheduler implementations
from shared.training.schedulers.lr_schedulers import (
    StepLR,
    CosineAnnealingLR,
    WarmupLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    WarmupCosineAnnealingLR,
    WarmupStepLR,
)

# Also export pure function implementations for backward compatibility
from shared.training.schedulers.step_decay import (
    step_lr,
    multistep_lr,
    exponential_lr,
    constant_lr,
)
