"""
Training Loops

Training loop implementations for common training patterns

Includes:
- TrainingLoop: Main training coordinator with support for both generic (callbacks-based)
  and manual (custom batch function) training patterns
- ValidationLoop: Training loop with validation support

The TrainingLoop consolidates patterns from all examples:
- Epoch iteration with configurable batch size
- Custom batch processing via compute_batch_loss callback
- Automatic progress reporting
- Evaluation support with custom eval function

Example Usage:
    # Create training loop with progress logging every 100 batches
    var loop = TrainingLoop(log_interval=100)

    # Run epoch with manual batch processing (AlexNet, VGG16, etc pattern)
    var avg_loss = loop.run_epoch_manual(
        train_images, train_labels, batch_size=128,
        compute_batch_loss=my_batch_fn,
        epoch=1, total_epochs=100
    )

Note:
    All symbols in this module are re-exported cleanly through the parent
    `shared.training` package. You may import directly from either location:

    ```mojo
    from shared.training.loops import TrainingLoop
    from shared.training import TrainingLoop  # also works
    ```

    No Mojo re-export limitation applies here (unlike `shared.training.callbacks`).
"""

# Export training loop implementations
from shared.training.loops.training_loop import (
    TrainingLoop,
    train_one_epoch,
    training_step,
)

# Export validation loop
from shared.training.loops.validation_loop import ValidationLoop
