"""Training argument utilities for common training script configuration.

This module provides a standardized TrainingArgs struct and parse_training_args()
function for consistent command-line argument handling across training scripts.

Features:
    - TrainingArgs struct for common hyperparameters
    - Automatic argument parsing with sensible defaults
    - Support for custom defaults via parse_training_args_with_defaults()
    - Validation of numeric ranges
    - Backward compatibility with direct sys.argv parsing

Example:
    from shared.utils import TrainingArgs, parse_training_args

    fn main() raises:
        var args = parse_training_args()
        print("Epochs:", args.epochs)
        print("Batch size:", args.batch_size)
        print("Learning rate:", args.learning_rate)
        print("Verbose:", args.verbose)
    ```
"""

from shared.utils.arg_parser import (
    create_training_parser,
    validate_positive_int,
    validate_positive_float,
    validate_range_float,
)


# ============================================================================
# Training Arguments Struct
# ============================================================================


@fieldwise_init
struct TrainingArgs(Copyable, Movable):
    """Container for common training hyperparameters and paths.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        momentum: Momentum factor for SGD.
        data_dir: Path to dataset directory.
        weights_dir: Path to save/load model weights.
        verbose: Whether to print verbose output.
        lr_decay_epochs: Decay LR every N epochs (0 = no decay).
        lr_decay_factor: Multiply LR by this factor when decaying.
    """

    var epochs: Int
    var batch_size: Int
    var learning_rate: Float64
    var momentum: Float64
    var data_dir: String
    var weights_dir: String
    var verbose: Bool
    var lr_decay_epochs: Int
    var lr_decay_factor: Float64

    fn __init__(out self):
        """Initialize with default training arguments."""
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.data_dir = "datasets"
        self.weights_dir = "weights"
        self.verbose = False
        self.lr_decay_epochs = 0
        self.lr_decay_factor = 0.1


# ============================================================================
# Argument Parsing Functions
# ============================================================================


fn parse_training_args() raises -> TrainingArgs:
    """Parse common training arguments from command line with defaults.

        Supported arguments:
            --epochs <int>: Number of training epochs (default: 10)
            --batch-size <int>: Batch size (default: 32)
            --lr <float>: Learning rate (default: 0.01)
            --momentum <float>: Momentum for SGD (default: 0.9)
            --data-dir <str>: Dataset directory (default: "datasets")
            --weights-dir <str>: Weights directory (default: "weights")
            --lr-decay-epochs <int>: Decay LR every N epochs (default: 0, disabled)
            --lr-decay-factor <float>: LR decay multiplier (default: 0.1)
            --verbose: Enable verbose output

    Returns:
            TrainingArgs struct with parsed and validated values.

    Raises:
            Error if argument validation fails.

        Example:
            ```mojo
           # Command line: mojo train.mojo --epochs 100 --lr 0.001 --verbose
            var args = parse_training_args()
            # args.epochs == 100, args.learning_rate == 0.001, args.verbose == True
            ```
    """
    return parse_training_args_with_defaults(
        default_epochs=10,
        default_batch_size=32,
        default_lr=0.01,
        default_momentum=0.9,
        default_data_dir="datasets",
        default_weights_dir="weights",
        default_lr_decay_epochs=0,
        default_lr_decay_factor=0.1,
    )


fn parse_training_args_with_defaults(
    default_epochs: Int = 10,
    default_batch_size: Int = 32,
    default_lr: Float64 = 0.01,
    default_momentum: Float64 = 0.9,
    default_data_dir: String = "datasets",
    default_weights_dir: String = "weights",
    default_lr_decay_epochs: Int = 0,
    default_lr_decay_factor: Float64 = 0.1,
) raises -> TrainingArgs:
    """Parse training arguments with custom defaults and validation.

        Allows each training script to specify model-appropriate defaults
        while still using shared parsing logic. Validates numeric ranges.

    Args:
            default_epochs: Default number of epochs (must be positive).
            default_batch_size: Default batch size (must be positive).
            default_lr: Default learning rate (must be positive).
            default_momentum: Default momentum (must be in [0.0, 1.0]).
            default_data_dir: Default dataset directory.
            default_weights_dir: Default weights directory.
            default_lr_decay_epochs: Default LR decay interval (0 = disabled).
            default_lr_decay_factor: Default LR decay multiplier (must be in (0.0, 1.0]).

    Returns:
            TrainingArgs struct with parsed and validated values.

    Raises:
            Error if argument validation fails.

        Example:
            ```mojo
            # AlexNet with custom defaults
            var args = parse_training_args_with_defaults(
                default_epochs=100,
                default_batch_size=128,
                default_lr=0.01,
                default_data_dir="datasets/cifar10",
                default_weights_dir="alexnet_weights",
                default_lr_decay_epochs=30,
                default_lr_decay_factor=0.1,
            )
            ```
    """
    var parser = create_training_parser()
    var parsed = parser.parse()

    # Extract values with defaults
    var epochs = parsed.get_int("epochs", default_epochs)
    var batch_size = parsed.get_int("batch-size", default_batch_size)
    var learning_rate = parsed.get_float("lr", default_lr)
    var momentum = parsed.get_float("momentum", default_momentum)
    var data_dir = parsed.get_string("data-dir", default_data_dir)
    var weights_dir = parsed.get_string("weights-dir", default_weights_dir)
    var lr_decay_epochs = parsed.get_int(
        "lr-decay-epochs", default_lr_decay_epochs
    )
    var lr_decay_factor = parsed.get_float(
        "lr-decay-factor", default_lr_decay_factor
    )
    var verbose = parsed.get_bool("verbose")

    # Validate numeric arguments
    validate_positive_int(epochs, "epochs")
    validate_positive_int(batch_size, "batch-size")
    validate_positive_float(learning_rate, "learning-rate")
    validate_range_float(momentum, 0.0, 1.0, "momentum")

    # Validate LR decay parameters
    if lr_decay_epochs < 0:
        raise Error(
            "lr-decay-epochs must be non-negative, got: "
            + String(lr_decay_epochs)
        )
    if lr_decay_factor <= 0.0 or lr_decay_factor > 1.0:
        raise Error(
            "lr-decay-factor must be in (0.0, 1.0], got: "
            + String(lr_decay_factor)
        )

    return TrainingArgs(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        data_dir=data_dir,
        weights_dir=weights_dir,
        verbose=verbose,
        lr_decay_epochs=lr_decay_epochs,
        lr_decay_factor=lr_decay_factor,
    )


# ============================================================================
# Validation Entry Point
# ============================================================================

# NOTE: main() function removed for mojo package compatibility
# (mojo package forbids main() in library files)
