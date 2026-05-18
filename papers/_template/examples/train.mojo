"""Example training script demonstrating configuration-driven training.

This script shows how to use the ML Odyssey configuration system to
set up and run model training. It demonstrates:
- Loading experiment configuration
- Creating model from config
- Setting up trainer from config
- Running training with config parameters

Usage:
    mojo train.mojo --paper lenet5 --experiment baseline
    mojo train.mojo --paper lenet5 --experiment augmented
"""

from projectodyssey.utils.config_loader import (
    load_experiment_config,
    validate_experiment_config,
)
from projectodyssey.utils.config import Config
from projectodyssey.utils.arg_parser import (
    create_training_parser,
    ArgumentParser,
    ParsedArgs,
)
from projectodyssey.training.trainer_interface import (
    TrainerConfig,
    TrainingMetrics,
)


# ============================================================================
# Model Configuration
# ============================================================================


@fieldwise_init
struct ModelConfig(Copyable, Movable):
    """Configuration for model creation.

    Holds all parameters needed to instantiate a model from configuration.
    """

    var name: String
    """Model name (e.g., 'lenet5', 'alexnet')."""
    var num_classes: Int
    """Number of output classes."""
    var input_channels: Int
    """Number of input channels (e.g., 1 for grayscale, 3 for RGB)."""
    var input_height: Int
    """Input image height."""
    var input_width: Int
    """Input image width."""
    var dtype: DType
    """Data type for model parameters."""
    var dropout: Float64
    """Dropout rate (0.0 = no dropout)."""


# ============================================================================
# Argument Parsing
# ============================================================================


def parse_arguments() raises -> Tuple[String, String]:
    """Parse command-line arguments for training.

    Uses the shared argument parser infrastructure with added --paper and
    --experiment arguments for configuration-driven training.

    Returns:
        Tuple of (paper_name, experiment_name).

    Raises:
        Error: If argument parsing fails or required arguments are missing.
    """
    # Create parser with standard ML training arguments
    var parser = create_training_parser()

    # Add paper-specific arguments
    parser.add_argument("paper", "string", "lenet5")
    parser.add_argument("experiment", "string", "baseline")

    # Parse command-line arguments
    var args = parser.parse()

    # Extract paper and experiment names
    var paper_name = args.get_string("paper", "lenet5")
    var experiment_name = args.get_string("experiment", "baseline")

    return (paper_name, experiment_name)


# ============================================================================
# Model Creation
# ============================================================================


def create_model_config(config: Config) raises -> ModelConfig:
    """Extract model configuration from experiment config.

    Reads model parameters from the configuration object and creates
    a ModelConfig struct for model instantiation.

    Args:
        config: Configuration containing model parameters.

    Returns:
        ModelConfig with extracted parameters.

    Raises:
        Error: If required model configuration keys are missing.
    """
    # Validate required model configuration keys
    var required_keys = List[String]()
    required_keys.append("model.name")
    required_keys.append("model.num_classes")
    config.validate(required_keys)

    # Extract model name and basic parameters
    var model_name = config.get_string("model.name")
    var num_classes = config.get_int("model.num_classes", default=10)

    # Extract input shape (channels, height, width)
    var input_channels = 1  # Default: grayscale
    var input_height = 28  # Default: MNIST-like
    var input_width = 28

    if config.has("model.input_shape"):
        var input_shape = config.get_list("model.input_shape")
        if len(input_shape) >= 1:
            input_channels = Int(input_shape[0])
        if len(input_shape) >= 2:
            input_height = Int(input_shape[1])
        if len(input_shape) >= 3:
            input_width = Int(input_shape[2])

    # Extract optional parameters with defaults
    var dropout = config.get_float("model.dropout", default=0.0)

    # Parse dtype from config (default to float32)
    var dtype = DType.float32
    if config.has("model.dtype"):
        var dtype_str = config.get_string("model.dtype", default="float32")
        if dtype_str == "float16":
            dtype = DType.float16
        elif dtype_str == "float64":
            dtype = DType.float64
        elif dtype_str == "bfloat16":
            dtype = DType.bfloat16

    return ModelConfig(
        name=model_name,
        num_classes=num_classes,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dtype=dtype,
        dropout=dropout,
    )


def create_model(config: Config) raises -> ModelConfig:
    """Create model from configuration.

    Extracts model configuration and prints summary information.
    In a full implementation, this would instantiate the actual model.

    Args:
        config: Configuration containing model parameters.

    Returns:
        ModelConfig with extracted parameters.

    Raises:
        Error: If model configuration is invalid.
    """
    print("Creating model from configuration...")

    # Extract model configuration
    var model_config = create_model_config(config)

    # Print model configuration summary
    print("  Model name:", model_config.name)
    print("  Number of classes:", model_config.num_classes)
    print(
        "  Input shape: [",
        model_config.input_channels,
        ",",
        model_config.input_height,
        ",",
        model_config.input_width,
        "]",
    )
    print("  Dropout:", model_config.dropout)
    print("  DType:", String(model_config.dtype))

    return model_config^


# ============================================================================
# Trainer Creation
# ============================================================================


def create_trainer_config(config: Config) raises -> TrainerConfig:
    """Extract trainer configuration from experiment config.

    Creates a TrainerConfig struct from the configuration object,
    extracting all training hyperparameters and settings.

    Args:
        config: Configuration containing training parameters.

    Returns:
        TrainerConfig with extracted parameters.

    Raises:
        Error: If required training configuration keys are missing.
    """
    # Extract training parameters with defaults
    var epochs = config.get_int("training.epochs", default=10)
    var batch_size = config.get_int("training.batch_size", default=32)
    var learning_rate = config.get_float(
        "optimizer.learning_rate", default=0.001
    )

    # Extract logging and validation intervals
    var log_interval = config.get_int("training.log_interval", default=10)
    var validate_interval = config.get_int(
        "training.validate_interval", default=1
    )

    # Extract checkpoint settings
    var save_checkpoints = config.get_bool(
        "training.save_checkpoints", default=False
    )
    var checkpoint_interval = config.get_int(
        "training.checkpoint_interval", default=5
    )

    # Extract learning rate scheduler settings
    var use_scheduler = config.get_bool("training.use_scheduler", default=False)
    var scheduler_type = config.get_string(
        "training.scheduler_type", default="none"
    )

    # Extract mixed precision settings
    var use_mixed_precision = config.get_bool(
        "training.use_mixed_precision", default=False
    )
    var precision_dtype = DType.float32
    if config.has("training.precision_dtype"):
        var dtype_str = config.get_string(
            "training.precision_dtype", default="float32"
        )
        if dtype_str == "float16":
            precision_dtype = DType.float16
        elif dtype_str == "bfloat16":
            precision_dtype = DType.bfloat16

    var loss_scale = Float32(
        config.get_float("training.loss_scale", default=65536.0)
    )
    var gradient_clip_norm = Float32(
        config.get_float("training.gradient_clip_norm", default=0.0)
    )

    return TrainerConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_interval=log_interval,
        validate_interval=validate_interval,
        save_checkpoints=save_checkpoints,
        checkpoint_interval=checkpoint_interval,
        use_scheduler=use_scheduler,
        scheduler_type=scheduler_type,
        use_mixed_precision=use_mixed_precision,
        precision_dtype=precision_dtype,
        loss_scale=loss_scale,
        gradient_clip_norm=gradient_clip_norm,
    )


def create_trainer(config: Config) raises -> TrainerConfig:
    """Create trainer from configuration.

    Extracts training configuration and prints summary information.
    In a full implementation, this would instantiate the actual trainer.

    Args:
        config: Configuration containing training parameters.

    Returns:
        TrainerConfig with extracted parameters.

    Raises:
        Error: If training configuration is invalid.
    """
    print("Creating trainer from configuration...")

    # Extract trainer configuration
    var trainer_config = create_trainer_config(config)

    # Extract optimizer name for display
    var optimizer_name = config.get_string("optimizer.name", default="sgd")

    # Print trainer configuration summary
    print("  Optimizer:", optimizer_name)
    print("  Learning rate:", trainer_config.learning_rate)
    print("  Batch size:", trainer_config.batch_size)
    print("  Epochs:", trainer_config.num_epochs)
    print("  Log interval:", trainer_config.log_interval)
    print("  Validate interval:", trainer_config.validate_interval)
    if trainer_config.save_checkpoints:
        print("  Checkpoint interval:", trainer_config.checkpoint_interval)
    if trainer_config.use_scheduler:
        print("  Scheduler:", trainer_config.scheduler_type)
    if trainer_config.use_mixed_precision:
        print("  Mixed precision:", String(trainer_config.precision_dtype))

    return trainer_config^


# ============================================================================
# Training Loop
# ============================================================================


def run_training(
    config: Config,
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
) raises -> TrainingMetrics:
    """Run the training loop.

    Implements the training loop skeleton using TrainingMetrics for tracking.
    In a full implementation, this would use actual model forward/backward passes.

    Args:
        config: Complete configuration for training.
        model_config: Model configuration.
        trainer_config: Trainer configuration.

    Returns:
        TrainingMetrics with final training statistics.

    Raises:
        Error: If training fails.
    """
    print("\nStarting training...")
    print("-" * 50)

    # Initialize training metrics
    var metrics = TrainingMetrics()
    var epochs = trainer_config.num_epochs

    # Training loop
    for epoch in range(epochs):
        metrics.current_epoch = epoch + 1
        metrics.reset_epoch()

        print("Epoch", epoch + 1, "/", epochs)

        # === Training Phase ===
        # In a full implementation, this would:
        # 1. Iterate over training batches
        # 2. Forward pass through model
        # 3. Compute loss
        # 4. Backward pass (compute gradients)
        # 5. Update weights

        # Placeholder: simulate training loss decreasing
        var train_loss = 1.0 / Float64(epoch + 1)
        var train_accuracy = Float64(epoch + 1) * 10.0 / Float64(epochs)
        if train_accuracy > 100.0:
            train_accuracy = 100.0

        metrics.update_train_metrics(train_loss, train_accuracy)
        print(
            "  Train Loss:",
            train_loss,
            "| Train Accuracy:",
            train_accuracy,
            "%",
        )

        # === Validation Phase ===
        # Validate at specified intervals
        if (epoch + 1) % trainer_config.validate_interval == 0:
            # Placeholder: simulate validation metrics
            var val_loss = train_loss * 1.1  # Slightly higher than train
            var val_accuracy = train_accuracy * 0.95  # Slightly lower

            metrics.update_val_metrics(val_loss, val_accuracy)
            print(
                "  Val Loss:",
                val_loss,
                "| Val Accuracy:",
                val_accuracy,
                "%",
            )

            # Check for improvement
            if val_loss < metrics.best_val_loss:
                print("  * New best validation loss!")

        # === Checkpointing ===
        if trainer_config.save_checkpoints:
            if (epoch + 1) % trainer_config.checkpoint_interval == 0:
                print(
                    "  Saving checkpoint at epoch",
                    epoch + 1,
                )
                # In full implementation: save model weights
                # model.save("checkpoint_epoch_" + String(epoch + 1) + ".pt")

    print("-" * 50)
    print("Training complete!")

    # Print final summary
    metrics.print_summary()

    return metrics^


# ============================================================================
# Main Entry Point
# ============================================================================


def main() raises:
    """Main entry point for training script.

    Orchestrates the complete training workflow:
    1. Parse command-line arguments
    2. Load experiment configuration
    3. Validate configuration
    4. Create model from configuration
    5. Create trainer from configuration
    6. Run training loop
    7. Report results
    """
    print("=" * 60)
    print("Configuration-Driven Training Example")
    print("=" * 60)

    # Step 1: Parse arguments
    var args = parse_arguments()
    var paper_name = args[0]
    var experiment_name = args[1]

    print("\nPaper:", paper_name)
    print("Experiment:", experiment_name)

    # Step 2: Load configuration
    print("\nLoading configuration...")
    var config = load_experiment_config(paper_name, experiment_name)
    print("  Configuration loaded successfully!")

    # Step 3: Validate configuration
    print("\nValidating configuration...")
    try:
        validate_experiment_config(config)
        print("  Configuration is valid!")
    except e:
        print("  ERROR: Configuration validation failed!")
        print("  ", String(e))
        return

    # Step 4: Create model from config
    var model_config = create_model(config)

    # Step 5: Create trainer from config
    var trainer_config = create_trainer(config)

    # Step 6: Run training
    var metrics = run_training(config, model_config, trainer_config)

    # Step 7: Report final results
    print("\n" + "=" * 60)
    print("Training script completed successfully!")
    print("=" * 60)
    print("\nFinal Results:")
    print("  Best Validation Loss:", metrics.best_val_loss)
    print("  Best Validation Accuracy:", metrics.best_val_accuracy, "%")
    print("  Best Epoch:", metrics.best_epoch)
    print("=" * 60)
