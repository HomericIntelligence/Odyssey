"""
Utility Functions Library

Provides logging, visualization, configuration, and other helper utilities

Modules:
    logging: Logging infrastructure for training and evaluation
    visualization: Plotting and visualization tools
    config: Configuration management utilities

Example:
    ```mojo
    from shared.utils import Logger, plot_training_curves, load_config

    # Create logger
    logger = Logger("training.log")
    logger.info("Starting training...")

    # Plot results
    plot_training_curves(train_losses, val_losses)

    # Load configuration
    config = load_config("experiment.yaml")
    ```

Import tests in tests/shared/test_imports.mojo are implemented and passing.
See Issue #3033: 4 tests for utils module imports — all tests pass.
"""

# Package version
from ..version import VERSION

# ============================================================================
# Exports - Implemented modules
# ============================================================================

# Logging utilities
from .logging import (
    Logger,  # Main logger class
    LogLevel,  # Log level enum
    get_logger,  # Get or create logger
    set_global_log_level,  # Set global log level
    StreamHandler,  # Console output handler
    FileHandler,  # File output handler
    LogRecord,  # Log record structure
    SimpleFormatter,  # Simple message formatter
    TimestampFormatter,  # Formatter with timestamps
    DetailedFormatter,  # Detailed formatter with location
    ColoredFormatter,  # Formatter with ANSI colors
)

# Configuration utilities
from .config import (
    Config,  # Configuration container
    load_config,  # Load from file (YAML/JSON)
    save_config,  # Save to file
    merge_configs,  # Merge multiple configs
    ConfigValidator,  # Validate configuration
    create_validator,  # Create validator instance
)

# File I/O utilities
from .file_io import (
    Checkpoint,  # Checkpoint container
    save_checkpoint,  # Save model checkpoint
    load_checkpoint,  # Load model checkpoint
    serialize_tensor,  # Serialize tensor
    deserialize_tensor,  # Deserialize tensor
    safe_write_file,  # Atomic file write
    safe_read_file,  # Safe file read
    create_backup,  # Backup creation
    file_exists,  # Check file existence
    directory_exists,  # Check directory existence
    create_directory,  # Create directory
)

# Visualization utilities
from .visualization import (
    plot_training_curves,  # Plot loss/accuracy curves
    plot_loss_only,  # Plot single loss curve
    plot_accuracy_only,  # Plot single accuracy curve
    plot_confusion_matrix,  # Plot confusion matrix
    visualize_model_architecture,  # Model architecture diagram
    show_images,  # Display image grid
    visualize_feature_maps,  # Feature map visualization
    save_figure,  # Save matplotlib figure
)

# Progress bar utilities
from .progress_bar import (
    ProgressBar,  # Simple progress bar
    ProgressBarWithMetrics,  # Progress bar with metrics display
    ProgressBarWithETA,  # Progress bar with time estimation
    format_duration,  # Format seconds as human-readable string
    create_progress_bar,  # Factory function
    create_progress_bar_with_metrics,  # Factory with metrics
    create_progress_bar_with_eta,  # Factory with ETA
)

# NOTE: Random utilities removed - use Mojo stdlib 'random' module directly
# Example: from random import random_float64, random_si64, seed

# Profiling utilities
from .profiling import (
    Timer,  # Context manager for timing
    memory_usage,  # Get current memory usage
    profile_function,  # Profile function execution
    benchmark_function,  # Benchmark function
    MemoryStats,  # Memory statistics
    TimingStats,  # Timing statistics
    ProfilingReport,  # Profiling report
)

# Tensor serialization utilities
from .serialization import (
    NamedTensor,  # Named tensor for checkpoint collections
    save_tensor,  # Save single tensor to file
    load_tensor,  # Load tensor from file
    load_tensor_with_name,  # Load tensor with associated name
    save_named_tensors,  # Save collection of named tensors
    load_named_tensors,  # Load collection from directory
    save_named_checkpoint,  # Save checkpoint with named tensors and metadata
    load_named_checkpoint,  # Load checkpoint with named tensors and metadata
    bytes_to_hex,  # Encode bytes to hex string
    hex_to_bytes,  # Decode hex string to bytes
    get_dtype_size,  # Get dtype size in bytes
    parse_dtype,  # Parse dtype string to enum
    dtype_to_string,  # Convert dtype enum to string
)

# Argument parsing utilities
from .arg_parser import (
    ArgumentParser,  # Main argument parser class
    ArgumentSpec,  # Argument specification
    ParsedArgs,  # Parsed arguments container
    create_parser,  # Create new parser
    create_training_parser,  # Create training-specific parser
    validate_positive_int,  # Validate positive integer arguments
    validate_positive_float,  # Validate positive float arguments
    validate_range_float,  # Validate float within range
)

# Training argument utilities
from .training_args import (
    TrainingArgs,  # Training hyperparameters container
    parse_training_args,  # Parse common training arguments
    parse_training_args_with_defaults,  # Parse with custom defaults
)

# Inference utilities
from .inference_utils import (
    InferenceConfig,  # Inference configuration container
    parse_inference_args,  # Parse common inference arguments
    parse_inference_args_with_defaults,  # Parse with custom defaults
    evaluate_accuracy,  # Calculate classification accuracy
    count_correct,  # Count correct predictions for batch processing
)

# ============================================================================
# Public API
# ============================================================================

# All symbols are exported via the imports above.
# Mojo does not support __all__ - exports are controlled by import statements.
