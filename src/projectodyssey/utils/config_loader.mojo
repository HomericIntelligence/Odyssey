"""Configuration loading utilities for ML Odyssey.

This module provides high-level functions for loading experiment configurations
with the standard three-tier merge pattern: defaults → paper → experiment.

Example:
    ```
    from projectodyssey.utils.config_loader import load_experiment_config

    var config = load_experiment_config("lenet5", "baseline")
    var lr = config.get_float("optimizer.learning_rate")
    var batch_size = config.get_int("training.batch_size")
    ```
"""

from projectodyssey.utils.config import Config, load_config, merge_configs
from std.python import Python


def load_default_config(config_type: String) raises -> Config:
    """Load a default configuration file.

    Args:
            config_type: Type of config to load (e.g., "training", "model", "data", "paths").

    Returns:
            Loaded default configuration.

    Raises:
            Error if config file not found or invalid.

        Example:
            ```mojo
            var training_defaults = load_default_config("training")
            var lr = training_defaults.get_float("optimizer.learning_rate")
            ```
    """
    var filepath = "configs/defaults/" + config_type + ".yaml"
    return load_config(filepath)


def load_paper_config(
    paper_name: String, config_type: String = "training"
) raises -> Config:
    """Load paper configuration with defaults merged in.

        Merges: defaults/{config_type}.yaml → papers/{paper_name}/{config_type}.yaml

    Args:
            paper_name: Name of the paper (e.g., "lenet5", "alexnet").
            config_type: Type of config to load (default: "training").

    Returns:
            Merged configuration with paper overrides.

    Raises:
            Error if config files not found or invalid.

        Example:
            ```mojo
            var lenet5_config = load_paper_config("lenet5", "model")
            var num_classes = lenet5_config.get_int("num_classes")
            ```
    """
    # Load defaults
    var defaults = Config()
    try:
        defaults = load_default_config(config_type)
    except e:
        # If no defaults exist, start with empty config
        pass

    # Load paper-specific config
    var paper_filepath = (
        "configs/papers/" + paper_name + "/" + config_type + ".yaml"
    )
    var paper_config = load_config(paper_filepath)

    # Merge: defaults → paper
    var result = merge_configs(defaults, paper_config)

    return result


def load_experiment_config(
    paper_name: String, experiment_name: String
) raises -> Config:
    """Load complete configuration for an experiment.

        Implements the three-tier merge pattern:
        1. Load defaults (training, model, data)
        2. Load paper-specific configs
        3. Load experiment config
        4. Merge: defaults → paper → experiment
        5. Substitute environment variables

    Args:
            paper_name: Name of the paper (e.g., "lenet5").
            experiment_name: Name of the experiment (e.g., "baseline", "augmented").

    Returns:
            Complete merged and validated configuration.

    Raises:
            Error if required config files not found or invalid.

        Example:
            ```mojo
            var config = load_experiment_config("lenet5", "baseline")
            var lr = config.get_float("optimizer.learning_rate")
            var batch_size = config.get_int("training.batch_size")
            var model_name = config.get_string("model.name")
            ```
    """
    # Step 1: Load all defaults (training, model, data)
    var config = Config()

    # Try to load each type of default config (graceful if missing)
    var default_types = List[String]()
    default_types.append("training")
    default_types.append("model")
    default_types.append("data")
    default_types.append("paths")

    for i in range(len(default_types)):
        var config_type = default_types[i]
        try:
            var default_config = load_default_config(config_type)
            config = merge_configs(config, default_config)
        except e:
            # Skip if default doesn't exist
            pass

    # Step 2: Load paper-specific configs
    for i in range(len(default_types)):
        var config_type = default_types[i]
        try:
            var paper_filepath = (
                "configs/papers/" + paper_name + "/" + config_type + ".yaml"
            )
            var paper_config = load_config(paper_filepath)
            config = merge_configs(config, paper_config)
        except e:
            # Skip if paper config doesn't exist
            pass

    # Step 3: Load experiment-specific config
    var exp_filepath = (
        "configs/experiments/" + paper_name + "/" + experiment_name + ".yaml"
    )
    var exp_config = load_config(exp_filepath)
    config = merge_configs(config, exp_config)

    # Step 4: Substitute environment variables
    config = config.substitute_env_vars()

    return config


def load_config_with_validation(
    filepath: String, required_keys: List[String]
) raises -> Config:
    """Load configuration with validation of required keys.

    Args:
            filepath: Path to configuration file.
            required_keys: List of required configuration keys.

    Returns:
            Loaded and validated configuration.

    Raises:
            Error if file not found, invalid, or missing required keys.

        Example:
            ```mojo
            var required = List[String]()
            required.append("optimizer.learning_rate")
            required.append("training.epochs")

            var config = load_config_with_validation(
                "configs/experiments/lenet5/baseline.yaml",
                required
            )
            ```
    """
    var config = load_config(filepath)
    config.validate(required_keys)
    return config


def create_experiment_config(
    paper_name: String, experiment_name: String, overrides: Config
) raises:
    """Create a new experiment configuration with overrides.

        Creates a new experiment config file that extends the paper config
        with specified overrides.

    Args:
            paper_name: Name of the paper.
            experiment_name: Name of the new experiment.
            overrides: Configuration overrides to apply.

    Raises:
            Error if experiment already exists or file cannot be written.

        Example:
            ```mojo
            var overrides = Config()
            overrides.set("optimizer.learning_rate", 0.01)
            overrides.set("training.batch_size", 64)

            create_experiment_config("lenet5", "high_lr", overrides)
            ```
    """
    # Create experiment config from paper config + overrides
    var paper_config = load_paper_config(paper_name, "training")
    var exp_config = merge_configs(paper_config, overrides)

    # Save to experiment directory
    var exp_filepath = (
        "configs/experiments/" + paper_name + "/" + experiment_name + ".yaml"
    )
    exp_config.to_yaml(exp_filepath)


def validate_experiment_config(config: Config) raises:
    """Validate that an experiment configuration has all required fields.

        Checks for common required fields across training, model, and data configs.

    Args:
            config: Configuration to validate.

    Raises:
            Error if any required field is missing or invalid.

        Example:
            ```mojo
            var config = load_experiment_config("lenet5", "baseline")
            validate_experiment_config(config)  # Raises if invalid
            ```
    """
    # Define required fields
    var required = List[String]()

    # Training requirements
    required.append("optimizer.name")
    required.append("optimizer.learning_rate")
    required.append("training.epochs")
    required.append("training.batch_size")

    # Validate presence of required fields
    config.validate(required)

    # Validate types
    config.validate_type("optimizer.name", "string")
    config.validate_type("optimizer.learning_rate", "float")
    config.validate_type("training.epochs", "int")
    config.validate_type("training.batch_size", "int")

    # Validate ranges
    config.validate_range("optimizer.learning_rate", 0.0, 1.0)
    config.validate_range("training.epochs", 1.0, 10000.0)
    config.validate_range("training.batch_size", 1.0, 10000.0)

    # Validate optimizer name
    var valid_optimizers = List[String]()
    valid_optimizers.append("sgd")
    valid_optimizers.append("adam")
    valid_optimizers.append("adamw")
    valid_optimizers.append("rmsprop")
    config.validate_enum("optimizer.name", valid_optimizers)
