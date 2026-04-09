"""Configuration fixtures for testing.

This module provides sample configuration strings (YAML/JSON) for testing
configuration parsing, validation, and error handling.

Key functions:
- valid_yaml_config(): Valid YAML configuration
- valid_json_config(): Valid JSON configuration
- invalid_config_*(): Various invalid configurations for error testing
- Configuration validation helpers

Use these fixtures to test config loading without managing test files.
"""


# ============================================================================
# Valid Configuration Fixtures
# ============================================================================


def valid_yaml_config() -> String:
    """Get valid YAML configuration for testing.

    Returns:
        String containing valid YAML configuration.

    Example:
        ```mojo
        var yaml = valid_yaml_config()
        # Test YAML parsing
        var config = parse_yaml(yaml)
        assert config.get("model", "name") == "TestModel"
        ```

    Note:
        This is a minimal but complete configuration covering common fields.
    """
    return """# Test Configuration
model:
  name: TestModel
  type: mlp
  input_dim: 784
  hidden_dims:
    - 256
    - 128
  output_dim: 10
  activation: relu
  use_dropout: false

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  optimizer: sgd
  momentum: 0.9
  weight_decay: 0.0001

data:
  dataset: mnist
  data_dir: ./data
  num_workers: 4
  shuffle: true
  validation_split: 0.2

logging:
  log_dir: ./logs
  log_level: INFO
  save_frequency: 5
  print_frequency: 100
"""


def valid_json_config() -> String:
    """Get valid JSON configuration for testing.

    Returns:
        String containing valid JSON configuration.

    Example:
        ```mojo
        var json = valid_json_config()
        # Test JSON parsing
        var config = parse_json(json)
        ```
    """
    return """{
  "model": {
    "name": "TestModel",
    "type": "mlp",
    "input_dim": 784,
    "hidden_dims": [256, 128],
    "output_dim": 10,
    "activation": "relu",
    "use_dropout": false
  },
  "training": {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0001
  },
  "data": {
    "dataset": "mnist",
    "data_dir": "./data",
    "num_workers": 4,
    "shuffle": true,
    "validation_split": 0.2
  },
  "logging": {
    "log_dir": "./logs",
    "log_level": "INFO",
    "save_frequency": 5,
    "print_frequency": 100
  }
}"""


def minimal_valid_config() -> String:
    """Get minimal valid configuration.

    Returns:
        Minimal configuration with only required fields.

    Example:
        ```mojo
        var config = minimal_valid_config()
        # Test that minimal config is accepted
        ```

    Note:
        Useful for testing required field validation and defaults.
    """
    return """model:
  name: MinimalModel
  input_dim: 10
  output_dim: 5

training:
  batch_size: 32
  epochs: 1
"""


# ============================================================================
# Invalid Configuration Fixtures (for error testing)
# ============================================================================


def invalid_config_missing_fields() -> String:
    """Get configuration with missing required fields.

    Returns:
        Configuration missing required fields.

    Example:
        ```mojo
        var config = invalid_config_missing_fields()
        # Test error handling
        try:
            parse_and_validate_config(config)
            # Should not reach here
        except:
            # Expected error for missing fields
            pass
        ```
    """
    return """model:
  name: IncompleteModel
  # Missing input_dim and output_dim

training:
  # Missing batch_size
  epochs: 10
"""


def invalid_config_wrong_types() -> String:
    """Get configuration with wrong field types.

    Returns:
        Configuration with type errors.

    Example:
        ```mojo
        var config = invalid_config_wrong_types()
        # Test type validation
        ```
    """
    return """model:
  name: BadTypesModel
  input_dim: "not_a_number"  # Should be Int
  output_dim: 10

training:
  batch_size: 32.5  # Should be Int
  epochs: "ten"  # Should be Int
  learning_rate: "high"  # Should be Float
"""


def invalid_config_negative_values() -> String:
    """Get configuration with invalid negative values.

    Returns:
        Configuration with negative values where positive required.

    Example:
        ```mojo
        var config = invalid_config_negative_values()
        # Test value range validation
        ```
    """
    return """model:
  name: NegativeValuesModel
  input_dim: -10  # Cannot be negative
  output_dim: 5

training:
  batch_size: -32  # Cannot be negative
  epochs: 10
  learning_rate: -0.001  # Cannot be negative
"""


def invalid_config_out_of_range() -> String:
    """Get configuration with out-of-range values.

    Returns:
        Configuration with values outside valid ranges.

    Example:
        ```mojo
        var config = invalid_config_out_of_range()
        # Test range validation
        ```
    """
    return """model:
  name: OutOfRangeModel
  input_dim: 10
  output_dim: 5

training:
  batch_size: 1000000  # Unreasonably large
  epochs: 10
  learning_rate: 100.0  # Too high for learning rate
  momentum: 1.5  # Should be in [0, 1]
"""


def invalid_yaml_syntax() -> String:
    """Get string with invalid YAML syntax.

    Returns:
        String with YAML syntax errors.

    Example:
        ```mojo
        var bad_yaml = invalid_yaml_syntax()
        try:
            parse_yaml(bad_yaml)
            # Should fail
        except:
            # Expected parse error
            pass
        ```
    """
    return """model:
  name: BadSyntaxModel
  input_dim: 10
    output_dim: 5  # Invalid indentation
  training:
  batch_size 32  # Missing colon
epochs: 10
"""


def invalid_json_syntax() -> String:
    """Get string with invalid JSON syntax.

    Returns:
        String with JSON syntax errors.

    Example:
        ```mojo
        var bad_json = invalid_json_syntax()
        try:
            parse_json(bad_json)
            # Should fail
        except:
            # Expected parse error
            pass
        ```
    """
    return """{
  "model": {
    "name": "BadSyntaxModel",
    "input_dim": 10,
    "output_dim": 5,  # Trailing comma before closing brace
  },
  "training": {
    "batch_size": 32
    "epochs": 10  # Missing comma
  }
}"""


# ============================================================================
# Configuration Templates
# ============================================================================


def config_template_classification() -> String:
    """Get configuration template for classification task.

    Returns:
        Configuration optimized for classification.

    Example:
        ```mojo
        var config = config_template_classification()
        # Use as starting point for classification tests
        ```
    """
    return """model:
  name: ClassificationModel
  type: mlp
  input_dim: 784
  hidden_dims: [256, 128]
  output_dim: 10
  activation: relu
  use_softmax: true

training:
  batch_size: 64
  epochs: 20
  learning_rate: 0.01
  optimizer: sgd
  loss: cross_entropy

data:
  dataset: mnist
  num_classes: 10
  normalize: true
"""


def config_template_regression() -> String:
    """Get configuration template for regression task.

    Returns:
        Configuration optimized for regression.

    Example:
        ```mojo
        var config = config_template_regression()
        # Use for regression tests
        ```
    """
    return """model:
  name: RegressionModel
  type: mlp
  input_dim: 20
  hidden_dims: [64, 32]
  output_dim: 1
  activation: relu
  use_softmax: false

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  loss: mse

data:
  normalize: true
  standardize: true
"""


# ============================================================================
# Configuration Validation Helpers
# ============================================================================


def has_required_fields(config_str: String) -> Bool:
    """Check if configuration string has required fields.

    Args:
        config_str: Configuration string to check.

    Returns:
        True if all required fields are present, False otherwise.

    Example:
        ```mojo
        var valid = has_required_fields(valid_yaml_config())  # True
        var invalid = has_required_fields(invalid_config_missing_fields())  # False
        ```

    Note:
        This is a simple string-based check, not full parsing.
        For production use, parse and validate properly.
    """
    # Check for required top-level sections
    var has_model = config_str.find("model:") != -1
    var has_training = config_str.find("training:") != -1

    if not (has_model and has_training):
        return False

    # Check for required model fields
    var has_name = config_str.find("name:") != -1
    var has_input_dim = config_str.find("input_dim:") != -1
    var has_output_dim = config_str.find("output_dim:") != -1

    if not (has_name and has_input_dim and has_output_dim):
        return False

    # Check for required training fields
    var has_batch_size = config_str.find("batch_size:") != -1
    var has_epochs = config_str.find("epochs:") != -1

    return has_batch_size and has_epochs


def is_valid_yaml_syntax(config_str: String) -> Bool:
    """Quick check if string looks like valid YAML.

    Args:
        config_str: String to check.

    Returns:
        True if basic YAML syntax appears valid, False otherwise.

    Example:
        ```mojo
        var valid = is_valid_yaml_syntax(valid_yaml_config())  # True
        var invalid = is_valid_yaml_syntax(invalid_yaml_syntax())  # False
        ```

    Note:
        This is NOT a full YAML parser - just basic sanity checking.
        For production use, use a proper YAML parser.
    """
    # Basic checks for common YAML syntax errors
    var lines = config_str.split("\n")

    for line in lines:
        var stripped = line[].strip()

        # Skip empty lines and comments
        if len(stripped) == 0 or stripped.startswith("#"):
            continue

        # Check for key-value pairs
        if ":" in stripped:
            # Make sure colon is not at the end (except for nested objects)
            var colon_idx = stripped.find(":")
            if colon_idx != -1:
                # Basic check - should have content before colon
                if colon_idx == 0:
                    return False

    return True


def is_valid_json_syntax(config_str: String) -> Bool:
    """Quick check if string looks like valid JSON.

    Args:
        config_str: String to check.

    Returns:
        True if basic JSON syntax appears valid, False otherwise.

    Example:
        ```mojo
        var valid = is_valid_json_syntax(valid_json_config())  # True
        var invalid = is_valid_json_syntax(invalid_json_syntax())  # False
        ```

    Note:
        This is NOT a full JSON parser - just basic sanity checking.
    """
    var trimmed = config_str.strip()

    # JSON should start with { or [
    if not (trimmed.startswith("{") or trimmed.startswith("[")):
        return False

    # JSON should end with } or ]
    if not (trimmed.endswith("}") or trimmed.endswith("]")):
        return False

    # Count braces and brackets
    var open_braces = 0
    var close_braces = 0
    var open_brackets = 0
    var close_brackets = 0

    for i in range(len(trimmed)):
        if trimmed[i] == "{":
            open_braces += 1
        elif trimmed[i] == "}":
            close_braces += 1
        elif trimmed[i] == "[":
            open_brackets += 1
        elif trimmed[i] == "]":
            close_brackets += 1

    # Braces and brackets should match
    return (open_braces == close_braces) and (open_brackets == close_brackets)
