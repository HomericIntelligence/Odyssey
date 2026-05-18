"""
Configuration Validation Tests.

Tests for validating required keys and value types in configurations.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from std.testing import assert_true, assert_false, assert_equal
from projectodyssey.utils.config import Config, load_config, create_validator


def test_validate_required_keys() raises:
    """Test validation of required configuration keys.

    Verifies that validation catches missing required fields.
    """
    var config = Config()
    config.set("learning_rate", 0.001)
    config.set("batch_size", 32)

    var required_keys = List[String]()
    required_keys.append("learning_rate")
    required_keys.append("batch_size")

    # Should not raise - all required keys present
    config.validate(required_keys)

    print("✓ test_validate_required_keys passed")


def test_validate_missing_required_key() raises:
    """Test validation fails when required key is missing.

    Verifies that missing required fields are detected.
    """
    var config = Config()
    config.set("learning_rate", 0.001)
    # Missing batch_size

    var required_keys = List[String]()
    required_keys.append("learning_rate")
    required_keys.append("batch_size")

    var error_raised = False
    try:
        config.validate(required_keys)
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for missing required key")

    print("✓ test_validate_missing_required_key passed")


def test_validate_training_config_required_fields() raises:
    """Test training configuration has all required fields.

    Verifies default training config meets requirements.
    """
    var config = load_config("configs/defaults/training.yaml")

    var required = List[String]()
    required.append("optimizer.name")
    required.append("optimizer.learning_rate")

    # Should validate successfully
    config.validate(required)

    print("✓ test_validate_training_config_required_fields passed")


def test_validate_type_string() raises:
    """Test string type validation.

    Verifies type checking for string values.
    """
    var config = Config()
    config.set("optimizer", "sgd")

    # Should validate as string
    config.validate_type("optimizer", "string")

    print("✓ test_validate_type_string passed")


def test_validate_type_int() raises:
    """Test integer type validation.

    Verifies type checking for integer values.
    """
    var config = Config()
    config.set("batch_size", 32)

    # Should validate as int
    config.validate_type("batch_size", "int")

    print("✓ test_validate_type_int passed")


def test_validate_type_float() raises:
    """Test float type validation.

    Verifies type checking for float values.
    """
    var config = Config()
    config.set("learning_rate", 0.001)

    # Should validate as float
    config.validate_type("learning_rate", "float")

    print("✓ test_validate_type_float passed")


def test_validate_type_bool() raises:
    """Test boolean type validation.

    Verifies type checking for boolean values.
    """
    var config = Config()
    config.set("use_cuda", True)

    # Should validate as bool
    config.validate_type("use_cuda", "bool")

    print("✓ test_validate_type_bool passed")


def test_validate_type_mismatch() raises:
    """Test type validation catches type mismatches.

    Verifies that wrong types are detected.
    """
    var config = Config()
    config.set("learning_rate", 0.001)  # Float

    var error_raised = False
    try:
        config.validate_type("learning_rate", "int")  # Expect int
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for type mismatch")

    print("✓ test_validate_type_mismatch passed")


def test_validate_range_valid() raises:
    """Test range validation with valid values.

    Verifies values within range pass validation.
    """
    var config = Config()
    config.set("learning_rate", 0.01)

    # Should be within range [0.0, 1.0]
    config.validate_range("learning_rate", 0.0, 1.0)

    print("✓ test_validate_range_valid passed")


def test_validate_range_out_of_bounds() raises:
    """Test range validation catches out-of-range values.

    Verifies values outside range are rejected.
    """
    var config = Config()
    config.set("learning_rate", -0.001)  # Negative

    var error_raised = False
    try:
        config.validate_range("learning_rate", 0.0, 1.0)
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for negative learning rate")

    print("✓ test_validate_range_out_of_bounds passed")


def test_validate_range_boundary_values() raises:
    """Test range validation with boundary values.

    Verifies that boundary values are accepted.
    """
    var config = Config()

    # Test lower boundary
    config.set("value", 0.0)
    config.validate_range("value", 0.0, 1.0)

    # Test upper boundary
    config.set("value", 1.0)
    config.validate_range("value", 0.0, 1.0)

    print("✓ test_validate_range_boundary_values passed")


def test_validate_range_int_values() raises:
    """Test range validation with integer values.

    Verifies range checking works for integers.
    """
    var config = Config()
    config.set("batch_size", 64)

    # Should be within range [1, 1024]
    config.validate_range("batch_size", 1.0, 1024.0)

    print("✓ test_validate_range_int_values passed")


def test_validate_enum_valid_value() raises:
    """Test enum validation with valid values.

    Verifies allowed values pass validation.
    """
    var config = Config()
    config.set("optimizer", "sgd")

    var valid_optimizers = List[String]()
    valid_optimizers.append("sgd")
    valid_optimizers.append("adam")
    valid_optimizers.append("rmsprop")

    config.validate_enum("optimizer", valid_optimizers)

    print("✓ test_validate_enum_valid_value passed")


def test_validate_enum_invalid_value() raises:
    """Test enum validation catches invalid values.

    Verifies disallowed values are rejected.
    """
    var config = Config()
    config.set("optimizer", "invalid_optimizer")

    var valid_optimizers = List[String]()
    valid_optimizers.append("sgd")
    valid_optimizers.append("adam")
    valid_optimizers.append("rmsprop")

    var error_raised = False
    try:
        config.validate_enum("optimizer", valid_optimizers)
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for invalid optimizer")

    print("✓ test_validate_enum_invalid_value passed")


def test_validate_activation_function() raises:
    """Test validation of activation function choices.

    Verifies activation function enum validation.
    """
    var config = Config()
    config.set("activation", "relu")

    var valid_activations = List[String]()
    valid_activations.append("relu")
    valid_activations.append("sigmoid")
    valid_activations.append("tanh")
    valid_activations.append("leaky_relu")

    config.validate_enum("activation", valid_activations)

    print("✓ test_validate_activation_function passed")


def test_validate_exclusive_none_set() raises:
    """Test exclusive validation when no keys are set.

    Verifies that having none of the exclusive keys is valid.
    """
    var config = Config()
    config.set("other", "value")

    var exclusive_keys = List[String]()
    exclusive_keys.append("option_a")
    exclusive_keys.append("option_b")

    # Should not raise - no exclusive keys present
    config.validate_exclusive(exclusive_keys)

    print("✓ test_validate_exclusive_none_set passed")


def test_validate_exclusive_one_set() raises:
    """Test exclusive validation when one key is set.

    Verifies that having exactly one exclusive key is valid.
    """
    var config = Config()
    config.set("option_a", "value")

    var exclusive_keys = List[String]()
    exclusive_keys.append("option_a")
    exclusive_keys.append("option_b")

    # Should not raise - only one exclusive key present
    config.validate_exclusive(exclusive_keys)

    print("✓ test_validate_exclusive_one_set passed")


def test_validate_exclusive_multiple_set() raises:
    """Test exclusive validation catches multiple exclusive keys.

    Verifies that having multiple exclusive keys is rejected.
    """
    var config = Config()
    config.set("option_a", "value1")
    config.set("option_b", "value2")

    var exclusive_keys = List[String]()
    exclusive_keys.append("option_a")
    exclusive_keys.append("option_b")

    var error_raised = False
    try:
        config.validate_exclusive(exclusive_keys)
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for multiple exclusive keys")

    print("✓ test_validate_exclusive_multiple_set passed")


def test_validate_complete_training_config() raises:
    """Test comprehensive validation of training configuration.

    Verifies all training config requirements are met.
    """
    var config = load_config("tests/configs/fixtures/valid_training.yaml")

    # Validate required fields
    var required = List[String]()
    required.append("optimizer.name")
    required.append("optimizer.learning_rate")
    required.append("training.epochs")
    required.append("training.batch_size")

    config.validate(required)

    # Validate optimizer is valid choice
    var valid_opts = List[String]()
    valid_opts.append("sgd")
    valid_opts.append("adam")
    valid_opts.append("rmsprop")
    config.validate_enum("optimizer.name", valid_opts)

    # Validate learning rate is in reasonable range
    config.validate_range("optimizer.learning_rate", 0.0, 1.0)

    # Validate epochs is positive
    config.validate_range("training.epochs", 1.0, 10000.0)

    # Validate batch size is reasonable
    config.validate_range("training.batch_size", 1.0, 2048.0)

    print("✓ test_validate_complete_training_config passed")


def test_validate_invalid_training_config() raises:
    """Test validation rejects invalid training configuration.

    Verifies that invalid configs are caught.
    """
    var config = load_config("tests/configs/fixtures/invalid_training.yaml")

    # Should have invalid optimizer
    var valid_opts = List[String]()
    valid_opts.append("sgd")
    valid_opts.append("adam")
    valid_opts.append("rmsprop")

    var error_raised = False
    try:
        config.validate_enum("optimizer.name", valid_opts)
    except:
        error_raised = True

    assert_true(error_raised, "Should reject invalid optimizer")

    print("✓ test_validate_invalid_training_config passed")


def test_validate_model_config() raises:
    """Test validation of model configuration.

    Verifies model config meets requirements.
    """
    var config = load_config("configs/papers/lenet5/model.yaml")

    # Validate required model fields
    var required = List[String]()
    required.append("model.name")
    required.append("model.output_classes")

    config.validate(required)

    # Validate output_classes is reasonable
    config.validate_range("model.output_classes", 2.0, 1000.0)

    print("✓ test_validate_model_config passed")


def test_create_validator() raises:
    """Test creating validator with builder pattern.

    Verifies validator construction and usage.
    """
    var validator = create_validator()

    var config = Config()
    config.set("learning_rate", 0.001)

    # Should validate empty validator
    var is_valid = validator.validate(config)
    assert_true(is_valid, "Empty validator should accept any config")

    print("✓ test_create_validator passed")


def test_validator_with_requirements() raises:
    """Test validator with required fields.

    Verifies validator correctly checks requirements.
    """
    var validator = create_validator()
    _ = validator.require("learning_rate")
    _ = validator.require("batch_size")

    var valid_config = Config()
    valid_config.set("learning_rate", 0.001)
    valid_config.set("batch_size", 32)

    var is_valid = validator.validate(valid_config)
    assert_true(is_valid, "Should validate config with all required fields")

    var invalid_config = Config()
    invalid_config.set("learning_rate", 0.001)
    # Missing batch_size

    var is_invalid = validator.validate(invalid_config)
    assert_false(is_invalid, "Should reject config missing required fields")

    print("✓ test_validator_with_requirements passed")


def main() raises:
    """Run all test_validation tests."""
    print("Running test_validation tests...")

    test_validate_required_keys()
    print("✓ test_validate_required_keys")

    test_validate_missing_required_key()
    print("✓ test_validate_missing_required_key")

    test_validate_training_config_required_fields()
    print("✓ test_validate_training_config_required_fields")

    test_validate_type_string()
    print("✓ test_validate_type_string")

    test_validate_type_int()
    print("✓ test_validate_type_int")

    test_validate_type_float()
    print("✓ test_validate_type_float")

    test_validate_type_bool()
    print("✓ test_validate_type_bool")

    test_validate_type_mismatch()
    print("✓ test_validate_type_mismatch")

    test_validate_range_valid()
    print("✓ test_validate_range_valid")

    test_validate_range_out_of_bounds()
    print("✓ test_validate_range_out_of_bounds")

    test_validate_range_boundary_values()
    print("✓ test_validate_range_boundary_values")

    test_validate_range_int_values()
    print("✓ test_validate_range_int_values")

    test_validate_enum_valid_value()
    print("✓ test_validate_enum_valid_value")

    test_validate_enum_invalid_value()
    print("✓ test_validate_enum_invalid_value")

    test_validate_activation_function()
    print("✓ test_validate_activation_function")

    test_validate_exclusive_none_set()
    print("✓ test_validate_exclusive_none_set")

    test_validate_exclusive_one_set()
    print("✓ test_validate_exclusive_one_set")

    test_validate_exclusive_multiple_set()
    print("✓ test_validate_exclusive_multiple_set")

    test_validate_complete_training_config()
    print("✓ test_validate_complete_training_config")

    test_validate_invalid_training_config()
    print("✓ test_validate_invalid_training_config")

    test_validate_model_config()
    print("✓ test_validate_model_config")

    test_create_validator()
    print("✓ test_create_validator")

    test_validator_with_requirements()
    print("✓ test_validator_with_requirements")

    print("\nAll test_validation tests passed!")
