"""
Configuration Validation Tests - Part 3: Exclusive, Complex, and Validator Tests

Tests for mutual exclusivity validation, complex scenarios, and validator builder.
Split from test_validation.mojo per ADR-009 to avoid heap corruption.

Run with: mojo test tests/configs/test_validation_part3.mojo

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config, create_validator


# ============================================================================
# Mutual Exclusivity Validation Tests
# ============================================================================


fn test_validate_exclusive_none_set() raises:
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


fn test_validate_exclusive_one_set() raises:
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


fn test_validate_exclusive_multiple_set() raises:
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


# ============================================================================
# Complex Validation Tests
# ============================================================================


fn test_validate_complete_training_config() raises:
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


fn test_validate_invalid_training_config() raises:
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


fn test_validate_model_config() raises:
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


# ============================================================================
# Validator Builder Tests
# ============================================================================


fn test_create_validator() raises:
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


fn test_validator_with_requirements() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run exclusive, complex, and validator builder tests."""
    print("\n" + "=" * 70)
    print("Running Configuration Validation Tests - Part 3")
    print("=" * 70 + "\n")

    # Exclusive validation tests
    print("Testing Mutual Exclusivity Validation...")
    test_validate_exclusive_none_set()
    test_validate_exclusive_one_set()
    test_validate_exclusive_multiple_set()

    # Complex validation tests
    print("\nTesting Complex Validation Scenarios...")
    test_validate_complete_training_config()
    test_validate_invalid_training_config()
    test_validate_model_config()

    # Validator builder tests
    print("\nTesting Validator Builder...")
    test_create_validator()
    test_validator_with_requirements()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Part 3 Configuration Validation Tests Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
