"""
Configuration Validation Tests - Part 1: Required Key and Type Validation

Tests for validating required keys and value types in configurations.
Split from test_validation.mojo per ADR-009 to avoid heap corruption.

Run with: mojo test tests/configs/test_validation_part1.mojo

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config, create_validator


# ============================================================================
# Required Key Validation Tests
# ============================================================================


fn test_validate_required_keys() raises:
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


fn test_validate_missing_required_key() raises:
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


fn test_validate_training_config_required_fields() raises:
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


# ============================================================================
# Type Validation Tests
# ============================================================================


fn test_validate_type_string() raises:
    """Test string type validation.

    Verifies type checking for string values.
    """
    var config = Config()
    config.set("optimizer", "sgd")

    # Should validate as string
    config.validate_type("optimizer", "string")

    print("✓ test_validate_type_string passed")


fn test_validate_type_int() raises:
    """Test integer type validation.

    Verifies type checking for integer values.
    """
    var config = Config()
    config.set("batch_size", 32)

    # Should validate as int
    config.validate_type("batch_size", "int")

    print("✓ test_validate_type_int passed")


fn test_validate_type_float() raises:
    """Test float type validation.

    Verifies type checking for float values.
    """
    var config = Config()
    config.set("learning_rate", 0.001)

    # Should validate as float
    config.validate_type("learning_rate", "float")

    print("✓ test_validate_type_float passed")


fn test_validate_type_bool() raises:
    """Test boolean type validation.

    Verifies type checking for boolean values.
    """
    var config = Config()
    config.set("use_cuda", True)

    # Should validate as bool
    config.validate_type("use_cuda", "bool")

    print("✓ test_validate_type_bool passed")


fn test_validate_type_mismatch() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run required key and type validation tests."""
    print("\n" + "=" * 70)
    print("Running Configuration Validation Tests - Part 1")
    print("=" * 70 + "\n")

    # Required key tests
    print("Testing Required Key Validation...")
    test_validate_required_keys()
    test_validate_missing_required_key()
    test_validate_training_config_required_fields()

    # Type validation tests
    print("\nTesting Type Validation...")
    test_validate_type_string()
    test_validate_type_int()
    test_validate_type_float()
    test_validate_type_bool()
    test_validate_type_mismatch()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Part 1 Configuration Validation Tests Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
