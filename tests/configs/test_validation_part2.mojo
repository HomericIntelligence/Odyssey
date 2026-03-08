"""
Configuration Validation Tests - Part 2: Range and Enum Validation

Tests for validating value ranges and enumerated choices in configurations.
Split from test_validation.mojo per ADR-009 to avoid heap corruption.

Run with: mojo test tests/configs/test_validation_part2.mojo

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config, create_validator


# ============================================================================
# Range Validation Tests
# ============================================================================


fn test_validate_range_valid() raises:
    """Test range validation with valid values.

    Verifies values within range pass validation.
    """
    var config = Config()
    config.set("learning_rate", 0.01)

    # Should be within range [0.0, 1.0]
    config.validate_range("learning_rate", 0.0, 1.0)

    print("✓ test_validate_range_valid passed")


fn test_validate_range_out_of_bounds() raises:
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


fn test_validate_range_boundary_values() raises:
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


fn test_validate_range_int_values() raises:
    """Test range validation with integer values.

    Verifies range checking works for integers.
    """
    var config = Config()
    config.set("batch_size", 64)

    # Should be within range [1, 1024]
    config.validate_range("batch_size", 1.0, 1024.0)

    print("✓ test_validate_range_int_values passed")


# ============================================================================
# Enum Validation Tests
# ============================================================================


fn test_validate_enum_valid_value() raises:
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


fn test_validate_enum_invalid_value() raises:
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


fn test_validate_activation_function() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run range and enum validation tests."""
    print("\n" + "=" * 70)
    print("Running Configuration Validation Tests - Part 2")
    print("=" * 70 + "\n")

    # Range validation tests
    print("Testing Range Validation...")
    test_validate_range_valid()
    test_validate_range_out_of_bounds()
    test_validate_range_boundary_values()
    test_validate_range_int_values()

    # Enum validation tests
    print("\nTesting Enum Validation...")
    test_validate_enum_valid_value()
    test_validate_enum_invalid_value()
    test_validate_activation_function()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Part 2 Configuration Validation Tests Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
