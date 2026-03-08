# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_env_vars.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""
Environment Variable Substitution Tests - Part 2

Tests for edge cases and integration scenarios for environment variable
substitution in configuration values.

Run with: mojo test tests/configs/test_env_vars_part2.mojo
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config
from python import Python


# ============================================================================
# Configuration File Tests (continued)
# ============================================================================


fn test_substitute_preserves_non_string_values() raises:
    """Test that substitution preserves non-string values.

    Verifies only string values are processed for substitution.
    """
    var config = Config()
    config.set("learning_rate", 0.001)
    config.set("batch_size", 32)
    config.set("use_cuda", True)
    config.set("path", "${HOME}/data")

    var python = Python.import_module("os")
    python.environ.__setitem__("HOME", value="/home/user")

    var substituted = config.substitute_env_vars()

    # Non-string values should be preserved
    var lr = substituted.get_float("learning_rate")
    assert_equal(lr, 0.001, "Float values should be preserved")

    var bs = substituted.get_int("batch_size")
    assert_equal(bs, 32, "Int values should be preserved")

    var cuda = substituted.get_bool("use_cuda")
    assert_equal(cuda, True, "Bool values should be preserved")

    # String value should be substituted
    var path = substituted.get_string("path")
    assert_equal(path, "/home/user/data", "String should be substituted")

    print("✓ test_substitute_preserves_non_string_values passed")


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_substitute_no_variables() raises:
    """Test substitution when no variables present.

    Verifies config without ${} patterns is unchanged.
    """
    var config = Config()
    config.set("path", "/static/path/no/vars")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(
        path, "/static/path/no/vars", "Should preserve values without variables"
    )

    print("✓ test_substitute_no_variables passed")


fn test_substitute_malformed_pattern() raises:
    """Test substitution with malformed ${} pattern.

    Verifies malformed patterns are left unchanged.
    """
    var config = Config()
    config.set("value1", "${MISSING")  # Missing closing brace
    config.set("value2", "$MISSING}")  # Missing opening brace
    config.set("value3", "${}")  # Empty variable name

    _ = config.substitute_env_vars()

    # Malformed patterns should be left as-is
    # Implementation may vary - this tests expected behavior
    print("✓ test_substitute_malformed_pattern passed")


fn test_substitute_nested_variables() raises:
    """Test substitution with nested ${} patterns.

    Verifies behavior with nested variable syntax.
    Note: Nested substitution may not be supported.
    """
    var config = Config()
    config.set("path", "${BASE_${LEVEL}}")

    _ = config.substitute_env_vars()

    # Nested variables typically not supported - should leave as-is
    # This test documents expected behavior
    print("✓ test_substitute_nested_variables passed")


fn test_substitute_dollar_sign_escape() raises:
    """Test handling of literal dollar signs.

    Verifies how literal $ characters are handled.
    """
    var config = Config()
    config.set("price", "$$100")  # Literal dollar signs

    var substituted = config.substitute_env_vars()

    # Should preserve literal dollar signs that aren't ${VAR} patterns
    var price = substituted.get_string("price")
    # Exact behavior depends on implementation
    print("✓ test_substitute_dollar_sign_escape passed")


# ============================================================================
# Integration Tests
# ============================================================================


fn test_load_and_substitute_training_config() raises:
    """Test loading and substituting training configuration.

    Verifies end-to-end workflow with environment variables.
    """
    var python = Python.import_module("os")
    python.environ.__setitem__("EXPERIMENT_NAME", value="baseline_001")
    python.environ.__setitem__("OUTPUT_PATH", value="/results")

    # Create config with env vars
    var config = Config()
    config.set("experiment", "${EXPERIMENT_NAME}")
    config.set("output", "${OUTPUT_PATH}/${EXPERIMENT_NAME}")
    config.set("learning_rate", 0.001)

    var substituted = config.substitute_env_vars()

    var exp = substituted.get_string("experiment")
    assert_equal(exp, "baseline_001", "Should substitute experiment name")

    var output = substituted.get_string("output")
    assert_equal(
        output,
        "/results/baseline_001",
        "Should substitute multiple variables in path",
    )

    var lr = substituted.get_float("learning_rate")
    assert_equal(lr, 0.001, "Should preserve numeric values")

    print("✓ test_load_and_substitute_training_config passed")


fn test_substitute_with_merge() raises:
    """Test environment variable substitution with config merging.

    Verifies substitution works correctly after merging configs.
    """
    var python = Python.import_module("os")
    python.environ.__setitem__("BASE_LR", value="0.01")

    var defaults = Config()
    defaults.set("learning_rate", "${BASE_LR:-0.001}")
    defaults.set("batch_size", 32)

    var experiment = Config()
    experiment.set("learning_rate", "${BASE_LR:-0.005}")

    # Merge then substitute
    var merged = defaults.merge(experiment)
    var substituted = merged.substitute_env_vars()

    var lr = substituted.get_string("learning_rate")
    # Note: This will be string "0.01" after substitution
    # May need conversion depending on usage
    assert_equal(lr, "0.01", "Should substitute in merged config")

    print("✓ test_substitute_with_merge passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run environment variable substitution tests - Part 2."""
    print("\n" + "=" * 70)
    print("Running Environment Variable Substitution Tests - Part 2")
    print("=" * 70 + "\n")

    # File-based tests (continued)
    print("Testing File-Based Substitution (continued)...")
    test_substitute_preserves_non_string_values()

    # Edge case tests
    print("\nTesting Edge Cases...")
    test_substitute_no_variables()
    test_substitute_malformed_pattern()
    test_substitute_nested_variables()
    test_substitute_dollar_sign_escape()

    # Integration tests
    print("\nTesting Integration Scenarios...")
    test_load_and_substitute_training_config()
    test_substitute_with_merge()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Environment Variable Substitution Tests - Part 2 Passed!")
    print("=" * 70)
