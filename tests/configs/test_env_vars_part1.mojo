# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_env_vars.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""
Environment Variable Substitution Tests - Part 1

Tests for substituting environment variables in configuration values.
Supports ${VAR} and ${VAR:-default} syntax.

Run with: mojo test tests/configs/test_env_vars_part1.mojo
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config
from python import Python


# ============================================================================
# Basic Environment Variable Substitution Tests
# ============================================================================


fn test_substitute_simple_env_var() raises:
    """Test basic environment variable substitution.

    Verifies ${VAR} is replaced with environment value.
    """
    # Set environment variable
    var python = Python.import_module("os")
    python.environ.__setitem__("TEST_VAR", value="test_value")

    var config = Config()
    config.set("path", "${TEST_VAR}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(path, "test_value", "Should substitute environment variable")

    print("✓ test_substitute_simple_env_var passed")


fn test_substitute_multiple_env_vars() raises:
    """Test substitution of multiple environment variables.

    Verifies multiple ${VAR} patterns are replaced.
    """
    var python = Python.import_module("os")
    python.environ.__setitem__("BASE_DIR", value="/home/user")
    python.environ.__setitem__("DATA_FOLDER", value="datasets")

    var config = Config()
    config.set("data_path", "${BASE_DIR}/${DATA_FOLDER}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("data_path")
    assert_equal(
        path,
        "/home/user/datasets",
        "Should substitute multiple environment variables",
    )

    print("✓ test_substitute_multiple_env_vars passed")


fn test_substitute_env_var_in_middle() raises:
    """Test substitution with variable in middle of string.

    Verifies ${VAR} can appear anywhere in value.
    """
    var python = Python.import_module("os")
    python.environ.__setitem__("MODEL_NAME", value="lenet5")

    var config = Config()
    config.set("path", "/models/${MODEL_NAME}/checkpoint.mojo")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(
        path,
        "/models/lenet5/checkpoint.mojo",
        "Should substitute variable in middle of string",
    )

    print("✓ test_substitute_env_var_in_middle passed")


# ============================================================================
# Default Value Syntax Tests
# ============================================================================


fn test_substitute_with_default_value() raises:
    """Test ${VAR:-default} syntax for missing variables.

    Verifies default value is used when variable not set.
    """
    var config = Config()
    config.set("output_dir", "${MISSING_VAR:-/tmp/output}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("output_dir")
    assert_equal(
        path, "/tmp/output", "Should use default value for missing variable"
    )

    print("✓ test_substitute_with_default_value passed")


fn test_substitute_with_default_when_var_exists() raises:
    """Test ${VAR:-default} when variable exists.

    Verifies actual value is used when variable is set.
    """
    var python = Python.import_module("os")
    python.environ.__setitem__("DATA_DIR", value="/actual/data")

    var config = Config()
    config.set("data_path", "${DATA_DIR:-/default/data}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("data_path")
    assert_equal(
        path, "/actual/data", "Should use actual value when variable exists"
    )

    print("✓ test_substitute_with_default_when_var_exists passed")


fn test_substitute_empty_default_value() raises:
    """Test ${VAR:-} syntax with empty default.

    Verifies empty string default is supported.
    """
    var config = Config()
    config.set("optional_param", "${MISSING:-}")

    var substituted = config.substitute_env_vars()

    var param = substituted.get_string("optional_param")
    assert_equal(param, "", "Should use empty string as default")

    print("✓ test_substitute_empty_default_value passed")


fn test_substitute_complex_default_value() raises:
    """Test ${VAR:-default} with complex default value.

    Verifies default can contain special characters.
    """
    var config = Config()
    config.set("path", "${MISSING:-/path/with-dashes_and.dots}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(
        path,
        "/path/with-dashes_and.dots",
        "Should handle complex default values",
    )

    print("✓ test_substitute_complex_default_value passed")


# ============================================================================
# Configuration File Tests
# ============================================================================


fn test_substitute_from_file() raises:
    """Test substitution from configuration file.

    Verifies environment variables in YAML files are substituted.
    """
    var python = Python.import_module("os")
    python.environ.__setitem__("DATA_DIR", value="/actual/data")

    var config = load_config("tests/configs/fixtures/env_vars.yaml")
    var substituted = config.substitute_env_vars()

    # data_dir should be substituted
    if substituted.has("data_dir"):
        var data_dir = substituted.get_string("data_dir")
        assert_equal(data_dir, "/actual/data", "Should substitute DATA_DIR")

    # output_dir should use default (MISSING env var)
    if substituted.has("output_dir"):
        var output_dir = substituted.get_string("output_dir")
        assert_equal(
            output_dir, "/tmp/output", "Should use default for OUTPUT_DIR"
        )

    print("✓ test_substitute_from_file passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run environment variable substitution tests - Part 1."""
    print("\n" + "=" * 70)
    print("Running Environment Variable Substitution Tests - Part 1")
    print("=" * 70 + "\n")

    # Basic substitution tests
    print("Testing Basic Substitution...")
    test_substitute_simple_env_var()
    test_substitute_multiple_env_vars()
    test_substitute_env_var_in_middle()

    # Default value tests
    print("\nTesting Default Value Syntax...")
    test_substitute_with_default_value()
    test_substitute_with_default_when_var_exists()
    test_substitute_empty_default_value()
    test_substitute_complex_default_value()

    # File-based tests
    print("\nTesting File-Based Substitution...")
    test_substitute_from_file()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Environment Variable Substitution Tests - Part 1 Passed!")
    print("=" * 70)
