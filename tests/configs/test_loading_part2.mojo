"""
Configuration Loading Tests - Part 2

Tests for JSON format, error handling, and complex configuration loading.
Split from test_loading.mojo per ADR-009.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_loading.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Run with: mojo test tests/configs/test_loading_part2.mojo
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config


# ============================================================================
# File Format Tests
# ============================================================================


fn test_load_json_config() raises:
    """Test loading JSON configuration file.

    Verifies JSON parsing works correctly.
    """
    # Create a simple JSON config for testing
    var config = Config()
    config.set("learning_rate", 0.001)
    config.set("batch_size", 32)
    config.to_json("tests/configs/fixtures/test_output.json")

    # Load it back
    var loaded = load_config("tests/configs/fixtures/test_output.json")

    assert_true(loaded.has("learning_rate"), "Should have learning_rate")
    assert_true(loaded.has("batch_size"), "Should have batch_size")

    print("✓ test_load_json_config passed")


# ============================================================================
# Error Handling Tests
# ============================================================================


fn test_load_missing_file() raises:
    """Test loading non-existent configuration file.

    Verifies proper error handling for missing files.
    """
    var error_raised = False

    try:
        _ = load_config("configs/nonexistent/file.yaml")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for missing file")

    print("✓ test_load_missing_file passed")


fn test_load_empty_file() raises:
    """Test loading empty configuration file.

    Verifies proper error handling for empty files.
    """
    # Create empty file
    with open("tests/configs/fixtures/empty.yaml", "w") as f:
        _ = f.write("")

    var error_raised = False
    try:
        _ = load_config("tests/configs/fixtures/empty.yaml")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for empty file")

    print("✓ test_load_empty_file passed")


fn test_load_invalid_format() raises:
    """Test loading file with invalid format.

    Verifies proper error handling for unsupported formats.
    """
    var error_raised = False

    try:
        _ = load_config("configs/test.txt")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for unsupported format")

    print("✓ test_load_invalid_format passed")


# ============================================================================
# Complex Configuration Tests
# ============================================================================


fn test_load_complex_nested_config() raises:
    """Test loading complex nested configuration.

    Verifies that nested structures are properly parsed.
    Note: Current implementation has limitations with nested structures.
    """
    var config = load_config("tests/configs/fixtures/complex.yaml")

    # Should load successfully even with nested structure
    # Nested access may be limited until full YAML parsing implemented
    assert_true(len(config.data) > 0, "Complex config should not be empty")

    print("✓ test_load_complex_nested_config passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run configuration loading tests - part 2."""
    print("\n" + "=" * 70)
    print("Running Configuration Loading Tests - Part 2")
    print("=" * 70 + "\n")

    # File format tests
    print("Testing File Format Support...")
    test_load_json_config()

    # Error handling tests
    print("\nTesting Error Handling...")
    test_load_missing_file()
    test_load_empty_file()
    test_load_invalid_format()

    # Complex config tests
    print("\nTesting Complex Configurations...")
    test_load_complex_nested_config()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Configuration Loading Tests - Part 2 Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
