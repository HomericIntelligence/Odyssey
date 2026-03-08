"""Tests for configuration management module - Part 1: Loading.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_config.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

This module tests configuration loading functionality including:
- YAML/JSON configuration file loading
- Nested configuration loading
- List value loading
- Error handling for missing/malformed files
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)


# ============================================================================
# Test Configuration Loading
# ============================================================================


fn test_load_yaml_config():
    """Test loading configuration from YAML file."""
    # TODO(#44): Implement when Config.from_yaml exists
    # Create temp YAML file with:
    # learning_rate: 0.001
    # batch_size: 32
    # epochs: 10
    # Load config
    # Verify values are parsed correctly
    # Clean up temp file
    pass


fn test_load_json_config():
    """Test loading configuration from JSON file."""
    # TODO(#44): Implement when Config.from_json exists
    # Create temp JSON file with:
    # {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
    # Load config
    # Verify values are parsed correctly
    # Clean up temp file
    pass


fn test_load_nested_config():
    """Test loading configuration with nested sections."""
    # TODO(#44): Implement when Config supports nested dicts
    # YAML:
    # model:
    #   layers: [64, 32, 10]
    #   activation: "relu"
    # optimizer:
    #   name: "sgd"
    #   lr: 0.01
    # Verify nested access: config.model.layers
    pass


fn test_load_config_with_lists():
    """Test loading configuration with list values."""
    # TODO(#44): Implement when Config supports lists
    # YAML:
    # layer_sizes: [64, 32, 10]
    # dropout_rates: [0.5, 0.3, 0.1]
    # Verify list values are parsed correctly
    pass


fn test_load_nonexistent_file():
    """Test loading nonexistent config file raises error."""
    # TODO(#44): Implement when Config.from_file exists
    # Try to load "nonexistent.yaml"
    # Verify FileNotFoundError is raised
    pass


fn test_load_malformed_yaml():
    """Test loading malformed YAML raises parse error."""
    # TODO(#44): Implement when Config.from_yaml exists
    # Create temp YAML with invalid syntax:
    # key: [unclosed list
    # Try to load
    # Verify ParseError is raised
    pass


# ============================================================================
# Test Configuration Validation (partial)
# ============================================================================


fn test_validate_required_fields():
    """Test validation ensures required fields are present."""
    # TODO(#44): Implement when Config.validate exists
    # Create config missing required field "learning_rate"
    # Call validate()
    # Verify ValidationError is raised
    pass


fn test_validate_field_types():
    """Test validation checks field types."""
    # TODO(#44): Implement when Config.validate exists
    # Create config with:
    # learning_rate: "not a number"  # Should be Float32
    # Call validate()
    # Verify TypeError is raised
    pass


fn main() raises:
    """Run all tests."""
    test_load_yaml_config()
    test_load_json_config()
    test_load_nested_config()
    test_load_config_with_lists()
    test_load_nonexistent_file()
    test_load_malformed_yaml()
    test_validate_required_fields()
    test_validate_field_types()
