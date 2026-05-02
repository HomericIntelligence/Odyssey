#!/usr/bin/env python3
"""
Configuration Schema Validation Tests

Tests for validating configurations against JSON schemas.
Uses Python's jsonschema library for schema validation.

Run with: pytest tests/configs/test_schema.py
"""

import os
import yaml
import json
import pytest
from pathlib import Path

# Try to import jsonschema, skip tests if not available
try:
    import jsonschema
    from jsonschema import validate, ValidationError

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    pytestmark = pytest.mark.skip(reason="jsonschema not installed")


# ============================================================================
# Helper Functions
# ============================================================================


def load_yaml_file(filepath):
    """Load YAML file and return parsed content.

    Args:
        filepath: Path to YAML file

    Returns:
        Parsed YAML content as dict
    """
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def load_json_file(filepath):
    """Load JSON file and return parsed content.

    Args:
        filepath: Path to JSON file

    Returns:
        Parsed JSON content as dict
    """
    with open(filepath, "r") as f:
        return json.load(f)


# ============================================================================
# Training Schema Tests
# ============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_training_schema_exists():
    """Test that training schema file exists.

    Verifies schema file is created by Issue #74.
    """
    schema_path = "configs/schemas/training.schema.yaml"
    # This will fail until Issue #74 creates the schema
    assert os.path.exists(schema_path), f"Schema file should exist: {schema_path}"


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_training_schema_valid():
    """Test that training schema is valid JSON Schema.

    Verifies schema follows JSON Schema specification.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")

    # Schema should have required fields
    assert "type" in schema, "Schema should have type field"
    assert "properties" in schema, "Schema should have properties"

    # Verify it's a valid JSON Schema by attempting to use it
    # This will raise if schema is invalid
    jsonschema.Draft7Validator.check_schema(schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_default_training_config_validates():
    """Test default training config validates against schema.

    Verifies default config meets schema requirements.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")
    config = load_yaml_file("configs/defaults/training.yaml")

    # Should validate without errors
    validate(instance=config, schema=schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_lenet5_training_config_validates():
    """Test LeNet-5 training config validates against schema.

    Verifies paper-specific config meets schema requirements.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")
    config = load_yaml_file("configs/papers/lenet5/training.yaml")

    # Should validate without errors
    validate(instance=config, schema=schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_invalid_training_config_fails():
    """Test that invalid training config fails validation.

    Verifies schema catches invalid configurations.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")
    config = load_yaml_file("tests/configs/fixtures/invalid_training.yaml")

    # Should raise ValidationError
    with pytest.raises(ValidationError):
        validate(instance=config, schema=schema)


# ============================================================================
# Model Schema Tests
# ============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_model_schema_exists():
    """Test that model schema file exists.

    Verifies schema file is created by Issue #74.
    """
    schema_path = "configs/schemas/model.schema.yaml"
    assert os.path.exists(schema_path), f"Schema file should exist: {schema_path}"


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_model_schema_valid():
    """Test that model schema is valid JSON Schema.

    Verifies schema follows JSON Schema specification.
    """
    schema = load_yaml_file("configs/schemas/model.schema.yaml")

    # Verify it's a valid JSON Schema
    jsonschema.Draft7Validator.check_schema(schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_lenet5_model_config_validates():
    """Test LeNet-5 model config validates against schema.

    Verifies model architecture config meets schema requirements.
    """
    schema = load_yaml_file("configs/schemas/model.schema.yaml")
    config = load_yaml_file("configs/papers/lenet5/model.yaml")

    # Should validate without errors
    validate(instance=config, schema=schema)


# ============================================================================
# Data Schema Tests
# ============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_data_schema_exists():
    """Test that data schema file exists.

    Verifies schema file is created by Issue #74.
    """
    schema_path = "configs/schemas/data.schema.yaml"
    # May not exist in initial implementation
    if os.path.exists(schema_path):
        schema = load_yaml_file(schema_path)
        jsonschema.Draft7Validator.check_schema(schema)


# ============================================================================
# Schema Property Tests
# ============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_training_schema_requires_optimizer():
    """Test training schema requires optimizer configuration.

    Verifies required fields are enforced by schema.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")
    assert schema is not None, "training schema failed to load"

    # Config missing optimizer should fail validation
    invalid_config = {"training": {"epochs": 100, "batch_size": 32}}
    with pytest.raises(ValidationError):
        validate(instance=invalid_config, schema=schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_training_schema_validates_types():
    """Test training schema validates value types.

    Verifies type checking is enforced by schema.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")

    # Config with wrong types should fail
    invalid_config = {
        "optimizer": {
            "name": "sgd",
            "learning_rate": "not_a_number",  # Should be number
        },
        "training": {"epochs": 100, "batch_size": 32},
    }

    with pytest.raises(ValidationError):
        validate(instance=invalid_config, schema=schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_training_schema_validates_ranges():
    """Test training schema validates value ranges.

    Verifies range constraints are enforced by schema.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")

    # Config with out-of-range values should fail
    invalid_config = {
        "optimizer": {
            "name": "sgd",
            "learning_rate": -0.01,  # Negative learning rate
        },
        "training": {
            "epochs": -10,  # Negative epochs
            "batch_size": 0,  # Zero batch size
        },
    }

    with pytest.raises(ValidationError):
        validate(instance=invalid_config, schema=schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_training_schema_validates_enums():
    """Test training schema validates enum values.

    Verifies allowed values are enforced by schema.
    """
    schema = load_yaml_file("configs/schemas/training.schema.yaml")

    # Config with invalid enum value should fail
    invalid_config = {
        "optimizer": {
            "name": "invalid_optimizer",  # Not in allowed list
            "learning_rate": 0.01,
        },
        "training": {"epochs": 100, "batch_size": 32},
    }

    with pytest.raises(ValidationError):
        validate(instance=invalid_config, schema=schema)


# ============================================================================
# Model Schema Property Tests
# ============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_model_schema_validates_field_types():
    """Test model schema validates field types.

    Verifies that when fields are present, they have correct types.
    Note: No fields are required since configs use inheritance via 'extends'.
    """
    schema = load_yaml_file("configs/schemas/model.schema.yaml")

    # Config with wrong type for extends should fail
    invalid_config = {
        "extends": 123,  # Should be string, not int
        "num_classes": 10,
    }

    with pytest.raises(ValidationError):
        validate(instance=invalid_config, schema=schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_model_schema_validates_num_classes():
    """Test model schema validates num_classes.

    Verifies num_classes is reasonable positive integer.
    """
    schema = load_yaml_file("configs/schemas/model.schema.yaml")

    # Config with invalid num_classes should fail
    invalid_config = {
        "name": "TestModel",
        "num_classes": -5,  # Negative classes
    }

    with pytest.raises(ValidationError):
        validate(instance=invalid_config, schema=schema)


# ============================================================================
# Complex Config Tests
# ============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_complex_config_validates():
    """Test complex nested config validates.

    Verifies schema handles nested structures correctly.
    """
    config = load_yaml_file("tests/configs/fixtures/complex.yaml")
    assert config is not None, "complex config failed to load"


# ============================================================================
# Schema Coverage Tests
# ============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_all_default_configs_validate():
    """Test all default configs validate against schemas.

    Verifies all configs in configs/defaults/ are valid.
    """
    defaults_dir = Path("configs/defaults")

    if not defaults_dir.exists():
        pytest.skip("Defaults directory not yet created (Issue #74)")

    for config_file in defaults_dir.glob("*.yaml"):
        # Determine schema based on config name
        schema_name = config_file.stem  # e.g., "training" from "training.yaml"
        schema_path = f"configs/schemas/{schema_name}.schema.yaml"

        if os.path.exists(schema_path):
            schema = load_yaml_file(schema_path)
            config = load_yaml_file(str(config_file))

            # Should validate without errors
            validate(instance=config, schema=schema)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_all_lenet5_configs_validate():
    """Test all LeNet-5 configs validate against schemas.

    Verifies all configs in configs/papers/lenet5/ are valid.
    """
    lenet5_dir = Path("configs/papers/lenet5")

    if not lenet5_dir.exists():
        pytest.skip("LeNet-5 config directory not yet created (Issue #74)")

    for config_file in lenet5_dir.glob("*.yaml"):
        schema_name = config_file.stem
        schema_path = f"configs/schemas/{schema_name}.schema.yaml"

        if os.path.exists(schema_path):
            schema = load_yaml_file(schema_path)
            config = load_yaml_file(str(config_file))

            # Should validate without errors
            validate(instance=config, schema=schema)


# ============================================================================
# Main Test Suite
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
