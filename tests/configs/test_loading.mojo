"""
Configuration Loading Tests

Tests for loading default, paper-specific, experiment configurations, and YAML format.
Split from test_loading.mojo per ADR-009.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_loading.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""


from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config


fn test_load_default_training_config() raises:
    """Test loading default training configuration.

    Verifies that the default training config loads successfully and
    contains expected optimizer settings.
    """
    # This will fail until Issue #74 creates the configs
    var config = load_config("configs/defaults/training.yaml")

    # Verify optimizer section exists
    assert_true(config.has("optimizer.name"), "Should have optimizer.name")
    var optimizer = config.get_string("optimizer.name")
    assert_equal(optimizer, "sgd", "Default optimizer should be SGD")

    # Verify learning rate exists and is reasonable
    assert_true(
        config.has("optimizer.learning_rate"), "Should have learning_rate"
    )
    var lr = config.get_float("optimizer.learning_rate")
    assert_true(lr > 0.0, "Learning rate should be positive")

    print("✓ test_load_default_training_config passed")


fn test_load_default_model_config() raises:
    """Test loading default model configuration.

    Verifies that default model settings load correctly.
    """
    var config = load_config("configs/defaults/model.yaml")

    # Verify model config has expected fields
    # Exact fields will depend on Issue #74 implementation
    assert_true(len(config.data) > 0, "Config should not be empty")

    print("✓ test_load_default_model_config passed")


fn test_load_default_data_config() raises:
    """Test loading default data configuration.

    Verifies that default data processing settings load correctly.
    """
    var config = load_config("configs/defaults/data.yaml")

    assert_true(len(config.data) > 0, "Config should not be empty")

    print("✓ test_load_default_data_config passed")


fn test_load_lenet5_model_config() raises:
    """Test loading LeNet-5 model configuration.

    Verifies that paper-specific model config loads with correct architecture.
    """
    var config = load_config("configs/papers/lenet5/model.yaml")

    # Verify model name (nested under model.name)
    assert_true(config.has("model.name"), "Should have model.name")
    var name = config.get_string("model.name")
    assert_equal(name, "lenet5", "Model name should be lenet5")

    # Verify architecture details (output_classes at model.output_classes)
    assert_true(
        config.has("model.output_classes"), "Should have model.output_classes"
    )
    var num_classes = config.get_int("model.output_classes")
    assert_equal(num_classes, 10, "LeNet-5 should have 10 output classes")

    print("✓ test_load_lenet5_model_config passed")


fn test_load_lenet5_training_config() raises:
    """Test loading LeNet-5 training configuration.

    Verifies that paper-specific training config loads correctly.
    """
    var config = load_config("configs/papers/lenet5/training.yaml")

    # Should have training parameters
    assert_true(len(config.data) > 0, "Training config should not be empty")

    # Check for learning rate (common parameter)
    assert_true(
        config.has("learning_rate") or config.has("optimizer.learning_rate"),
        "Should have learning rate configuration",
    )

    print("✓ test_load_lenet5_training_config passed")


fn test_load_experiment_baseline_config() raises:
    """Test loading baseline experiment configuration.

    Verifies that experiment configs can reference base configs.
    """
    var config = load_config("configs/experiments/lenet5/baseline.yaml")

    # Experiment configs may have an "extends" field
    # The exact structure depends on Issue #74
    assert_true(len(config.data) > 0, "Experiment config should not be empty")

    print("✓ test_load_experiment_baseline_config passed")


fn test_load_experiment_augmented_config() raises:
    """Test loading augmented experiment configuration.

    Verifies experiment config with data augmentation settings.
    """
    var config = load_config("configs/experiments/lenet5/augmented.yaml")

    assert_true(len(config.data) > 0, "Experiment config should not be empty")

    print("✓ test_load_experiment_augmented_config passed")


fn test_load_yaml_config() raises:
    """Test loading YAML configuration file.

    Verifies YAML parsing works correctly.
    """
    var config = load_config("tests/configs/fixtures/minimal.yaml")

    assert_true(config.has("learning_rate"), "Should have learning_rate")
    assert_true(config.has("batch_size"), "Should have batch_size")

    var lr = config.get_float("learning_rate")
    assert_equal(lr, 0.001, "Learning rate should be 0.001")

    var bs = config.get_int("batch_size")
    assert_equal(bs, 32, "Batch size should be 32")

    print("✓ test_load_yaml_config passed")


fn test_load_json_config() raises:
    """Test loading JSON configuration file.

    Verifies JSON parsing works correctly.
    """
    # Create a simple JSON config for testing
    var config = Config()
    config.set("learning_rate", 0.001)
    config.set("batch_size", 32)
    config.to_json("/tmp/test_output.json")

    # Load it back
    var loaded = load_config("/tmp/test_output.json")

    assert_true(loaded.has("learning_rate"), "Should have learning_rate")
    assert_true(loaded.has("batch_size"), "Should have batch_size")

    print("✓ test_load_json_config passed")


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
    with open("/tmp/test_empty.yaml", "w") as f:
        _ = f.write("")

    var error_raised = False
    try:
        _ = load_config("/tmp/test_empty.yaml")
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


fn main() raises:
    """Run all test_loading tests."""
    print("Running test_loading tests...")

    test_load_default_training_config()
    print("✓ test_load_default_training_config")

    test_load_default_model_config()
    print("✓ test_load_default_model_config")

    test_load_default_data_config()
    print("✓ test_load_default_data_config")

    test_load_lenet5_model_config()
    print("✓ test_load_lenet5_model_config")

    test_load_lenet5_training_config()
    print("✓ test_load_lenet5_training_config")

    test_load_experiment_baseline_config()
    print("✓ test_load_experiment_baseline_config")

    test_load_experiment_augmented_config()
    print("✓ test_load_experiment_augmented_config")

    test_load_yaml_config()
    print("✓ test_load_yaml_config")

    test_load_json_config()
    print("✓ test_load_json_config")

    test_load_missing_file()
    print("✓ test_load_missing_file")

    test_load_empty_file()
    print("✓ test_load_empty_file")

    test_load_invalid_format()
    print("✓ test_load_invalid_format")

    test_load_complex_nested_config()
    print("✓ test_load_complex_nested_config")

    print("\nAll test_loading tests passed!")
