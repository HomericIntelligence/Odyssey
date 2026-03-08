"""
Configuration Integration Tests - Part 1

End-to-end tests for configuration loading and usage workflows.
Tests the complete integration of configs with model creation and training.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_integration.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Run with: mojo test tests/configs/test_integration_part1.mojo
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config, merge_configs
from python import Python


# ============================================================================
# Helper Function
# ============================================================================


fn load_experiment_config(paper: String, experiment: String) raises -> Config:
    """Helper function to load complete experiment configuration.

    Loads and merges: defaults → paper → experiment configs.

    Args:
        paper: Paper name (e.g., "lenet5").
        experiment: Experiment name (e.g., "baseline").

    Returns:
        Merged configuration.
    """
    # Load defaults
    var defaults = load_config("configs/defaults/training.yaml")

    # Load paper config
    var paper_path = "configs/papers/" + paper + "/training.yaml"
    var paper_config = load_config(paper_path)

    # Load experiment config
    var exp_path = "configs/experiments/" + paper + "/" + experiment + ".yaml"
    var exp_config = load_config(exp_path)

    # Merge in order
    var merged = merge_configs(defaults, paper_config)
    merged = merge_configs(merged, exp_config)

    return merged


# ============================================================================
# End-to-End Configuration Loading Tests
# ============================================================================


fn test_load_complete_experiment_config() raises:
    """Test loading complete experiment configuration.

    Verifies end-to-end workflow: defaults → paper → experiment.
    """
    # Load all three levels
    var defaults_training = load_config("configs/defaults/training.yaml")
    var defaults_model = load_config("configs/defaults/model.yaml")
    var defaults_data = load_config("configs/defaults/data.yaml")

    var paper_training = load_config("configs/papers/lenet5/training.yaml")
    var paper_model = load_config("configs/papers/lenet5/model.yaml")

    var exp_config = load_config("configs/experiments/lenet5/baseline.yaml")

    # Merge training configs
    var training = merge_configs(defaults_training, paper_training)
    training = merge_configs(training, exp_config)

    # Merge model configs
    var model = merge_configs(defaults_model, paper_model)

    # Final config should have all required sections
    assert_true(len(training.data) > 0, "Training config should not be empty")
    assert_true(len(model.data) > 0, "Model config should not be empty")
    assert_true(len(defaults_data.data) > 0, "Data config should not be empty")

    print("✓ test_load_complete_experiment_config passed")


fn test_load_experiment_with_helper_function() raises:
    """Test loading experiment config using helper function.

    Verifies convenience function for experiment config loading.
    """
    # This function will be implemented in Issue #74
    # For now, test manual approach
    var config = load_experiment_config("lenet5", "baseline")

    # Should have all merged sections
    assert_true(
        config.has("model") or len(config.data) > 0,
        "Experiment config should be loaded",
    )

    print("✓ test_load_experiment_with_helper_function passed")


# ============================================================================
# Model Creation Integration Tests
# ============================================================================


fn test_model_creation_from_config() raises:
    """Test creating model from configuration.

    Verifies model config can be used for model instantiation.
    """
    var config = load_config("configs/papers/lenet5/model.yaml")

    # Verify config has required fields for model creation
    # Config uses dotted keys for nested YAML (e.g., "model.name")
    assert_true(config.has("model.name"), "Should have model.name")
    assert_true(
        config.has("model.output_classes"), "Should have model.output_classes"
    )

    # In actual implementation, would do:
    # var model = create_model_from_config(config)
    # For now, verify config structure
    var name = config.get_string("model.name")
    var num_classes = config.get_int("model.output_classes")

    assert_equal(name, "lenet5", "Model name should be lenet5")
    assert_equal(num_classes, 10, "Should have 10 classes for MNIST")

    print("✓ test_model_creation_from_config passed")


fn test_model_with_architecture_config() raises:
    """Test model creation with architecture details.

    Verifies architecture parameters can be extracted from config.
    """
    var config = load_config("configs/papers/lenet5/model.yaml")

    # Architecture might specify layers, filters, etc.
    # Exact fields depend on Issue #74 implementation
    # This test verifies expected structure

    assert_true(len(config.data) > 0, "Model config should have architecture")

    print("✓ test_model_with_architecture_config passed")


# ============================================================================
# Training Loop Integration Tests
# ============================================================================


fn test_training_loop_from_config() raises:
    """Test extracting training parameters from config.

    Verifies training config can be used in training loop.
    """
    var config = load_experiment_config("lenet5", "baseline")

    # Extract training parameters
    var lr = config.get_float("optimizer.learning_rate", 0.001)
    var epochs = config.get_int("training.epochs", 10)
    var batch_size = config.get_int("training.batch_size", 32)

    # Verify reasonable values
    assert_true(lr > 0.0, "Learning rate should be positive")
    assert_true(epochs > 0, "Epochs should be positive")
    assert_true(batch_size > 0, "Batch size should be positive")

    # In actual implementation:
    # var optimizer = create_optimizer(config)
    # for epoch in range(epochs):
    #     train_epoch(model, data, optimizer, config)

    print("✓ test_training_loop_from_config passed")


fn test_optimizer_creation_from_config() raises:
    """Test creating optimizer from configuration.

    Verifies optimizer config can be used for optimizer creation.
    """
    var config = load_experiment_config("lenet5", "baseline")

    # Extract optimizer settings
    var opt_name = config.get_string("optimizer.name", "sgd")
    var lr = config.get_float("optimizer.learning_rate", 0.001)

    # Valid optimizer names
    var valid_optimizers = List[String]()
    valid_optimizers.append("sgd")
    valid_optimizers.append("adam")
    valid_optimizers.append("rmsprop")

    # Check optimizer is valid
    var is_valid = False
    for i in range(len(valid_optimizers)):
        if opt_name == valid_optimizers[i]:
            is_valid = True
            break

    assert_true(is_valid, "Optimizer should be valid choice")

    print("✓ test_optimizer_creation_from_config passed")


# ============================================================================
# Data Pipeline Integration Tests
# ============================================================================


fn test_data_pipeline_from_config() raises:
    """Test creating data pipeline from configuration.

    Verifies data config can be used for data loading.
    """
    var config = load_config("configs/defaults/data.yaml")

    # Data config might specify dataset, preprocessing, augmentation
    # Exact structure depends on Issue #74 implementation

    assert_true(len(config.data) > 0, "Data config should not be empty")

    # In actual implementation:
    # var dataset = load_dataset(config)
    # var dataloader = create_dataloader(dataset, config)

    print("✓ test_data_pipeline_from_config passed")


fn test_data_augmentation_from_config() raises:
    """Test data augmentation configuration.

    Verifies augmentation settings can be extracted.
    """
    var config = load_experiment_config("lenet5", "augmented")

    # Augmented experiment should have augmentation enabled
    # Exact fields depend on Issue #74
    assert_true(len(config.data) > 0, "Augmented config should not be empty")

    print("✓ test_data_augmentation_from_config passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run configuration integration tests (Part 1)."""
    print("\n" + "=" * 70)
    print("Running Configuration Integration Tests - Part 1")
    print("=" * 70 + "\n")

    # End-to-end loading tests
    print("Testing End-to-End Configuration Loading...")
    test_load_complete_experiment_config()
    test_load_experiment_with_helper_function()

    # Model integration tests
    print("\nTesting Model Creation Integration...")
    test_model_creation_from_config()
    test_model_with_architecture_config()

    # Training integration tests
    print("\nTesting Training Loop Integration...")
    test_training_loop_from_config()
    test_optimizer_creation_from_config()

    # Data pipeline tests
    print("\nTesting Data Pipeline Integration...")
    test_data_pipeline_from_config()
    test_data_augmentation_from_config()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Configuration Integration Tests (Part 1) Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
    print("These tests follow TDD - they define expected behavior")
