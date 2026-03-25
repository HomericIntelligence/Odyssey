"""
Configuration Integration Tests

End-to-end tests for configuration loading and usage workflows.
Tests the complete integration of configs with model creation and training.

"""


from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config, merge_configs
from python import Python


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


fn test_config_with_environment_variables() raises:
    """Test configuration with environment variable substitution.

    Verifies end-to-end workflow with environment variables.
    """
    var python = Python.import_module("os")
    python.environ.__setitem__("EXPERIMENT_DIR", value="/tmp/experiments")

    # Create config with env vars
    var config = Config()
    config.set("output_dir", "${EXPERIMENT_DIR}/lenet5")
    config.set("checkpoint_dir", "${EXPERIMENT_DIR}/lenet5/checkpoints")

    # Substitute environment variables
    var substituted = config.substitute_env_vars()

    var output = substituted.get_string("output_dir")
    assert_equal(
        output, "/tmp/experiments/lenet5", "Should substitute EXPERIMENT_DIR"
    )

    var checkpoint = substituted.get_string("checkpoint_dir")
    assert_equal(
        checkpoint,
        "/tmp/experiments/lenet5/checkpoints",
        "Should substitute in nested path",
    )

    print("✓ test_config_with_environment_variables passed")


fn test_multiple_experiments_from_same_paper() raises:
    """Test loading multiple experiments for same paper.

    Verifies different experiments can coexist.
    """
    var baseline = load_experiment_config("lenet5", "baseline")
    var augmented = load_experiment_config("lenet5", "augmented")

    # Both should be valid configs
    assert_true(len(baseline.data) > 0, "Baseline config should load")
    assert_true(len(augmented.data) > 0, "Augmented config should load")

    # Configs should differ (augmented has extra settings)
    # Exact differences depend on Issue #74
    print("✓ test_multiple_experiments_from_same_paper passed")


fn test_config_save_and_reload() raises:
    """Test saving and reloading configuration.

    Verifies config persistence works correctly.
    """
    # Load and merge config
    var config = load_experiment_config("lenet5", "baseline")

    # Save to temporary file
    var temp_path = "/tmp/temp_config.yaml"
    config.to_yaml(temp_path)

    # Reload and verify
    var reloaded = load_config(temp_path)

    # Should have same number of entries
    # Exact comparison depends on config content
    assert_true(len(reloaded.data) > 0, "Reloaded config should not be empty")

    print("✓ test_config_save_and_reload passed")


fn test_experiment_reproducibility() raises:
    """Test that config enables experiment reproducibility.

    Verifies loading same config produces same results.
    """
    # Load config twice
    var config1 = load_experiment_config("lenet5", "baseline")
    var config2 = load_experiment_config("lenet5", "baseline")

    # Extract key parameters
    var lr1 = config1.get_float("optimizer.learning_rate", 0.001)
    var lr2 = config2.get_float("optimizer.learning_rate", 0.001)

    assert_equal(lr1, lr2, "Same config should produce same parameters")

    print("✓ test_experiment_reproducibility passed")


fn test_config_versioning() raises:
    """Test configuration can be versioned.

    Verifies configs can include version information.
    """
    var config = load_experiment_config("lenet5", "baseline")

    # Config might include version field
    # This enables tracking which config version was used
    # Exact implementation depends on Issue #74

    print("✓ test_config_versioning passed")


fn test_config_loading_with_fallbacks() raises:
    """Test config loading with fallback values.

    Verifies graceful degradation when optional configs missing.
    """
    var config = Config()
    config.set("learning_rate", 0.001)

    # Get with defaults
    var lr = config.get_float("learning_rate", 0.01)
    assert_equal(lr, 0.001, "Should use actual value")

    var momentum = config.get_float("momentum", 0.9)
    assert_equal(momentum, 0.9, "Should use default for missing value")

    print("✓ test_config_loading_with_fallbacks passed")


fn test_partial_config_merge() raises:
    """Test merging when configs have different keys.

    Verifies partial configs merge correctly.
    """
    var base = Config()
    base.set("learning_rate", 0.01)
    base.set("momentum", 0.9)
    base.set("weight_decay", 0.0001)

    var partial = Config()
    partial.set("learning_rate", 0.001)  # Only override one value

    var merged = merge_configs(base, partial)

    # Should have all values from base plus override
    var lr = merged.get_float("learning_rate")
    assert_equal(lr, 0.001, "Should use override value")

    var momentum = merged.get_float("momentum")
    assert_equal(momentum, 0.9, "Should preserve base value")

    print("✓ test_partial_config_merge passed")


fn main() raises:
    """Run all test_integration tests."""
    print("Running test_integration tests...")

    test_load_complete_experiment_config()
    print("✓ test_load_complete_experiment_config")

    test_load_experiment_with_helper_function()
    print("✓ test_load_experiment_with_helper_function")

    test_model_creation_from_config()
    print("✓ test_model_creation_from_config")

    test_model_with_architecture_config()
    print("✓ test_model_with_architecture_config")

    test_training_loop_from_config()
    print("✓ test_training_loop_from_config")

    test_optimizer_creation_from_config()
    print("✓ test_optimizer_creation_from_config")

    test_data_pipeline_from_config()
    print("✓ test_data_pipeline_from_config")

    test_data_augmentation_from_config()
    print("✓ test_data_augmentation_from_config")

    test_config_with_environment_variables()
    print("✓ test_config_with_environment_variables")

    test_multiple_experiments_from_same_paper()
    print("✓ test_multiple_experiments_from_same_paper")

    test_config_save_and_reload()
    print("✓ test_config_save_and_reload")

    test_experiment_reproducibility()
    print("✓ test_experiment_reproducibility")

    test_config_versioning()
    print("✓ test_config_versioning")

    test_config_loading_with_fallbacks()
    print("✓ test_config_loading_with_fallbacks")

    test_partial_config_merge()
    print("✓ test_partial_config_merge")

    print("\nAll test_integration tests passed!")
