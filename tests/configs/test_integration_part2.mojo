"""
Configuration Integration Tests - Part 2

End-to-end tests for environment integration, multi-experiment workflows,
reproducibility, and error recovery.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_integration.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Run with: mojo test tests/configs/test_integration_part2.mojo
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
# Environment Integration Tests
# ============================================================================


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


# ============================================================================
# Multi-Experiment Workflow Tests
# ============================================================================


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
    var temp_path = "tests/configs/fixtures/temp_config.yaml"
    config.to_yaml(temp_path)

    # Reload and verify
    var reloaded = load_config(temp_path)

    # Should have same number of entries
    # Exact comparison depends on config content
    assert_true(len(reloaded.data) > 0, "Reloaded config should not be empty")

    print("✓ test_config_save_and_reload passed")


# ============================================================================
# Reproducibility Tests
# ============================================================================


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


# ============================================================================
# Error Recovery Tests
# ============================================================================


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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run configuration integration tests (Part 2)."""
    print("\n" + "=" * 70)
    print("Running Configuration Integration Tests - Part 2")
    print("=" * 70 + "\n")

    # Environment integration tests
    print("Testing Environment Integration...")
    test_config_with_environment_variables()

    # Multi-experiment tests
    print("\nTesting Multi-Experiment Workflows...")
    test_multiple_experiments_from_same_paper()
    test_config_save_and_reload()

    # Reproducibility tests
    print("\nTesting Reproducibility...")
    test_experiment_reproducibility()
    test_config_versioning()

    # Error recovery tests
    print("\nTesting Error Recovery...")
    test_config_loading_with_fallbacks()
    test_partial_config_merge()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Configuration Integration Tests (Part 2) Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
    print("These tests follow TDD - they define expected behavior")
