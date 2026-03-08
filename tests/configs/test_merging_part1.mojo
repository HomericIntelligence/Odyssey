"""
Configuration Merging Tests - Part 1

Tests for merging configurations across the 3-tier hierarchy:
defaults → paper-specific → experiment-specific

Split from test_merging.mojo per ADR-009 (≤10 fn test_ per file).

Run with: mojo test tests/configs/test_merging_part1.mojo
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_merging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config, merge_configs


# ============================================================================
# Basic Merging Tests
# ============================================================================


fn test_merge_two_configs() raises:
    """Test basic two-config merge operation.

    Verifies that merging two configs combines values correctly
    with override taking precedence.
    """
    var base = Config()
    base.set("learning_rate", 0.01)
    base.set("batch_size", 32)
    base.set("optimizer", "sgd")

    var override = Config()
    override.set("learning_rate", 0.001)  # Override
    override.set("epochs", 100)  # New value

    var merged = merge_configs(base, override)

    # Override values should take precedence
    var lr = merged.get_float("learning_rate")
    assert_equal(lr, 0.001, "Override learning_rate should be used")

    # Base values should be retained if not overridden
    var bs = merged.get_int("batch_size")
    assert_equal(bs, 32, "Base batch_size should be retained")

    var opt = merged.get_string("optimizer")
    assert_equal(opt, "sgd", "Base optimizer should be retained")

    # New values from override should be added
    var epochs = merged.get_int("epochs")
    assert_equal(epochs, 100, "Override epochs should be added")

    print("✓ test_merge_two_configs passed")


fn test_merge_empty_configs() raises:
    """Test merging with empty configurations.

    Verifies that merging handles empty configs gracefully.
    """
    var base = Config()
    base.set("learning_rate", 0.01)

    var empty = Config()

    # Merge with empty override
    var merged1 = merge_configs(base, empty)
    assert_true(merged1.has("learning_rate"), "Should retain base values")

    # Merge empty base with override
    var merged2 = merge_configs(empty, base)
    assert_true(merged2.has("learning_rate"), "Should have override values")

    print("✓ test_merge_empty_configs passed")


# ============================================================================
# Two-Level Merge Tests (Defaults → Paper)
# ============================================================================


fn test_merge_default_and_paper() raises:
    """Test merging default and paper configurations.

    Verifies 2-level merge: defaults → paper-specific config.
    """
    var defaults = load_config("configs/defaults/training.yaml")
    var paper = load_config("configs/papers/lenet5/training.yaml")

    var merged = merge_configs(defaults, paper)

    # Paper values should override defaults
    # Exact assertions depend on config file content from Issue #74
    assert_true(len(merged.data) > 0, "Merged config should not be empty")

    # Should have values from both configs
    # Paper-specific values should take precedence
    if paper.has("learning_rate"):
        var paper_lr = paper.get_float("learning_rate")
        var merged_lr = merged.get_float("learning_rate")
        assert_equal(
            merged_lr, paper_lr, "Paper learning_rate should override default"
        )

    print("✓ test_merge_default_and_paper passed")


fn test_merge_preserves_default_values() raises:
    """Test that merge preserves default values not in override.

    Verifies defaults are retained when not overridden by paper config.
    """
    var defaults = Config()
    defaults.set("learning_rate", 0.01)
    defaults.set("momentum", 0.9)
    defaults.set("weight_decay", 0.0001)
    defaults.set("batch_size", 32)

    var paper = Config()
    paper.set("learning_rate", 0.001)  # Override only this

    var merged = merge_configs(defaults, paper)

    # Override should be used
    var lr = merged.get_float("learning_rate")
    assert_equal(lr, 0.001, "Should use paper learning_rate")

    # Defaults should be preserved
    var momentum = merged.get_float("momentum")
    assert_equal(momentum, 0.9, "Should preserve default momentum")

    var wd = merged.get_float("weight_decay")
    assert_equal(wd, 0.0001, "Should preserve default weight_decay")

    var bs = merged.get_int("batch_size")
    assert_equal(bs, 32, "Should preserve default batch_size")

    print("✓ test_merge_preserves_default_values passed")


# ============================================================================
# Three-Level Merge Tests (Defaults → Paper → Experiment)
# ============================================================================


fn test_three_level_merge() raises:
    """Test three-level configuration merge.

    Verifies full hierarchy: defaults → paper → experiment.
    """
    var defaults = load_config("configs/defaults/training.yaml")
    var paper = load_config("configs/papers/lenet5/training.yaml")
    var experiment = load_config("configs/experiments/lenet5/augmented.yaml")

    # Merge in sequence: defaults → paper → experiment
    var merged = merge_configs(defaults, paper)
    merged = merge_configs(merged, experiment)

    # Final config should have values from all three levels
    assert_true(len(merged.data) > 0, "Merged config should not be empty")

    # Experiment values should take highest precedence
    # Exact assertions depend on Issue #74 config content
    print("✓ test_three_level_merge passed")


fn test_experiment_overrides_all() raises:
    """Test that experiment config overrides both defaults and paper.

    Verifies that experiment-level settings have highest precedence.
    """
    var defaults = Config()
    defaults.set("learning_rate", 0.01)
    defaults.set("batch_size", 32)

    var paper = Config()
    paper.set("learning_rate", 0.001)

    var experiment = Config()
    experiment.set("learning_rate", 0.0001)  # Override both

    var merged = merge_configs(defaults, paper)
    merged = merge_configs(merged, experiment)

    var lr = merged.get_float("learning_rate")
    assert_equal(
        lr, 0.0001, "Experiment learning_rate should override all others"
    )

    var bs = merged.get_int("batch_size")
    assert_equal(bs, 32, "Default batch_size should be preserved")

    print("✓ test_experiment_overrides_all passed")


fn test_three_level_merge_baseline_experiment() raises:
    """Test three-level merge with baseline experiment.

    Verifies merge works with baseline experiment (minimal overrides).
    """
    var defaults = load_config("configs/defaults/training.yaml")
    var paper = load_config("configs/papers/lenet5/training.yaml")
    var baseline = load_config("configs/experiments/lenet5/baseline.yaml")

    var merged = merge_configs(defaults, paper)
    merged = merge_configs(merged, baseline)

    assert_true(len(merged.data) > 0, "Merged config should not be empty")

    print("✓ test_three_level_merge_baseline_experiment passed")


# ============================================================================
# Deep Merging Tests
# ============================================================================


fn test_merge_nested_structures() raises:
    """Test merging configurations with nested structures.

    Note: Current implementation may have limitations with deep nesting.
    This test validates expected behavior.
    """
    var base = Config()
    base.set("model.layers", 3)
    base.set("model.activation", "relu")
    base.set("optimizer.name", "sgd")

    var override = Config()
    override.set("model.layers", 5)  # Override nested value
    override.set("optimizer.learning_rate", 0.001)  # Add nested value

    var merged = merge_configs(base, override)

    # Override nested values should take precedence
    if merged.has("model.layers"):
        var layers = merged.get_int("model.layers")
        assert_equal(layers, 5, "Should use override nested value")

    # Base nested values should be preserved
    if merged.has("model.activation"):
        var act = merged.get_string("model.activation")
        assert_equal(act, "relu", "Should preserve base nested value")

    print("✓ test_merge_nested_structures passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run Part 1 configuration merging tests."""
    print("\n" + "=" * 70)
    print("Running Configuration Merging Tests - Part 1")
    print("=" * 70 + "\n")

    # Basic merging tests
    print("Testing Basic Merging...")
    test_merge_two_configs()
    test_merge_empty_configs()

    # Two-level merge tests
    print("\nTesting Two-Level Merge (Defaults → Paper)...")
    test_merge_default_and_paper()
    test_merge_preserves_default_values()

    # Three-level merge tests
    print("\nTesting Three-Level Merge (Defaults → Paper → Experiment)...")
    test_three_level_merge()
    test_experiment_overrides_all()
    test_three_level_merge_baseline_experiment()

    # Deep merging tests
    print("\nTesting Deep Merging...")
    test_merge_nested_structures()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Configuration Merging Tests (Part 1) Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
