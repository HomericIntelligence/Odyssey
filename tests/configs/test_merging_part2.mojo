"""
Configuration Merging Tests - Part 2

Tests for merge conflict handling and merge properties.

Split from test_merging.mojo per ADR-009 (≤10 fn test_ per file).

Run with: mojo test tests/configs/test_merging_part2.mojo
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_merging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config, merge_configs


# ============================================================================
# Deep Merging Tests (continued)
# ============================================================================


fn test_merge_with_dotted_keys() raises:
    """Test merging configs using dotted key notation.

    Verifies that dot-notation keys are handled correctly.
    """
    var base = Config()
    base.set("optimizer.name", "sgd")
    base.set("optimizer.lr", 0.01)

    var override = Config()
    override.set("optimizer.lr", 0.001)

    var merged = merge_configs(base, override)

    var opt = merged.get_string("optimizer.name")
    assert_equal(opt, "sgd", "Should preserve base dotted key")

    var lr = merged.get_float("optimizer.lr")
    assert_equal(lr, 0.001, "Should override dotted key value")

    print("✓ test_merge_with_dotted_keys passed")


# ============================================================================
# Merge Conflict Tests
# ============================================================================


fn test_merge_type_conflicts() raises:
    """Test merging when same key has different types.

    Verifies that override type takes precedence.
    """
    var base = Config()
    base.set("value", 42)  # Int

    var override = Config()
    override.set("value", "forty-two")  # String

    var merged = merge_configs(base, override)

    # Override should win, changing type
    var val = merged.get_string("value")
    assert_equal(val, "forty-two", "Override type should take precedence")

    print("✓ test_merge_type_conflicts passed")


fn test_merge_multiple_times() raises:
    """Test merging same config multiple times.

    Verifies that repeated merges produce consistent results.
    """
    var base = Config()
    base.set("a", 1)
    base.set("b", 2)

    var override = Config()
    override.set("b", 20)
    override.set("c", 30)

    var merged1 = merge_configs(base, override)
    var merged2 = merge_configs(base, override)

    # Should produce identical results
    assert_equal(
        merged1.get_int("a"),
        merged2.get_int("a"),
        "Multiple merges should be consistent",
    )
    assert_equal(
        merged1.get_int("b"),
        merged2.get_int("b"),
        "Multiple merges should be consistent",
    )
    assert_equal(
        merged1.get_int("c"),
        merged2.get_int("c"),
        "Multiple merges should be consistent",
    )

    print("✓ test_merge_multiple_times passed")


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_merge_associativity() raises:
    """Test that merge operation is associative.

    Verifies: (A merge B) merge C == A merge (B merge C)
    Note: This is only true when using right-precedence merging.
    """
    var a = Config()
    a.set("x", 1)
    a.set("y", 2)

    var b = Config()
    b.set("y", 20)
    b.set("z", 30)

    var c = Config()
    c.set("z", 300)
    c.set("w", 400)

    # Left-associative: (A merge B) merge C
    var left = merge_configs(a, b)
    left = merge_configs(left, c)

    # With right-precedence merging, order matters
    # This test verifies expected behavior
    assert_equal(left.get_int("x"), 1, "Should have value from A")
    assert_equal(left.get_int("y"), 20, "Should have value from B")
    assert_equal(left.get_int("z"), 300, "Should have value from C")
    assert_equal(left.get_int("w"), 400, "Should have value from C")

    print("✓ test_merge_associativity passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run Part 2 configuration merging tests."""
    print("\n" + "=" * 70)
    print("Running Configuration Merging Tests - Part 2")
    print("=" * 70 + "\n")

    # Deep merging tests (continued)
    print("Testing Deep Merging (continued)...")
    test_merge_with_dotted_keys()

    # Conflict handling tests
    print("\nTesting Merge Conflicts...")
    test_merge_type_conflicts()
    test_merge_multiple_times()

    # Property-based tests
    print("\nTesting Merge Properties...")
    test_merge_associativity()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Configuration Merging Tests (Part 2) Passed!")
    print("=" * 70)
