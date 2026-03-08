# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_text_augmentations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Tests for text augmentation transforms - Part 2: RandomSwap edge cases and RandomDeletion.

Tests remaining RandomSwap operations and RandomDeletion operations.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_false,
    TestFixtures,
)
from shared.data.text_transforms import (
    TextTransform,
    RandomSwap,
    RandomDeletion,
    RandomInsertion,
    RandomSynonymReplacement,
    # TextCompose,  # Commented out - Issue #2086
    # TextPipeline,  # Commented out - Issue #2086
    split_words,
    join_words,
)


# ============================================================================
# RandomSwap Tests (edge cases)
# ============================================================================


fn test_random_swap_empty_text() raises:
    """Test RandomSwap handles empty text."""
    var text = String("")
    var swap = RandomSwap(1.0, 1)
    var result = swap(text)

    assert_equal(result, "")


fn test_random_swap_single_word() raises:
    """Test RandomSwap handles single word."""
    var text = String("hello")
    var swap = RandomSwap(1.0, 1)
    var result = swap(text)

    assert_equal(result, "hello")


fn test_random_swap_deterministic() raises:
    """Test RandomSwap is deterministic with seed."""
    var text = String("the quick brown fox jumps")

    TestFixtures.set_seed()
    var swap1 = RandomSwap(0.5, 2)
    var result1 = swap1(text)

    TestFixtures.set_seed()
    var swap2 = RandomSwap(0.5, 2)
    var result2 = swap2(text)

    assert_equal(result1, result2)


# ============================================================================
# RandomDeletion Tests
# ============================================================================


fn test_random_deletion_basic() raises:
    """Test RandomDeletion deletes some words."""
    var text = String("the quick brown fox jumps over lazy dog")

    # With p=0.5, some words should be deleted
    var delete = RandomDeletion(0.5)

    TestFixtures.set_seed()
    var result = delete(text)

    # Result should have fewer or equal words
    var original_words = split_words(text)
    var result_words = split_words(result)

    assert_true(len(result_words) <= len(original_words))
    assert_true(len(result_words) >= 1)  # At least one word remains


fn test_random_deletion_probability_never() raises:
    """Test RandomDeletion with p=0.0 never deletes."""
    var text = String("the quick brown fox")

    var delete = RandomDeletion(0.0)
    var result = delete(text)

    assert_equal(result, text)


fn test_random_deletion_preserves_one_word() raises:
    """Test RandomDeletion always keeps at least one word."""
    var text = String("the quick brown fox")

    # Even with p=1.0, at least one word should remain
    var delete = RandomDeletion(1.0)
    var result = delete(text)

    var words = split_words(result)
    assert_true(len(words) >= 1)


fn test_random_deletion_empty_text() raises:
    """Test RandomDeletion handles empty text."""
    var text = String("")
    var delete = RandomDeletion(0.5)
    var result = delete(text)

    assert_equal(result, "")


fn test_random_deletion_single_word() raises:
    """Test RandomDeletion preserves single word."""
    var text = String("hello")
    var delete = RandomDeletion(1.0)
    var result = delete(text)

    assert_equal(result, "hello")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all text augmentation part 2 tests."""
    print("Running text augmentation tests (part 2)...")

    # RandomSwap edge case tests
    test_random_swap_empty_text()
    print("  ✓ test_random_swap_empty_text")
    test_random_swap_single_word()
    print("  ✓ test_random_swap_single_word")
    test_random_swap_deterministic()
    print("  ✓ test_random_swap_deterministic")

    # RandomDeletion tests
    test_random_deletion_basic()
    print("  ✓ test_random_deletion_basic")
    test_random_deletion_probability_never()
    print("  ✓ test_random_deletion_probability_never")
    test_random_deletion_preserves_one_word()
    print("  ✓ test_random_deletion_preserves_one_word")
    test_random_deletion_empty_text()
    print("  ✓ test_random_deletion_empty_text")
    test_random_deletion_single_word()
    print("  ✓ test_random_deletion_single_word")

    print("\n✓ All 8 text augmentation part 2 tests passed!")
