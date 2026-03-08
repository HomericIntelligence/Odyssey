# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_text_augmentations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Tests for text augmentation transforms - Part 1: Helper functions and RandomSwap basics.

Tests split_words, join_words helper functions and basic RandomSwap operations.
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
# Helper Function Tests
# ============================================================================


fn test_split_words_basic() raises:
    """Test basic word splitting on spaces."""
    var text = String("the quick brown fox")
    var words = split_words(text)

    assert_equal(len(words), 4)
    assert_equal(words[0], "the")
    assert_equal(words[1], "quick")
    assert_equal(words[2], "brown")
    assert_equal(words[3], "fox")


fn test_split_words_empty() raises:
    """Test splitting empty string returns empty list."""
    var text = String("")
    var words = split_words(text)

    assert_equal(len(words), 0)


fn test_split_words_single() raises:
    """Test splitting single word."""
    var text = String("hello")
    var words = split_words(text)

    assert_equal(len(words), 1)
    assert_equal(words[0], "hello")


fn test_join_words_basic() raises:
    """Test basic word joining with spaces."""
    var words = List[String]()
    words.append("the")
    words.append("quick")
    words.append("brown")
    words.append("fox")

    var text = join_words(words)
    assert_equal(text, "the quick brown fox")


fn test_join_words_empty() raises:
    """Test joining empty list returns empty string."""
    var words = List[String]()
    var text = join_words(words)

    assert_equal(text, "")


fn test_join_words_single() raises:
    """Test joining single word."""
    var words = List[String]()
    words.append("hello")

    var text = join_words(words)
    assert_equal(text, "hello")


# ============================================================================
# RandomSwap Tests (basic)
# ============================================================================


fn test_random_swap_basic() raises:
    """Test RandomSwap swaps word positions."""
    var text = String("the quick brown fox")

    # With p=1.0, swaps should always occur
    var swap = RandomSwap(1.0, 1)

    TestFixtures.set_seed()
    var result = swap(text)

    # Result should still have same number of words
    var words = split_words(result)
    assert_equal(len(words), 4)


fn test_random_swap_probability() raises:
    """Test RandomSwap respects probability."""
    var text = String("the quick brown fox")

    # With p=0.0, no swaps should occur
    var swap = RandomSwap(0.0, 10)
    var result = swap(text)

    assert_equal(result, text)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all text augmentation part 1 tests."""
    print("Running text augmentation tests (part 1)...")

    # Helper function tests
    test_split_words_basic()
    print("  ✓ test_split_words_basic")
    test_split_words_empty()
    print("  ✓ test_split_words_empty")
    test_split_words_single()
    print("  ✓ test_split_words_single")
    test_join_words_basic()
    print("  ✓ test_join_words_basic")
    test_join_words_empty()
    print("  ✓ test_join_words_empty")
    test_join_words_single()
    print("  ✓ test_join_words_single")

    # RandomSwap basic tests
    test_random_swap_basic()
    print("  ✓ test_random_swap_basic")
    test_random_swap_probability()
    print("  ✓ test_random_swap_probability")

    print("\n✓ All 8 text augmentation part 1 tests passed!")
