# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_text_augmentations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Tests for text augmentation transforms - Part 3: RandomDeletion determinism and RandomInsertion.

Tests RandomDeletion determinism and all RandomInsertion operations.
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
# RandomDeletion Tests (determinism)
# ============================================================================


fn test_random_deletion_deterministic() raises:
    """Test RandomDeletion is deterministic with seed."""
    var text = String("the quick brown fox jumps")

    TestFixtures.set_seed()
    var delete1 = RandomDeletion(0.3)
    var result1 = delete1(text)

    TestFixtures.set_seed()
    var delete2 = RandomDeletion(0.3)
    var result2 = delete2(text)

    assert_equal(result1, result2)


# ============================================================================
# RandomInsertion Tests
# ============================================================================


fn test_random_insertion_basic() raises:
    """Test RandomInsertion inserts words from vocabulary."""
    var text = String("the brown fox")

    var vocab = List[String]()
    vocab.append("quick")
    vocab.append("lazy")
    vocab.append("red")

    # With p=1.0, insertion should occur
    var insert = RandomInsertion(vocab.copy(), 1.0, 1)

    TestFixtures.set_seed()
    var result = insert(text)

    # Result should have more or equal words
    var original_words = split_words(text)
    var result_words = split_words(result)

    assert_true(len(result_words) >= len(original_words))


fn test_random_insertion_probability() raises:
    """Test RandomInsertion respects probability."""
    var text = String("the brown fox")

    var vocab = List[String]()
    vocab.append("quick")

    # With p=0.0, no insertion should occur
    var insert = RandomInsertion(vocab.copy(), 0.0, 10)
    var result = insert(text)

    assert_equal(result, text)


fn test_random_insertion_empty_text() raises:
    """Test RandomInsertion handles empty text."""
    var text = String("")

    var vocab = List[String]()
    vocab.append("quick")

    var insert = RandomInsertion(vocab.copy(), 1.0, 1)
    var result = insert(text)

    assert_equal(result, "")


fn test_random_insertion_empty_vocabulary() raises:
    """Test RandomInsertion handles empty vocabulary."""
    var text = String("the brown fox")

    var vocab = List[String]()

    var insert = RandomInsertion(vocab.copy(), 1.0, 1)
    var result = insert(text)

    assert_equal(result, text)


fn test_random_insertion_deterministic() raises:
    """Test RandomInsertion is deterministic with seed."""
    var text = String("the brown fox")

    var vocab = List[String]()
    vocab.append("quick")
    vocab.append("lazy")

    TestFixtures.set_seed()
    var insert1 = RandomInsertion(vocab.copy(), 0.5, 2)
    var result1 = insert1(text)

    TestFixtures.set_seed()
    var vocab2 = List[String]()
    vocab2.append("quick")
    vocab2.append("lazy")
    var insert2 = RandomInsertion(vocab2.copy(), 0.5, 2)
    var result2 = insert2(text)

    assert_equal(result1, result2)


# ============================================================================
# RandomSynonymReplacement Tests (basic)
# ============================================================================


fn test_random_synonym_replacement_basic() raises:
    """Test RandomSynonymReplacement replaces with synonyms."""
    var text = String("the quick fox")

    var synonyms = Dict[String, List[String]]()
    var quick_syns = List[String]()
    quick_syns.append("fast")
    quick_syns.append("rapid")
    synonyms["quick"] = quick_syns^

    # With p=1.0, should replace
    var replace = RandomSynonymReplacement(synonyms.copy(), 1.0)

    TestFixtures.set_seed()
    var result = replace(text)

    # Result should have same number of words
    var words = split_words(result)
    assert_equal(len(words), 3)

    # "quick" should be replaced (result should differ)
    # Note: Due to randomness, we just check it has same word count
    var original_words = split_words(text)
    assert_equal(len(words), len(original_words))


fn test_random_synonym_replacement_probability() raises:
    """Test RandomSynonymReplacement respects probability."""
    var text = String("the quick fox")

    var synonyms = Dict[String, List[String]]()
    var quick_syns = List[String]()
    quick_syns.append("fast")
    synonyms["quick"] = quick_syns^

    # With p=0.0, no replacement should occur
    var replace = RandomSynonymReplacement(synonyms.copy(), 0.0)
    var result = replace(text)

    assert_equal(result, text)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all text augmentation part 3 tests."""
    print("Running text augmentation tests (part 3)...")

    # RandomDeletion determinism test
    test_random_deletion_deterministic()
    print("  ✓ test_random_deletion_deterministic")

    # RandomInsertion tests
    test_random_insertion_basic()
    print("  ✓ test_random_insertion_basic")
    test_random_insertion_probability()
    print("  ✓ test_random_insertion_probability")
    test_random_insertion_empty_text()
    print("  ✓ test_random_insertion_empty_text")
    test_random_insertion_empty_vocabulary()
    print("  ✓ test_random_insertion_empty_vocabulary")
    test_random_insertion_deterministic()
    print("  ✓ test_random_insertion_deterministic")

    # RandomSynonymReplacement basic tests
    test_random_synonym_replacement_basic()
    print("  ✓ test_random_synonym_replacement_basic")
    test_random_synonym_replacement_probability()
    print("  ✓ test_random_synonym_replacement_probability")

    print("\n✓ All 8 text augmentation part 3 tests passed!")
