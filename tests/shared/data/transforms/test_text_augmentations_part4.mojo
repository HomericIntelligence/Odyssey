# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_text_augmentations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Tests for text augmentation transforms - Part 4: RandomSynonymReplacement edge cases.

Tests remaining RandomSynonymReplacement operations (no-match, empty text, determinism).
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
# RandomSynonymReplacement Tests (edge cases)
# ============================================================================


fn test_random_synonym_replacement_no_synonyms() raises:
    """Test RandomSynonymReplacement with no matching synonyms."""
    var text = String("the quick fox")

    var synonyms = Dict[String, List[String]]()
    var slow_syns = List[String]()
    slow_syns.append("sluggish")
    synonyms["slow"] = slow_syns^  # "slow" not in text

    var replace = RandomSynonymReplacement(synonyms.copy(), 1.0)
    var result = replace(text)

    # No words should be replaced
    assert_equal(result, text)


fn test_random_synonym_replacement_empty_text() raises:
    """Test RandomSynonymReplacement handles empty text."""
    var text = String("")

    var synonyms = Dict[String, List[String]]()
    var quick_syns = List[String]()
    quick_syns.append("fast")
    synonyms["quick"] = quick_syns^

    var replace = RandomSynonymReplacement(synonyms.copy(), 1.0)
    var result = replace(text)

    assert_equal(result, "")


fn test_random_synonym_replacement_deterministic() raises:
    """Test RandomSynonymReplacement is deterministic with seed."""
    var text = String("the quick brown fox")

    var synonyms = Dict[String, List[String]]()
    var quick_syns = List[String]()
    quick_syns.append("fast")
    quick_syns.append("rapid")
    synonyms["quick"] = quick_syns^

    var brown_syns = List[String]()
    brown_syns.append("dark")
    brown_syns.append("tan")
    synonyms["brown"] = brown_syns^

    TestFixtures.set_seed()
    var replace1 = RandomSynonymReplacement(synonyms.copy(), 0.5)
    var result1 = replace1(text)

    TestFixtures.set_seed()
    var synonyms2 = Dict[String, List[String]]()
    var quick_syns2 = List[String]()
    quick_syns2.append("fast")
    quick_syns2.append("rapid")
    synonyms2["quick"] = quick_syns2^

    var brown_syns2 = List[String]()
    brown_syns2.append("dark")
    brown_syns2.append("tan")
    synonyms2["brown"] = brown_syns2^

    var replace2 = RandomSynonymReplacement(synonyms2.copy(), 0.5)
    var result2 = replace2(text)

    assert_equal(result1, result2)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all text augmentation part 4 tests."""
    print("Running text augmentation tests (part 4)...")

    # RandomSynonymReplacement edge case tests
    test_random_synonym_replacement_no_synonyms()
    print("  ✓ test_random_synonym_replacement_no_synonyms")
    test_random_synonym_replacement_empty_text()
    print("  ✓ test_random_synonym_replacement_empty_text")
    test_random_synonym_replacement_deterministic()
    print("  ✓ test_random_synonym_replacement_deterministic")

    print("\n✓ All 3 text augmentation part 4 tests passed!")
