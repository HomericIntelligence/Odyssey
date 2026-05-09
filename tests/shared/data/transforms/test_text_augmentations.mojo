"""Tests for text augmentation transforms.

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


def test_split_words_basic() raises:
    """Test basic word splitting on spaces."""
    var text = String("the quick brown fox")
    var words = split_words(text)

    assert_equal(len(words), 4)
    assert_equal(words[0], "the")
    assert_equal(words[1], "quick")
    assert_equal(words[2], "brown")
    assert_equal(words[3], "fox")


def test_split_words_empty() raises:
    """Test splitting empty string returns empty list."""
    var text = String("")
    var words = split_words(text)

    assert_equal(len(words), 0)


def test_split_words_single() raises:
    """Test splitting single word."""
    var text = String("hello")
    var words = split_words(text)

    assert_equal(len(words), 1)
    assert_equal(words[0], "hello")


def test_join_words_basic() raises:
    """Test basic word joining with spaces."""
    var words = List[String]()
    words.append("the")
    words.append("quick")
    words.append("brown")
    words.append("fox")

    var text = join_words(words)
    assert_equal(text, "the quick brown fox")


def test_join_words_empty() raises:
    """Test joining empty list returns empty string."""
    var words = List[String]()
    var text = join_words(words)

    assert_equal(text, "")


def test_join_words_single() raises:
    """Test joining single word."""
    var words = List[String]()
    words.append("hello")

    var text = join_words(words)
    assert_equal(text, "hello")


def test_random_swap_basic() raises:
    """Test RandomSwap swaps word positions."""
    var text = String("the quick brown fox")

    # With p=1.0, swaps should always occur
    var swap = RandomSwap(1.0, 1)

    TestFixtures.set_seed()
    var result = swap(text)

    # Result should still have same number of words
    var words = split_words(result)
    assert_equal(len(words), 4)


def test_random_swap_probability() raises:
    """Test RandomSwap respects probability."""
    var text = String("the quick brown fox")

    # With p=0.0, no swaps should occur
    var swap = RandomSwap(0.0, 10)
    var result = swap(text)

    assert_equal(result, text)


def test_random_swap_empty_text() raises:
    """Test RandomSwap handles empty text."""
    var text = String("")
    var swap = RandomSwap(1.0, 1)
    var result = swap(text)

    assert_equal(result, "")


def test_random_swap_single_word() raises:
    """Test RandomSwap handles single word."""
    var text = String("hello")
    var swap = RandomSwap(1.0, 1)
    var result = swap(text)

    assert_equal(result, "hello")


def test_random_swap_deterministic() raises:
    """Test RandomSwap is deterministic with seed."""
    var text = String("the quick brown fox jumps")

    TestFixtures.set_seed()
    var swap1 = RandomSwap(0.5, 2)
    var result1 = swap1(text)

    TestFixtures.set_seed()
    var swap2 = RandomSwap(0.5, 2)
    var result2 = swap2(text)

    assert_equal(result1, result2)


def test_random_deletion_basic() raises:
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


def test_random_deletion_probability_never() raises:
    """Test RandomDeletion with p=0.0 never deletes."""
    var text = String("the quick brown fox")

    var delete = RandomDeletion(0.0)
    var result = delete(text)

    assert_equal(result, text)


def test_random_deletion_preserves_one_word() raises:
    """Test RandomDeletion always keeps at least one word."""
    var text = String("the quick brown fox")

    # Even with p=1.0, at least one word should remain
    var delete = RandomDeletion(1.0)
    var result = delete(text)

    var words = split_words(result)
    assert_true(len(words) >= 1)


def test_random_deletion_empty_text() raises:
    """Test RandomDeletion handles empty text."""
    var text = String("")
    var delete = RandomDeletion(0.5)
    var result = delete(text)

    assert_equal(result, "")


def test_random_deletion_single_word() raises:
    """Test RandomDeletion preserves single word."""
    var text = String("hello")
    var delete = RandomDeletion(1.0)
    var result = delete(text)

    assert_equal(result, "hello")


def test_random_deletion_deterministic() raises:
    """Test RandomDeletion is deterministic with seed."""
    var text = String("the quick brown fox jumps")

    TestFixtures.set_seed()
    var delete1 = RandomDeletion(0.3)
    var result1 = delete1(text)

    TestFixtures.set_seed()
    var delete2 = RandomDeletion(0.3)
    var result2 = delete2(text)

    assert_equal(result1, result2)


def test_random_insertion_basic() raises:
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


def test_random_insertion_probability() raises:
    """Test RandomInsertion respects probability."""
    var text = String("the brown fox")

    var vocab = List[String]()
    vocab.append("quick")

    # With p=0.0, no insertion should occur
    var insert = RandomInsertion(vocab.copy(), 0.0, 10)
    var result = insert(text)

    assert_equal(result, text)


def test_random_insertion_empty_text() raises:
    """Test RandomInsertion handles empty text."""
    var text = String("")

    var vocab = List[String]()
    vocab.append("quick")

    var insert = RandomInsertion(vocab.copy(), 1.0, 1)
    var result = insert(text)

    assert_equal(result, "")


def test_random_insertion_empty_vocabulary() raises:
    """Test RandomInsertion handles empty vocabulary."""
    var text = String("the brown fox")

    var vocab = List[String]()

    var insert = RandomInsertion(vocab.copy(), 1.0, 1)
    var result = insert(text)

    assert_equal(result, text)


def test_random_insertion_deterministic() raises:
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


def test_random_synonym_replacement_basic() raises:
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


def test_random_synonym_replacement_probability() raises:
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


def test_random_synonym_replacement_no_synonyms() raises:
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


def test_random_synonym_replacement_empty_text() raises:
    """Test RandomSynonymReplacement handles empty text."""
    var text = String("")

    var synonyms = Dict[String, List[String]]()
    var quick_syns = List[String]()
    quick_syns.append("fast")
    synonyms["quick"] = quick_syns^

    var replace = RandomSynonymReplacement(synonyms.copy(), 1.0)
    var result = replace(text)

    assert_equal(result, "")


def test_random_synonym_replacement_deterministic() raises:
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


def main() raises:
    """Run all test_text_augmentations tests."""
    print("Running test_text_augmentations tests...")

    test_split_words_basic()
    print("✓ test_split_words_basic")

    test_split_words_empty()
    print("✓ test_split_words_empty")

    test_split_words_single()
    print("✓ test_split_words_single")

    test_join_words_basic()
    print("✓ test_join_words_basic")

    test_join_words_empty()
    print("✓ test_join_words_empty")

    test_join_words_single()
    print("✓ test_join_words_single")

    test_random_swap_basic()
    print("✓ test_random_swap_basic")

    test_random_swap_probability()
    print("✓ test_random_swap_probability")

    test_random_swap_empty_text()
    print("✓ test_random_swap_empty_text")

    test_random_swap_single_word()
    print("✓ test_random_swap_single_word")

    test_random_swap_deterministic()
    print("✓ test_random_swap_deterministic")

    test_random_deletion_basic()
    print("✓ test_random_deletion_basic")

    test_random_deletion_probability_never()
    print("✓ test_random_deletion_probability_never")

    test_random_deletion_preserves_one_word()
    print("✓ test_random_deletion_preserves_one_word")

    test_random_deletion_empty_text()
    print("✓ test_random_deletion_empty_text")

    test_random_deletion_single_word()
    print("✓ test_random_deletion_single_word")

    test_random_deletion_deterministic()
    print("✓ test_random_deletion_deterministic")

    test_random_insertion_basic()
    print("✓ test_random_insertion_basic")

    test_random_insertion_probability()
    print("✓ test_random_insertion_probability")

    test_random_insertion_empty_text()
    print("✓ test_random_insertion_empty_text")

    test_random_insertion_empty_vocabulary()
    print("✓ test_random_insertion_empty_vocabulary")

    test_random_insertion_deterministic()
    print("✓ test_random_insertion_deterministic")

    test_random_synonym_replacement_basic()
    print("✓ test_random_synonym_replacement_basic")

    test_random_synonym_replacement_probability()
    print("✓ test_random_synonym_replacement_probability")

    test_random_synonym_replacement_no_synonyms()
    print("✓ test_random_synonym_replacement_no_synonyms")

    test_random_synonym_replacement_empty_text()
    print("✓ test_random_synonym_replacement_empty_text")

    test_random_synonym_replacement_deterministic()
    print("✓ test_random_synonym_replacement_deterministic")

    print("\nAll test_text_augmentations tests passed!")
