"""Text transformation and augmentation utilities.

This module provides transformations for augmenting text data for NLP tasks.
Implements basic word-level augmentations including synonym replacement,
random insertion, random swap, and random deletion.

All Random* text transforms use RandomTransformBase for probability handling.

Limitations:
- Basic word-level operations (split on spaces)
- English-centric approach
- Simple synonym dictionary (not semantic embeddings)
- May produce ungrammatical text
- No advanced NLP features
"""

from std.random import random_si64
from odyssey.data.random_transform_base import (
    RandomTransformBase,
    random_float,
)


# ============================================================================
# TextTransform Trait
# ============================================================================


trait TextTransform:
    """Base interface for text transforms.

    Text transforms modify string data and return transformed copies.
    Unlike the Transform trait which works with AnyTensor, this works with String.
    """

    def __call__(self, text: String) raises -> String:
        """Apply the transform to text.

        Args:
            text: Input text string.

        Returns:
            Transformed text string.

        Raises:
            Error: If transform cannot be applied.
        """
        ...


# ============================================================================
# Helper Functions
# ============================================================================


def split_words(text: String) raises -> List[String]:
    """Split text into words by spaces.

    Simple space-based tokenization. Does not handle punctuation specially.

    Args:
        text: Input text to split.

    Returns:
        List of words (space-separated tokens).

    Raises:
        Error: If splitting operation fails.
    """
    # Use built-in split method
    var parts = text.split(" ")

    # Filter out empty strings that may result from multiple spaces
    var words = List[String]()
    for i in range(len(parts)):
        if (String(parts[i])).byte_length() > 0:
            words.append(String(parts[i]))

    return words^


def join_words(words: List[String]) raises -> String:
    """Join words into text with spaces.

    Args:
        words: List of words to join.

    Returns:
        Joined text with spaces between words.

    Raises:
        Error: If joining operation fails.
    """
    if len(words) == 0:
        return String("")

    var result = words[0]
    for i in range(1, len(words)):
        result += " " + words[i]

    return result


# ============================================================================
# Text Augmentation Transforms
# ============================================================================


struct RandomSwap(Copyable, Movable, TextTransform):
    """Randomly swap positions of word pairs.

    Swaps adjacent or nearby word positions with configurable probability.
    Helps create variations while preserving overall meaning.

    Uses RandomTransformBase for probability handling.

    Example:
        ```mojo
        the quick brown fox" -> "quick the brown fox" (first two swapped).
        ```
    """

    var base: RandomTransformBase
    """Random transform base for probability handling."""
    var n: Int
    """Number of swap operations to attempt."""

    def __init__(out self, p: Float64 = 0.15, n: Int = 2):
        """Create random swap transform.

        Args:
            p: Probability of performing each swap (0.0 to 1.0).
            n: Number of swap operations to attempt.
        """
        self.base = RandomTransformBase(p)
        self.n = n

    def __call__(self, text: String) raises -> String:
        """Randomly swap word pairs in text.

        Args:
            text: Input text.

        Returns:
            Text with randomly swapped words.

        Raises:
            Error: If operation fails.
        """
        # Handle empty or single-word text
        if text.byte_length() == 0:
            return text

        var words = split_words(text)
        if len(words) <= 1:
            return text

        # Perform n swap operations
        for _ in range(self.n):
            # Check probability - skip if should_apply returns False
            if not self.base.should_apply():
                continue

            # Pick two random positions
            var idx1 = Int(random_si64(Int64(0), Int64(len(words) - 1)))
            var idx2 = Int(random_si64(Int64(0), Int64(len(words) - 1)))

            # Ensure different positions
            if idx1 != idx2:
                # Swap
                var temp = words[idx1]
                words[idx1] = words[idx2]
                words[idx2] = temp

        return join_words(words)


struct RandomDeletion(Copyable, Movable, TextTransform):
    """Randomly delete words from text.

    Deletes words with specified probability while ensuring at least
    one word remains. Helps create shorter variations.

    Uses RandomTransformBase for probability handling.

    Example:
        ```mojo
        the quick brown fox" -> "quick brown fox" (deleted "the").
        ```
    """

    var base: RandomTransformBase
    """Random transform base for probability handling."""

    def __init__(out self, p: Float64 = 0.1):
        """Create random deletion transform.

        Args:
            p: Probability of deleting each word (0.0 to 1.0).
        """
        self.base = RandomTransformBase(p)

    def __call__(self, text: String) raises -> String:
        """Randomly delete words from text.

        Ensures at least one word remains even if all would be deleted.

        Args:
            text: Input text.

        Returns:
            Text with some words randomly deleted.

        Raises:
            Error: If operation fails.
        """
        # Handle empty text
        if text.byte_length() == 0:
            return text

        var words = split_words(text)
        if len(words) == 0:
            return text

        # If only one word, don't delete
        if len(words) == 1:
            return text

        # Decide which words to keep
        var kept_words = List[String]()
        for i in range(len(words)):
            if not self.base.should_apply():
                # Keep this word if should_apply returns False
                kept_words.append(words[i])

        # Ensure at least one word remains
        if len(kept_words) == 0:
            # Keep a random word
            var idx = Int(random_si64(Int64(0), Int64(len(words) - 1)))
            kept_words.append(words[idx])

        return join_words(kept_words)


struct RandomInsertion(Copyable, Movable, TextTransform):
    """Insert random words from vocabulary into text.

    Inserts words from a predefined vocabulary at random positions.
    Helps increase lexical diversity.

    Uses RandomTransformBase for probability handling.

    Example:
        ```mojo
        the brown fox" -> "the quick brown fox" (inserted "quick").
        ```
    """

    var base: RandomTransformBase
    """Random transform base for probability handling."""
    var n: Int
    """Number of words to insert."""
    var vocabulary: List[String]
    """List of words to choose from for insertion."""

    def __init__(
        out self, var vocabulary: List[String], p: Float64 = 0.1, n: Int = 1
    ):
        """Create random insertion transform.

        Args:
            vocabulary: List of words to choose from for insertion.
            p: Probability of performing insertion (0.0 to 1.0).
            n: Number of words to insert.
        """
        self.base = RandomTransformBase(p)
        self.vocabulary = vocabulary^
        self.n = n

    def __call__(self, text: String) raises -> String:
        """Insert random words from vocabulary into text.

        Args:
            text: Input text.

        Returns:
            Text with randomly inserted words.

        Raises:
            Error: If operation fails.
        """
        # Handle empty text or empty vocabulary
        if text.byte_length() == 0 or len(self.vocabulary) == 0:
            return text

        var words = split_words(text)
        if len(words) == 0:
            return text

        # Perform n insertion operations
        for _ in range(self.n):
            # Check probability - skip if should_apply returns False
            if not self.base.should_apply():
                continue

            # Pick random word from vocabulary
            var vocab_idx = Int(
                random_si64(Int64(0), Int64(len(self.vocabulary) - 1))
            )
            var word_to_insert = self.vocabulary[vocab_idx]

            # Pick random position to insert (0 to len(words) inclusive)
            var insert_pos = Int(random_si64(Int64(0), Int64(len(words) + 1)))

            # Insert word at position
            var new_words = List[String]()
            for i in range(len(words)):
                if i == insert_pos:
                    new_words.append(word_to_insert)
                new_words.append(words[i])

            # Handle case where insert_pos == len(words)
            if insert_pos == len(words):
                new_words.append(word_to_insert)

            words = new_words^

        return join_words(words)


struct RandomSynonymReplacement(Copyable, Movable, TextTransform):
    """Replace random words with synonyms from dictionary.

    Uses a simple synonym dictionary to replace words with alternatives.
    This is a conservative augmentation that preserves meaning well.

    Uses RandomTransformBase for probability handling.

    Example:
        ```mojo
        the quick fox" -> "the fast fox" (replaced "quick" with "fast").
        ```
    """

    var base: RandomTransformBase
    """Random transform base for probability handling."""
    var synonyms: Dict[String, List[String]]
    """Dictionary mapping words to lists of synonyms."""

    def __init__(
        out self, var synonyms: Dict[String, List[String]], p: Float64 = 0.2
    ):
        """Create random synonym replacement transform.

        Args:
            synonyms: Dictionary mapping words to lists of synonyms.
            p: Probability of replacing each word (0.0 to 1.0).
        """
        self.base = RandomTransformBase(p)
        self.synonyms = synonyms^

    def __call__(self, text: String) raises -> String:
        """Replace random words with synonyms.

        Args:
            text: Input text.

        Returns:
            Text with some words replaced by synonyms.

        Raises:
            Error: If operation fails.
        """
        # Handle empty text
        if text.byte_length() == 0:
            return text

        var words = split_words(text)
        if len(words) == 0:
            return text

        # Process each word
        var result_words = List[String]()
        for i in range(len(words)):
            var word = words[i]

            # Check if should replace (if should_apply returns True and word has synonyms)
            if self.base.should_apply() and word in self.synonyms:
                # Get synonyms for this word
                var syns = self.synonyms[word].copy()
                if len(syns) > 0:
                    # Pick random synonym
                    var syn_idx = Int(
                        random_si64(Int64(0), Int64(len(syns) - 1))
                    )
                    result_words.append(syns[syn_idx])
                else:
                    # No synonyms available, keep original
                    result_words.append(word)
            else:
                # Don't replace
                result_words.append(word)

        return join_words(result_words)


# ============================================================================
# Text Pipeline (Compose)
# ============================================================================

# COMMENTED OUT: Issue #2086 - Mojo trait storage limitation prevents List[TextTransform]
# This will need to be refactored to use a different approach (e.g., union of concrete types,
# or wait for Mojo to support trait objects in collections)

# struct TextCompose(TextTransform, Copyable, Movable):
#     """Compose multiple text transforms sequentially.
#
#     Applies text transforms in order, passing output of each to the next.
#     Similar to the Compose transform for tensors but works with strings.
#     """
#
#     var transforms: List[TextTransform]
#
#     fn __init__(out self, var transforms: List[TextTransform]):
#         """Create composition of text transforms.
#
#         Args:
#             transforms: List of text transforms to apply in order.
#         """
#         self.transforms = transforms^
#
#     fn __call__(self, text: String) raises -> String:
#         """Apply all text transforms sequentially.
#
#         Args:
#            text: Input text.
#
#         Returns:
#            Transformed text after all transforms.
#
#         Raises:
#            Error if any transform cannot be applied.
#         """
#         var result = text
#         for t in self.transforms:
#             result = t(result)
#         return result
#
#     fn __len__(self) -> Int:
#         """Return number of transforms."""
#         return len(self.transforms)
#
#     fn append(mut self, transform: TextTransform):
#         """Add a transform to the pipeline.
#
#         Args:
#             transform: Text transform to add.
#         """
#         self.transforms.append(transform)
#
#
# # Type comptime for more intuitive naming
# comptime TextPipeline = TextCompose
