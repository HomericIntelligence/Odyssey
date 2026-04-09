"""Base patterns and utilities for probabilistic transforms.

This module provides shared probability handling for all Random* transforms
(RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomErasing, etc.).

The RandomTransformBase struct encapsulates the probability field and the
should_apply() method to determine whether a random transform should be
executed based on its configured probability.

Pattern:
    All Random* transforms that use probability-based application should:
    1. Use RandomTransformBase to handle probability logic.
    2. Call should_apply() to decide whether to apply the transform.
    3. Keep transform-specific logic separate from probability logic.

Example:
    ```
    var transform = RandomHorizontalFlip(p=0.5)
    if transform.should_apply():
        # Apply horizontal flip logic
    ```
"""

from std.random import random_si64


# ============================================================================
# Random Float Generation
# ============================================================================


def random_float() -> Float64:
    """Generate random float in [0, 1) with high precision.

        Uses 1 billion possible values for better probability distribution.

    Returns:
            Random float in range [0.0, 1.0).
    """
    return Float64(random_si64(0, 1000000000)) / 1000000000.0


# ============================================================================
# RandomTransformBase
# ============================================================================


struct RandomTransformBase(Copyable, Movable):
    """Base struct for probabilistic transforms.

    Encapsulates probability handling and decision logic for Random* transforms
    All random transforms that apply conditionally based on probability should
    use this pattern.

    Fields:
        p: Probability of applying the transform (0.0 to 1.0).

    Example:
        ```mojo
        var base = RandomTransformBase(0.5)
        if base.should_apply():
            # Apply the transform
        ```
    """

    var p: Float64

    def __init__(out self, p: Float64 = 0.5):
        """Create probabilistic transform base.

        Args:
            p: Probability of applying the transform (0.0 to 1.0).
        """
        self.p = p

    def should_apply(self) -> Bool:
        """Determine if transform should be applied based on probability.

        Generates a random value in [0, 1) and compares against the
        configured probability. Returns True if transform should apply.

        Returns:
            True if transform should be applied, False otherwise.
        """
        var rand_val = random_float()
        return rand_val < self.p
