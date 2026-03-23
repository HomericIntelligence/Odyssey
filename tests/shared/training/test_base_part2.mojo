# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_base.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for training base module (part 2 of 2).

This module tests the gradient clipping and loss validation functions:
- is_valid_loss (Inf case)
- clip_gradients: Legacy gradient clipping for Float64 lists

Split from test_base.mojo per ADR-009 (≤10 fn test_ per file).
"""

from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
)
from shared.core.numerical_safety import has_nan, has_inf
from shared.training.base import (
    has_nan_or_inf,
    compute_gradient_norm,
    is_valid_loss,
    clip_gradients,
)
from math import nan, inf, sqrt
from collections import List


fn test_is_valid_loss_inf() raises:
    """Test is_valid_loss with Inf."""
    print("Testing is_valid_loss with Inf...")

    var result = is_valid_loss(inf[DType.float64]())

    if result:
        raise Error("is_valid_loss should return False for Inf")

    print("  ✓ is_valid_loss correctly rejects Inf")


fn test_clip_gradients_no_clipping_needed() raises:
    """Test clip_gradients when norm is below threshold."""
    print("Testing clip_gradients with norm below threshold...")

    var gradients: List[Float64] = []
    gradients.append(0.1)
    gradients.append(0.2)
    gradients.append(0.3)

    var original_sum = gradients[0] + gradients[1] + gradients[2]

    var clipped = clip_gradients(gradients^, 10.0)

    var new_sum = clipped[0] + clipped[1] + clipped[2]

    # Gradients should be unchanged when norm < max_norm
    if (new_sum < original_sum - 0.01) or (new_sum > original_sum + 0.01):
        raise Error("Gradients should be unchanged when norm < max_norm")

    print("  Original sum: ", original_sum)
    print("  Clipped sum: ", new_sum)
    print("  ✓ Gradients correctly unchanged when no clipping needed")


fn test_clip_gradients_clipping_needed() raises:
    """Test clip_gradients scales gradients when norm exceeds threshold."""
    print("Testing clip_gradients with norm above threshold...")

    var gradients: List[Float64] = []
    gradients.append(3.0)
    gradients.append(4.0)

    var clipped = clip_gradients(gradients^, 1.0)

    # Norm of [3.0, 4.0] is 5.0, so scaled by 1.0/5.0 = 0.2
    # Result should be [0.6, 0.8]
    # Check that clipped norm is 1.0
    var norm_sq = clipped[0] * clipped[0] + clipped[1] * clipped[1]
    var norm = sqrt(norm_sq)

    if norm < 0.99 or norm > 1.01:
        raise Error("Clipped norm should be 1.0, got " + String(norm))

    print("  Original norm: 5.0, Clipped norm:", norm)
    print("  ✓ Gradients correctly clipped to max_norm")


fn main() raises:
    """Run all tests."""
    print("Running gradient utility tests (part 2)...\n")

    try:
        test_is_valid_loss_inf()
        test_clip_gradients_no_clipping_needed()
        test_clip_gradients_clipping_needed()

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except e:
        print("\nTest failed with error:", e)
        raise
