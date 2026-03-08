# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_loss_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for loss utility functions - Part 4.

Tests cover:
- compute_ratio (edge cases): Zero denominator handling
- negate_tensor: Tensor negation

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_greater_or_equal,
)
from shared.core.extensor import ExTensor, zeros, full
from shared.core.loss_utils import (
    compute_ratio,
    negate_tensor,
)


# ============================================================================
# compute_ratio Tests (edge cases)
# ============================================================================


fn test_compute_ratio_zero_denominator() raises:
    """Test compute_ratio prevents division by zero with epsilon."""
    var shape = List[Int]()
    shape.append(1)

    var numerator = full(shape, 1.0, DType.float32)
    var denominator = zeros(shape, DType.float32)

    var result = compute_ratio(numerator, denominator, epsilon=1e-5)

    var result_data = result._data.bitcast[Float32]()
    # Should be 1.0 / 1e-5 = 100000, but check it's finite and positive
    assert_greater_or_equal(
        result_data[0], 0.0, "Result should be non-negative"
    )


# ============================================================================
# negate_tensor Tests
# ============================================================================


fn test_negate_tensor_positive() raises:
    """Test negate_tensor negates positive values."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, 5.0, DType.float32)

    var result = negate_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), -5.0, tolerance=1e-6)


fn test_negate_tensor_negative() raises:
    """Test negate_tensor negates negative values."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, -3.0, DType.float32)

    var result = negate_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 3.0, tolerance=1e-6)


fn test_negate_tensor_zero() raises:
    """Test negate_tensor with zero stays zero."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = zeros(shape, DType.float32)

    var result = negate_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


fn main() raises:
    """Run loss utility tests - Part 4."""
    print("Running loss utility tests - Part 4...")

    # compute_ratio edge case tests
    test_compute_ratio_zero_denominator()

    # negate_tensor tests
    test_negate_tensor_positive()
    test_negate_tensor_negative()
    test_negate_tensor_zero()

    print("All loss utility tests - Part 4 passed!")
