# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_loss_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for loss utility functions - Part 2.

Tests cover:
- validate_tensor_dtypes: Dtype validation
- compute_one_minus_tensor: 1-x computation
- compute_sign_tensor: Sign computation

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.loss_utils import (
    validate_tensor_dtypes,
    compute_one_minus_tensor,
    compute_sign_tensor,
)


# ============================================================================
# validate_tensor_dtypes Tests
# ============================================================================


fn test_validate_tensor_dtypes_matching() raises:
    """Test validate_tensor_dtypes accepts matching dtypes."""
    var shape = List[Int]()
    shape.append(3)

    var tensor1 = zeros(shape, DType.float32)
    var tensor2 = ones(shape, DType.float32)

    # Should not raise
    validate_tensor_dtypes(tensor1, tensor2, "test_op")


fn test_validate_tensor_dtypes_mismatch() raises:
    """Test validate_tensor_dtypes raises on dtype mismatch."""
    var shape = List[Int]()
    shape.append(3)

    var tensor1 = zeros(shape, DType.float32)
    var tensor2 = zeros(shape, DType.float64)

    var error_raised = False
    try:
        validate_tensor_dtypes(tensor1, tensor2, "test_op")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error on dtype mismatch")


# ============================================================================
# compute_one_minus_tensor Tests
# ============================================================================


fn test_compute_one_minus_tensor_half() raises:
    """Test compute_one_minus_tensor with 0.5 gives 0.5."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, 0.5, DType.float32)

    var result = compute_one_minus_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.5, tolerance=1e-6)


fn test_compute_one_minus_tensor_zero() raises:
    """Test compute_one_minus_tensor with 0.0 gives 1.0."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = zeros(shape, DType.float32)

    var result = compute_one_minus_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 1.0, tolerance=1e-6)


fn test_compute_one_minus_tensor_one() raises:
    """Test compute_one_minus_tensor with 1.0 gives 0.0."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, 1.0, DType.float32)

    var result = compute_one_minus_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


# ============================================================================
# compute_sign_tensor Tests
# ============================================================================


fn test_compute_sign_tensor_positive() raises:
    """Test compute_sign_tensor with positive values."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, 5.0, DType.float32)

    var result = compute_sign_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 1.0, tolerance=1e-6)


fn test_compute_sign_tensor_negative() raises:
    """Test compute_sign_tensor with negative values."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, -3.0, DType.float32)

    var result = compute_sign_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), -1.0, tolerance=1e-6)


fn test_compute_sign_tensor_zero() raises:
    """Test compute_sign_tensor with zero."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = zeros(shape, DType.float32)

    var result = compute_sign_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


fn main() raises:
    """Run loss utility tests - Part 2."""
    print("Running loss utility tests - Part 2...")

    # validate_tensor_dtypes tests
    test_validate_tensor_dtypes_matching()
    test_validate_tensor_dtypes_mismatch()

    # compute_one_minus_tensor tests
    test_compute_one_minus_tensor_half()
    test_compute_one_minus_tensor_zero()
    test_compute_one_minus_tensor_one()

    # compute_sign_tensor tests
    test_compute_sign_tensor_positive()
    test_compute_sign_tensor_negative()
    test_compute_sign_tensor_zero()

    print("All loss utility tests - Part 2 passed!")
