# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_loss_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for loss utility functions - Part 3.

Tests cover:
- blend_tensors: Tensor blending with mask
- compute_difference: Tensor subtraction
- compute_product: Element-wise multiplication
- compute_ratio (basic): Division utility

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.loss_utils import (
    blend_tensors,
    compute_difference,
    compute_product,
    compute_ratio,
)


# ============================================================================
# blend_tensors Tests
# ============================================================================


fn test_blend_tensors_all_first() raises:
    """Test blend_tensors selects first tensor with all 1s mask."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 1.0, DType.float32)
    var tensor2 = full(shape, 2.0, DType.float32)
    var mask = ones(shape, DType.float32)  # All 1s

    var result = blend_tensors(tensor1, tensor2, mask)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 1.0, tolerance=1e-6)


fn test_blend_tensors_all_second() raises:
    """Test blend_tensors selects second tensor with all 0s mask."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 1.0, DType.float32)
    var tensor2 = full(shape, 2.0, DType.float32)
    var mask = zeros(shape, DType.float32)  # All 0s

    var result = blend_tensors(tensor1, tensor2, mask)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 2.0, tolerance=1e-6)


fn test_blend_tensors_mixed() raises:
    """Test blend_tensors with mixed mask values."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 10.0, DType.float32)
    var tensor2 = full(shape, 20.0, DType.float32)

    var mask = zeros(shape, DType.float32)
    var mask_data = mask._data.bitcast[Float32]()
    mask_data[0] = 1.0  # First element selects tensor1
    mask_data[1] = 0.0  # Second element selects tensor2

    var result = blend_tensors(tensor1, tensor2, mask)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 10.0, tolerance=1e-6)
    assert_almost_equal(Float64(result_data[1]), 20.0, tolerance=1e-6)


# ============================================================================
# compute_difference Tests
# ============================================================================


fn test_compute_difference_basic() raises:
    """Test compute_difference computes tensor1 - tensor2."""
    var shape = List[Int]()
    shape.append(3)

    var tensor1 = full(shape, 5.0, DType.float32)
    var tensor2 = full(shape, 2.0, DType.float32)

    var result = compute_difference(tensor1, tensor2)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 3.0, tolerance=1e-6)


fn test_compute_difference_shape_mismatch() raises:
    """Test compute_difference raises on shape mismatch."""
    var shape1 = List[Int]()
    shape1.append(2)

    var shape2 = List[Int]()
    shape2.append(3)

    var tensor1 = zeros(shape1, DType.float32)
    var tensor2 = zeros(shape2, DType.float32)

    var error_raised = False
    try:
        _ = compute_difference(tensor1, tensor2)
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error on shape mismatch")


# ============================================================================
# compute_product Tests
# ============================================================================


fn test_compute_product_basic() raises:
    """Test compute_product computes element-wise multiplication."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 3.0, DType.float32)
    var tensor2 = full(shape, 4.0, DType.float32)

    var result = compute_product(tensor1, tensor2)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 12.0, tolerance=1e-6)


fn test_compute_product_with_zero() raises:
    """Test compute_product with zero gives zero."""
    var shape = List[Int]()
    shape.append(1)

    var tensor1 = full(shape, 5.0, DType.float32)
    var tensor2 = zeros(shape, DType.float32)

    var result = compute_product(tensor1, tensor2)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


# ============================================================================
# compute_ratio Tests (basic)
# ============================================================================


fn test_compute_ratio_basic() raises:
    """Test compute_ratio computes numerator / denominator."""
    var shape = List[Int]()
    shape.append(1)

    var numerator = full(shape, 6.0, DType.float32)
    var denominator = full(shape, 2.0, DType.float32)

    var result = compute_ratio(numerator, denominator)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 3.0, tolerance=1e-6)


fn main() raises:
    """Run loss utility tests - Part 3."""
    print("Running loss utility tests - Part 3...")

    # blend_tensors tests
    test_blend_tensors_all_first()
    test_blend_tensors_all_second()
    test_blend_tensors_mixed()

    # compute_difference tests
    test_compute_difference_basic()
    test_compute_difference_shape_mismatch()

    # compute_product tests
    test_compute_product_basic()
    test_compute_product_with_zero()

    # compute_ratio tests (basic)
    test_compute_ratio_basic()

    print("All loss utility tests - Part 3 passed!")
