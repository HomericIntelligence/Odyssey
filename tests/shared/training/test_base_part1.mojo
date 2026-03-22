# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_base.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for training base module (part 1 of 2).

This module tests the gradient utilities and numerical safety functions:
- has_nan_or_inf: NaN/Inf detection for numerical stability
- compute_gradient_norm: L1/L2 norm computation for gradient clipping and diagnostics
- is_valid_loss (valid and NaN cases)

Split from test_base.mojo per ADR-009 (≤10 fn test_ per file).
"""

from shared.core.any_tensor import (
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


fn test_has_nan_or_inf_no_issues() raises:
    """Test has_nan_or_inf returns False for normal values."""
    print("Testing has_nan_or_inf with normal values...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var tensor = full(shape, 0.5, DType.float64)
    var result = has_nan_or_inf(tensor)

    if result:
        raise Error("has_nan_or_inf should return False for normal values")

    print("  ✓ has_nan_or_inf correctly returns False for normal values")


fn test_has_nan_or_inf_with_nan() raises:
    """Test has_nan_or_inf detects NaN values."""
    print("Testing has_nan_or_inf with NaN...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var tensor = full(shape, 1.0, DType.float64)

    # Insert a NaN value
    var ptr = tensor._data.bitcast[Float64]()
    ptr[0] = nan[DType.float64]()

    var result = has_nan_or_inf(tensor)

    if not result:
        raise Error("has_nan_or_inf should detect NaN values")

    print("  ✓ has_nan_or_inf correctly detects NaN values")


fn test_has_nan_or_inf_with_inf() raises:
    """Test has_nan_or_inf detects Inf values."""
    print("Testing has_nan_or_inf with Inf...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var tensor = full(shape, 1.0, DType.float64)

    # Insert an Inf value
    var ptr = tensor._data.bitcast[Float64]()
    ptr[1] = inf[DType.float64]()

    var result = has_nan_or_inf(tensor)

    if not result:
        raise Error("has_nan_or_inf should detect Inf values")

    print("  ✓ has_nan_or_inf correctly detects Inf values")


fn test_compute_gradient_norm_l2() raises:
    """Test compute_gradient_norm with L2 norm (default)."""
    print("Testing compute_gradient_norm L2 norm...")

    # Create test gradients: [3.0, 4.0] should have L2 norm = 5.0
    var shape = List[Int]()
    shape.append(2)

    var tensor = full(shape, 0.0, DType.float64)
    var ptr = tensor._data.bitcast[Float64]()
    ptr[0] = 3.0
    ptr[1] = 4.0

    var params: List[AnyTensor] = []
    params.append(tensor)

    var norm = compute_gradient_norm(params, "L2")

    # Check if L2 norm is approximately 5.0
    if norm < 4.9 or norm > 5.1:
        raise Error("L2 norm should be 5.0, got " + String(norm))

    print("  Expected L2 norm: 5.0, Got:", norm)
    print("  ✓ L2 norm computation correct")


fn test_compute_gradient_norm_l1() raises:
    """Test compute_gradient_norm with L1 norm."""
    print("Testing compute_gradient_norm L1 norm...")

    # Create test gradients: [1.0, 2.0, 3.0] should have L1 norm = 6.0
    var shape = List[Int]()
    shape.append(3)

    var tensor = full(shape, 0.0, DType.float64)
    var ptr = tensor._data.bitcast[Float64]()
    ptr[0] = 1.0
    ptr[1] = 2.0
    ptr[2] = 3.0

    var params: List[AnyTensor] = []
    params.append(tensor)

    var norm = compute_gradient_norm(params, "L1")

    # Check if L1 norm is 6.0
    if norm < 5.9 or norm > 6.1:
        raise Error("L1 norm should be 6.0, got " + String(norm))

    print("  Expected L1 norm: 6.0, Got:", norm)
    print("  ✓ L1 norm computation correct")


fn test_compute_gradient_norm_multiple_tensors() raises:
    """Test compute_gradient_norm with multiple tensors."""
    print("Testing compute_gradient_norm with multiple tensors...")

    # Create two tensors: [3.0, 4.0] and [0.0] for combined L2 norm = 5.0
    var shape1 = List[Int]()
    shape1.append(2)

    var tensor1 = full(shape1, 0.0, DType.float64)
    var ptr1 = tensor1._data.bitcast[Float64]()
    ptr1[0] = 3.0
    ptr1[1] = 4.0

    var shape2 = List[Int]()
    shape2.append(1)

    var tensor2 = full(shape2, 0.0, DType.float64)

    var params: List[AnyTensor] = []
    params.append(tensor1)
    params.append(tensor2)

    var norm = compute_gradient_norm(params, "L2")

    # Check combined norm
    if norm < 4.9 or norm > 5.1:
        raise Error("Combined L2 norm should be 5.0, got " + String(norm))

    print("  Combined L2 norm: 5.0, Got:", norm)
    print("  ✓ Multiple tensor norm computation correct")


fn test_is_valid_loss_valid() raises:
    """Test is_valid_loss with valid loss values."""
    print("Testing is_valid_loss with valid values...")

    var result = is_valid_loss(0.5)

    if not result:
        raise Error("is_valid_loss should return True for valid values")

    print("  ✓ is_valid_loss correctly validates normal loss")


fn test_is_valid_loss_nan() raises:
    """Test is_valid_loss with NaN."""
    print("Testing is_valid_loss with NaN...")

    var result = is_valid_loss(nan[DType.float64]())

    if result:
        raise Error("is_valid_loss should return False for NaN")

    print("  ✓ is_valid_loss correctly rejects NaN")


fn main() raises:
    """Run all tests."""
    print("Running gradient utility tests (part 1)...\n")

    try:
        test_has_nan_or_inf_no_issues()
        test_has_nan_or_inf_with_nan()
        test_has_nan_or_inf_with_inf()
        test_compute_gradient_norm_l2()
        test_compute_gradient_norm_l1()
        test_compute_gradient_norm_multiple_tensors()
        test_is_valid_loss_valid()
        test_is_valid_loss_nan()

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except e:
        print("\nTest failed with error:", e)
        raise
