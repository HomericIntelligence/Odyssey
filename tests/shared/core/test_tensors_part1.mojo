"""Tests for tensor creation operations (Part 1 of 3).

Tests cover:
- Tensor creation: zeros, ones, full, empty, arange, eye, linspace

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_tensors.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal_int,
    assert_close_float,
    assert_equal,
    assert_almost_equal,
    assert_dtype_equal,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    empty,
    arange,
    eye,
    linspace,
    zeros_like,
    ones_like,
    full_like,
)


# ============================================================================
# Tensor Creation Tests
# ============================================================================


fn test_zeros_creation() raises:
    """Test zeros tensor creation."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    # Check shape
    assert_equal_int(t.shape()[0], 3)
    assert_equal_int(t.shape()[1], 4)

    # Check all values are zero
    for i in range(12):
        assert_close_float(
            Float64(t._data.bitcast[Float32]()[i]), Float64(0.0), atol=1e-5
        )


fn test_ones_creation() raises:
    """Test ones tensor creation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = ones(shape, DType.float32)

    # Check shape
    assert_equal(t.shape()[0], 2)
    assert_equal(t.shape()[1], 3)

    # Check all values are one
    for i in range(6):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


fn test_full_creation() raises:
    """Test full tensor creation with specified value."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var t = full(shape, 3.14, DType.float32)

    # Check shape
    assert_equal(t.shape()[0], 2)
    assert_equal(t.shape()[1], 2)

    # Check all values are 3.14
    for i in range(4):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(3.14), tolerance=1e-5
        )


fn test_empty_creation() raises:
    """Test empty tensor creation (uninitialized memory)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var t = empty(shape, DType.float32)

    # Check shape is correct (values are uninitialized)
    assert_equal(t.shape()[0], 3)
    assert_equal(t.shape()[1], 3)
    assert_equal(t.numel(), 9)


fn test_arange_creation() raises:
    """Test arange tensor creation (sequential values)."""
    var t = arange(0, 10, 1, DType.float32)

    # Check values: [0, 1, 2, ..., 9]
    assert_equal(t.numel(), 10)
    for i in range(10):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(i), tolerance=1e-5
        )


fn test_arange_with_step() raises:
    """Test arange with non-unit step."""
    var t = arange(0, 10, 2, DType.float32)

    # Check values: [0, 2, 4, 6, 8]
    assert_equal(t.numel(), 5)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(6.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(8.0), tolerance=1e-5
    )


fn test_eye_creation() raises:
    """Test identity matrix creation."""
    var t = eye(3, 3, 0, DType.float32)

    # Check shape (3x3)
    assert_equal(t.shape()[0], 3)
    assert_equal(t.shape()[1], 3)

    # Check diagonal is 1, off-diagonal is 0
    for i in range(3):
        for j in range(3):
            var idx = i * 3 + j
            var val = t._data.bitcast[Float32]()[idx]
            if i == j:
                assert_almost_equal(val, Float32(1.0), tolerance=1e-5)
            else:
                assert_almost_equal(val, Float32(0.0), tolerance=1e-5)


fn test_linspace_creation() raises:
    """Test linspace tensor creation (evenly spaced values)."""
    var t = linspace(0.0, 1.0, 5, DType.float32)

    # Check values: [0.0, 0.25, 0.5, 0.75, 1.0]
    assert_equal(t.numel(), 5)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(0.75), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(1.0), tolerance=1e-5
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_tensors_part1.mojo")
    print("=" * 70 + "\n")

    # test_zeros_creation
    total += 1
    try:
        test_zeros_creation()
        passed += 1
        print("  ✓ test_zeros_creation")
    except e:
        failed += 1
        print("  ✗ test_zeros_creation:", e)

    # test_ones_creation
    total += 1
    try:
        test_ones_creation()
        passed += 1
        print("  ✓ test_ones_creation")
    except e:
        failed += 1
        print("  ✗ test_ones_creation:", e)

    # test_full_creation
    total += 1
    try:
        test_full_creation()
        passed += 1
        print("  ✓ test_full_creation")
    except e:
        failed += 1
        print("  ✗ test_full_creation:", e)

    # test_empty_creation
    total += 1
    try:
        test_empty_creation()
        passed += 1
        print("  ✓ test_empty_creation")
    except e:
        failed += 1
        print("  ✗ test_empty_creation:", e)

    # test_arange_creation
    total += 1
    try:
        test_arange_creation()
        passed += 1
        print("  ✓ test_arange_creation")
    except e:
        failed += 1
        print("  ✗ test_arange_creation:", e)

    # test_arange_with_step
    total += 1
    try:
        test_arange_with_step()
        passed += 1
        print("  ✓ test_arange_with_step")
    except e:
        failed += 1
        print("  ✗ test_arange_with_step:", e)

    # test_eye_creation
    total += 1
    try:
        test_eye_creation()
        passed += 1
        print("  ✓ test_eye_creation")
    except e:
        failed += 1
        print("  ✗ test_eye_creation:", e)

    # test_linspace_creation
    total += 1
    try:
        test_linspace_creation()
        passed += 1
        print("  ✓ test_linspace_creation")
    except e:
        failed += 1
        print("  ✗ test_linspace_creation:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
