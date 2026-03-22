# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise logical_xor operation.

Tests cover:
- Full XOR truth table: (F,F)→F, (F,T)→T, (T,F)→T, (T,T)→F
- Shape preservation
- All-false inputs
- All-true inputs (T XOR T = F)
- Identity property (A XOR 0 = bool(A))

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import AnyTensor, zeros
from shared.core.elementwise import logical_xor


# ============================================================================
# logical_xor Tests
# ============================================================================


fn test_logical_xor_values() raises:
    """Truth table: (F,F)→F, (F,T)→T, (T,F)→T, (T,T)→F."""
    var shape = List[Int]()
    shape.append(4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # a = [0, 0, 1, 1], b = [0, 1, 0, 1]
    a._data.bitcast[Float32]()[0] = 0.0
    a._data.bitcast[Float32]()[1] = 0.0
    a._data.bitcast[Float32]()[2] = 1.0
    a._data.bitcast[Float32]()[3] = 1.0
    b._data.bitcast[Float32]()[0] = 0.0
    b._data.bitcast[Float32]()[1] = 1.0
    b._data.bitcast[Float32]()[2] = 0.0
    b._data.bitcast[Float32]()[3] = 1.0

    var result = logical_xor(a, b)

    # XOR truth table: [0, 1, 1, 0]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(0.0), tolerance=1e-5
    )


fn test_logical_xor_shape_preserved() raises:
    """Output shape matches input shape."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var result = logical_xor(a, b)
    assert_true(len(result.shape()) == 1)
    assert_equal(result.shape()[0], 3)


fn test_logical_xor_all_false() raises:
    """zeros XOR zeros → all zeros."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var result = logical_xor(a, b)
    for i in range(3):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-5
        )


fn test_logical_xor_all_true() raises:
    """ones XOR ones → all zeros (T XOR T = F)."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    for i in range(3):
        a._data.bitcast[Float32]()[i] = 1.0
        b._data.bitcast[Float32]()[i] = 1.0
    var result = logical_xor(a, b)
    for i in range(3):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-5
        )


fn test_logical_xor_identity() raises:
    """tensor XOR zeros → bool(tensor) (XOR with False is identity)."""
    var shape = List[Int]()
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    a._data.bitcast[Float32]()[0] = 0.0
    a._data.bitcast[Float32]()[1] = 1.0
    a._data.bitcast[Float32]()[2] = 1.0
    var result = logical_xor(a, b)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run logical_xor elementwise operation tests."""
    print("Running logical_xor elementwise operation tests...")

    test_logical_xor_values()
    print("✓ test_logical_xor_values")

    test_logical_xor_shape_preserved()
    print("✓ test_logical_xor_shape_preserved")

    test_logical_xor_all_false()
    print("✓ test_logical_xor_all_false")

    test_logical_xor_all_true()
    print("✓ test_logical_xor_all_true")

    test_logical_xor_identity()
    print("✓ test_logical_xor_identity")

    print("\nAll logical_xor tests passed!")
