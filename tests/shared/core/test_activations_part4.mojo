"""Tests for activation functions - Part 4: Softmax (continued), GELU (basic).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests: test_softmax_one_hot, test_softmax_sum_to_one, test_softmax_numerical_stability,
       test_softmax_backward, test_gelu_basic, test_gelu_positive,
       test_gelu_shape, test_gelu_approximate
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    ones_like,
)
from shared.core.activation import (
    softmax,
    gelu,
    softmax_backward,
)
from shared.testing import (
    check_gradient,
)


# ============================================================================
# Softmax Tests (continued)
# ============================================================================


fn test_softmax_one_hot() raises:
    """Test softmax with large difference (one-hot-like)."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 10.0
    x._data.bitcast[Float32]()[2] = 0.0

    var y = softmax(x, axis=1)

    # Middle value should be ~1.0, others ~0.0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-3
    )


fn test_softmax_sum_to_one() raises:
    """Test softmax probabilities sum to 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)
    var x = zeros(shape, DType.float32)

    # Set random values
    for i in range(8):
        x._data.bitcast[Float32]()[i] = Float32(i % 5) - 2.0

    var y = softmax(x, axis=1)

    # Each row should sum to 1.0
    var sum_row0 = Float32(0.0)
    var sum_row1 = Float32(0.0)
    for i in range(4):
        sum_row0 += y._data.bitcast[Float32]()[i]
        sum_row1 += y._data.bitcast[Float32]()[4 + i]

    assert_almost_equal(sum_row0, Float32(1.0), tolerance=1e-5)
    assert_almost_equal(sum_row1, Float32(1.0), tolerance=1e-5)


fn test_softmax_numerical_stability() raises:
    """Test softmax with large values (numerical stability)."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1000.0
    x._data.bitcast[Float32]()[1] = 1001.0
    x._data.bitcast[Float32]()[2] = 1002.0

    var y = softmax(x, axis=-1)

    # Should still sum to 1 (no overflow)
    var sum: Float32 = 0.0
    for i in range(3):
        sum += y._data.bitcast[Float32]()[i]

    assert_almost_equal(sum, Float32(1.0), tolerance=1e-5)

    # Largest value should have largest probability
    assert_true(y._data.bitcast[Float32]()[2] > y._data.bitcast[Float32]()[1])
    assert_true(y._data.bitcast[Float32]()[1] > y._data.bitcast[Float32]()[0])


fn test_softmax_backward() raises:
    """Test softmax gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set test values
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0
    x._data.bitcast[Float32]()[3] = -0.5
    x._data.bitcast[Float32]()[4] = 0.5
    x._data.bitcast[Float32]()[5] = 1.5

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return softmax(x, axis=1)

    var y = softmax(x, axis=1)
    var grad_out = ones_like(y)

    # Note: softmax_backward takes output y, not input x
    fn backward_fn(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var out = softmax(x, axis=1)  # Recompute output inside wrapper
        return softmax_backward(grad, out, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3, atol=5e-4 is needed for float32 softmax gradients
    # Softmax involves exp() and division which amplify numerical errors,
    # especially at the edges of the distribution
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=5e-4)


# ============================================================================
# GELU Tests
# ============================================================================


fn test_gelu_basic() raises:
    """Test GELU with known value at x=0."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = gelu(x)

    # GELU(0) = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


fn test_gelu_positive() raises:
    """Test GELU with positive values."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0

    var y = gelu(x)

    # For positive x, GELU(x) ≈ x (asymptotically)
    # GELU(1) ≈ 0.84, GELU(2) ≈ 1.96
    assert_true(y._data.bitcast[Float32]()[0] > 0.8)
    assert_true(y._data.bitcast[Float32]()[0] < 1.0)
    assert_true(y._data.bitcast[Float32]()[1] > 1.9)
    assert_true(y._data.bitcast[Float32]()[1] < 2.0)


fn test_gelu_shape() raises:
    """Test GELU preserves shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = gelu(x)

    assert_equal(y.shape()[0], 3)
    assert_equal(y.shape()[1], 4)


fn test_gelu_approximate() raises:
    """Test GELU with tanh approximation."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )

    # GELU is NOT symmetric (unlike relu). For x < 0, GELU(x) is close to 0.
    # For x > 0, GELU(x) is close to x.
    var val_neg2 = y._data.bitcast[Float32]()[0]  # GELU(-2.0) ≈ -0.045
    var val_pos2 = y._data.bitcast[Float32]()[4]  # GELU(2.0) ≈ 1.954

    # For large positive x, GELU(x) ≈ x
    assert_true(val_pos2 > 1.9, "GELU(2.0) should be close to 2.0")

    # For large negative x, GELU(x) ≈ 0 (small negative value)
    assert_true(abs(val_neg2) < 0.1, "GELU(-2.0) should be close to 0")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_activations_part4.mojo")
    print("=" * 70 + "\n")

    # test_softmax_one_hot
    total += 1
    try:
        test_softmax_one_hot()
        passed += 1
        print("  ✓ test_softmax_one_hot")
    except e:
        failed += 1
        print("  ✗ test_softmax_one_hot:", e)

    # test_softmax_sum_to_one
    total += 1
    try:
        test_softmax_sum_to_one()
        passed += 1
        print("  ✓ test_softmax_sum_to_one")
    except e:
        failed += 1
        print("  ✗ test_softmax_sum_to_one:", e)

    # test_softmax_numerical_stability
    total += 1
    try:
        test_softmax_numerical_stability()
        passed += 1
        print("  ✓ test_softmax_numerical_stability")
    except e:
        failed += 1
        print("  ✗ test_softmax_numerical_stability:", e)

    # test_softmax_backward
    total += 1
    try:
        test_softmax_backward()
        passed += 1
        print("  ✓ test_softmax_backward")
    except e:
        failed += 1
        print("  ✗ test_softmax_backward:", e)

    # test_gelu_basic
    total += 1
    try:
        test_gelu_basic()
        passed += 1
        print("  ✓ test_gelu_basic")
    except e:
        failed += 1
        print("  ✗ test_gelu_basic:", e)

    # test_gelu_positive
    total += 1
    try:
        test_gelu_positive()
        passed += 1
        print("  ✓ test_gelu_positive")
    except e:
        failed += 1
        print("  ✗ test_gelu_positive:", e)

    # test_gelu_shape
    total += 1
    try:
        test_gelu_shape()
        passed += 1
        print("  ✓ test_gelu_shape")
    except e:
        failed += 1
        print("  ✗ test_gelu_shape:", e)

    # test_gelu_approximate
    total += 1
    try:
        test_gelu_approximate()
        passed += 1
        print("  ✓ test_gelu_approximate")
    except e:
        failed += 1
        print("  ✗ test_gelu_approximate:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
