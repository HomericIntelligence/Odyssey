"""Tests for activation functions - Part 5: GELU (continued), Swish, Mish (basic).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests: test_gelu_exact, test_gelu_comparison, test_gelu_float16,
       test_gelu_backward_gradient, test_swish_basic, test_swish_positive,
       test_swish_backward_gradient, test_mish_basic
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.core.any_tensor import (
    AnyTensor,
    zeros,
    ones_like,
)
from shared.core.activation import (
    gelu,
    swish,
    mish,
    gelu_backward,
    swish_backward,
)
from shared.testing import (
    check_gradient,
)


# ============================================================================
# GELU Tests (continued)
# ============================================================================


fn test_gelu_exact() raises:
    """Test GELU with exact erf implementation."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = gelu(x, approximate=False)

    # GELU(0) = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )

    # For large positive x, GELU(x) ≈ x
    assert_true(y._data.bitcast[Float32]()[4] > 1.9)

    # For large negative x, GELU(x) ≈ 0
    assert_true(abs(y._data.bitcast[Float32]()[0]) < 0.1)


fn test_gelu_comparison() raises:
    """Compare approximate and exact GELU implementations."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y_approx = gelu(x, approximate=True)
    var y_exact = gelu(x, approximate=False)

    # Approximate and exact should be close
    for i in range(5):
        var approx_val = y_approx._data.bitcast[Float32]()[i]
        var exact_val = y_exact._data.bitcast[Float32]()[i]
        var diff = abs(approx_val - exact_val)

        # Approximation error should be small (< 1%)
        if abs(exact_val) > 0.01:
            var rel_error = diff / abs(exact_val)
            assert_true(rel_error < 0.01)


fn test_gelu_float16() raises:
    """Test GELU with float16."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float16)
    x._data.bitcast[Float16]()[0] = Float16(-1.0)
    x._data.bitcast[Float16]()[1] = Float16(0.0)
    x._data.bitcast[Float16]()[2] = Float16(1.0)

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, Float32(0.0), tolerance=0.01)


fn test_gelu_backward_gradient() raises:
    """Test GELU backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return gelu(x, approximate=False)

    var y = gelu(x, approximate=False)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        return gelu_backward(grad, x, approximate=False)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Swish Tests
# ============================================================================


fn test_swish_basic() raises:
    """Test swish with known values."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = swish(x)

    # swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


fn test_swish_positive() raises:
    """Test swish with large positive value."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 10.0

    var y = swish(x)

    # swish(10) ≈ 10 * sigmoid(10) ≈ 10 * 1 ≈ 10
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(10.0), tolerance=0.01
    )


fn test_swish_backward_gradient() raises:
    """Test Swish backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return swish(x)

    var y = swish(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        return swish_backward(grad, x)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Mish Tests
# ============================================================================


fn test_mish_basic() raises:
    """Test mish with known values."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0

    var y = mish(x)

    # mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(log(2)) ≈ 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.01
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
    print("Running tests from: test_activations_part5.mojo")
    print("=" * 70 + "\n")

    # test_gelu_exact
    total += 1
    try:
        test_gelu_exact()
        passed += 1
        print("  ✓ test_gelu_exact")
    except e:
        failed += 1
        print("  ✗ test_gelu_exact:", e)

    # test_gelu_comparison
    total += 1
    try:
        test_gelu_comparison()
        passed += 1
        print("  ✓ test_gelu_comparison")
    except e:
        failed += 1
        print("  ✗ test_gelu_comparison:", e)

    # test_gelu_float16
    total += 1
    try:
        test_gelu_float16()
        passed += 1
        print("  ✓ test_gelu_float16")
    except e:
        failed += 1
        print("  ✗ test_gelu_float16:", e)

    # test_gelu_backward_gradient
    total += 1
    try:
        test_gelu_backward_gradient()
        passed += 1
        print("  ✓ test_gelu_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_gelu_backward_gradient:", e)

    # test_swish_basic
    total += 1
    try:
        test_swish_basic()
        passed += 1
        print("  ✓ test_swish_basic")
    except e:
        failed += 1
        print("  ✗ test_swish_basic:", e)

    # test_swish_positive
    total += 1
    try:
        test_swish_positive()
        passed += 1
        print("  ✓ test_swish_positive")
    except e:
        failed += 1
        print("  ✗ test_swish_positive:", e)

    # test_swish_backward_gradient
    total += 1
    try:
        test_swish_backward_gradient()
        passed += 1
        print("  ✓ test_swish_backward_gradient")
    except e:
        failed += 1
        print("  ✗ test_swish_backward_gradient:", e)

    # test_mish_basic
    total += 1
    try:
        test_mish_basic()
        passed += 1
        print("  ✓ test_mish_basic")
    except e:
        failed += 1
        print("  ✗ test_mish_basic:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
