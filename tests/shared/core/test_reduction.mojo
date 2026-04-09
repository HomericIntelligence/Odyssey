"""Tests for sum and mean reduction operations with gradient checking.

Tests cover:
- Sum reduction along axes
- Mean reduction along axes
- Numerical gradient checking using finite differences

Gradient checking formula:
    numerical_grad ≈ (f(x + ε) - f(x - ε)) / (2ε)

All tests validate backward passes produce correct gradient values.
"""


from tests.shared.conftest import (
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.reduction import (
    sum,
    mean,
    max_reduce,
    min_reduce,
    sum_backward,
    mean_backward,
    max_reduce_backward,
    min_reduce_backward,
)
from shared.testing import check_gradient
from shared.core.reduction import (
    variance,
    std as std_op,
    variance_backward,
    std_backward,
)
from shared.core.reduction import (
    median,
    percentile,
    median_backward,
    percentile_backward,
)


def test_sum_backward_shapes() raises:
    """Test that sum_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = sum(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = sum_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)
    assert_equal(gi_shape[2], 4)


def test_sum_backward_gradient() raises:
    """Test sum_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(-0.3))
    x.set(2, Float32(1.2))
    x.set(3, Float32(-0.8))
    x.set(4, Float32(0.1))
    x.set(5, Float32(0.7))

    # Forward function wrapper (sum along axis 1)
    def forward(inp: AnyTensor) raises unified {read} -> AnyTensor:
        return sum(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    def backward(grad: AnyTensor, inp: AnyTensor) raises unified {read} -> AnyTensor:
        return sum_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision in sum accumulation
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


def test_mean_backward_shapes() raises:
    """Test that mean_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = mean(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = mean_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)
    assert_equal(gi_shape[2], 4)


def test_mean_backward_gradient() raises:
    """Test mean_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(-0.3))
    x.set(2, Float32(1.2))
    x.set(3, Float32(-0.8))
    x.set(4, Float32(0.1))
    x.set(5, Float32(0.7))

    # Forward function wrapper (mean along axis 1)
    def forward(inp: AnyTensor) raises unified {read} -> AnyTensor:
        return mean(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    def backward(grad: AnyTensor, inp: AnyTensor) raises unified {read} -> AnyTensor:
        return mean_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision in division
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


def test_max_reduce_backward_shapes() raises:
    """Test that max_reduce_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = max_reduce(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = max_reduce_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 3)
    assert_equal(gi_shape[1], 4)


def test_max_reduce_backward_gradient() raises:
    """Test max_reduce_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(-0.3))
    x.set(2, Float32(1.2))
    x.set(3, Float32(-0.8))
    x.set(4, Float32(0.1))
    x.set(5, Float32(0.7))

    # Forward function wrapper (max along axis 1)
    def forward(inp: AnyTensor) raises unified {read} -> AnyTensor:
        return max_reduce(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    def backward(grad: AnyTensor, inp: AnyTensor) raises unified {read} -> AnyTensor:
        return max_reduce_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


def test_min_reduce_backward_shapes() raises:
    """Test that min_reduce_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = min_reduce(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = min_reduce_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 3)
    assert_equal(gi_shape[1], 4)


def test_min_reduce_backward_gradient() raises:
    """Test min_reduce_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(0.5))
    x.set(1, Float32(-0.3))
    x.set(2, Float32(1.2))
    x.set(3, Float32(-0.8))
    x.set(4, Float32(0.1))
    x.set(5, Float32(0.7))

    # Forward function wrapper (min along axis 1)
    def forward(inp: AnyTensor) raises unified {read} -> AnyTensor:
        return min_reduce(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    def backward(grad: AnyTensor, inp: AnyTensor) raises unified {read} -> AnyTensor:
        return min_reduce_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


def test_var_forward_uniform() raises:
    """Test variance of uniform values (should be 0)."""
    var shape = List[Int]()
    shape.append(5)
    var x = ones(shape, DType.float32)

    var result = variance(x, axis=-1)
    assert_close_float(result._get_float64(0), 0.0, rtol=1e-5, atol=1e-7)


def test_var_forward_simple() raises:
    """Test variance with known result."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(3.0))

    # Mean = 2.0, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    var result = variance(x, axis=-1, ddof=0)
    assert_close_float(result._get_float64(0), 2.0 / 3.0, rtol=1e-5, atol=1e-7)


def test_var_forward_with_ddof() raises:
    """Test sample variance with ddof=1."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(3.0))

    # Sample variance with ddof=1: var = 2 / 2 = 1.0
    var result = variance(x, axis=-1, ddof=1)
    assert_close_float(result._get_float64(0), 1.0, rtol=1e-5, atol=1e-7)


def test_var_forward_axis() raises:
    """Test variance along specific axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(3.0))
    x.set(3, Float32(4.0))
    x.set(4, Float32(5.0))
    x.set(5, Float32(6.0))

    var result = variance(x, axis=1, ddof=0)
    var result_shape = result.shape()
    assert_equal(result_shape[0], 2)


def test_var_backward_shapes() raises:
    """Test that var_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    for i in range(6):
        x.set(i, Float32(Float32(i) + 1.0))

    var result = variance(x, axis=1, ddof=0)
    var grad_output = ones_like(result)
    var grad_input = variance_backward(grad_output, x, axis=1, ddof=0)

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)


def test_var_backward_gradient() raises:
    """Test var_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(0.5))
    x.set(1, Float32(-0.3))
    x.set(2, Float32(1.2))
    x.set(3, Float32(-0.8))
    x.set(4, Float32(0.1))
    x.set(5, Float32(0.7))

    def forward(inp: AnyTensor) raises unified {read} -> AnyTensor:
        return variance(inp, axis=1, ddof=0)

    var y = forward(x)
    var grad_out = ones_like(y)

    def backward(grad: AnyTensor, inp: AnyTensor) raises unified {read} -> AnyTensor:
        return variance_backward(grad, inp, axis=1, ddof=0)

    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


def test_std_forward_simple() raises:
    """Test standard deviation with known result."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(3.0))

    # std = sqrt(var) = sqrt(2/3)
    var result = std_op(x, axis=-1, ddof=0)
    var expected = (2.0 / 3.0) ** 0.5
    assert_close_float(result._get_float64(0), expected, rtol=1e-5, atol=1e-7)


def test_std_backward_gradient() raises:
    """Test std_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(0.5))
    x.set(1, Float32(0.3))
    x.set(2, Float32(1.2))
    x.set(3, Float32(0.8))
    x.set(4, Float32(0.1))
    x.set(5, Float32(0.7))

    def forward(inp: AnyTensor) raises unified {read} -> AnyTensor:
        return std_op(inp, axis=1, ddof=0)

    var y = forward(x)
    var grad_out = ones_like(y)

    def backward(grad: AnyTensor, inp: AnyTensor) raises unified {read} -> AnyTensor:
        return std_backward(grad, inp, axis=1, ddof=0)

    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


def test_median_forward_odd() raises:
    """Test median with odd count."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(3.0))
    x.set(1, Float32(1.0))
    x.set(2, Float32(4.0))
    x.set(3, Float32(2.0))
    x.set(4, Float32(5.0))

    var result = median(x, axis=-1)
    assert_close_float(result._get_float64(0), 3.0, rtol=1e-5, atol=1e-7)


def test_median_forward_even() raises:
    """Test median with even count (average of two middle values)."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(3.0))
    x.set(3, Float32(4.0))

    var result = median(x, axis=-1)
    assert_close_float(result._get_float64(0), 2.5, rtol=1e-5, atol=1e-7)


def test_median_backward_shapes() raises:
    """Test that median_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    for i in range(6):
        x.set(i, Float32(Float32(i) + 1.0))

    var result = median(x, axis=1)
    var grad_output = ones_like(result)
    var grad_input = median_backward(grad_output, x, axis=1)

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)


def test_percentile_forward_p50() raises:
    """Test that 50th percentile equals median."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(3.0))
    x.set(3, Float32(4.0))
    x.set(4, Float32(5.0))

    var result = percentile(x, 50.0, axis=-1)
    assert_close_float(result._get_float64(0), 3.0, rtol=1e-5, atol=1e-7)


def test_percentile_forward_p0_p100() raises:
    """Test that 0th and 100th percentiles equal min and max."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))
    x.set(2, Float32(3.0))
    x.set(3, Float32(4.0))
    x.set(4, Float32(5.0))

    var p0 = percentile(x, 0.0, axis=-1)
    assert_close_float(p0._get_float64(0), 1.0, rtol=1e-5, atol=1e-7)

    var p100 = percentile(x, 100.0, axis=-1)
    assert_close_float(p100._get_float64(0), 5.0, rtol=1e-5, atol=1e-7)


def test_percentile_backward_shapes() raises:
    """Test that percentile_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    for i in range(6):
        x.set(i, Float32(Float32(i) + 1.0))

    var result = percentile(x, 50.0, axis=1)
    var grad_output = ones_like(result)
    var grad_input = percentile_backward(grad_output, x, 50.0, axis=1)

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)


def main() raises:
    """Run all test_reduction tests."""
    print("Running test_reduction tests...")

    test_sum_backward_shapes()
    print("✓ test_sum_backward_shapes")

    test_sum_backward_gradient()
    print("✓ test_sum_backward_gradient")

    test_mean_backward_shapes()
    print("✓ test_mean_backward_shapes")

    test_mean_backward_gradient()
    print("✓ test_mean_backward_gradient")

    test_max_reduce_backward_shapes()
    print("✓ test_max_reduce_backward_shapes")

    test_max_reduce_backward_gradient()
    print("✓ test_max_reduce_backward_gradient")

    test_min_reduce_backward_shapes()
    print("✓ test_min_reduce_backward_shapes")

    test_min_reduce_backward_gradient()
    print("✓ test_min_reduce_backward_gradient")

    test_var_forward_uniform()
    print("✓ test_var_forward_uniform")

    test_var_forward_simple()
    print("✓ test_var_forward_simple")

    test_var_forward_with_ddof()
    print("✓ test_var_forward_with_ddof")

    test_var_forward_axis()
    print("✓ test_var_forward_axis")

    test_var_backward_shapes()
    print("✓ test_var_backward_shapes")

    test_var_backward_gradient()
    print("✓ test_var_backward_gradient")

    test_std_forward_simple()
    print("✓ test_std_forward_simple")

    test_std_backward_gradient()
    print("✓ test_std_backward_gradient")

    test_median_forward_odd()
    print("✓ test_median_forward_odd")

    test_median_forward_even()
    print("✓ test_median_forward_even")

    test_median_backward_shapes()
    print("✓ test_median_backward_shapes")

    test_percentile_forward_p50()
    print("✓ test_percentile_forward_p50")

    test_percentile_forward_p0_p100()
    print("✓ test_percentile_forward_p0_p100")

    test_percentile_backward_shapes()
    print("✓ test_percentile_backward_shapes")

    print("\nAll test_reduction tests passed!")
