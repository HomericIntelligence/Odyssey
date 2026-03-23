# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_pooling.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for pooling layer operations - Part 3.

Tests cover:
- global_avgpool2d_backward: Zero gradient and forward-backward consistency
- avgpool2d_backward: Output shape

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
    avgpool2d,
    avgpool2d_backward,
    global_avgpool2d,
    global_avgpool2d_backward,
)


# ============================================================================
# Global AvgPool2D Backward Tests (edge cases and consistency)
# ============================================================================


fn test_global_avgpool2d_backward_zero_gradient() raises:
    """Test that zero gradient produces zero output."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(3)
    input_shape.append(3)
    var input = ones(input_shape, DType.float32)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, DType.float32)

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()
    for i in range(9):
        assert_almost_equal(grad_data[i], 0.0, tolerance=1e-6)


fn test_global_avgpool2d_backward_forward_backward_consistency() raises:
    """Test consistency between forward and backward passes.

    Forward: x -> global_avgpool2d -> y
    Backward: grad_y -> global_avgpool2d_backward -> grad_x

    The sum of grad_x should equal grad_y (gradient conservation).
    """
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    input_shape.append(2)
    input_shape.append(2)
    var input = full(input_shape, 2.0, DType.float32)

    var output = global_avgpool2d(input)

    var grad_output_shape = output.shape()
    var grad_output = full(grad_output_shape, 1.0, DType.float32)

    var grad_input = global_avgpool2d_backward(grad_output, input)

    var grad_data = grad_input._data.bitcast[Float32]()

    # Each spatial position should get 1.0 / (2*2) = 0.25
    var expected_per_position = Float32(1.0 / 4.0)
    var total_elements = 2 * 3 * 2 * 2

    for i in range(total_elements):
        assert_almost_equal(grad_data[i], expected_per_position, Float32(1e-6))


# ============================================================================
# AvgPool2D Backward Tests
# ============================================================================


fn test_avgpool2d_backward_output_shape() raises:
    """Test avgpool2d_backward produces correct gradient shape."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(4)
    input_shape.append(4)
    var input = ones(input_shape, DType.float32)

    var output = avgpool2d(input, kernel_size=2, stride=2, padding=0)

    var grad_output_shape = output.shape()
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = avgpool2d_backward(
        grad_output, input, kernel_size=2, stride=2, padding=0
    )

    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], 1, "Batch size mismatch")
    assert_equal(grad_shape[1], 1, "Channels mismatch")
    assert_equal(grad_shape[2], 4, "Gradient height mismatch")
    assert_equal(grad_shape[3], 4, "Gradient width mismatch")


fn main() raises:
    """Run pooling tests part 3."""
    print("Running pooling tests part 3...")

    test_global_avgpool2d_backward_zero_gradient()
    print("✓ test_global_avgpool2d_backward_zero_gradient")

    test_global_avgpool2d_backward_forward_backward_consistency()
    print("✓ test_global_avgpool2d_backward_forward_backward_consistency")

    test_avgpool2d_backward_output_shape()
    print("✓ test_avgpool2d_backward_output_shape")

    print("\nAll pooling tests part 3 passed!")
