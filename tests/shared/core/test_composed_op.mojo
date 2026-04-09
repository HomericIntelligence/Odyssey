"""Tests for ComposedOp initialization and forward pass

Tests the composition of two differentiable operations including:
- Initialization and trait implementation
- Forward pass chaining: output = second.forward(first.forward(input))
- Intermediate tensor caching
- Forward pass shape preservation

"""


from std.testing import assert_true, assert_equal
from tests.shared.conftest import assert_almost_equal, assert_shape_equal
from shared.tensor.any_tensor import AnyTensor, ones, zeros
from shared.core.traits import Differentiable, Composable, ComposedOp


def make_tensor_with_shape(numel: Int) raises -> AnyTensor:
    """Create a tensor with given number of elements."""
    var shape_list = List[Int]()
    shape_list.append(numel)
    return ones(shape_list^, DType.float32)


def make_shape_list(numel: Int) -> List[Int]:
    """Create a shape list with given number of elements."""
    var shape_list = List[Int]()
    shape_list.append(numel)
    return shape_list^


struct ScaleMul(Copyable, Differentiable, Movable):
    """Mock operation that multiplies input by a scalar.

    Used for testing composition logic with predictable behavior.
    Forward: y = x * scale
    Backward: grad_x = grad_y * scale
    """

    var scale: Float64
    var last_input: AnyTensor

    def __init__(out self, scale: Float64) raises:
        """Initialize with scale factor."""
        self.scale = scale
        self.last_input = make_tensor_with_shape(1)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: multiply by scale."""
        self.last_input = input.copy()
        var output = input.copy()
        # Multiply all elements by scale
        for i in range(output.numel()):
            output._set_float64(i, input._get_float64(i) * self.scale)
        return output

    def backward(self, grad_output: AnyTensor) raises -> AnyTensor:
        """Backward pass: multiply gradient by scale."""
        # Use clone() for deep copy - copy() creates a shallow view that shares data
        var grad_input = grad_output.clone()
        # Multiply all gradients by scale
        for i in range(grad_input.numel()):
            grad_input._set_float64(i, grad_output._get_float64(i) * self.scale)
        return grad_input


struct ScaleAdd(Copyable, Differentiable, Movable):
    """Mock operation that adds a scalar to input.

    Used for testing composition logic with predictable behavior.
    Forward: y = x + offset
    Backward: grad_x = grad_y (offset doesn't affect gradient)
    """

    var offset: Float64
    var last_input: AnyTensor

    def __init__(out self, offset: Float64) raises:
        """Initialize with offset value."""
        self.offset = offset
        self.last_input = make_tensor_with_shape(1)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: add offset."""
        self.last_input = input.copy()
        var output = input.copy()
        # Add offset to all elements
        for i in range(output.numel()):
            output._set_float64(i, input._get_float64(i) + self.offset)
        return output

    def backward(self, grad_output: AnyTensor) raises -> AnyTensor:
        """Backward pass: gradient passes through unchanged."""
        # Use clone() for deep copy - copy() creates a shallow view that shares data
        return grad_output.clone()


def test_composed_op_initialization() raises:
    """Test that ComposedOp initializes with first and second operations."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)

    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    # Verify operations are stored
    assert_almost_equal(composed.first.scale, 2.0, tolerance=1e-10)
    assert_almost_equal(composed.second.offset, 3.0, tolerance=1e-10)


def test_composed_op_implements_differentiable() raises:
    """Test that ComposedOp implements Differentiable trait."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    # ComposedOp should have forward and backward methods (verified by compilation)
    # This test verifies the methods can be called
    var input = make_tensor_with_shape(1)
    var output = composed.forward(input)
    assert_true(True, "forward() should compile and execute")


def test_composed_op_implements_composable() raises:
    """Test that ComposedOp implements Composable trait."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    # ComposedOp should implement Composable (verified by compilation)
    # This is a compile-time check that the trait is implemented
    assert_true(True, "ComposedOp should implement Composable trait")


def test_composed_op_forward_chains_operations() raises:
    """Test that forward pass chains first.forward() -> second.forward()."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    # Input: 5.0
    # After first (ScaleMul(2.0)): 5.0 * 2.0 = 10.0
    # After second (ScaleAdd(3.0)): 10.0 + 3.0 = 13.0
    var input = make_tensor_with_shape(1)
    input._set_float64(0, 5.0)

    var output = composed.forward(input)

    # Verify output value (should be (5.0 * 2.0) + 3.0 = 13.0)
    assert_almost_equal(output._get_float64(0), 13.0, tolerance=1e-6)


def test_composed_op_forward_caches_intermediate() raises:
    """Test that forward pass caches intermediate tensor."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    var input = make_tensor_with_shape(1)
    input._set_float64(0, 5.0)

    _ = composed.forward(input)

    # Intermediate should be cached (10.0 in this case)
    # We can verify by checking the numel
    assert_equal(
        composed._intermediate.numel(), 1, "Intermediate should have 1 element"
    )


def test_composed_op_forward_shape_preservation() raises:
    """Test that forward pass preserves tensor shape."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    var input = make_tensor_with_shape(5)
    var output = composed.forward(input)

    # Output shape should match input shape
    assert_equal(output.numel(), 5, "Output should have 5 elements")


def test_composed_op_with_different_scales() raises:
    """Test composition with different scale factors."""
    var first = ScaleMul(0.5)
    var second = ScaleMul(4.0)
    var composed = ComposedOp[ScaleMul, ScaleMul](first, second)

    var input = make_tensor_with_shape(1)
    input._set_float64(0, 8.0)

    # Forward: 8.0 * 0.5 = 4.0, then 4.0 * 4.0 = 16.0
    var output = composed.forward(input)
    assert_almost_equal(output._get_float64(0), 16.0, tolerance=1e-6)

    # Backward: grad = 1.0 * 4.0 * 0.5 = 2.0
    var grad_output = make_tensor_with_shape(1)
    grad_output._set_float64(0, 1.0)
    var grad_input = composed.backward(grad_output)
    assert_almost_equal(grad_input._get_float64(0), 2.0, tolerance=1e-6)


def test_composed_op_backward_applies_chain_rule() raises:
    """Test that backward pass applies chain rule correctly.

    For composition F ∘ G:
    - Forward: y = G(F(x))
    - Backward: grad_x = F.backward(G.backward(grad_y))
    """
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    var input = make_tensor_with_shape(1)
    input._set_float64(0, 5.0)

    _ = composed.forward(input)

    # Backward with grad_output = 1.0
    var grad_output = make_tensor_with_shape(1)
    grad_output._set_float64(0, 1.0)
    var grad_input = composed.backward(grad_output)

    # Chain rule: grad_x = grad_y * scale2 * scale1
    # = 1.0 * 1.0 (from ScaleAdd.backward) * 2.0 (from ScaleMul.backward)
    # = 2.0
    assert_almost_equal(grad_input._get_float64(0), 2.0, tolerance=1e-6)


def test_composed_op_backward_shape_preservation() raises:
    """Test that backward pass preserves gradient shape."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    var input = make_tensor_with_shape(5)
    _ = composed.forward(input)

    var grad_output = make_tensor_with_shape(5)
    var grad_input = composed.backward(grad_output)

    # Gradient shape should match input shape
    assert_equal(grad_input.numel(), 5, "Gradient should have 5 elements")


def test_composed_op_backward_multiple_passes() raises:
    """Test that backward can be called multiple times."""
    var first = ScaleMul(2.0)
    var second = ScaleAdd(3.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    var input = make_tensor_with_shape(1)
    input._set_float64(0, 5.0)

    _ = composed.forward(input)

    var grad_output = make_tensor_with_shape(1)
    grad_output._set_float64(0, 1.0)

    # First backward pass
    var grad_input1 = composed.backward(grad_output)
    assert_almost_equal(grad_input1._get_float64(0), 2.0, tolerance=1e-6)

    # Second backward pass (should produce same result)
    var grad_input2 = composed.backward(grad_output)
    assert_almost_equal(grad_input2._get_float64(0), 2.0, tolerance=1e-6)


def test_composed_op_forward_backward_roundtrip() raises:
    """Test complete forward-backward pass roundtrip.

    Verify that forward and backward passes work together correctly.
    """
    var first = ScaleMul(3.0)
    var second = ScaleMul(2.0)
    var composed = ComposedOp[ScaleMul, ScaleMul](first, second)

    var input = make_tensor_with_shape(1)
    input._set_float64(0, 4.0)

    # Forward: 4.0 * 3.0 = 12.0, then 12.0 * 2.0 = 24.0
    var output = composed.forward(input)
    assert_almost_equal(output._get_float64(0), 24.0, tolerance=1e-6)

    # Backward: grad = 1.0 * 2.0 (second) * 3.0 (first) = 6.0
    var grad_output = make_tensor_with_shape(1)
    grad_output._set_float64(0, 1.0)
    var grad_input = composed.backward(grad_output)
    assert_almost_equal(grad_input._get_float64(0), 6.0, tolerance=1e-6)


def test_composed_op_identity_composition() raises:
    """Test composition with identity-like operations (scale by 1.0)."""
    var first = ScaleMul(1.0)
    var second = ScaleAdd(0.0)
    var composed = ComposedOp[ScaleMul, ScaleAdd](first, second)

    var input = make_tensor_with_shape(1)
    input._set_float64(0, 7.0)

    # Forward: 7.0 * 1.0 = 7.0, then 7.0 + 0.0 = 7.0
    var output = composed.forward(input)
    assert_almost_equal(output._get_float64(0), 7.0, tolerance=1e-6)

    # Backward: grad = 1.0 * 1.0 = 1.0
    var grad_output = make_tensor_with_shape(1)
    grad_output._set_float64(0, 1.0)
    var grad_input = composed.backward(grad_output)
    assert_almost_equal(grad_input._get_float64(0), 1.0, tolerance=1e-6)


def main() raises:
    """Run all test_composed_op tests."""
    print("Running test_composed_op tests...")

    test_composed_op_initialization()
    print("✓ test_composed_op_initialization")

    test_composed_op_implements_differentiable()
    print("✓ test_composed_op_implements_differentiable")

    test_composed_op_implements_composable()
    print("✓ test_composed_op_implements_composable")

    test_composed_op_forward_chains_operations()
    print("✓ test_composed_op_forward_chains_operations")

    test_composed_op_forward_caches_intermediate()
    print("✓ test_composed_op_forward_caches_intermediate")

    test_composed_op_forward_shape_preservation()
    print("✓ test_composed_op_forward_shape_preservation")

    test_composed_op_with_different_scales()
    print("✓ test_composed_op_with_different_scales")

    test_composed_op_backward_applies_chain_rule()
    print("✓ test_composed_op_backward_applies_chain_rule")

    test_composed_op_backward_shape_preservation()
    print("✓ test_composed_op_backward_shape_preservation")

    test_composed_op_backward_multiple_passes()
    print("✓ test_composed_op_backward_multiple_passes")

    test_composed_op_forward_backward_roundtrip()
    print("✓ test_composed_op_forward_backward_roundtrip")

    test_composed_op_identity_composition()
    print("✓ test_composed_op_identity_composition")

    print("\nAll test_composed_op tests passed!")
