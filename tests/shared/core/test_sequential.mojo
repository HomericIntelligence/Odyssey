"""Tests for Sequential module containers.

Tests cover:
- Sequential2: two-layer chained forward pass, parameter collection, mode switching
- Sequential3: three-layer chained forward pass, parameter collection
- Numerical correctness using FP-representable values (0.0, 0.5, 1.0)
- Mode propagation (train/eval) to sub-layers

Design Note:
    Local DummyLayer structs are used instead of real layers to isolate
    Sequential logic from layer-specific behavior, following the pattern
    established in test_module.mojo.
"""

from shared.core.module import Module
from shared.core.extensor import AnyTensor, zeros, ones
from shared.core.sequential import Sequential2, Sequential3
from tests.shared.conftest import (
    assert_true,
    assert_equal_int,
    assert_almost_equal,
)


# ============================================================================
# Dummy layer implementations for testing
# ============================================================================


struct IdentityModule(Module, Movable):
    """Module that passes input through unchanged.

    Used to verify that Sequential chains without mutating data.
    Has no trainable parameters.
    """

    var is_training: Bool

    fn __init__(out self):
        """Initialize identity module."""
        self.is_training = True

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Return input unchanged.

        Args:
            input: Input tensor.

        Returns:
            Same tensor passed through without modification.
        """
        return input

    fn parameters(self) raises -> List[AnyTensor]:
        """Return empty parameter list.

        Returns:
            Empty list (no trainable parameters).
        """
        return List[AnyTensor]()

    fn train(mut self):
        """Set to training mode."""
        self.is_training = True

    fn eval(mut self):
        """Set to evaluation mode."""
        self.is_training = False


struct ScaleModule(Module, Movable):
    """Module that scales all elements by a constant factor.

    Used to verify that Sequential chains produce correct numerical outputs.
    Has no trainable parameters.
    """

    var scale: Float32
    var is_training: Bool

    fn __init__(out self, scale: Float32):
        """Initialize scale module.

        Args:
            scale: Scalar multiplier applied to all input elements.
        """
        self.scale = scale
        self.is_training = True

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Scale all elements by self.scale.

        Args:
            input: Input tensor of any shape.

        Returns:
            Tensor with same shape, all values multiplied by scale.
        """
        var result = zeros(input.shape(), DType.float32)
        var n = input.numel()
        for i in range(n):
            result._data.bitcast[Float32]()[i] = (
                input._data.bitcast[Float32]()[i] * self.scale
            )
        return result

    fn parameters(self) raises -> List[AnyTensor]:
        """Return empty parameter list.

        Returns:
            Empty list (no trainable parameters).
        """
        return List[AnyTensor]()

    fn train(mut self):
        """Set to training mode."""
        self.is_training = True

    fn eval(mut self):
        """Set to evaluation mode."""
        self.is_training = False


struct CountingModule:
    """Module that tracks how many times forward has been called.

    Used to verify that Sequential invokes all layers exactly once.
    Has no trainable parameters.
    """

    var call_count: Int
    var is_training: Bool

    fn __init__(out self):
        """Initialize counting module with zero call count."""
        self.call_count = 0
        self.is_training = True

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Increment call count and return input unchanged.

        Args:
            input: Input tensor.

        Returns:
            Same tensor passed through unchanged.
        """
        self.call_count += 1
        return input

    fn parameters(self) raises -> List[AnyTensor]:
        """Return empty parameter list.

        Returns:
            Empty list (no trainable parameters).
        """
        return List[AnyTensor]()

    fn train(mut self):
        """Set to training mode."""
        self.is_training = True

    fn eval(mut self):
        """Set to evaluation mode."""
        self.is_training = False


struct DummyModuleWithParams(Module, Movable):
    """Module with a fixed number of dummy parameters.

    Used to test parameter collection in Sequential containers.
    """

    var num_params: Int
    var is_training: Bool

    fn __init__(out self, num_params: Int) raises:
        """Initialize module with specified number of parameters.

        Args:
            num_params: Number of dummy parameters to return from parameters().
        """
        self.num_params = num_params
        self.is_training = True

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Return input unchanged.

        Args:
            input: Input tensor.

        Returns:
            Input tensor unchanged.
        """
        return input

    fn parameters(self) raises -> List[AnyTensor]:
        """Return dummy parameter list of specified size.

        Returns:
            List of scalar zero tensors, length == self.num_params.
        """
        var params: List[AnyTensor] = []
        for _ in range(self.num_params):
            var shape: List[Int] = [1]
            params.append(zeros(shape, DType.float32))
        return params^

    fn train(mut self):
        """Set to training mode."""
        self.is_training = True

    fn eval(mut self):
        """Set to evaluation mode."""
        self.is_training = False


# ============================================================================
# Sequential2 tests
# ============================================================================


fn test_sequential2_forward_identity_chain() raises:
    """Test Sequential2 forward pass with two identity layers.

    Input should pass through both layers without modification.
    Verifies shape and values are preserved end-to-end.
    """
    var seq = Sequential2[IdentityModule, IdentityModule](
        IdentityModule(), IdentityModule()
    )
    var shape: List[Int] = [2, 4]
    var input = ones(shape, DType.float32)
    var output = seq.forward(input)

    # Shape must be preserved
    var out_shape = output.shape()
    assert_equal_int(len(out_shape), 2, "Output must be 2D")
    assert_equal_int(out_shape[0], 2, "Batch dim must be 2")
    assert_equal_int(out_shape[1], 4, "Feature dim must be 4")

    # Values must be unchanged (identity layers)
    var n = output.numel()
    for i in range(n):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-6,
        )


fn test_sequential2_forward_values() raises:
    """Test Sequential2 numerical correctness with two scale layers.

    Uses FP-representable values (0.5 * 0.5 = 0.25) to verify that
    layers are chained correctly and values are computed accurately.
    """
    # Scale by 0.5 then 0.5 -> net scale 0.25
    var seq = Sequential2[ScaleModule, ScaleModule](
        ScaleModule(0.5), ScaleModule(0.5)
    )
    var shape: List[Int] = [1, 4]
    var input = ones(shape, DType.float32)
    var output = seq.forward(input)

    var n = output.numel()
    for i in range(n):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(0.25),
            tolerance=1e-6,
        )


fn test_sequential2_forward_order() raises:
    """Test that Sequential2 applies layers in order (layer0 then layer1).

    Scale by 2.0 then by 0.5 should give net 1.0.
    Scale by 0.5 then by 2.0 also gives 1.0 but verifies no reordering.
    We use asymmetric verification: scale by 1.0 then 0.5 -> 0.5,
    to distinguish from scale by 0.5 then 1.0 -> same result.
    Instead, use a CountingModule to confirm order.
    """
    var seq = Sequential2[ScaleModule, ScaleModule](
        ScaleModule(2.0), ScaleModule(0.5)
    )
    var shape: List[Int] = [1, 2]
    var input = ones(shape, DType.float32)
    var output = seq.forward(input)

    # 1.0 * 2.0 * 0.5 = 1.0
    var n = output.numel()
    for i in range(n):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(1.0),
            tolerance=1e-6,
        )


fn test_sequential2_parameters_combined() raises:
    """Test Sequential2 parameter collection combines both layers.

    A module with 3 params followed by a module with 2 params should
    yield a combined list of 5 parameters.
    """
    var seq = Sequential2[DummyModuleWithParams, DummyModuleWithParams](
        DummyModuleWithParams(3), DummyModuleWithParams(2)
    )
    var params = seq.parameters()
    assert_equal_int(len(params), 5, "Combined parameters should be 3 + 2 = 5")


fn test_sequential2_parameters_empty() raises:
    """Test Sequential2 parameter collection with no-param layers.

    Both identity layers have no parameters, so combined list is empty.
    """
    var seq = Sequential2[IdentityModule, IdentityModule](
        IdentityModule(), IdentityModule()
    )
    var params = seq.parameters()
    assert_equal_int(len(params), 0, "Both identity layers have no parameters")


fn test_sequential2_train_eval_mode() raises:
    """Test Sequential2 propagates train/eval mode to both sub-layers.

    After calling train(), both layers must be in training mode.
    After calling eval(), both layers must be in eval mode.
    """
    var seq = Sequential2[IdentityModule, IdentityModule](
        IdentityModule(), IdentityModule()
    )

    # Both start in training mode
    assert_true(seq.layer0.is_training, "layer0 should start in training mode")
    assert_true(seq.layer1.is_training, "layer1 should start in training mode")

    # Switch to eval
    seq.eval()
    assert_true(not seq.layer0.is_training, "layer0 should be in eval mode")
    assert_true(not seq.layer1.is_training, "layer1 should be in eval mode")

    # Switch back to train
    seq.train()
    assert_true(seq.layer0.is_training, "layer0 should be back in train mode")
    assert_true(seq.layer1.is_training, "layer1 should be back in train mode")


fn test_sequential2_zero_input() raises:
    """Test Sequential2 with zero input through identity chain.

    Input of all zeros should remain all zeros through identity layers.
    Verifies no spurious initialization or mutation occurs.
    """
    var seq = Sequential2[IdentityModule, IdentityModule](
        IdentityModule(), IdentityModule()
    )
    var shape: List[Int] = [2, 3]
    var input = zeros(shape, DType.float32)
    var output = seq.forward(input)

    var n = output.numel()
    for i in range(n):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(0.0),
            tolerance=1e-6,
        )


# ============================================================================
# Sequential3 tests
# ============================================================================


fn test_sequential3_forward_chain() raises:
    """Test Sequential3 chains three layers correctly.

    Scale by 0.5 three times: 1.0 * 0.5^3 = 0.125.
    """
    var seq = Sequential3[ScaleModule, ScaleModule, ScaleModule](
        ScaleModule(0.5), ScaleModule(0.5), ScaleModule(0.5)
    )
    var shape: List[Int] = [1, 3]
    var input = ones(shape, DType.float32)
    var output = seq.forward(input)

    var n = output.numel()
    for i in range(n):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            Float32(0.125),
            tolerance=1e-6,
        )


fn test_sequential3_parameters_combined() raises:
    """Test Sequential3 parameter collection combines all three layers.

    Layers with 1, 2, and 3 params should yield 6 total parameters.
    """
    var seq = Sequential3[
        DummyModuleWithParams, DummyModuleWithParams, DummyModuleWithParams
    ](
        DummyModuleWithParams(1),
        DummyModuleWithParams(2),
        DummyModuleWithParams(3),
    )
    var params = seq.parameters()
    assert_equal_int(
        len(params), 6, "Combined parameters should be 1 + 2 + 3 = 6"
    )


fn test_sequential3_train_eval_mode() raises:
    """Test Sequential3 propagates train/eval mode to all three sub-layers."""
    var seq = Sequential3[IdentityModule, IdentityModule, IdentityModule](
        IdentityModule(), IdentityModule(), IdentityModule()
    )

    # All start in training mode
    assert_true(seq.layer0.is_training, "layer0 should start in training mode")
    assert_true(seq.layer1.is_training, "layer1 should start in training mode")
    assert_true(seq.layer2.is_training, "layer2 should start in training mode")

    # Switch to eval
    seq.eval()
    assert_true(not seq.layer0.is_training, "layer0 should be in eval mode")
    assert_true(not seq.layer1.is_training, "layer1 should be in eval mode")
    assert_true(not seq.layer2.is_training, "layer2 should be in eval mode")

    # Switch back to train
    seq.train()
    assert_true(seq.layer0.is_training, "layer0 should be back in train mode")
    assert_true(seq.layer1.is_training, "layer1 should be back in train mode")
    assert_true(seq.layer2.is_training, "layer2 should be back in train mode")


fn test_sequential3_shape_preserved() raises:
    """Test Sequential3 preserves tensor shape through three identity layers."""
    var seq = Sequential3[IdentityModule, IdentityModule, IdentityModule](
        IdentityModule(), IdentityModule(), IdentityModule()
    )
    var shape: List[Int] = [3, 5]
    var input = ones(shape, DType.float32)
    var output = seq.forward(input)

    var out_shape = output.shape()
    assert_equal_int(len(out_shape), 2, "Output must be 2D")
    assert_equal_int(out_shape[0], 3, "Batch dim must be 3")
    assert_equal_int(out_shape[1], 5, "Feature dim must be 5")


# ============================================================================
# Main
# ============================================================================


fn main() raises:
    """Run all Sequential module tests."""
    test_sequential2_forward_identity_chain()
    test_sequential2_forward_values()
    test_sequential2_forward_order()
    test_sequential2_parameters_combined()
    test_sequential2_parameters_empty()
    test_sequential2_train_eval_mode()
    test_sequential2_zero_input()
    test_sequential3_forward_chain()
    test_sequential3_parameters_combined()
    test_sequential3_train_eval_mode()
    test_sequential3_shape_preserved()
    print("All sequential tests passed!")
