"""Tests for updated trait signatures using AnyTensor (Phase 5a).

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Phase 5a changes trait signatures from AnyTensor to AnyTensor:
- Module.forward(mut self, input: AnyTensor) -> AnyTensor
- Module.parameters(self) -> List[AnyTensor]
- Differentiable.forward/backward -> AnyTensor
- Parameterized.parameters/gradients -> List[AnyTensor]

Constraint H7: Module.forward() stays on AnyTensor. Layers convert at
boundaries internally, but the trait interface is type-erased.

Tests cover:
- Module.forward accepts AnyTensor input and returns AnyTensor
- Module.parameters returns List[AnyTensor]
- Module train/eval mode switching
- Linear forward shape correctness through AnyTensor boundary
- ReLU forward pass zeroes negative values via AnyTensor boundary
"""

from testing import assert_true, assert_almost_equal
from shared.core.any_tensor import AnyTensor, zeros, ones
from shared.core.module import Module
from shared.core.layers.linear import Linear
from shared.core.layers.relu import ReLULayer


fn test_module_forward_accepts_anytensor() raises:
    """Module.forward accepts AnyTensor input and returns AnyTensor."""
    var layer = Linear(4, 2)
    var input: AnyTensor = zeros([1, 4], DType.float32)
    var output = layer.forward(input)
    assert_true(output.numel() > 0, "forward produces output")
    print("PASS: test_module_forward_accepts_anytensor")


fn test_module_forward_returns_anytensor() raises:
    """Module.forward return type is AnyTensor (H7 boundary)."""
    var layer = Linear(4, 2)
    var input = zeros([1, 4], DType.float32)
    # The return value must be assignable to AnyTensor
    var output: AnyTensor = layer.forward(input)
    assert_true(output.dtype() == DType.float32, "output dtype preserved")
    assert_true(output.numel() == 2, "output has correct numel")
    print("PASS: test_module_forward_returns_anytensor")


fn test_module_parameters_returns_anytensor_list() raises:
    """Module.parameters() returns List[AnyTensor]."""
    var layer = Linear(4, 2)
    var params = layer.parameters()
    assert_true(len(params) == 2, "Linear has weight and bias")
    # Each parameter should be an AnyTensor with elements
    for i in range(len(params)):
        assert_true(params[i].numel() > 0, "param has elements")
    print("PASS: test_module_parameters_returns_anytensor_list")


fn test_module_train_mode() raises:
    """Module train mode switching works without error."""
    var layer = Linear(4, 2)
    layer.train()
    # Verify layer still works after mode switch
    var input = zeros([1, 4], DType.float32)
    var output = layer.forward(input)
    assert_true(output.numel() > 0, "forward works after train mode")
    print("PASS: test_module_train_mode")


fn test_linear_forward_shape_batched() raises:
    """Linear(4, 2) produces correct output shape [3, 2] for batch=3."""
    var layer = Linear(4, 2)
    var input = zeros([3, 4], DType.float32)
    var output = layer.forward(input)
    var s = output.shape()
    assert_true(s[0] == 3, "batch dim preserved")
    assert_true(s[1] == 2, "output features correct")
    print("PASS: test_linear_forward_shape_batched")


fn test_relu_forward_zeroes_negatives() raises:
    """ReLU forward pass zeroes negative values via AnyTensor boundary."""
    var layer = ReLULayer()
    var input = zeros([4], DType.float32)
    # Set test values: 1.0, -0.5, 0.25, -1.0
    var data = input._data.bitcast[Float32]()
    data[0] = 1.0
    data[1] = -0.5
    data[2] = 0.25
    data[3] = -1.0
    var output = layer.forward(input)
    var out_data = output._data.bitcast[Float32]()
    assert_almost_equal(out_data[0], Float32(1.0), atol=1e-6)
    assert_almost_equal(out_data[1], Float32(0.0), atol=1e-6)
    assert_almost_equal(out_data[2], Float32(0.25), atol=1e-6)
    assert_almost_equal(out_data[3], Float32(0.0), atol=1e-6)
    print("PASS: test_relu_forward_zeroes_negatives")


fn test_relu_parameters_empty() raises:
    """ReLU has no trainable parameters."""
    var layer = ReLULayer()
    var params = layer.parameters()
    assert_true(len(params) == 0, "ReLU has no parameters")
    print("PASS: test_relu_parameters_empty")


fn main() raises:
    test_module_forward_accepts_anytensor()
    test_module_forward_returns_anytensor()
    test_module_parameters_returns_anytensor_list()
    test_module_train_mode()
    test_linear_forward_shape_batched()
    test_relu_forward_zeroes_negatives()
    test_relu_parameters_empty()
    print("\nAll 7 trait signature tests passed")
