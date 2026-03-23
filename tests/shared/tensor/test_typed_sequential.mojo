"""Tests for Sequential with AnyTensor boundaries (Phase 4b).

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Phase 4b updates Sequential2-5 to chain layers through AnyTensor:
- Sequential.forward accepts and returns AnyTensor
- Sequential.parameters returns List[AnyTensor]
- Inter-layer communication uses AnyTensor (H7 boundary)
- train/mode propagated to sub-layers

Tests cover:
- Sequential2 forward pass chaining Linear + ReLU
- Sequential2 parameter collection from sub-layers
- Sequential2 train mode propagation
- Sequential2 output shape correctness
- Sequential3 forward pass chaining three layers
"""

from testing import assert_true, assert_almost_equal
from shared.core.sequential import Sequential2, Sequential3
from shared.core.layers.linear import Linear
from shared.core.layers.relu import ReLULayer
from shared.tensor.any_tensor import AnyTensor, zeros, ones


fn test_sequential2_forward_linear_relu() raises:
    """Sequential2 chains Linear + ReLU via AnyTensor boundary."""
    var model = Sequential2[Linear, ReLULayer](
        Linear(4, 2),
        ReLULayer(),
    )
    var input = zeros([1, 4], DType.float32)
    var output = model.forward(input)
    assert_true(output.numel() > 0, "produces output")
    # Output shape should be [1, 2] after Linear(4, 2)
    var s = output.shape()
    assert_true(s[0] == 1, "batch dim preserved")
    assert_true(s[1] == 2, "output features from Linear")
    print("PASS: test_sequential2_forward_linear_relu")


fn test_sequential2_parameters_collected() raises:
    """Sequential2 collects parameters from all layers."""
    var model = Sequential2[Linear, ReLULayer](
        Linear(4, 2),
        ReLULayer(),
    )
    var params = model.parameters()
    # Linear has weight + bias = 2 params, ReLU has 0
    assert_true(len(params) == 2, "has 2 parameters from Linear")
    # Weight shape: (4, 2)
    var ws = params[0].shape()
    assert_true(ws[0] == 4, "weight rows")
    assert_true(ws[1] == 2, "weight cols")
    print("PASS: test_sequential2_parameters_collected")


fn test_sequential2_train_propagation() raises:
    """Sequential2 propagates train mode to sub-layers."""
    var model = Sequential2[Linear, ReLULayer](
        Linear(4, 2),
        ReLULayer(),
    )
    # Should not raise
    model.train()
    # Verify model still works after mode switch
    var input = zeros([1, 4], DType.float32)
    var output = model.forward(input)
    assert_true(output.numel() > 0, "forward works after train mode")
    print("PASS: test_sequential2_train_propagation")


fn test_sequential2_output_dtype_preserved() raises:
    """Sequential2 output preserves dtype through AnyTensor chain."""
    var model = Sequential2[Linear, ReLULayer](
        Linear(4, 2),
        ReLULayer(),
    )
    var input = zeros([1, 4], DType.float32)
    var output = model.forward(input)
    assert_true(
        output.dtype() == DType.float32, "dtype preserved through chain"
    )
    print("PASS: test_sequential2_output_dtype_preserved")


fn test_sequential2_relu_clips_negatives() raises:
    """Sequential2[Linear, ReLU] produces non-negative outputs.

    After ReLU, all output values must be >= 0.
    Uses ones input to get deterministic matmul output from Linear,
    then verifies ReLU zeroes any negative values.
    """
    var model = Sequential2[Linear, ReLULayer](
        Linear(4, 2),
        ReLULayer(),
    )
    var input = ones([1, 4], DType.float32)
    var output = model.forward(input)
    var n = output.numel()
    var out_data = output._data.bitcast[Float32]()
    for i in range(n):
        assert_true(
            out_data[i] >= 0.0,
            "ReLU output must be non-negative",
        )
    print("PASS: test_sequential2_relu_clips_negatives")


fn test_sequential3_forward_chain() raises:
    """Sequential3 chains Linear + ReLU + Linear via AnyTensor."""
    var model = Sequential3[Linear, ReLULayer, Linear](
        Linear(4, 3),
        ReLULayer(),
        Linear(3, 2),
    )
    var input = zeros([1, 4], DType.float32)
    var output = model.forward(input)
    var s = output.shape()
    assert_true(s[0] == 1, "batch dim")
    assert_true(s[1] == 2, "output features from final Linear")
    print("PASS: test_sequential3_forward_chain")


fn test_sequential3_parameters_combined() raises:
    """Sequential3 collects parameters from all sub-layers."""
    var model = Sequential3[Linear, ReLULayer, Linear](
        Linear(4, 3),
        ReLULayer(),
        Linear(3, 2),
    )
    var params = model.parameters()
    # Linear(4,3): weight + bias = 2; ReLU: 0; Linear(3,2): weight + bias = 2
    assert_true(len(params) == 4, "has 4 parameters total (2 per Linear)")
    print("PASS: test_sequential3_parameters_combined")


fn main() raises:
    test_sequential2_forward_linear_relu()
    test_sequential2_parameters_collected()
    test_sequential2_train_propagation()
    test_sequential2_output_dtype_preserved()
    test_sequential2_relu_clips_negatives()
    test_sequential3_forward_chain()
    test_sequential3_parameters_combined()
    print("\nAll 7 sequential tests passed")
