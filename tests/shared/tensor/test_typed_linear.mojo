"""Tests for Linear[dtype] with typed Tensor weights (Phase 4b).

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Phase 4b parameterizes Linear[dtype: DType = DType.float32]:
- Internal weights stored as Tensor[dtype] for type safety
- forward() accepts AnyTensor, returns AnyTensor (H7 boundary)
- parameters() returns List[AnyTensor] (Module trait compliance)
- Supports multiple dtype instantiations (float32, float64)

Tests cover:
- Default dtype (float32)
- Explicit float64 parameterization
- parameters() returns List[AnyTensor] (Module trait compliance)
- forward() AnyTensor boundary (H7 constraint)
- Numerical correctness with identity weights and known bias
"""

from testing import assert_true, assert_almost_equal
from shared.core.layers.linear import Linear
from shared.core.any_tensor import AnyTensor, zeros, ones


fn test_linear_default_dtype() raises:
    """Linear defaults to float32 weights."""
    var layer = Linear(4, 2)
    # Weight dtype should be float32 by default
    var params = layer.parameters()
    assert_true(params[0].dtype() == DType.float32, "default weight dtype is float32")
    assert_true(params[1].dtype() == DType.float32, "default bias dtype is float32")
    print("PASS: test_linear_default_dtype")


fn test_linear_float64() raises:
    """Linear[DType.float64] uses float64 weights."""
    var layer = Linear[DType.float64](4, 2)
    var params = layer.parameters()
    assert_true(params[0].dtype() == DType.float64, "weight dtype is float64")
    assert_true(params[1].dtype() == DType.float64, "bias dtype is float64")
    print("PASS: test_linear_float64")


fn test_linear_parameters_as_anytensor() raises:
    """parameters() returns List[AnyTensor] (Module trait compliance)."""
    var layer = Linear(4, 2)
    var params = layer.parameters()
    assert_true(len(params) == 2, "has weight and bias params")
    # Weight shape: (4, 2)
    var ws = params[0].shape()
    assert_true(ws[0] == 4, "weight rows")
    assert_true(ws[1] == 2, "weight cols")
    # Bias shape: (2,)
    var bs = params[1].shape()
    assert_true(bs[0] == 2, "bias size")
    print("PASS: test_linear_parameters_as_anytensor")


fn test_linear_forward_anytensor_boundary() raises:
    """forward() accepts AnyTensor, returns AnyTensor (H7 boundary)."""
    var layer = Linear(4, 2)
    var input: AnyTensor = zeros([1, 4], DType.float32)
    var output: AnyTensor = layer.forward(input)
    assert_true(output.dtype() == DType.float32, "output dtype preserved")
    var s = output.shape()
    assert_true(s[0] == 1, "batch dim")
    assert_true(s[1] == 2, "output features")
    print("PASS: test_linear_forward_anytensor_boundary")


fn test_linear_identity_weight_correctness() raises:
    """Linear with identity weights and known bias produces correct output.

    Input: [1.0, 0.5] (1x2)
    Weight: [[1, 0], [0, 1]] (2x2 identity)
    Bias: [0.25, 0.5]
    Expected: [1.25, 1.0]
    """
    var layer = Linear(2, 2)

    # Set weight to identity
    var weight_data = layer.weight._data.bitcast[Float32]()
    weight_data[0] = 1.0
    weight_data[1] = 0.0
    weight_data[2] = 0.0
    weight_data[3] = 1.0

    # Set bias to [0.25, 0.5]
    var bias_data = layer.bias._data.bitcast[Float32]()
    bias_data[0] = 0.25
    bias_data[1] = 0.5

    # Create input [1.0, 0.5]
    var input = ones([1, 2], DType.float32)
    var in_data = input._data.bitcast[Float32]()
    in_data[0] = 1.0
    in_data[1] = 0.5

    var output = layer.forward(input)
    var out_data = output._data.bitcast[Float32]()
    # Expected: [1.0*1 + 0.5*0 + 0.25, 1.0*0 + 0.5*1 + 0.5] = [1.25, 1.0]
    assert_almost_equal(out_data[0], Float32(1.25), atol=1e-5)
    assert_almost_equal(out_data[1], Float32(1.0), atol=1e-5)
    print("PASS: test_linear_identity_weight_correctness")


fn test_linear_zero_bias() raises:
    """Linear with zero bias produces matmul-only output."""
    var layer = Linear(3, 2)

    # Zero out bias
    var bias_data = layer.bias._data.bitcast[Float32]()
    bias_data[0] = 0.0
    bias_data[1] = 0.0

    # Set weight to known values: [[1, 0], [0, 1], [1, 1]]
    var weight_data = layer.weight._data.bitcast[Float32]()
    weight_data[0] = 1.0
    weight_data[1] = 0.0
    weight_data[2] = 0.0
    weight_data[3] = 1.0
    weight_data[4] = 1.0
    weight_data[5] = 1.0

    # Input: [1.0, 0.5, 0.25]
    var input = zeros([1, 3], DType.float32)
    var in_data = input._data.bitcast[Float32]()
    in_data[0] = 1.0
    in_data[1] = 0.5
    in_data[2] = 0.25

    var output = layer.forward(input)
    var out_data = output._data.bitcast[Float32]()
    # Expected: [1*1 + 0.5*0 + 0.25*1, 1*0 + 0.5*1 + 0.25*1] = [1.25, 0.75]
    assert_almost_equal(out_data[0], Float32(1.25), atol=1e-5)
    assert_almost_equal(out_data[1], Float32(0.75), atol=1e-5)
    print("PASS: test_linear_zero_bias")


fn main() raises:
    test_linear_default_dtype()
    test_linear_float64()
    test_linear_parameters_as_anytensor()
    test_linear_forward_anytensor_boundary()
    test_linear_identity_weight_correctness()
    test_linear_zero_bias()
    print("\nAll 6 typed linear tests passed")
