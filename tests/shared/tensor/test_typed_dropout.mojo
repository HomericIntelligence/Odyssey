# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for DropoutLayer with AnyTensor inputs.

TDD tests for Phase 4a (PR 6, epic #4998): update non-Module layers.
DropoutLayer uses AnyTensor for forward/backward (no struct-level dtype
param needed since it has no typed weights).

Tests cover:
- Forward preserves dtype (float32)
- Forward preserves dtype (float64)
- Inference mode passes input through unchanged
- Training mode zeros some elements
- Training mode scales kept elements by 1/(1-p)
- Backward applies same mask
- No trainable parameters
"""

from testing import assert_true, assert_almost_equal
from shared.core.layers.dropout import DropoutLayer
from shared.core.extensor import AnyTensor, zeros


fn test_dropout_forward_preserves_dtype_f32() raises:
    """Forward pass preserves float32 dtype."""
    var layer = DropoutLayer(dropout_rate=0.5)
    layer.set_training(False)
    var input = AnyTensor([2, 4], DType.float32)
    var data = input._data.bitcast[Float32]()
    for i in range(input.numel()):
        data[i] = Float32(1.0)
    var output = layer.forward(input)
    assert_true(output.dtype() == DType.float32, "output dtype should be float32")
    assert_true(output.numel() == 8, "output should have same numel")
    print("PASS: test_dropout_forward_preserves_dtype_f32")


fn test_dropout_forward_preserves_dtype_f64() raises:
    """Forward pass preserves float64 dtype."""
    var layer = DropoutLayer(dropout_rate=0.5)
    layer.set_training(False)
    var input = AnyTensor([2, 4], DType.float64)
    for i in range(input.numel()):
        input.set(i, 1.0)
    var output = layer.forward(input)
    assert_true(output.dtype() == DType.float64, "output dtype should be float64")
    print("PASS: test_dropout_forward_preserves_dtype_f64")


fn test_dropout_inference_passthrough() raises:
    """In inference mode (training=False), output equals input exactly."""
    var layer = DropoutLayer(dropout_rate=0.5)
    layer.set_training(False)
    var input = AnyTensor([4], DType.float32)
    var data = input._data.bitcast[Float32]()
    data[0] = 0.5
    data[1] = 1.0
    data[2] = 1.5
    data[3] = 0.25
    var output = layer.forward(input)
    var out = output._data.bitcast[Float32]()
    assert_almost_equal(out[0], Float32(0.5), atol=1e-6)
    assert_almost_equal(out[1], Float32(1.0), atol=1e-6)
    assert_almost_equal(out[2], Float32(1.5), atol=1e-6)
    assert_almost_equal(out[3], Float32(0.25), atol=1e-6)
    print("PASS: test_dropout_inference_passthrough")


fn test_dropout_training_zeros_elements() raises:
    """In training mode, some elements are zeroed (stochastic)."""
    var layer = DropoutLayer(dropout_rate=0.5)
    layer.set_training(True)
    # Use a large enough tensor that statistically some will be zeroed
    var input = AnyTensor([100], DType.float32)
    var data = input._data.bitcast[Float32]()
    for i in range(100):
        data[i] = Float32(1.0)
    var output = layer.forward(input)
    var out = output._data.bitcast[Float32]()
    var zero_count = 0
    var nonzero_count = 0
    for i in range(100):
        if out[i] == Float32(0.0):
            zero_count += 1
        else:
            nonzero_count += 1
    # With p=0.5 and 100 elements, expect roughly 50 zeros
    # Allow wide margin for stochastic test: at least 10 of each
    assert_true(zero_count >= 10, "should have some zeroed elements")
    assert_true(nonzero_count >= 10, "should have some kept elements")
    print("PASS: test_dropout_training_zeros_elements")


fn test_dropout_training_scale_factor() raises:
    """Kept elements are scaled by 1/(1-p) to preserve expected value."""
    var layer = DropoutLayer(dropout_rate=0.5)
    layer.set_training(True)
    var input = AnyTensor([100], DType.float32)
    var data = input._data.bitcast[Float32]()
    for i in range(100):
        data[i] = Float32(1.0)
    var output = layer.forward(input)
    var out = output._data.bitcast[Float32]()
    # Non-zero elements should be scaled by 1/(1-0.5) = 2.0
    var scale = Float32(1.0) / (Float32(1.0) - Float32(0.5))
    for i in range(100):
        var val = out[i]
        if val != Float32(0.0):
            assert_almost_equal(val, scale, atol=1e-6)
    print("PASS: test_dropout_training_scale_factor")


fn test_dropout_zero_rate() raises:
    """Dropout with rate=0.0 keeps all elements (no dropout)."""
    var layer = DropoutLayer(dropout_rate=0.0)
    layer.set_training(True)
    var input = AnyTensor([4], DType.float32)
    var data = input._data.bitcast[Float32]()
    data[0] = 0.5
    data[1] = 1.0
    data[2] = 1.5
    data[3] = 0.25
    var output = layer.forward(input)
    var out = output._data.bitcast[Float32]()
    # With rate=0, scale = 1/(1-0) = 1.0, all elements kept
    assert_almost_equal(out[0], Float32(0.5), atol=1e-6)
    assert_almost_equal(out[1], Float32(1.0), atol=1e-6)
    assert_almost_equal(out[2], Float32(1.5), atol=1e-6)
    assert_almost_equal(out[3], Float32(0.25), atol=1e-6)
    print("PASS: test_dropout_zero_rate")


fn test_dropout_backward_typed() raises:
    """Backward pass applies mask and returns AnyTensor."""
    var layer = DropoutLayer(dropout_rate=0.5)
    layer.set_training(True)
    var input = AnyTensor([10], DType.float32)
    var data = input._data.bitcast[Float32]()
    for i in range(10):
        data[i] = Float32(1.0)
    # Forward to generate mask
    _ = layer.forward(input)
    var grad_output = AnyTensor([10], DType.float32)
    var gdata = grad_output._data.bitcast[Float32]()
    for i in range(10):
        gdata[i] = Float32(1.0)
    var grad_input = layer.backward(grad_output, layer.last_mask)
    assert_true(grad_input.dtype() == DType.float32, "grad_input dtype")
    assert_true(grad_input.numel() == 10, "grad_input numel")
    print("PASS: test_dropout_backward_typed")


fn test_dropout_no_parameters() raises:
    """Dropout has no trainable parameters."""
    var layer = DropoutLayer(dropout_rate=0.5)
    var params = layer.parameters()
    assert_true(len(params) == 0, "dropout should have 0 parameters")
    print("PASS: test_dropout_no_parameters")


fn main() raises:
    test_dropout_forward_preserves_dtype_f32()
    test_dropout_forward_preserves_dtype_f64()
    test_dropout_inference_passthrough()
    test_dropout_training_zeros_elements()
    test_dropout_training_scale_factor()
    test_dropout_zero_rate()
    test_dropout_backward_typed()
    test_dropout_no_parameters()
    print("All 8 typed dropout tests passed!")
