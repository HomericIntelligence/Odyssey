"""Tests for gradient container types with AnyTensor.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Tests verify that gradient container types (GradientPair, GradientTriple,
GradientQuad) store AnyTensor fields correctly. Since AnyTensor is now an
alias for AnyTensor, these tests confirm the migration preserves correct
storage and retrieval of gradient tensors.

Tests cover:
- GradientPair: Stores two AnyTensor gradients (grad_a, grad_b)
- GradientTriple: Stores three AnyTensor gradients (grad_input, grad_weights, grad_bias)
- GradientQuad: Stores four AnyTensor gradients
- Field values preserved after construction
"""

from std.testing import assert_true, assert_almost_equal
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.gradient_types import GradientPair, GradientTriple, GradientQuad


def test_gradient_pair_stores_anytensor() raises:
    """GradientPair stores two AnyTensor gradient fields."""
    var g_a: AnyTensor = zeros([4, 3], DType.float32)
    var g_b: AnyTensor = ones([4, 3], DType.float32)
    var pair = GradientPair(g_a, g_b)
    assert_true(pair.grad_a.numel() == 12, "grad_a has 12 elements")
    assert_true(pair.grad_b.numel() == 12, "grad_b has 12 elements")
    assert_true(pair.grad_a.dtype() == DType.float32, "grad_a dtype preserved")
    assert_true(pair.grad_b.dtype() == DType.float32, "grad_b dtype preserved")
    print("PASS: test_gradient_pair_stores_anytensor")


def test_gradient_pair_preserves_values() raises:
    """GradientPair preserves tensor values through construction."""
    var g_a: AnyTensor = zeros([2], DType.float32)
    g_a._set_float64(0, 0.5)
    g_a._set_float64(1, 1.0)
    var g_b: AnyTensor = zeros([2], DType.float32)
    g_b._set_float64(0, 1.5)
    g_b._set_float64(1, -0.5)
    var pair = GradientPair(g_a, g_b)
    assert_almost_equal(
        Float64(pair.grad_a._get_float64(0)), 0.5, atol=1e-6, msg="grad_a[0]"
    )
    assert_almost_equal(
        Float64(pair.grad_a._get_float64(1)), 1.0, atol=1e-6, msg="grad_a[1]"
    )
    assert_almost_equal(
        Float64(pair.grad_b._get_float64(0)), 1.5, atol=1e-6, msg="grad_b[0]"
    )
    assert_almost_equal(
        Float64(pair.grad_b._get_float64(1)), -0.5, atol=1e-6, msg="grad_b[1]"
    )
    print("PASS: test_gradient_pair_preserves_values")


def test_gradient_pair_different_shapes() raises:
    """GradientPair holds AnyTensors of different shapes."""
    var g_a: AnyTensor = zeros([3], DType.float32)
    var g_b: AnyTensor = zeros([2, 4], DType.float32)
    var pair = GradientPair(g_a, g_b)
    assert_true(pair.grad_a.numel() == 3, "grad_a has 3 elements")
    assert_true(pair.grad_b.numel() == 8, "grad_b has 8 elements")
    var shape_b = pair.grad_b.shape()
    assert_true(shape_b[0] == 2, "grad_b dim 0")
    assert_true(shape_b[1] == 4, "grad_b dim 1")
    print("PASS: test_gradient_pair_different_shapes")


def test_gradient_triple_stores_anytensor() raises:
    """GradientTriple stores three AnyTensor gradient fields."""
    var g_input: AnyTensor = zeros([4, 3], DType.float32)
    var g_weights: AnyTensor = ones([3, 2], DType.float32)
    var g_bias: AnyTensor = zeros([2], DType.float32)
    var triple = GradientTriple(g_input, g_weights, g_bias)
    assert_true(triple.grad_input.numel() == 12, "grad_input has 12 elements")
    assert_true(triple.grad_weights.numel() == 6, "grad_weights has 6 elements")
    assert_true(triple.grad_bias.numel() == 2, "grad_bias has 2 elements")
    print("PASS: test_gradient_triple_stores_anytensor")


def test_gradient_triple_preserves_values() raises:
    """GradientTriple preserves tensor values through construction."""
    var g_input: AnyTensor = zeros([2], DType.float32)
    g_input._set_float64(0, 1.0)
    var g_weights: AnyTensor = zeros([2], DType.float32)
    g_weights._set_float64(0, 0.5)
    var g_bias: AnyTensor = zeros([1], DType.float32)
    g_bias._set_float64(0, -1.0)
    var triple = GradientTriple(g_input, g_weights, g_bias)
    assert_almost_equal(
        Float64(triple.grad_input._get_float64(0)),
        1.0,
        atol=1e-6,
        msg="grad_input[0]",
    )
    assert_almost_equal(
        Float64(triple.grad_weights._get_float64(0)),
        0.5,
        atol=1e-6,
        msg="grad_weights[0]",
    )
    assert_almost_equal(
        Float64(triple.grad_bias._get_float64(0)),
        -1.0,
        atol=1e-6,
        msg="grad_bias[0]",
    )
    print("PASS: test_gradient_triple_preserves_values")


def test_gradient_quad_stores_anytensor() raises:
    """GradientQuad stores four AnyTensor gradient fields."""
    var g_a: AnyTensor = zeros([3], DType.float32)
    var g_b: AnyTensor = zeros([3], DType.float32)
    var g_c: AnyTensor = zeros([3], DType.float32)
    var g_d: AnyTensor = zeros([3], DType.float32)
    var quad = GradientQuad(g_a, g_b, g_c, g_d)
    assert_true(quad.grad_a.numel() == 3, "grad_a has 3 elements")
    assert_true(quad.grad_b.numel() == 3, "grad_b has 3 elements")
    assert_true(quad.grad_c.numel() == 3, "grad_c has 3 elements")
    assert_true(quad.grad_d.numel() == 3, "grad_d has 3 elements")
    print("PASS: test_gradient_quad_stores_anytensor")


def test_gradient_quad_preserves_dtype() raises:
    """GradientQuad preserves dtype of each AnyTensor field."""
    var g_a: AnyTensor = zeros([2], DType.float32)
    var g_b: AnyTensor = zeros([2], DType.float32)
    var g_c: AnyTensor = zeros([2], DType.float32)
    var g_d: AnyTensor = zeros([2], DType.float32)
    var quad = GradientQuad(g_a, g_b, g_c, g_d)
    assert_true(quad.grad_a.dtype() == DType.float32, "grad_a dtype")
    assert_true(quad.grad_b.dtype() == DType.float32, "grad_b dtype")
    assert_true(quad.grad_c.dtype() == DType.float32, "grad_c dtype")
    assert_true(quad.grad_d.dtype() == DType.float32, "grad_d dtype")
    print("PASS: test_gradient_quad_preserves_dtype")


def main() raises:
    test_gradient_pair_stores_anytensor()
    test_gradient_pair_preserves_values()
    test_gradient_pair_different_shapes()
    test_gradient_triple_stores_anytensor()
    test_gradient_triple_preserves_values()
    test_gradient_quad_stores_anytensor()
    test_gradient_quad_preserves_dtype()
    print("\n7 gradient types AnyTensor tests passed\n")
