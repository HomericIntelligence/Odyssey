"""Tests for utility helper functions.

This module tests the utility functions:
- ones_like
- zeros_like
- full_like
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import (
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)


def test_ones_like_shape() raises:
    """Test that ones_like creates tensor with correct shape."""
    print("Testing ones_like shape...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var template = zeros(shape, DType.float32)
    var result = ones_like(template)

    # Check shape matches
    if result.dim() != template.dim():
        raise Error("ones_like should preserve number of dimensions")

    for i in range(template.dim()):
        if result.shape()[i] != template.shape()[i]:
            raise Error("ones_like should preserve all dimensions")

    print(
        "  Template shape:",
        template.shape()[0],
        "x",
        template.shape()[1],
        "x",
        template.shape()[2],
    )
    print(
        "  Result shape:",
        result.shape()[0],
        "x",
        result.shape()[1],
        "x",
        result.shape()[2],
    )
    print("  ✓ ones_like shape test passed")


def test_ones_like_values() raises:
    """Test that ones_like fills tensor with ones."""
    print("Testing ones_like values...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var template = zeros(shape, DType.float32)
    var result = ones_like(template)

    # Check all values are 1
    for i in range(result.numel()):
        var val = result._get_float64(i)
        if val != 1.0:
            raise Error("ones_like should fill with 1.0")

    print("  All", result.numel(), "values are 1.0")
    print("  ✓ ones_like values test passed")


def test_ones_like_dtype() raises:
    """Test that ones_like preserves dtype."""
    print("Testing ones_like dtype...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    # Test float32
    var template_f32 = zeros(shape, DType.float32)
    var result_f32 = ones_like(template_f32)

    if result_f32.dtype() != DType.float32:
        raise Error("ones_like should preserve float32 dtype")

    # Test float64
    var template_f64 = zeros(shape, DType.float64)
    var result_f64 = ones_like(template_f64)

    if result_f64.dtype() != DType.float64:
        raise Error("ones_like should preserve float64 dtype")

    print("  float32 dtype preserved: ✓")
    print("  float64 dtype preserved: ✓")
    print("  ✓ ones_like dtype test passed")


def test_zeros_like_values() raises:
    """Test that zeros_like fills tensor with zeros."""
    print("Testing zeros_like values...")

    var shape = List[Int]()
    shape.append(3)
    shape.append(2)

    var template = ones(shape, DType.float32)
    var result = zeros_like(template)

    # Check all values are 0
    for i in range(result.numel()):
        var val = result._get_float64(i)
        if val != 0.0:
            raise Error("zeros_like should fill with 0.0")

    print("  All", result.numel(), "values are 0.0")
    print("  ✓ zeros_like values test passed")


def test_full_like_custom_value() raises:
    """Test that full_like fills with custom value."""
    print("Testing full_like with custom value...")

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var template = zeros(shape, DType.float32)
    var fill_value = 3.14

    var result = full_like(template, fill_value)

    # Check all values are fill_value
    for i in range(result.numel()):
        var val = result._get_float64(i)
        var diff = abs(val - fill_value)
        if diff > 0.001:
            raise Error("full_like should fill with custom value")

    print("  All", result.numel(), "values are", fill_value)
    print("  ✓ full_like test passed")


def test_utility_with_different_shapes() raises:
    """Test utilities with various tensor shapes."""
    print("Testing utilities with various shapes...")

    # Scalar (0D)
    var scalar_shape = List[Int]()
    var scalar = ones(scalar_shape, DType.float32)
    var scalar_zeros = zeros_like(scalar)

    # 1D
    var vec_shape = List[Int]()
    vec_shape.append(5)
    var vec = zeros(vec_shape, DType.float32)
    var vec_ones = ones_like(vec)

    # 2D
    var mat_shape = List[Int]()
    mat_shape.append(3)
    mat_shape.append(4)
    var mat = zeros(mat_shape, DType.float32)
    var mat_full = full_like(mat, 2.0)

    # 3D
    var tensor_shape = List[Int]()
    tensor_shape.append(2)
    tensor_shape.append(3)
    tensor_shape.append(4)
    var tensor = zeros(tensor_shape, DType.float32)
    var tensor_ones = ones_like(tensor)

    print("  Scalar (0D): ✓")
    print("  Vector (1D):", vec_ones.shape()[0], "elements ✓")
    print("  Matrix (2D):", mat_full.shape()[0], "x", mat_full.shape()[1], "✓")
    print(
        "  Tensor (3D):",
        tensor_ones.shape()[0],
        "x",
        tensor_ones.shape()[1],
        "x",
        tensor_ones.shape()[2],
        "✓",
    )
    print("  ✓ Various shapes test passed")


def run_all_tests() raises:
    """Run all utility function tests."""
    print("=" * 60)
    print("Utility Functions Test Suite")
    print("=" * 60)

    test_ones_like_shape()
    test_ones_like_values()
    test_ones_like_dtype()
    test_zeros_like_values()
    test_full_like_custom_value()
    test_utility_with_different_shapes()

    print("=" * 60)
    print("All utility function tests passed! ✓")
    print("=" * 60)


def main() raises:
    """Entry point for utility tests."""
    run_all_tests()
