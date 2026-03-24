"""Tests for collection operations with AnyTensor.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests verify that collection ops (concatenate, stack, split) work correctly
when called with AnyTensor values. Since AnyTensor is now an alias for AnyTensor,
these tests confirm the migration preserves correct behavior.

Tests cover:
- concatenate: Join tensors along existing axis
- concatenate axis=1: Join along non-default axis
- stack: Join tensors along new axis
- split: Divide tensor into equal parts
- split axis=1: Split along non-default axis
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.shape import concatenate, stack, split


fn test_concatenate_axis0() raises:
    """Concatenate joins AnyTensor list along axis 0."""
    var a: AnyTensor = ones([2, 3], DType.float32)
    var b: AnyTensor = ones([3, 3], DType.float32)
    var tensors = List[AnyTensor]()
    tensors.append(a)
    tensors.append(b)
    var result = concatenate(tensors, axis=0)
    var s = result.shape()
    assert_true(s[0] == 5, "concatenated dim should be 2+3=5")
    assert_true(s[1] == 3, "other dim preserved")
    assert_true(result.numel() == 15, "total elements 5*3=15")
    assert_true(result.dtype() == DType.float32, "dtype preserved")
    print("PASS: test_concatenate_axis0")


fn test_concatenate_axis1() raises:
    """Concatenate joins AnyTensor list along axis 1."""
    var a: AnyTensor = ones([2, 3], DType.float32)
    var b: AnyTensor = ones([2, 4], DType.float32)
    var tensors = List[AnyTensor]()
    tensors.append(a)
    tensors.append(b)
    var result = concatenate(tensors, axis=1)
    var s = result.shape()
    assert_true(s[0] == 2, "non-concat dim preserved")
    assert_true(s[1] == 7, "concatenated dim should be 3+4=7")
    print("PASS: test_concatenate_axis1")


fn test_concatenate_single_tensor() raises:
    """Concatenate with single tensor returns equivalent tensor."""
    var a: AnyTensor = zeros([3, 4], DType.float32)
    var tensors = List[AnyTensor]()
    tensors.append(a)
    var result = concatenate(tensors, axis=0)
    var s = result.shape()
    assert_true(s[0] == 3, "shape preserved dim 0")
    assert_true(s[1] == 4, "shape preserved dim 1")
    print("PASS: test_concatenate_single_tensor")


fn test_stack_axis0() raises:
    """Stack creates new axis 0 from AnyTensor list."""
    var a: AnyTensor = ones([3, 4], DType.float32)
    var b: AnyTensor = ones([3, 4], DType.float32)
    var tensors = List[AnyTensor]()
    tensors.append(a)
    tensors.append(b)
    var result = stack(tensors, axis=0)
    var s = result.shape()
    assert_true(s[0] == 2, "stacked dim should be 2")
    assert_true(s[1] == 3, "original dim 0")
    assert_true(s[2] == 4, "original dim 1")
    assert_true(result.dtype() == DType.float32, "dtype preserved")
    print("PASS: test_stack_axis0")


fn test_stack_1d_tensors() raises:
    """Stack 1D AnyTensors creates 2D result."""
    var a: AnyTensor = zeros([4], DType.float32)
    var b: AnyTensor = zeros([4], DType.float32)
    var c: AnyTensor = zeros([4], DType.float32)
    var tensors = List[AnyTensor]()
    tensors.append(a)
    tensors.append(b)
    tensors.append(c)
    var result = stack(tensors, axis=0)
    var s = result.shape()
    assert_true(s[0] == 3, "stacked 3 tensors")
    assert_true(s[1] == 4, "original length")
    print("PASS: test_stack_1d_tensors")


fn test_split_equal_parts() raises:
    """Split divides AnyTensor into equal parts along axis 0."""
    var t: AnyTensor = ones([6, 4], DType.float32)
    var parts = split(t, 3, axis=0)
    assert_true(len(parts) == 3, "should produce 3 parts")
    for i in range(len(parts)):
        var s = parts[i].shape()
        assert_true(s[0] == 2, "split dim should be 6/3=2")
        assert_true(s[1] == 4, "other dim preserved")
    print("PASS: test_split_equal_parts")


fn test_split_axis1() raises:
    """Split along axis 1 produces correct shapes."""
    var t: AnyTensor = ones([4, 6], DType.float32)
    var parts = split(t, 2, axis=1)
    assert_true(len(parts) == 2, "should produce 2 parts")
    for i in range(len(parts)):
        var s = parts[i].shape()
        assert_true(s[0] == 4, "non-split dim preserved")
        assert_true(s[1] == 3, "split dim should be 6/2=3")
    print("PASS: test_split_axis1")


fn test_split_into_single() raises:
    """Split with num_splits=1 returns the whole tensor."""
    var t: AnyTensor = zeros([4, 3], DType.float32)
    var parts = split(t, 1, axis=0)
    assert_true(len(parts) == 1, "should produce 1 part")
    var s = parts[0].shape()
    assert_true(s[0] == 4, "full dim 0")
    assert_true(s[1] == 3, "full dim 1")
    print("PASS: test_split_into_single")


fn main() raises:
    test_concatenate_axis0()
    test_concatenate_axis1()
    test_concatenate_single_tensor()
    test_stack_axis0()
    test_stack_1d_tensors()
    test_split_equal_parts()
    test_split_axis1()
    test_split_into_single()
    print("\n8 collection ops AnyTensor tests passed\n")
