# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""Tests for AnyTensor method-style API: split and split_with_indices.

Verifies that the thin wrapper methods on AnyTensor produce identical results
to the functional implementations in shared.core.shape. Follows #3243 and #3804.
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.shape import split, split_with_indices

from tests.shared.conftest import (
    assert_numel,
    assert_dim,
    assert_value_at,
)


def test_split_method_equal() raises:
    """Test split() method splits into equal parts matching free function."""
    var a = arange(0.0, 12.0, 1.0, DType.float32)

    var method_parts = a.split(3)
    var free_parts = split(a, 3)

    if len(method_parts) != 3:
        raise Error("split method should return 3 parts")
    if len(free_parts) != 3:
        raise Error("split free function should return 3 parts")

    for i in range(3):
        assert_numel(
            method_parts[i], 4, "Each part should have 4 elements"
        )

    # Verify method matches free function values
    assert_value_at(
        method_parts[0], 0, 0.0, message="Part 0, index 0 should be 0.0"
    )
    assert_value_at(
        method_parts[0], 3, 3.0, message="Part 0, index 3 should be 3.0"
    )
    assert_value_at(
        method_parts[1], 0, 4.0, message="Part 1, index 0 should be 4.0"
    )
    assert_value_at(
        method_parts[1], 3, 7.0, message="Part 1, index 3 should be 7.0"
    )
    assert_value_at(
        method_parts[2], 0, 8.0, message="Part 2, index 0 should be 8.0"
    )
    assert_value_at(
        method_parts[2], 3, 11.0, message="Part 2, index 3 should be 11.0"
    )


def test_split_method_axis() raises:
    """Test split() method on 2D tensor along axis=1."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(6)
    var a = arange(0.0, 24.0, 1.0, DType.float32).reshape(shape)

    var parts = a.split(3, 1)  # Split 6 cols into 3 parts of 2

    if len(parts) != 3:
        raise Error("Should split into 3 parts along axis=1")

    for i in range(3):
        assert_dim(parts[i], 2, "Each part should be 2D")
        assert_numel(parts[i], 8, "Each part should have 8 elements (4x2)")


def test_split_with_indices_method_basic() raises:
    """Test split_with_indices() method on 1D tensor at [3, 7]."""
    var a = arange(0.0, 10.0, 1.0, DType.float32)
    var indices = List[Int]()
    indices.append(3)
    indices.append(7)

    var parts = a.split_with_indices(indices)

    if len(parts) != 3:
        raise Error("Should split into 3 sections")

    assert_numel(parts[0], 3, "First section: indices 0-2 (3 elements)")
    assert_numel(parts[1], 4, "Second section: indices 3-6 (4 elements)")
    assert_numel(parts[2], 3, "Third section: indices 7-9 (3 elements)")

    # Verify values: [0,1,2], [3,4,5,6], [7,8,9]
    assert_value_at(parts[0], 0, 0.0, message="Part 0, index 0 should be 0.0")
    assert_value_at(parts[0], 2, 2.0, message="Part 0, index 2 should be 2.0")
    assert_value_at(parts[1], 0, 3.0, message="Part 1, index 0 should be 3.0")
    assert_value_at(parts[1], 3, 6.0, message="Part 1, index 3 should be 6.0")
    assert_value_at(parts[2], 0, 7.0, message="Part 2, index 0 should be 7.0")
    assert_value_at(parts[2], 2, 9.0, message="Part 2, index 2 should be 9.0")


def test_split_with_indices_method_2d() raises:
    """Test split_with_indices() method on 2D tensor along axis=0."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(5)
    var a = ones(shape, DType.float32)
    var indices = List[Int]()
    indices.append(3)
    indices.append(7)

    var parts = a.split_with_indices(indices, 0)

    if len(parts) != 3:
        raise Error("Should split into 3 sections along axis=0")

    assert_numel(
        parts[0], 15, "First section: 3 rows x 5 cols = 15 elements"
    )
    assert_numel(
        parts[1], 20, "Second section: 4 rows x 5 cols = 20 elements"
    )
    assert_numel(
        parts[2], 15, "Third section: 3 rows x 5 cols = 15 elements"
    )


def test_split_with_indices_method_vs_free_fn() raises:
    """Test that method result matches free function result (symmetry test)."""
    var a = arange(0.0, 10.0, 1.0, DType.float32)
    var indices = List[Int]()
    indices.append(3)
    indices.append(7)

    var method_parts = a.split_with_indices(indices)
    var free_parts = split_with_indices(a, indices)

    if len(method_parts) != len(free_parts):
        raise Error(
            "Method and free function should return same number of parts"
        )

    for i in range(len(method_parts)):
        if method_parts[i].numel() != free_parts[i].numel():
            raise Error(
                "Part sizes should match between method and free function"
            )

    # Spot-check values match
    assert_value_at(method_parts[0], 0, 0.0, message="Method part 0[0] == 0.0")
    assert_value_at(free_parts[0], 0, 0.0, message="Free fn part 0[0] == 0.0")
    assert_value_at(method_parts[2], 2, 9.0, message="Method part 2[2] == 9.0")
    assert_value_at(free_parts[2], 2, 9.0, message="Free fn part 2[2] == 9.0")


def main() raises:
    """Run AnyTensor method API tests for split and split_with_indices."""
    print("Testing AnyTensor method API: split and split_with_indices...")

    print("  Testing split() method (equal splits)...")
    test_split_method_equal()

    print("  Testing split() method (axis=1)...")
    test_split_method_axis()

    print("  Testing split_with_indices() method (basic 1D)...")
    test_split_with_indices_method_basic()

    print("  Testing split_with_indices() method (2D)...")
    test_split_with_indices_method_2d()

    print("  Testing split_with_indices() method vs free function...")
    test_split_with_indices_method_vs_free_fn()

    print("All AnyTensor method API tests passed!")
