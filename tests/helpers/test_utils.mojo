"""Unit tests for test utilities.

Tests the utility functions in utils.mojo.
"""

from tests.helpers.utils import (
    print_tensor,
    tensor_summary,
    compare_tensors,
    benchmark,
)
from tests.helpers.fixtures import random_tensor, sequential_tensor


def test_print_tensor() raises:
    """Test print_tensor output format."""
    var tensor = sequential_tensor([2, 3], DType.float32)
    var output = print_tensor(tensor)

    # Verify output contains expected components
    if "Tensor" not in output:
        raise Error("print_tensor should contain 'Tensor' in output")
    if "shape=" not in output:
        raise Error("print_tensor should contain 'shape=' in output")
    if "dtype=" not in output:
        raise Error("print_tensor should contain 'dtype=' in output")


def test_tensor_summary() raises:
    """Test tensor_summary statistics calculation."""
    var tensor = sequential_tensor([3, 3], DType.float32)
    var output = tensor_summary(tensor)

    # Verify output contains expected statistics
    if "Shape:" not in output:
        raise Error("tensor_summary should contain 'Shape:'")
    if "Min:" not in output:
        raise Error("tensor_summary should contain 'Min:'")
    if "Max:" not in output:
        raise Error("tensor_summary should contain 'Max:'")
    if "Mean:" not in output:
        raise Error("tensor_summary should contain 'Mean:'")
    if "Std:" not in output:
        raise Error("tensor_summary should contain 'Std:'")


def test_compare_tensors_identical() raises:
    """Test compare_tensors with identical tensors."""
    var t1 = sequential_tensor([3, 3], DType.float32)
    var t2 = sequential_tensor([3, 3], DType.float32)

    var output = compare_tensors(t1, t2)

    # Verify shapes and dtypes match
    if "Shapes match: ✓" not in output:
        raise Error("compare_tensors should report matching shapes")
    if "DTypes match: ✓" not in output:
        raise Error("compare_tensors should report matching dtypes")


def test_compare_tensors_different_shapes() raises:
    """Test compare_tensors with different shapes."""
    var t1 = sequential_tensor([2, 3], DType.float32)
    var t2 = sequential_tensor([3, 2], DType.float32)

    var output = compare_tensors(t1, t2)

    # Verify shape mismatch is detected
    if "Shapes match: ✗" not in output:
        raise Error("compare_tensors should detect shape mismatch")


def test_benchmark_function() raises:
    """Test benchmark function execution."""

    def dummy_test() raises:
        pass

    # Just verify benchmark doesn't crash
    var result = benchmark[dummy_test](10)
    # Result should be 0.0 (placeholder)
    if result < 0.0:
        raise Error("benchmark should return non-negative time")


def main() raises:
    """Run all utility tests."""
    test_print_tensor()
    print("✓ test_print_tensor passed")

    test_tensor_summary()
    print("✓ test_tensor_summary passed")

    test_compare_tensors_identical()
    print("✓ test_compare_tensors_identical passed")

    test_compare_tensors_different_shapes()
    print("✓ test_compare_tensors_different_shapes passed")

    test_benchmark_function()
    print("✓ test_benchmark_function passed")

    print("\nAll utility tests passed!")
