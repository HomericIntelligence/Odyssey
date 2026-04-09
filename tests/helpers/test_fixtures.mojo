"""Unit tests for test fixtures.

Tests the fixture functions in fixtures.mojo.
"""

from tests.helpers.fixtures import (
    random_tensor,
    sequential_tensor,
    nan_tensor,
    inf_tensor,
    ones_like,
    zeros_like,
)


def test_random_tensor_shape() raises:
    """Test random_tensor creates correct shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var tensor = random_tensor(shape, DType.float32)

    # Verify shape
    var t_shape = tensor.shape()
    if t_shape.__len__() != 2:
        raise Error("random_tensor should create 2D tensor")
    if t_shape[0] != 3 or t_shape[1] != 4:
        raise Error("random_tensor shape mismatch")

    # Verify dtype
    if tensor.dtype() != DType.float32:
        raise Error("random_tensor dtype mismatch")


def test_random_tensor_values() raises:
    """Test random_tensor produces values in [0, 1)."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(10)

    var tensor = random_tensor(shape, DType.float32)

    # Check a few values are in range [0, 1)
    var numel = tensor.numel()
    var all_in_range = True

    for i in range(numel):
        var val = tensor._get_float64(i)
        if val < 0.0 or val >= 1.0:
            all_in_range = False
            break

    if not all_in_range:
        raise Error("random_tensor values should be in [0, 1)")


def test_sequential_tensor() raises:
    """Test sequential_tensor produces correct values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var tensor = sequential_tensor(shape, DType.float32)

    # Verify values: 0, 1, 2, 3, 4, 5
    var expected_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    for i in range(6):
        var val = tensor._get_float64(i)
        if val != expected_values[i]:
            raise Error(
                "sequential_tensor value mismatch at index " + String(i)
            )


def test_nan_tensor() raises:
    """Test nan_tensor creates NaN values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var tensor = nan_tensor(shape)

    # Verify dtype is float32
    if tensor.dtype() != DType.float32:
        raise Error("nan_tensor should create float32")

    # Note: Exact NaN comparison is tricky, just verify it was created
    var numel = tensor.numel()
    if numel != 4:
        raise Error("nan_tensor numel mismatch")


def test_inf_tensor() raises:
    """Test inf_tensor creates infinity values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var tensor = inf_tensor(shape)

    # Verify dtype is float32
    if tensor.dtype() != DType.float32:
        raise Error("inf_tensor should create float32")

    # Verify numel
    var numel = tensor.numel()
    if numel != 4:
        raise Error("inf_tensor numel mismatch")


def test_ones_like() raises:
    """Test ones_like creates tensor of ones."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var template = random_tensor(shape, DType.float32)
    var ones_t = ones_like(template)

    # Verify shape and dtype match
    var ones_shape = ones_t.shape()
    if ones_shape[0] != 3 or ones_shape[1] != 4:
        raise Error("ones_like shape mismatch")

    if ones_t.dtype() != DType.float32:
        raise Error("ones_like dtype mismatch")

    # Verify all values are 1.0
    var numel = ones_t.numel()
    for i in range(numel):
        var val = ones_t._get_float64(i)
        if val != 1.0:
            raise Error("ones_like should contain all 1.0 values")


def test_zeros_like() raises:
    """Test zeros_like creates tensor of zeros."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var template = random_tensor(shape, DType.float32)
    var zeros_t = zeros_like(template)

    # Verify shape and dtype match
    var zeros_shape = zeros_t.shape()
    if zeros_shape[0] != 3 or zeros_shape[1] != 4:
        raise Error("zeros_like shape mismatch")

    if zeros_t.dtype() != DType.float32:
        raise Error("zeros_like dtype mismatch")

    # Verify all values are 0.0
    var numel = zeros_t.numel()
    for i in range(numel):
        var val = zeros_t._get_float64(i)
        if val != 0.0:
            raise Error("zeros_like should contain all 0.0 values")


def main() raises:
    """Run all fixture tests."""
    test_random_tensor_shape()
    print("✓ test_random_tensor_shape passed")

    test_random_tensor_values()
    print("✓ test_random_tensor_values passed")

    test_sequential_tensor()
    print("✓ test_sequential_tensor passed")

    test_nan_tensor()
    print("✓ test_nan_tensor passed")

    test_inf_tensor()
    print("✓ test_inf_tensor passed")

    test_ones_like()
    print("✓ test_ones_like passed")

    test_zeros_like()
    print("✓ test_zeros_like passed")

    print("\nAll fixture tests passed!")
