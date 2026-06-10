"""Integration tests for AnyTensor operations.

Tests chained arithmetic operations and creation + arithmetic patterns.

"""


from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import (
    zeros,
    ones,
    full,
    arange,
    eye,
    linspace,
)
from projectodyssey.core.arithmetic import add, subtract, multiply
from tests.projectodyssey.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


def test_chained_add_operations() raises:
    """Test chaining multiple add operations."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)  # [1, 1, 1, 1, 1]
    var b = full(shape, 2.0, DType.float32)  # [2, 2, 2, 2, 2]
    var c = full(shape, 3.0, DType.float32)  # [3, 3, 3, 3, 3]

    var result = add(add(a, b), c)  # (1+2)+3 = 6

    assert_all_values(result, 6.0, 1e-6, "Chained additions should work")


def test_mixed_arithmetic_operations() raises:
    """Test mixing different arithmetic operations."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = full(shape, 4.0, DType.float32)

    # (a + b) * c = (2 + 3) * 4 = 20
    var sum_ab = add(a, b)
    var result = multiply(sum_ab, c)

    assert_all_values(result, 20.0, 1e-6, "Mixed operations should work")


def test_arithmetic_with_operator_overloading() raises:
    """Test using operator overloading for complex expressions."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = full(shape, 3.0, DType.float32)

    # a + b * c = 1 + 2 * 3 = 1 + 6 = 7
    var result = a + b * c

    assert_all_values(result, 7.0, 1e-6, "Operator precedence should work")


def test_complex_expression() raises:
    """Test complex arithmetic expression."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = full(shape, 3.0, DType.float32)
    var d = full(shape, 4.0, DType.float32)

    # ((a + b) * c) - d = ((1 + 2) * 3) - 4 = 9 - 4 = 5
    var result = ((a + b) * c) - d

    assert_all_values(result, 5.0, 1e-6, "Complex expressions should work")


def test_identity_matrix_operations() raises:
    """Test operations with identity matrix."""
    var I = eye(3, 3, 0, DType.float32)
    var A = full(List[Int](), 2.0, DType.float32)  # Will need reshaping
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var B = full(shape, 2.0, DType.float32)

    # I + B should give all 3s on diagonal, 2s elsewhere
    var result = add(I, B)

    assert_numel(result, 9, "Result should be 3x3")
    # Check diagonal
    assert_value_at(result, 0, 3.0, 1e-6, "Diagonal [0,0]")
    assert_value_at(result, 4, 3.0, 1e-6, "Diagonal [1,1]")
    assert_value_at(result, 8, 3.0, 1e-6, "Diagonal [2,2]")
    # Check off-diagonal
    assert_value_at(result, 1, 2.0, 1e-6, "Off-diagonal [0,1]")
    assert_value_at(result, 3, 2.0, 1e-6, "Off-diagonal [1,0]")


def test_arange_arithmetic() raises:
    """Test arithmetic with arange-created tensors."""
    var a = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var shape = List[Int]()
    shape.append(5)
    var b = ones(shape, DType.float32)  # [1, 1, 1, 1, 1]

    var result = add(a, b)  # [1, 2, 3, 4, 5]

    assert_value_at(result, 0, 1.0, 1e-6, "0 + 1 = 1")
    assert_value_at(result, 2, 3.0, 1e-6, "2 + 1 = 3")
    assert_value_at(result, 4, 5.0, 1e-6, "4 + 1 = 5")


def test_linspace_operations() raises:
    """Test operations with linspace-created tensors."""
    var a = linspace(0.0, 4.0, 5, DType.float32)  # [0, 1, 2, 3, 4]
    var b = linspace(5.0, 9.0, 5, DType.float32)  # [5, 6, 7, 8, 9]

    var result = add(a, b)  # [5, 7, 9, 11, 13]

    assert_value_at(result, 0, 5.0, 1e-6, "0 + 5 = 5")
    assert_value_at(result, 2, 9.0, 1e-6, "2 + 7 = 9")
    assert_value_at(result, 4, 13.0, 1e-6, "4 + 9 = 13")


def test_same_dtype_consistency() raises:
    """Test that operations preserve dtype consistently."""
    var shape = List[Int]()
    shape.append(5)

    var a32 = ones(shape, DType.float32)
    var b32 = ones(shape, DType.float32)
    var result32 = add(a32, b32)
    assert_dtype(result32, DType.float32, "float32 + float32 should be float32")

    var a64 = ones(shape, DType.float64)
    var b64 = ones(shape, DType.float64)
    var result64 = add(a64, b64)
    assert_dtype(result64, DType.float64, "float64 + float64 should be float64")


def test_int_dtype_operations() raises:
    """Test operations with integer dtypes."""
    var shape = List[Int]()
    shape.append(5)

    var a = full(shape, 3.0, DType.int32)
    var b = full(shape, 2.0, DType.int32)
    var result = add(a, b)

    assert_dtype(result, DType.int32, "int32 + int32 should be int32")
    assert_all_values(result, 5.0, 1e-6, "3 + 2 = 5")


def test_2d_elementwise_operations() raises:
    """Test element-wise operations on 2D tensors."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = subtract(a, b)  # All 2s

    assert_numel(result, 12, "Result should have 12 elements")
    assert_all_values(result, 2.0, 1e-6, "5 - 3 = 2 for all elements")


def test_3d_operations() raises:
    """Test operations on 3D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = full(shape, 0.5, DType.float32)

    var result = multiply(a, b)  # All 0.5s

    assert_numel(result, 24, "Result should have 24 elements")
    assert_all_values(result, 0.5, 1e-6, "1 * 0.5 = 0.5 for all elements")


def test_linear_transformation_pattern() raises:
    """Test pattern similar to linear layer: W*x + b."""
    var shape = List[Int]()
    shape.append(5)

    # Simulate weights, input, and bias
    var W = full(shape, 2.0, DType.float32)
    var x = ones(shape, DType.float32)
    var b = full(shape, 0.5, DType.float32)

    # Linear transformation: W*x + b
    var Wx = multiply(W, x)  # 2 * 1 = 2
    var result = add(Wx, b)  # 2 + 0.5 = 2.5

    assert_all_values(result, 2.5, 1e-6, "Linear transformation result")


def test_gradient_descent_update_pattern() raises:
    """Test pattern similar to gradient descent: w - lr * grad."""
    var shape = List[Int]()
    shape.append(5)

    var w = ones(shape, DType.float32)  # weights
    var grad = full(shape, 0.2, DType.float32)  # gradients
    var lr = full(shape, 0.1, DType.float32)  # learning rate

    # Update: w - lr * grad
    var lr_grad = multiply(lr, grad)  # 0.1 * 0.2 = 0.02
    var new_w = subtract(w, lr_grad)  # 1 - 0.02 = 0.98

    assert_all_values(new_w, 0.98, 1e-6, "Weight update pattern")


def test_batch_normalization_pattern() raises:
    """Test pattern similar to batch normalization: (x - mean) * scale."""
    var shape = List[Int]()
    shape.append(5)

    var x = full(shape, 5.0, DType.float32)
    var mean = full(shape, 3.0, DType.float32)
    var scale = full(shape, 2.0, DType.float32)

    # (x - mean) * scale
    var centered = subtract(x, mean)  # 5 - 3 = 2
    var result = multiply(centered, scale)  # 2 * 2 = 4

    assert_all_values(result, 4.0, 1e-6, "Batch norm pattern")


def test_additive_identity() raises:
    """Test that adding zero doesn't change values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 7.5, DType.float32)
    var zero = zeros(shape, DType.float32)

    var result = add(a, zero)

    assert_all_values(result, 7.5, 1e-6, "x + 0 = x")


def test_multiplicative_identity() raises:
    """Test that multiplying by one doesn't change values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 7.5, DType.float32)
    var one = ones(shape, DType.float32)

    var result = multiply(a, one)

    assert_all_values(result, 7.5, 1e-6, "x * 1 = x")


def test_multiplicative_zero() raises:
    """Test that multiplying by zero gives zero."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 99.9, DType.float32)
    var zero = zeros(shape, DType.float32)

    var result = multiply(a, zero)

    assert_all_values(result, 0.0, 1e-8, "x * 0 = 0")


def test_scalar_operations() raises:
    """Test operations with scalar tensors."""
    var shape_scalar = List[Int]()
    var a = full(shape_scalar, 5.0, DType.float32)
    var b = full(shape_scalar, 3.0, DType.float32)

    var result = add(a, b)

    assert_dim(result, 0, "Result should be scalar")
    assert_value_at(result, 0, 8.0, 1e-6, "5 + 3 = 8")


def test_large_tensor_operations() raises:
    """Test operations on large tensors."""
    var shape = List[Int]()
    shape.append(10000)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = multiply(a, b)

    assert_numel(result, 10000, "Result should have 10000 elements")
    # Spot check a few values
    assert_value_at(result, 0, 2.0, 1e-6, "First element")
    assert_value_at(result, 5000, 2.0, 1e-6, "Middle element")
    assert_value_at(result, 9999, 2.0, 1e-6, "Last element")


def main() raises:
    """Run all test_integration tests."""
    print("Running test_integration tests...")

    test_chained_add_operations()
    print("✓ test_chained_add_operations")

    test_mixed_arithmetic_operations()
    print("✓ test_mixed_arithmetic_operations")

    test_arithmetic_with_operator_overloading()
    print("✓ test_arithmetic_with_operator_overloading")

    test_complex_expression()
    print("✓ test_complex_expression")

    test_identity_matrix_operations()
    print("✓ test_identity_matrix_operations")

    test_arange_arithmetic()
    print("✓ test_arange_arithmetic")

    test_linspace_operations()
    print("✓ test_linspace_operations")

    test_same_dtype_consistency()
    print("✓ test_same_dtype_consistency")

    test_int_dtype_operations()
    print("✓ test_int_dtype_operations")

    test_2d_elementwise_operations()
    print("✓ test_2d_elementwise_operations")

    test_3d_operations()
    print("✓ test_3d_operations")

    test_linear_transformation_pattern()
    print("✓ test_linear_transformation_pattern")

    test_gradient_descent_update_pattern()
    print("✓ test_gradient_descent_update_pattern")

    test_batch_normalization_pattern()
    print("✓ test_batch_normalization_pattern")

    test_additive_identity()
    print("✓ test_additive_identity")

    test_multiplicative_identity()
    print("✓ test_multiplicative_identity")

    test_multiplicative_zero()
    print("✓ test_multiplicative_zero")

    test_scalar_operations()
    print("✓ test_scalar_operations")

    test_large_tensor_operations()
    print("✓ test_large_tensor_operations")

    print("\nAll test_integration tests passed!")
