"""Matrix multiplication tests

Tests cover:
- Correctness: Baseline (Stage 0) tests
- Edge Cases: Small sizes, trivial cases
"""


from tests.shared.conftest import (
    assert_all_close,
    assert_all_values,
    assert_almost_equal,
    assert_close_float,
    assert_dim,
    assert_dtype,
    assert_equal,
    assert_equal_int,
    assert_matrices_equal,
    assert_numel,
    assert_shape,
    assert_true,
    assert_value_at,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    full,
    arange,
    eye,
)
from shared.core.matrix import matmul
from tests.shared.conftest import (
    assert_all_close,
    assert_all_values,
    assert_almost_equal,
    assert_close_float,
    assert_dim,
    assert_dtype,
    assert_equal,
    assert_equal_int,
    assert_numel,
    assert_shape,
    assert_true,
    assert_value_at,
)


fn test_matmul_baseline_2x2() raises:
    """Test baseline matmul with simple 2x2 matrices (reference test)."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(2)
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(2)

    var a = zeros(shape_a, DType.float32)
    var b = zeros(shape_b, DType.float32)

    # A = [[1, 2], [3, 4]]
    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0
    a._data.bitcast[Float32]()[3] = 4.0

    # B = [[5, 6], [7, 8]]
    b._data.bitcast[Float32]()[0] = 5.0
    b._data.bitcast[Float32]()[1] = 6.0
    b._data.bitcast[Float32]()[2] = 7.0
    b._data.bitcast[Float32]()[3] = 8.0

    var result = matmul(a, b)

    # Result = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    #        = [[19, 22], [43, 50]]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(19.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(22.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(43.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(50.0), tolerance=1e-5
    )


fn test_matmul_baseline_identity() raises:
    """Test baseline matmul with identity matrix."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    var a = zeros(shape, DType.float32)
    var identity = zeros(shape, DType.float32)

    # A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(9):
        a._data.bitcast[Float32]()[i] = Float32(i + 1)

    # Identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    identity._data.bitcast[Float32]()[0] = 1.0  # (0, 0)
    identity._data.bitcast[Float32]()[4] = 1.0  # (1, 1)
    identity._data.bitcast[Float32]()[8] = 1.0  # (2, 2)

    var result = matmul(a, identity)

    # A @ I = A
    for i in range(9):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            a._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


fn test_matmul_baseline_zero_matrix() raises:
    """Test baseline matmul with zero matrix."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    var a = zeros(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should be 3x3")
    assert_all_values(c, 0.0, 1e-6, "Zero matrix @ anything = zero matrix")


fn test_matmul_trivial_1x1() raises:
    """Test 1x1 matrix multiplication (trivial case)."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(1)

    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 4.0, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 1, "Result should be 1x1 (1 element)")
    assert_value_at(c, 0, 12.0, 1e-6, "1x1 @ 1x1 = 3*4")


fn test_matmul_vector_matrix_1x64x1() raises:
    """Test vector-matrix multiplication (1x64) @ (64x1)."""
    var shape_a = List[Int]()
    shape_a.append(1)
    shape_a.append(64)

    var shape_b = List[Int]()
    shape_b.append(64)
    shape_b.append(1)

    var a = ones(shape_a, DType.float32)
    var b = full(shape_b, 2.0, DType.float32)
    var c = matmul(a, b)

    # Result: 1x1 matrix with value 64*2 = 128
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 1, "Result should be 1x1")
    assert_value_at(c, 0, 128.0, 1e-5, "Sum of 64 products of 1*2")


fn test_matmul_matrix_vector_64x1x64() raises:
    """Test matrix-vector multiplication (64x1) @ (1x64)."""
    var shape_a = List[Int]()
    shape_a.append(64)
    shape_a.append(1)

    var shape_b = List[Int]()
    shape_b.append(1)
    shape_b.append(64)

    var a = full(shape_a, 2.0, DType.float32)
    var b = ones(shape_b, DType.float32)
    var c = matmul(a, b)

    # Result: 64x64 matrix with all elements = 2*1 = 2
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4096, "Result should be 64x64")
    assert_all_values(c, 2.0, 1e-6, "All elements should be 2.0")


fn test_matmul_smaller_than_simd_7x7x7() raises:
    """Test matrices smaller than SIMD width (7x7)."""
    var shape = List[Int]()
    shape.append(7)
    shape.append(7)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = matmul(a, b)

    # Each element = 1*2 + ... (7 times) = 14
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 49, "Result should be 7x7")
    assert_all_values(c, 14.0, 1e-5, "Each element should be 14")


fn test_matmul_non_power_of_2() raises:
    """Test non-power-of-2 sizes (63x65) @ (65x67)."""
    var shape_a = List[Int]()
    shape_a.append(63)
    shape_a.append(65)

    var shape_b = List[Int]()
    shape_b.append(65)
    shape_b.append(67)

    var a = ones(shape_a, DType.float32)
    var b = full(shape_b, 0.5, DType.float32)
    var c = matmul(a, b)

    # Each element = 1*0.5 + ... (65 times) = 32.5
    assert_dim(c, 2, "Result should be 2D")
    assert_equal(c.shape()[0], 63, "First dimension should be 63")
    assert_equal(c.shape()[1], 67, "Second dimension should be 67")
    assert_numel(c, 4221, "Result should be 63x67")

    # Check first and last elements
    assert_value_at(c, 0, 32.5, 1e-4, "First element should be 32.5")
    assert_value_at(c, 4220, 32.5, 1e-4, "Last element should be 32.5")


fn test_matmul_exact_block_size_64x64x64() raises:
    """Test matrices matching typical cache block size (64x64)."""
    var shape = List[Int]()
    shape.append(64)
    shape.append(64)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = matmul(a, b)

    # Each element = 1*2 + ... (64 times) = 128
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4096, "Result should be 64x64")
    assert_all_values(c, 128.0, 1e-4, "Each element should be 128")


fn test_matmul_large_rectangular() raises:
    """Test large rectangular matrices (1024x512) @ (512x2048)."""
    var shape_a = List[Int]()
    shape_a.append(1024)
    shape_a.append(512)

    var shape_b = List[Int]()
    shape_b.append(512)
    shape_b.append(2048)

    var a = full(shape_a, 0.1, DType.float32)
    var b = full(shape_b, 0.2, DType.float32)
    var c = matmul(a, b)

    # Each element = 0.1*0.2 + ... (512 times) = 10.24
    assert_dim(c, 2, "Result should be 2D")
    assert_equal(c.shape()[0], 1024, "First dimension should be 1024")
    assert_equal(c.shape()[1], 2048, "Second dimension should be 2048")

    # Check spot values (avoid checking all 2M+ elements)
    assert_value_at(c, 0, 10.24, 1e-3, "First element should be 10.24")
    assert_value_at(c, 2048, 10.24, 1e-3, "Second row first element")
    # Note: Using lower tolerance for accumulated floating-point errors


fn test_matmul_dtype_float32() raises:
    """Test matmul with Float32 dtype."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = matmul(a, b)

    assert_dtype(c, DType.float32, "Result should be Float32")
    assert_all_values(c, 8.0, 1e-5, "Each element should be 8.0")


fn test_matmul_dtype_float64() raises:
    """Test matmul with Float64 dtype."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)

    var a = ones(shape, DType.float64)
    var b = full(shape, 2.0, DType.float64)
    var c = matmul(a, b)

    assert_dtype(c, DType.float64, "Result should be Float64")
    assert_all_values(c, 8.0, 1e-8, "Each element should be 8.0")


fn test_matmul_dtype_float16() raises:
    """Test matmul with Float16 dtype."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)

    var a = ones(shape, DType.float16)
    var b = full(shape, 2.0, DType.float16)
    var c = matmul(a, b)

    assert_dtype(c, DType.float16, "Result should be Float16")
    # Note: Float16 has lower precision, use looser tolerance
    assert_all_values(c, 8.0, 1e-2, "Each element should be ~8.0")


fn test_matmul_dtype_preserves_type() raises:
    """Test that matmul preserves input dtype across all types."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    # Float32
    var a32 = ones(shape, DType.float32)
    var b32 = ones(shape, DType.float32)
    var c32 = matmul(a32, b32)
    assert_dtype(c32, DType.float32, "Float32 should be preserved")

    # Float64
    var a64 = ones(shape, DType.float64)
    var b64 = ones(shape, DType.float64)
    var c64 = matmul(a64, b64)
    assert_dtype(c64, DType.float64, "Float64 should be preserved")

    # Float16
    var a16 = ones(shape, DType.float16)
    var b16 = ones(shape, DType.float16)
    var c16 = matmul(a16, b16)
    assert_dtype(c16, DType.float16, "Float16 should be preserved")


fn test_matmul_dtype_mismatch_error() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)  # Different dtype

    var error_raised = False
    try:
        var c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in matmul")


fn test_matmul_incompatible_shapes() raises:
    """Test that incompatible shapes raise error."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)

    var shape_b = List[Int]()
    shape_b.append(5)
    shape_b.append(2)  # Incompatible: 4 != 5

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for incompatible matmul shapes (3,4) @"
            " (5,2)"
        )


fn test_matmul_1d_error() raises:
    """Test that 1D inputs raise error."""
    var shape = List[Int]()
    shape.append(5)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var error_raised = False
    try:
        var c = matmul(a, b)  # matmul requires 2D+
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for 1D inputs to matmul")


fn test_matmul_additional_rectangular_sizes() raises:
    """Test rectangular matrices with various dimensions."""
    # Test case 1: Wide matrix (M < K < N)
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(8)
    var shape_b = List[Int]()
    shape_b.append(8)
    shape_b.append(16)

    var a = ones(shape_a, DType.float32)
    var b = full(shape_b, 2.0, DType.float32)
    var c = matmul(a, b)

    var expected_shape = List[Int]()
    expected_shape.append(4)
    expected_shape.append(16)
    assert_shape(c, expected_shape, "Result shape should be (4, 16)")
    assert_all_values(c, 16.0, 1e-5, "Each element should be 8*2 = 16")


fn test_matmul_accumulation_precision_float32() raises:
    """Test accumulation precision with many terms (float32)."""
    # Test with 128 terms to check accumulation precision
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(128)
    var shape_b = List[Int]()
    shape_b.append(128)
    shape_b.append(4)

    var a = full(shape_a, 0.1, DType.float32)
    var b = full(shape_b, 0.1, DType.float32)
    var c = matmul(a, b)

    # Expected: 128 * (0.1 * 0.1) = 128 * 0.01 = 1.28
    assert_value_at(c, 0, 1.28, 1e-5, "Accumulated result should match")


fn test_matmul_accumulation_precision_float64() raises:
    """Test accumulation precision with many terms (float64)."""
    # Test with 256 terms for float64
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(256)
    var shape_b = List[Int]()
    shape_b.append(256)
    shape_b.append(4)

    var a = full(shape_a, 0.1, DType.float64)
    var b = full(shape_b, 0.1, DType.float64)
    var c = matmul(a, b)

    # Expected: 256 * 0.01 = 2.56
    assert_value_at(c, 0, 2.56, 1e-8, "Float64 accumulation should be precise")


fn main() raises:
    """Run all test_matmul tests."""
    print("Running test_matmul tests...")

    test_matmul_baseline_2x2()
    print("✓ test_matmul_baseline_2x2")

    test_matmul_baseline_identity()
    print("✓ test_matmul_baseline_identity")

    test_matmul_baseline_zero_matrix()
    print("✓ test_matmul_baseline_zero_matrix")

    test_matmul_trivial_1x1()
    print("✓ test_matmul_trivial_1x1")

    test_matmul_vector_matrix_1x64x1()
    print("✓ test_matmul_vector_matrix_1x64x1")

    test_matmul_matrix_vector_64x1x64()
    print("✓ test_matmul_matrix_vector_64x1x64")

    test_matmul_smaller_than_simd_7x7x7()
    print("✓ test_matmul_smaller_than_simd_7x7x7")

    test_matmul_non_power_of_2()
    print("✓ test_matmul_non_power_of_2")

    test_matmul_exact_block_size_64x64x64()
    print("✓ test_matmul_exact_block_size_64x64x64")

    test_matmul_large_rectangular()
    print("✓ test_matmul_large_rectangular")

    test_matmul_dtype_float32()
    print("✓ test_matmul_dtype_float32")

    test_matmul_dtype_float64()
    print("✓ test_matmul_dtype_float64")

    test_matmul_dtype_float16()
    print("✓ test_matmul_dtype_float16")

    test_matmul_dtype_preserves_type()
    print("✓ test_matmul_dtype_preserves_type")

    test_matmul_dtype_mismatch_error()
    print("✓ test_matmul_dtype_mismatch_error")

    test_matmul_incompatible_shapes()
    print("✓ test_matmul_incompatible_shapes")

    test_matmul_1d_error()
    print("✓ test_matmul_1d_error")

    test_matmul_additional_rectangular_sizes()
    print("✓ test_matmul_additional_rectangular_sizes")

    test_matmul_accumulation_precision_float32()
    print("✓ test_matmul_accumulation_precision_float32")

    test_matmul_accumulation_precision_float64()
    print("✓ test_matmul_accumulation_precision_float64")

    print("\nAll test_matmul tests passed!")
