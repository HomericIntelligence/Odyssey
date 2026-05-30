"""Test AnyTensor initialization with various large shapes.

This test focuses on line 107 in any_tensor.mojo which is where the crash occurs.
Line 107: self._strides.append(0)  # Preallocate

The crash happens during stride calculation in AnyTensor.__init__.
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from std.testing import assert_equal, assert_true


def test_anytensor_init_simple() raises:
    """Test basic AnyTensor initialization."""
    print("\n=== Test 1: Basic AnyTensor Initialization ===")

    # Small tensor
    var shape1 = List[Int]()
    shape1.append(2)
    shape1.append(3)

    print("Creating tensor shape (2, 3)...")
    var t1 = AnyTensor(shape1, DType.float32)
    print("SUCCESS: numel =", t1.numel(), "dim =", t1.dim())

    # Check strides (need explicit copy since List[Int] isn't ImplicitlyCopyable)
    var strides = t1._strides.copy()
    print("Strides:", strides[0], strides[1])
    assert_equal(strides[0], 3)  # Row-major: 3 elements per row
    assert_equal(strides[1], 1)


def test_anytensor_init_large_shapes() raises:
    """Test AnyTensor with progressively larger shapes."""
    print("\n=== Test 2: Large Shape Initialization ===")

    # Test case 1: (10, 10)
    print("\nTesting shape 1: (10, 10)")
    var shape1 = List[Int]()
    shape1.append(10)
    shape1.append(10)
    test_shape_creation(shape1)

    # Test case 2: (100, 100)
    print("\nTesting shape 2: (100, 100)")
    var shape2 = List[Int]()
    shape2.append(100)
    shape2.append(100)
    test_shape_creation(shape2)

    # Test case 3: (1000, 100)
    print("\nTesting shape 3: (1000, 100)")
    var shape3 = List[Int]()
    shape3.append(1000)
    shape3.append(100)
    test_shape_creation(shape3)

    # Test case 4: (2, 47) - Exact crash case
    print("\nTesting shape 4: (2, 47) - CRASH CASE")
    var shape4 = List[Int]()
    shape4.append(2)
    shape4.append(47)
    test_shape_creation(shape4)


def test_shape_creation(shape: List[Int]) raises:
    """Helper to test tensor creation for a shape."""
    var numel = 1
    for j in range(len(shape)):
        numel *= shape[j]

    print("  numel =", numel)
    try:
        var t = AnyTensor(shape, DType.float32)
        print("  SUCCESS: Created tensor with", t.dim(), "dimensions")

        # Verify strides were calculated correctly
        var expected_stride = 1
        for j in range(len(shape) - 1, -1, -1):
            var actual_stride = t._strides[j]
            print(
                "    Dimension",
                j,
                "stride:",
                actual_stride,
                "expected:",
                expected_stride,
            )
            assert_equal(actual_stride, expected_stride)
            expected_stride *= shape[j]

    except e:
        print("  CRASH during initialization:", e)
        raise e^


def test_anytensor_init_multidimensional() raises:
    """Test AnyTensor with various dimensionalities."""
    print("\n=== Test 3: Multi-Dimensional Tensors ===")

    # 1D
    print("\n1D tensor (100)...")
    var shape1 = List[Int]()
    shape1.append(100)
    var t1 = AnyTensor(shape1, DType.float32)
    print("  SUCCESS: strides =", t1._strides[0])

    # 2D
    print("\n2D tensor (10, 20)...")
    var shape2 = List[Int]()
    shape2.append(10)
    shape2.append(20)
    var t2 = AnyTensor(shape2, DType.float32)
    print("  SUCCESS: strides =", t2._strides[0], t2._strides[1])

    # 3D
    print("\n3D tensor (5, 10, 15)...")
    var shape3 = List[Int]()
    shape3.append(5)
    shape3.append(10)
    shape3.append(15)
    var t3 = AnyTensor(shape3, DType.float32)
    print(
        "  SUCCESS: strides =", t3._strides[0], t3._strides[1], t3._strides[2]
    )

    # 4D (like MNIST batch)
    print("\n4D tensor (2, 1, 28, 28)...")
    var shape4 = List[Int]()
    shape4.append(2)
    shape4.append(1)
    shape4.append(28)
    shape4.append(28)
    var t4 = AnyTensor(shape4, DType.float32)
    print(
        "  SUCCESS: strides =",
        t4._strides[0],
        t4._strides[1],
        t4._strides[2],
        t4._strides[3],
    )


def test_anytensor_stride_preallocate() raises:
    """Test the exact stride preallocation pattern that crashes.

    This directly tests lines 104-110 in any_tensor.mojo:
        self._strides = List[Int]()
        var stride = 1
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides.append(0)  # Preallocate <- LINE 107 (CRASH)
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides[i] = stride
            stride *= self._shape[i]
    """
    print("\n=== Test 4: Stride Preallocation Pattern ===")

    var shapes = List[List[Int]]()

    # Various shapes to test the loop
    var s1 = List[Int]()
    s1.append(2)
    shapes.append(s1.copy())

    var s2 = List[Int]()
    s2.append(2)
    s2.append(3)
    shapes.append(s2.copy())

    var s3 = List[Int]()
    s3.append(2)
    s3.append(3)
    s3.append(4)
    shapes.append(s3.copy())

    var s4 = List[Int]()
    s4.append(2)
    s4.append(47)  # Exact crash case
    shapes.append(s4.copy())

    for i in range(len(shapes)):
        var shape = shapes[i].copy()
        print(
            "\nTesting stride preallocation for shape with",
            len(shape),
            "dimensions...",
        )

        # Manually simulate the stride calculation
        var strides_test = List[Int]()
        print("  Step 1: Preallocate strides...")
        for j in range(len(shape) - 1, -1, -1):
            print("    Appending 0 for dimension", j)
            strides_test.append(0)  # This is line 107

        print("  Step 2: Calculate actual strides...")
        var stride = 1
        for j in range(len(shape) - 1, -1, -1):
            print("    Setting stride[", j, "] =", stride)
            strides_test[j] = stride
            stride *= shape[j]

        print("  SUCCESS: Strides calculated:", end=" ")
        for j in range(len(strides_test)):
            print(strides_test[j], end=" ")
        print()

        # Now test actual AnyTensor creation
        print("  Creating AnyTensor with same shape...")
        var t = AnyTensor(shape, DType.float32)
        print("  SUCCESS")


def test_memory_limits() raises:
    """Test tensors near memory limits."""
    print("\n=== Test 5: Memory Limit Validation ===")

    # Small tensor well within limits
    print("\nSmall tensor (100 bytes)...")
    var shape1 = List[Int]()
    shape1.append(25)  # 25 * 4 bytes = 100 bytes
    var t1 = AnyTensor(shape1, DType.float32)
    print("  SUCCESS:", t1.numel() * 4, "bytes")

    # Medium tensor (1 MB)
    print("\nMedium tensor (~1 MB)...")
    var shape2 = List[Int]()
    shape2.append(250000)  # 250k * 4 bytes = 1 MB
    var t2 = AnyTensor(shape2, DType.float32)
    print("  SUCCESS:", t2.numel() * 4, "bytes")

    # Large tensor (100 MB)
    print("\nLarge tensor (~100 MB)...")
    var shape3 = List[Int]()
    shape3.append(25000000)  # 25M * 4 bytes = 100 MB
    var t3 = AnyTensor(shape3, DType.float32)
    print("  SUCCESS:", t3.numel() * 4, "bytes")

    # Very large tensor (500 MB - warning threshold)
    print("\nVery large tensor (~500 MB) - should warn...")
    var shape4 = List[Int]()
    shape4.append(125000000)  # 125M * 4 bytes = 500 MB
    var t4 = AnyTensor(shape4, DType.float32)
    print("  SUCCESS:", t4.numel() * 4, "bytes")

    # Exceeds limit (should raise error)
    print("\nExceeds limit (>2 GB) - should raise error...")
    try:
        var shape5 = List[Int]()
        shape5.append(
            1000000000
        )  # 1B * 4 bytes = 4 GB (exceeds MAX_TENSOR_BYTES)
        var t5 = AnyTensor(shape5, DType.float32)
        print("  ERROR: Should have raised exception!")
    except e:
        print("  SUCCESS: Caught expected error:", e)


def main() raises:
    """Run all AnyTensor initialization tests."""
    print("=" * 60)
    print("ANYTENSOR INITIALIZATION TESTS")
    print("=" * 60)

    test_anytensor_init_simple()
    test_anytensor_init_large_shapes()
    test_anytensor_init_multidimensional()
    test_anytensor_stride_preallocate()
    test_memory_limits()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
