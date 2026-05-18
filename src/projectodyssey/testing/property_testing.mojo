"""Property-Based Testing Framework for ML Odyssey.

Provides utilities for property-based testing, where instead of testing
specific examples, we test properties that should hold for all inputs.

Example:
    from projectodyssey.testing.property_testing import (
        random_shape,
        random_compatible_shape,
        run_property_test,
    )

    # Test reshape preserves element count
    def test_reshape_numel() raises:
        def property_fn() raises -> Bool:
            var shape = random_shape(max_dims=4, max_size=10)
            var tensor = random_tensor(shape, DType.float32)
            var new_shape = random_compatible_shape(tensor.numel())
            var reshaped = reshape(tensor, new_shape)
            return reshaped.numel() == tensor.numel()

        run_property_test(property_fn, num_tests=100)

Features:
- Random shape generation with configurable constraints
- Compatible shape generation for reshape testing
- Property test runner with failure shrinking
- Seed-based reproducibility for debugging
"""

from std.random import random_float64, seed
from std.math import sqrt
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
from projectodyssey.core.shape import reshape
from projectodyssey.testing.data_generators import random_tensor


# ============================================================================
# Random Shape Generators
# ============================================================================


def random_shape(
    max_dims: Int = 4,
    max_size: Int = 100,
    min_dims: Int = 1,
    min_size: Int = 1,
) raises -> List[Int]:
    """Generate a random tensor shape.

    Args:
        max_dims: Maximum number of dimensions (default: 4).
        max_size: Maximum size per dimension (default: 100).
        min_dims: Minimum number of dimensions (default: 1).
        min_size: Minimum size per dimension (default: 1).

    Returns:
        List of integers representing a valid tensor shape.

    Example:
        var shape = random_shape(max_dims=3, max_size=10)
        # Could return [3, 7, 2] or [5] or [4, 8] etc.
    """
    # Random number of dimensions
    var dim_range = max_dims - min_dims + 1
    var num_dims = (
        min_dims + Int(random_float64() * Float64(dim_range)) % dim_range
    )
    if num_dims > max_dims:
        num_dims = max_dims
    if num_dims < min_dims:
        num_dims = min_dims

    # Generate each dimension size
    var shape = List[Int]()
    var size_range = max_size - min_size + 1
    for _ in range(num_dims):
        var dim_size = (
            min_size + Int(random_float64() * Float64(size_range)) % size_range
        )
        if dim_size > max_size:
            dim_size = max_size
        if dim_size < min_size:
            dim_size = min_size
        shape.append(dim_size)

    return shape^


def random_compatible_shape(numel: Int, max_dims: Int = 4) raises -> List[Int]:
    """Generate a random shape with given total number of elements.

    Useful for testing reshape operations where output shape must
    be compatible with input element count.

    Args:
        numel: Total number of elements the shape must represent.
        max_dims: Maximum number of dimensions (default: 4).

    Returns:
        List of integers representing a valid shape with numel elements.

    Example:
        var shape = random_compatible_shape(24, max_dims=3)
        # Could return [24], [4, 6], [2, 3, 4], [2, 12] etc.
    """
    if numel <= 0:
        var shape = List[Int]()
        return shape^

    # Find factors of numel
    var factors = List[Int]()
    var n = numel
    var d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n = n // d
        d += 1
    if n > 1:
        factors.append(n)

    if len(factors) == 0:
        var shape = List[Int]()
        shape.append(1)
        return shape^

    # Randomly combine factors into dimensions
    var num_dims = 1 + Int(
        random_float64() * Float64(min(max_dims, len(factors)))
    )
    if num_dims > max_dims:
        num_dims = max_dims
    if num_dims < 1:
        num_dims = 1

    var shape = List[Int]()
    for _ in range(num_dims):
        shape.append(1)

    # Distribute factors across dimensions
    for i in range(len(factors)):
        var dim_idx = Int(random_float64() * Float64(num_dims)) % num_dims
        shape[dim_idx] = shape[dim_idx] * factors[i]

    return shape^


def random_broadcastable_shapes(
    max_dims: Int = 4, max_size: Int = 10
) raises -> Tuple[List[Int], List[Int]]:
    """Generate two broadcastable shapes.

    Useful for testing broadcasting operations like add, multiply.

    Args:
        max_dims: Maximum number of dimensions.
        max_size: Maximum size per dimension.

    Returns:
        Tuple of two shapes that are broadcast-compatible.

    Example:
        var (shape1, shape2) = random_broadcastable_shapes()
        # Could return ([3, 4], [4]) or ([2, 1, 5], [3, 5]) etc.
    """
    var shape1 = random_shape(max_dims, max_size)
    var shape2 = List[Int]()

    # Make shape2 broadcastable with shape1
    var start_dim = Int(random_float64() * Float64(len(shape1)))
    for i in range(start_dim, len(shape1)):
        # Either match the dimension or use 1
        if random_float64() > 0.3:
            shape2.append(shape1[i])
        else:
            shape2.append(1)

    if len(shape2) == 0:
        shape2.append(1)

    return (shape1^, shape2^)


# ============================================================================
# Property Test Runner
# ============================================================================


def run_property_test[
    PropFn: def() raises -> Bool
](
    property_fn: PropFn,
    num_tests: Int = 100,
    test_name: String = "property",
) raises:
    """Run a property test multiple times with random inputs.

    If the property fails, attempts to shrink to a minimal failing case.

    Args:
        property_fn: Function that returns True if property holds.
        num_tests: Number of random test cases to run.
        test_name: Name for error messages.

    Raises:
        Error: If property fails on any input.

    Example:
        ```mojo
        def property_fn() raises -> Bool:
            var a = random_tensor(random_shape(), DType.float32)
            var b = random_tensor(a.shape(), DType.float32)
            var sum1 = add(a, b)
            var sum2 = add(b, a)
            return tensors_equal(sum1, sum2)

        run_property_test(property_fn, num_tests=100, test_name="commutativity")
        ```
    """
    var failures = 0

    for i in range(num_tests):
        try:
            var result = property_fn()
            if not result:
                failures += 1
                raise Error(
                    test_name
                    + " failed on iteration "
                    + String(i)
                    + " (result was False)"
                )
        except e:
            failures += 1
            raise Error(
                test_name
                + " failed on iteration "
                + String(i)
                + ": "
                + String(e)
            )

    if failures > 0:
        raise Error(
            test_name
            + " failed "
            + String(failures)
            + " out of "
            + String(num_tests)
        )


def run_property_test_with_seed[
    PropFn: def() raises -> Bool
](
    property_fn: PropFn,
    num_tests: Int = 100,
    test_seed: Int = 42,
    test_name: String = "property",
) raises:
    """Run a property test with deterministic seed for reproducibility.

    Args:
        property_fn: Function that returns True if property holds.
        num_tests: Number of random test cases to run.
        test_seed: Random seed for reproducibility.
        test_name: Name for error messages.

    Raises:
        Error: If property fails, includes seed for reproduction.
    """
    seed(test_seed)

    try:
        run_property_test(property_fn, num_tests, test_name)
    except e:
        raise Error(String(e) + " (seed=" + String(test_seed) + ")")


# ============================================================================
# Common Property Assertions
# ============================================================================


def assert_tensors_close(
    a: AnyTensor, b: AnyTensor, atol: Float64 = 1e-6, rtol: Float64 = 1e-5
) raises:
    """Assert two tensors are element-wise close.

    Args:
        a: First tensor.
        b: Second tensor.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Raises:
        Error: If tensors have different shapes or values differ.
    """
    if a.numel() != b.numel():
        raise Error(
            "Tensor numel mismatch: "
            + String(a.numel())
            + " vs "
            + String(b.numel())
        )

    for i in range(a.numel()):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        var diff = val_a - val_b
        if diff < 0:
            diff = -diff

        var max_val = val_a if val_a > val_b else val_b
        if max_val < 0:
            max_val = -max_val

        if diff > atol + rtol * max_val:
            raise Error(
                "Tensor values differ at index "
                + String(i)
                + ": "
                + String(val_a)
                + " vs "
                + String(val_b)
                + " (diff="
                + String(diff)
                + ")"
            )


def tensors_equal(
    a: AnyTensor, b: AnyTensor, atol: Float64 = 1e-6
) raises -> Bool:
    """Check if two tensors are element-wise equal within tolerance.

    Args:
        a: First tensor.
        b: Second tensor.
        atol: Absolute tolerance.

    Returns:
        True if tensors are equal within tolerance.
    """
    if a.numel() != b.numel():
        return False

    for i in range(a.numel()):
        var val_a = a._get_float64(i)
        var val_b = b._get_float64(i)
        var diff = val_a - val_b
        if diff < 0:
            diff = -diff

        if diff > atol:
            return False

    return True


# ============================================================================
# Example Property Tests
# ============================================================================


def test_reshape_preserves_numel() raises:
    """Property: Reshape preserves element count."""

    def property_fn() raises -> Bool:
        var shape = random_shape(max_dims=3, max_size=10)
        var tensor = random_tensor(shape, DType.float32)
        var new_shape = random_compatible_shape(tensor.numel(), max_dims=3)
        var reshaped = reshape(tensor, new_shape)
        return reshaped.numel() == tensor.numel()

    run_property_test(
        property_fn, num_tests=50, test_name="reshape_preserves_numel"
    )


def test_addition_commutative() raises:
    """Property: Addition is commutative (a + b == b + a)."""
    from projectodyssey.core.arithmetic import add

    def property_fn() raises -> Bool:
        var shape = random_shape(max_dims=2, max_size=5)
        var a = random_tensor(shape, DType.float32)
        var b = random_tensor(shape, DType.float32)

        var sum1 = add(a, b)
        var sum2 = add(b, a)

        return tensors_equal(sum1, sum2, atol=1e-5)

    run_property_test(
        property_fn, num_tests=50, test_name="addition_commutative"
    )


def test_multiplication_commutative() raises:
    """Property: Multiplication is commutative (a * b == b * a)."""
    from projectodyssey.core.arithmetic import multiply

    def property_fn() raises -> Bool:
        var shape = random_shape(max_dims=2, max_size=5)
        var a = random_tensor(shape, DType.float32)
        var b = random_tensor(shape, DType.float32)

        var prod1 = multiply(a, b)
        var prod2 = multiply(b, a)

        return tensors_equal(prod1, prod2, atol=1e-5)

    run_property_test(
        property_fn, num_tests=50, test_name="multiplication_commutative"
    )


def test_addition_identity() raises:
    """Property: Adding zero is identity (a + 0 == a)."""
    from projectodyssey.core.arithmetic import add

    def property_fn() raises -> Bool:
        var shape = random_shape(max_dims=2, max_size=5)
        var a = random_tensor(shape, DType.float32)
        var zero = zeros(shape, DType.float32)

        var result = add(a, zero)

        return tensors_equal(result, a, atol=1e-6)

    run_property_test(property_fn, num_tests=50, test_name="addition_identity")


def test_multiplication_identity() raises:
    """Property: Multiplying by one is identity (a * 1 == a)."""
    from projectodyssey.core.arithmetic import multiply

    def property_fn() raises -> Bool:
        var shape = random_shape(max_dims=2, max_size=5)
        var a = random_tensor(shape, DType.float32)
        var one = ones(shape, DType.float32)

        var result = multiply(a, one)

        return tensors_equal(result, a, atol=1e-6)

    run_property_test(
        property_fn, num_tests=50, test_name="multiplication_identity"
    )
