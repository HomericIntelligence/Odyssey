"""Shape Fuzzing Utilities for ML Odyssey.

Provides utilities for generating random and edge case tensor shapes for
property-based testing. Enables discovering shape-related bugs in tensor
operations through systematic shape exploration.

Key Features:
    - Random shape generation with configurable bounds
    - Edge case shapes (empty, scalar, 1D, high-dimensional)
    - Broadcast-compatible shape pair generation
    - Shape validation utilities

Shape Categories:
    1. Scalar (0D): [] - single value
    2. Empty: [0], [3, 0], [0, 5] - zero elements
    3. Small: [1], [2, 3] - quick tests
    4. Medium: [10, 20], [5, 10, 15] - realistic tests
    5. Large: [100, 100], [50, 50, 50] - stress tests
    6. Edge: [1, 1, 1, 1], [1000000] - boundary cases

Usage:
    from projectodyssey.testing.fuzz_shapes import (
        ShapeFuzzer,
        generate_random_shape,
        generate_edge_case_shapes,
        generate_broadcast_shapes,
    )

    # Generate random shapes
    var fuzzer = ShapeFuzzer(seed=42)
    var shape = fuzzer.random_shape(max_ndim=4, max_dim=100)

    # Generate broadcast-compatible pair
    var (shape_a, shape_b) = fuzzer.broadcast_compatible_pair()

Example:
    ```mojo
    var fuzzer = ShapeFuzzer(seed=42)

    # Test with various shapes
    for _ in range(100):
        var shape = fuzzer.random_shape()
        var tensor = zeros(shape, DType.float32)
        # Test operations...
    ```
"""

from std.random import random_float64, seed as random_seed
from projectodyssey.testing.fuzz_core import SeededRNG


# ============================================================================
# Shape Fuzzer
# ============================================================================


struct ShapeFuzzer(Copyable, Movable):
    """Generator for random and edge case tensor shapes.

    Provides methods for generating various types of shapes useful for
    testing tensor operations.

    Attributes:
        rng: Seeded random number generator.
        max_dim: Default maximum dimension size.
        max_ndim: Default maximum number of dimensions.
        max_numel: Maximum total elements to avoid memory issues.

    Example:
        ```mojo
        var fuzzer = ShapeFuzzer(seed=42)
        var shape = fuzzer.random_shape()
        var edge_shapes = fuzzer.edge_case_shapes()
        ```
    """

    var rng: SeededRNG
    var max_dim: Int
    var max_ndim: Int
    var max_numel: Int

    def __init__(
        out self,
        seed: Int = 42,
        max_dim: Int = 100,
        max_ndim: Int = 6,
        max_numel: Int = 10000000,
    ):
        """Initialize shape fuzzer.

        Args:
            seed: Random seed for reproducibility.
            max_dim: Maximum size for any dimension.
            max_ndim: Maximum number of dimensions.
            max_numel: Maximum total elements (to avoid memory issues).
        """
        self.rng = SeededRNG(seed)
        self.max_dim = max_dim
        self.max_ndim = max_ndim
        self.max_numel = max_numel

    def next_iteration(mut self):
        """Advance to next iteration for fresh randomness."""
        self.rng.next_iteration()

    def random_shape(
        mut self,
        min_ndim: Int = 0,
        max_ndim: Int = -1,
        min_dim: Int = 1,
        max_dim: Int = -1,
    ) -> List[Int]:
        """Generate a random shape.

        Args:
            min_ndim: Minimum number of dimensions (default: 0 for scalar).
            max_ndim: Maximum number of dimensions (default: use self.max_ndim).
            min_dim: Minimum size for each dimension (default: 1).
            max_dim: Maximum size for each dimension (default: use self.max_dim).

        Returns:
            Random shape as List[Int].

        Example:
            ```mojo
            var shape = fuzzer.random_shape(min_ndim=1, max_ndim=4)
            # Might return [3, 5, 2] or [10] etc.
            ```
        """
        var actual_max_ndim = max_ndim if max_ndim > 0 else self.max_ndim
        var actual_max_dim = max_dim if max_dim > 0 else self.max_dim

        # Random number of dimensions
        var ndim = self.rng.random_int(min_ndim, actual_max_ndim + 1)

        var shape = List[Int]()
        var numel = 1

        for _ in range(ndim):
            # Calculate max allowed dim to stay under max_numel
            var allowed_dim = actual_max_dim
            if numel > 0 and self.max_numel > 0:
                allowed_dim = min(allowed_dim, self.max_numel // numel)
            allowed_dim = max(allowed_dim, min_dim)

            var dim_size = self.rng.random_int(min_dim, allowed_dim + 1)
            shape.append(dim_size)
            numel *= dim_size

        return shape^

    def scalar_shape(self) -> List[Int]:
        """Generate a 0D scalar shape.

        Returns:
            Empty list representing scalar shape.

        Example:
            ```mojo
            var shape = fuzzer.scalar_shape()  # Returns []
            ```
        """
        return List[Int]()

    def empty_shape_1d(mut self) -> List[Int]:
        """Generate a 1D empty shape [0].

        Returns:
            Shape [0] representing empty 1D tensor.

        Example:
            ```mojo
            var shape = fuzzer.empty_shape_1d()  # Returns [0]
            ```
        """
        var shape = List[Int]()
        shape.append(0)
        return shape^

    def empty_shape_nd(mut self, ndim: Int = -1) -> List[Int]:
        """Generate an N-D shape with one zero dimension.

        Args:
            ndim: Number of dimensions (default: random 1-4).

        Returns:
            Shape with at least one zero dimension.

        Example:
            ```mojo
            var shape = fuzzer.empty_shape_nd(3)  # Might return [3, 0, 5]
            ```
        """
        var actual_ndim = ndim
        if actual_ndim < 1:
            actual_ndim = self.rng.random_int(1, 5)

        var shape = List[Int]()
        var zero_pos = self.rng.random_int(0, actual_ndim)

        for i in range(actual_ndim):
            if i == zero_pos:
                shape.append(0)
            else:
                shape.append(self.rng.random_int(1, 10))

        return shape^

    def single_element_shape(mut self, ndim: Int = -1) -> List[Int]:
        """Generate a shape with only one element.

        Args:
            ndim: Number of dimensions (default: random 1-4).

        Returns:
            Shape where all dimensions are 1.

        Example:
            ```mojo
            var shape = fuzzer.single_element_shape(3)  # Returns [1, 1, 1]
            ```
        """
        var actual_ndim = ndim
        if actual_ndim < 0:
            actual_ndim = self.rng.random_int(1, 5)

        var shape = List[Int]()
        for _ in range(actual_ndim):
            shape.append(1)
        return shape^

    def large_1d_shape(
        mut self, min_size: Int = 10000, max_size: Int = 100000
    ) -> List[Int]:
        """Generate a large 1D shape.

        Args:
            min_size: Minimum number of elements.
            max_size: Maximum number of elements.

        Returns:
            1D shape with many elements.

        Example:
            ```mojo
            var shape = fuzzer.large_1d_shape()  # Might return [50000]
            ```
        """
        var size = self.rng.random_int(min_size, max_size + 1)
        size = min(size, self.max_numel)

        var shape = List[Int]()
        shape.append(size)
        return shape^

    def high_dimensional_shape(
        mut self, min_ndim: Int = 5, max_ndim: Int = 10, dim_size: Int = 2
    ) -> List[Int]:
        """Generate a high-dimensional shape with small dimensions.

        Args:
            min_ndim: Minimum number of dimensions.
            max_ndim: Maximum number of dimensions.
            dim_size: Size of each dimension (small to avoid memory issues).

        Returns:
            High-dimensional shape like [2, 2, 2, 2, 2, 2].

        Example:
            ```mojo
            var shape = fuzzer.high_dimensional_shape(6, 8, 2)
            # Might return [2, 2, 2, 2, 2, 2] (6D)
            ```
        """
        var ndim = self.rng.random_int(min_ndim, max_ndim + 1)

        # Ensure we don't exceed max_numel
        var max_allowed_dims = 1
        var temp_numel = 1
        while (
            temp_numel * dim_size <= self.max_numel and max_allowed_dims < ndim
        ):
            temp_numel *= dim_size
            max_allowed_dims += 1
        ndim = min(ndim, max_allowed_dims)

        var shape = List[Int]()
        for _ in range(ndim):
            shape.append(dim_size)
        return shape^

    def edge_case_shapes(mut self) -> List[List[Int]]:
        """Generate a collection of edge case shapes.

        Returns:
            List of shapes covering various edge cases:
            - Scalar (0D)
            - Empty (1D and 2D)
            - Single element
            - 1D various sizes
            - High dimensional
            - Large 1D

        Example:
            ```mojo
            var shapes = fuzzer.edge_case_shapes()
            for shape in shapes:
                var tensor = zeros(shape, DType.float32)
                # Test with each edge case...
            ```
        """
        var shapes = List[List[Int]]()

        # 0D scalar
        shapes.append(self.scalar_shape())

        # Empty shapes
        shapes.append(self.empty_shape_1d())
        shapes.append(self.empty_shape_nd(2))

        # Single element
        var single1 = List[Int]()
        single1.append(1)
        shapes.append(single1^)

        var single2 = List[Int]()
        single2.append(1)
        single2.append(1)
        shapes.append(single2^)

        # Simple 1D
        var simple1d = List[Int]()
        simple1d.append(5)
        shapes.append(simple1d^)

        # 2D square
        var square = List[Int]()
        square.append(3)
        square.append(3)
        shapes.append(square^)

        # 2D rectangular
        var rect = List[Int]()
        rect.append(2)
        rect.append(5)
        shapes.append(rect^)

        # 3D
        var threed = List[Int]()
        threed.append(2)
        threed.append(3)
        threed.append(4)
        shapes.append(threed^)

        # High dimensional (small dims)
        shapes.append(self.high_dimensional_shape(6, 6, 2))

        # Large 1D (moderate size for testing)
        var large = List[Int]()
        large.append(10000)
        shapes.append(large^)

        return shapes^


# ============================================================================
# Broadcast-Compatible Shape Generation
# ============================================================================


def generate_broadcast_shapes(
    mut fuzzer: ShapeFuzzer,
) -> Tuple[List[Int], List[Int]]:
    """Generate a pair of broadcast-compatible shapes.

    Creates two shapes that can be broadcast together according to
    NumPy broadcasting rules.

    Args:
        fuzzer: Shape fuzzer for random generation.

    Returns:
        Tuple of two broadcast-compatible shapes.

    Example:
        ```mojo
        var (shape_a, shape_b) = generate_broadcast_shapes(fuzzer)
        var a = ones(shape_a, DType.float32)
        var b = ones(shape_b, DType.float32)
        var c = add(a, b)  # Should work without error
        ```

    Broadcasting Rules:
        - Shapes are compared from trailing dimensions
        - Dimensions are compatible if equal or one is 1
        - Smaller shape is padded with 1s on the left.
    """
    # Generate base shape
    var base_shape = fuzzer.random_shape(
        min_ndim=1, max_ndim=4, min_dim=1, max_dim=10
    )

    # Create compatible shape by randomly:
    # 1. Keeping some dims the same
    # 2. Changing some dims to 1
    # 3. Possibly adding/removing leading dims
    var compat_shape = List[Int]()

    # Randomly decide if we add leading 1s
    var add_leading = fuzzer.rng.random_bool()
    if add_leading:
        var num_leading = fuzzer.rng.random_int(1, 3)
        for _ in range(num_leading):
            compat_shape.append(1)

    # Process each dimension
    for i in range(len(base_shape)):
        var choice = fuzzer.rng.random_int(0, 3)
        if choice == 0:
            # Keep same dimension
            compat_shape.append(base_shape[i])
        elif choice == 1:
            # Change to 1 (broadcastable)
            compat_shape.append(1)
        else:
            # Keep same dimension
            compat_shape.append(base_shape[i])

    return (base_shape^, compat_shape^)


def generate_matmul_shapes(
    mut fuzzer: ShapeFuzzer,
) -> Tuple[List[Int], List[Int]]:
    """Generate a pair of shapes compatible for matrix multiplication.

    Creates two 2D shapes where the inner dimensions match.

    Args:
        fuzzer: Shape fuzzer for random generation.

    Returns:
        Tuple of two matmul-compatible shapes.

    Example:
        ```mojo
        var (shape_a, shape_b) = generate_matmul_shapes(fuzzer)
        # shape_a = [M, K], shape_b = [K, N]
        var a = ones(shape_a, DType.float32)
        var b = ones(shape_b, DType.float32)
        var c = matmul(a, b)  # Should work, result is [M, N]
        ```
    """
    var m = fuzzer.rng.random_int(1, 20)
    var k = fuzzer.rng.random_int(1, 20)
    var n = fuzzer.rng.random_int(1, 20)

    var shape_a = List[Int]()
    shape_a.append(m)
    shape_a.append(k)

    var shape_b = List[Int]()
    shape_b.append(k)
    shape_b.append(n)

    return (shape_a^, shape_b^)


def generate_same_shape_pair(
    mut fuzzer: ShapeFuzzer,
) -> Tuple[List[Int], List[Int]]:
    """Generate a pair of identical shapes.

    Args:
        fuzzer: Shape fuzzer for random generation.

    Returns:
        Tuple of two identical shapes.

    Example:
        ```mojo
        var (shape_a, shape_b) = generate_same_shape_pair(fuzzer)
        # shape_a == shape_b
        ```
    """
    var shape = fuzzer.random_shape(min_ndim=1, max_ndim=4)

    # Copy shape for second
    var shape_copy = List[Int]()
    for i in range(len(shape)):
        shape_copy.append(shape[i])

    return (shape^, shape_copy^)


# ============================================================================
# Shape Validation Utilities
# ============================================================================


def is_valid_shape(shape: List[Int]) -> Bool:
    """Check if a shape is valid (all dimensions non-negative).

    Args:
        shape: Shape to validate.

    Returns:
        True if all dimensions are >= 0.

    Example:
        ```mojo
        var valid = is_valid_shape([3, 4, 5])  # True
        var invalid = is_valid_shape([3, -1, 5])  # False
        ```
    """
    for i in range(len(shape)):
        if shape[i] < 0:
            return False
    return True


def is_empty_shape(shape: List[Int]) -> Bool:
    """Check if a shape represents an empty tensor (0 elements).

    Args:
        shape: Shape to check.

    Returns:
        True if any dimension is 0.

    Example:
        ```mojo
        var empty = is_empty_shape([3, 0, 5])  # True
        var non_empty = is_empty_shape([3, 4, 5])  # False
        ```
    """
    for i in range(len(shape)):
        if shape[i] == 0:
            return True
    return False


def is_scalar_shape(shape: List[Int]) -> Bool:
    """Check if a shape represents a scalar (0D tensor).

    Args:
        shape: Shape to check.

    Returns:
        True if shape has no dimensions.

    Example:
        ```mojo
        var scalar = is_scalar_shape([])  # True
        var vector = is_scalar_shape([5])  # False
        ```
    """
    return len(shape) == 0


def compute_numel(shape: List[Int]) -> Int:
    """Compute total number of elements from shape.

    Args:
        shape: Shape to compute numel for.

    Returns:
        Product of all dimensions.

    Example:
        ```mojo
        var numel = compute_numel([3, 4, 5])  # 60
        ```
    """
    var numel = 1
    for i in range(len(shape)):
        numel *= shape[i]
    return numel


def shapes_equal(shape_a: List[Int], shape_b: List[Int]) -> Bool:
    """Check if two shapes are equal.

    Args:
        shape_a: First shape.
        shape_b: Second shape.

    Returns:
        True if shapes have same dimensions.

    Example:
        ```mojo
        var equal = shapes_equal([3, 4], [3, 4])  # True
        var not_equal = shapes_equal([3, 4], [4, 3])  # False
        ```
    """
    if len(shape_a) != len(shape_b):
        return False

    for i in range(len(shape_a)):
        if shape_a[i] != shape_b[i]:
            return False

    return True


def are_broadcast_compatible(shape_a: List[Int], shape_b: List[Int]) -> Bool:
    """Check if two shapes are broadcast-compatible.

    Args:
        shape_a: First shape.
        shape_b: Second shape.

    Returns:
        True if shapes can be broadcast together.

    Example:
        ```mojo
        var compat = are_broadcast_compatible([3, 4], [1, 4])  # True
        var incompat = are_broadcast_compatible([3, 4], [2, 4])  # False
        ```
    """
    var ndim_a = len(shape_a)
    var ndim_b = len(shape_b)
    var max_ndim = max(ndim_a, ndim_b)

    # Compare from trailing dimensions
    for i in range(max_ndim):
        var dim_a = 1
        var dim_b = 1

        if i < ndim_a:
            dim_a = shape_a[ndim_a - 1 - i]
        if i < ndim_b:
            dim_b = shape_b[ndim_b - 1 - i]

        # Dimensions must be equal or one must be 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    return True


def compute_broadcast_shape(
    shape_a: List[Int], shape_b: List[Int]
) raises -> List[Int]:
    """Compute the result shape of broadcasting two shapes.

    Args:
        shape_a: First shape.
        shape_b: Second shape.

    Returns:
        Broadcast result shape.

    Raises:
        Error: If shapes are not broadcast-compatible.

    Example:
        ```mojo
        var result = compute_broadcast_shape([3, 1], [1, 4])
        # result = [3, 4]
        ```
    """
    if not are_broadcast_compatible(shape_a, shape_b):
        raise Error("Shapes are not broadcast-compatible")

    var ndim_a = len(shape_a)
    var ndim_b = len(shape_b)
    var max_ndim = max(ndim_a, ndim_b)

    var result = List[Int]()

    # Build result shape from trailing dimensions
    for i in range(max_ndim):
        var dim_a = 1
        var dim_b = 1

        if i < ndim_a:
            dim_a = shape_a[ndim_a - 1 - i]
        if i < ndim_b:
            dim_b = shape_b[ndim_b - 1 - i]

        # Result dimension is max of the two
        result.append(max(dim_a, dim_b))

    # Reverse to get correct order
    var reversed_result = List[Int]()
    for i in range(len(result) - 1, -1, -1):
        reversed_result.append(result[i])

    return reversed_result^


def shape_to_string(shape: List[Int]) -> String:
    """Convert shape to string representation.

    Args:
        shape: Shape to convert.

    Returns:
        String like "[3, 4, 5]".

    Example:
        ```mojo
        var s = shape_to_string([3, 4, 5])  # "[3, 4, 5]"
        ```
    """
    var result = String("[")
    for i in range(len(shape)):
        if i > 0:
            result += ", "
        result += String(shape[i])
    result += "]"
    return result^
