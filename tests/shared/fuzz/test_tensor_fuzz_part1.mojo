# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_fuzz.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Fuzz Tests for Tensor Operations - Part 1: Creation and Arithmetic

Property-based tests using fuzzing to discover edge cases and unexpected
behaviors in tensor creation and arithmetic operations.

Test Strategy:
    1. Generate random inputs within configurable bounds
    2. Execute operations under test
    3. Verify invariants (no crashes, valid outputs, numeric properties)
    4. Report failures with reproducible seeds

Fuzzing Categories (Part 1):
    - Tensor Creation: Random shapes, dtypes, edge case values
    - Arithmetic Operations: Add, subtract, multiply, divide
    - Edge Cases: Empty tensors

Usage:
    mojo test tests/shared/fuzz/test_tensor_fuzz_part1.mojo

Note:
    These tests use deterministic seeds for reproducibility.
    If a test fails, the seed is printed for reproduction.
"""

from random import seed as random_seed
from math import isnan, isinf

from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    add,
    subtract,
    multiply,
    divide,
)

from shared.testing.fuzz_core import (
    FuzzConfig,
    FuzzResult,
    SeededRNG,
    create_random_tensor,
    create_edge_case_tensor,
    has_nan,
    has_inf,
    is_finite,
    check_numeric_invariants,
    verify_shape_preserved,
    verify_dtype_preserved,
    format_failure_message,
)

from shared.testing.fuzz_shapes import (
    ShapeFuzzer,
    generate_broadcast_shapes,
    generate_same_shape_pair,
    is_empty_shape,
    is_scalar_shape,
    compute_numel,
    shape_to_string,
)

from shared.testing.fuzz_dtypes import (
    DTypeFuzzer,
    get_float_dtypes,
    get_edge_values,
    is_float_dtype,
    supports_nan_inf,
    dtype_to_string,
)

from tests.shared.conftest import assert_true, assert_equal_int


# ============================================================================
# Configuration
# ============================================================================

# Default number of iterations for fuzz tests
comptime DEFAULT_ITERATIONS: Int = 50

# Seed for reproducible fuzzing
comptime DEFAULT_SEED: Int = 42


# ============================================================================
# Tensor Creation Fuzz Tests
# ============================================================================


fn test_fuzz_tensor_creation_random_shapes() raises:
    """Fuzz test: tensor creation with random shapes should not crash.

    Verifies that creating tensors with various random shapes:
    - Does not crash
    - Produces tensors with correct shape
    - Produces tensors with correct dtype
    - Produces tensors with expected number of elements
    """
    print("  test_fuzz_tensor_creation_random_shapes...")

    var config = FuzzConfig(
        seed=DEFAULT_SEED,
        iterations=DEFAULT_ITERATIONS,
        max_dim=50,  # Keep dimensions small for speed
        max_ndim=5,
    )

    var shape_fuzzer = ShapeFuzzer(seed=config.seed, max_dim=config.max_dim)
    var result = FuzzResult()

    for i in range(config.iterations):
        shape_fuzzer.next_iteration()
        var iteration_seed = config.seed + i

        try:
            var shape = shape_fuzzer.random_shape(
                min_ndim=0, max_ndim=config.max_ndim
            )
            var tensor = zeros(shape, DType.float32)

            # Verify shape
            var tensor_shape = tensor.shape()
            if len(tensor_shape) != len(shape):
                result.add_failure(
                    "Shape dimension mismatch: got "
                    + String(len(tensor_shape))
                    + " expected "
                    + String(len(shape)),
                    iteration_seed,
                )
                continue

            for j in range(len(shape)):
                if tensor_shape[j] != shape[j]:
                    result.add_failure(
                        "Shape value mismatch at dim " + String(j),
                        iteration_seed,
                    )
                    continue

            # Verify numel
            var expected_numel = compute_numel(shape)
            if tensor.numel() != expected_numel:
                result.add_failure(
                    "Numel mismatch: got "
                    + String(tensor.numel())
                    + " expected "
                    + String(expected_numel),
                    iteration_seed,
                )
                continue

            result.add_success()

        except e:
            # Some shapes might cause expected errors (e.g., too large)
            var err_str = String(e)
            if "too large" in err_str or "memory" in err_str:
                result.add_expected_error()
            else:
                result.add_failure(String(e), iteration_seed)

    if config.verbose or not result.is_success():
        result.print_summary()

    assert_true(
        result.is_success(),
        (
            "Tensor creation with random shapes should not produce unexpected"
            " failures"
        ),
    )


fn test_fuzz_tensor_creation_edge_shapes() raises:
    """Fuzz test: tensor creation with edge case shapes.

    Tests edge case shapes:
    - Scalar (0D)
    - Empty (0 elements)
    - Single element
    - High dimensional
    - Large 1D
    """
    print("  test_fuzz_tensor_creation_edge_shapes...")

    var shape_fuzzer = ShapeFuzzer(seed=DEFAULT_SEED)
    var edge_shapes = shape_fuzzer.edge_case_shapes()
    var result = FuzzResult()

    for i in range(len(edge_shapes)):
        var shape = edge_shapes[i].copy()
        var iteration_seed = DEFAULT_SEED + i

        try:
            var tensor = zeros(shape, DType.float32)

            # Verify basic properties
            var expected_numel = compute_numel(shape)
            if tensor.numel() != expected_numel:
                result.add_failure(
                    "Numel mismatch for shape "
                    + shape_to_string(shape)
                    + ": got "
                    + String(tensor.numel())
                    + " expected "
                    + String(expected_numel),
                    iteration_seed,
                )
                continue

            result.add_success()

        except e:
            var err_str = String(e)
            # Large tensors may fail due to memory constraints
            if "too large" in err_str or "memory" in err_str:
                result.add_expected_error()
            else:
                result.add_failure(
                    "Failed for shape "
                    + shape_to_string(shape)
                    + ": "
                    + err_str,
                    iteration_seed,
                )

    assert_true(
        result.is_success(),
        "Tensor creation with edge shapes should handle all cases",
    )


fn test_fuzz_tensor_creation_all_dtypes() raises:
    """Fuzz test: tensor creation works for all data types.

    Verifies that tensors can be created with all supported dtypes:
    - float16, float32, float64, bfloat16
    - int8, int16, int32, int64
    - uint8, uint16, uint32, uint64
    - bool
    """
    print("  test_fuzz_tensor_creation_all_dtypes...")

    var dtypes = List[DType]()
    dtypes.append(DType.float16)
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.bfloat16)
    dtypes.append(DType.int8)
    dtypes.append(DType.int16)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)
    dtypes.append(DType.uint8)
    dtypes.append(DType.uint16)
    dtypes.append(DType.uint32)
    dtypes.append(DType.uint64)
    dtypes.append(DType.bool)

    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var result = FuzzResult()

    for i in range(len(dtypes)):
        var dtype = dtypes[i]
        var iteration_seed = DEFAULT_SEED + i

        try:
            var tensor = zeros(shape, dtype)

            # Verify dtype
            if tensor.dtype() != dtype:
                result.add_failure(
                    "DType mismatch: got "
                    + String(tensor.dtype())
                    + " expected "
                    + String(dtype),
                    iteration_seed,
                )
                continue

            # Verify shape preserved
            if tensor.numel() != 12:  # 3 * 4 = 12
                result.add_failure(
                    "Numel mismatch for dtype " + String(dtype),
                    iteration_seed,
                )
                continue

            result.add_success()

        except e:
            result.add_failure(
                "Failed for dtype " + String(dtype) + ": " + String(e),
                iteration_seed,
            )

    assert_true(
        result.is_success(),
        "Tensor creation should work for all dtypes",
    )


# ============================================================================
# Arithmetic Operation Fuzz Tests
# ============================================================================


fn test_fuzz_add_same_shape() raises:
    """Fuzz test: addition with same shape tensors.

    Verifies that adding tensors of the same shape:
    - Does not crash
    - Produces result with same shape
    - Produces result with same dtype
    - Produces finite results for finite inputs
    """
    print("  test_fuzz_add_same_shape...")

    var config = FuzzConfig(
        seed=DEFAULT_SEED,
        iterations=DEFAULT_ITERATIONS,
        max_dim=20,
        max_ndim=4,
    )

    var shape_fuzzer = ShapeFuzzer(seed=config.seed, max_dim=config.max_dim)
    var rng = SeededRNG(config.seed)
    var result = FuzzResult()

    for i in range(config.iterations):
        shape_fuzzer.next_iteration()
        rng.next_iteration()
        var iteration_seed = config.seed + i

        try:
            var shape = shape_fuzzer.random_shape(
                min_ndim=1, max_ndim=config.max_ndim
            )

            # Skip if shape is empty
            if is_empty_shape(shape):
                result.add_expected_error()
                continue

            var a = create_random_tensor(rng, shape, DType.float32, -10.0, 10.0)
            var b = create_random_tensor(rng, shape, DType.float32, -10.0, 10.0)

            var c = add(a, b)

            # Verify shape preserved
            if not verify_shape_preserved(a, c):
                result.add_failure(
                    "Shape not preserved in addition",
                    iteration_seed,
                )
                continue

            # Verify dtype preserved
            if not verify_dtype_preserved(a, c):
                result.add_failure(
                    "DType not preserved in addition",
                    iteration_seed,
                )
                continue

            # Verify result is finite (inputs were finite)
            if not is_finite(c):
                result.add_failure(
                    "Addition of finite inputs produced non-finite result",
                    iteration_seed,
                )
                continue

            result.add_success()

        except e:
            result.add_failure(String(e), iteration_seed)

    assert_true(
        result.is_success(),
        "Addition with same shape should work correctly",
    )


fn test_fuzz_subtract_same_shape() raises:
    """Fuzz test: subtraction with same shape tensors."""
    print("  test_fuzz_subtract_same_shape...")

    var config = FuzzConfig(
        seed=DEFAULT_SEED,
        iterations=DEFAULT_ITERATIONS,
        max_dim=20,
        max_ndim=4,
    )

    var shape_fuzzer = ShapeFuzzer(seed=config.seed, max_dim=config.max_dim)
    var rng = SeededRNG(config.seed)
    var result = FuzzResult()

    for i in range(config.iterations):
        shape_fuzzer.next_iteration()
        rng.next_iteration()
        var iteration_seed = config.seed + i

        try:
            var shape = shape_fuzzer.random_shape(
                min_ndim=1, max_ndim=config.max_ndim
            )

            if is_empty_shape(shape):
                result.add_expected_error()
                continue

            var a = create_random_tensor(rng, shape, DType.float32, -10.0, 10.0)
            var b = create_random_tensor(rng, shape, DType.float32, -10.0, 10.0)

            var c = subtract(a, b)

            if not verify_shape_preserved(a, c):
                result.add_failure("Shape not preserved", iteration_seed)
                continue

            if not is_finite(c):
                result.add_failure(
                    "Subtraction produced non-finite result",
                    iteration_seed,
                )
                continue

            result.add_success()

        except e:
            result.add_failure(String(e), iteration_seed)

    assert_true(
        result.is_success(),
        "Subtraction with same shape should work correctly",
    )


fn test_fuzz_multiply_same_shape() raises:
    """Fuzz test: multiplication with same shape tensors."""
    print("  test_fuzz_multiply_same_shape...")

    var config = FuzzConfig(
        seed=DEFAULT_SEED,
        iterations=DEFAULT_ITERATIONS,
        max_dim=20,
        max_ndim=4,
    )

    var shape_fuzzer = ShapeFuzzer(seed=config.seed, max_dim=config.max_dim)
    var rng = SeededRNG(config.seed)
    var result = FuzzResult()

    for i in range(config.iterations):
        shape_fuzzer.next_iteration()
        rng.next_iteration()
        var iteration_seed = config.seed + i

        try:
            var shape = shape_fuzzer.random_shape(
                min_ndim=1, max_ndim=config.max_ndim
            )

            if is_empty_shape(shape):
                result.add_expected_error()
                continue

            # Use smaller range to avoid overflow
            var a = create_random_tensor(rng, shape, DType.float32, -10.0, 10.0)
            var b = create_random_tensor(rng, shape, DType.float32, -10.0, 10.0)

            var c = multiply(a, b)

            if not verify_shape_preserved(a, c):
                result.add_failure("Shape not preserved", iteration_seed)
                continue

            if not is_finite(c):
                result.add_failure(
                    "Multiplication produced non-finite result",
                    iteration_seed,
                )
                continue

            result.add_success()

        except e:
            result.add_failure(String(e), iteration_seed)

    assert_true(
        result.is_success(),
        "Multiplication with same shape should work correctly",
    )


fn test_fuzz_divide_same_shape() raises:
    """Fuzz test: division with same shape tensors.

    Note: Division by zero produces Inf/NaN which is expected.
    """
    print("  test_fuzz_divide_same_shape...")

    var config = FuzzConfig(
        seed=DEFAULT_SEED,
        iterations=DEFAULT_ITERATIONS,
        max_dim=20,
        max_ndim=4,
    )

    var shape_fuzzer = ShapeFuzzer(seed=config.seed, max_dim=config.max_dim)
    var rng = SeededRNG(config.seed)
    var result = FuzzResult()

    for i in range(config.iterations):
        shape_fuzzer.next_iteration()
        rng.next_iteration()
        var iteration_seed = config.seed + i

        try:
            var shape = shape_fuzzer.random_shape(
                min_ndim=1, max_ndim=config.max_ndim
            )

            if is_empty_shape(shape):
                result.add_expected_error()
                continue

            var a = create_random_tensor(rng, shape, DType.float32, -10.0, 10.0)
            # Avoid zero in divisor
            var b = create_random_tensor(rng, shape, DType.float32, 0.1, 10.0)

            var c = divide(a, b)

            if not verify_shape_preserved(a, c):
                result.add_failure("Shape not preserved", iteration_seed)
                continue

            # With non-zero divisor, result should be finite
            if not is_finite(c):
                result.add_failure(
                    "Division with non-zero divisor produced non-finite result",
                    iteration_seed,
                )
                continue

            result.add_success()

        except e:
            result.add_failure(String(e), iteration_seed)

    assert_true(
        result.is_success(),
        "Division with same shape (non-zero divisor) should work correctly",
    )


# ============================================================================
# Edge Case Fuzz Tests (Part 1)
# ============================================================================


fn test_fuzz_operations_on_empty_tensors() raises:
    """Fuzz test: operations on empty tensors should not crash."""
    print("  test_fuzz_operations_on_empty_tensors...")

    var result = FuzzResult()

    # Create empty tensor
    var shape = List[Int]()
    shape.append(0)

    try:
        var a = zeros(shape, DType.float32)
        var b = zeros(shape, DType.float32)

        # All operations should work on empty tensors
        var c = add(a, b)
        if c.numel() != 0:
            result.add_failure("Empty add should give empty result", 0)
        else:
            result.add_success()

        var d = subtract(a, b)
        if d.numel() != 0:
            result.add_failure("Empty subtract should give empty result", 1)
        else:
            result.add_success()

        var e = multiply(a, b)
        if e.numel() != 0:
            result.add_failure("Empty multiply should give empty result", 2)
        else:
            result.add_success()

        var f = divide(a, b)
        if f.numel() != 0:
            result.add_failure("Empty divide should give empty result", 3)
        else:
            result.add_success()

    except e:
        result.add_failure("Empty tensor operations crashed: " + String(e), 0)

    assert_true(
        result.is_success(),
        "Operations on empty tensors should not crash",
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all fuzz tests (part 1)."""
    print("Running tensor fuzz tests (part 1)...")

    # Tensor creation tests
    print("\n=== Tensor Creation Fuzz Tests ===")
    test_fuzz_tensor_creation_random_shapes()
    test_fuzz_tensor_creation_edge_shapes()
    test_fuzz_tensor_creation_all_dtypes()

    # Arithmetic operation tests
    print("\n=== Arithmetic Operation Fuzz Tests ===")
    test_fuzz_add_same_shape()
    test_fuzz_subtract_same_shape()
    test_fuzz_multiply_same_shape()
    test_fuzz_divide_same_shape()

    # Edge case tests (part 1)
    print("\n=== Edge Case Fuzz Tests (Part 1) ===")
    test_fuzz_operations_on_empty_tensors()

    print("\nAll tensor fuzz tests (part 1) completed!")
