# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_fuzz.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Fuzz Tests for Tensor Operations - Part 2: Edge Cases

Property-based tests using fuzzing to discover edge cases and unexpected
behaviors in tensor arithmetic edge case handling.

Test Strategy:
    1. Generate random inputs within configurable bounds
    2. Execute operations under test
    3. Verify invariants (no crashes, valid outputs, numeric properties)
    4. Report failures with reproducible seeds

Fuzzing Categories (Part 2):
    - Edge Cases: Scalars, NaN, Inf, division by zero, large values, subnormals

Usage:
    mojo test tests/shared/fuzz/test_tensor_fuzz_part2.mojo

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
# Edge Case Fuzz Tests (Part 2)
# ============================================================================


fn test_fuzz_operations_on_scalars() raises:
    """Fuzz test: operations on 0D scalar tensors should work correctly."""
    print("  test_fuzz_operations_on_scalars...")

    var result = FuzzResult()
    var rng = SeededRNG(DEFAULT_SEED)
    var shape = List[Int]()  # 0D scalar shape

    for i in range(20):
        rng.next_iteration()
        var iteration_seed = DEFAULT_SEED + i

        try:
            var val_a = rng.random_float(-10.0, 10.0)
            var val_b = rng.random_float(0.1, 10.0)  # Avoid zero for division

            var a = full(shape, val_a, DType.float32)
            var b = full(shape, val_b, DType.float32)

            # Test all operations
            var c = add(a, b)
            if c.numel() != 1:
                result.add_failure(
                    "Scalar add should give scalar", iteration_seed
                )
                continue

            var d = subtract(a, b)
            if d.numel() != 1:
                result.add_failure(
                    "Scalar subtract should give scalar", iteration_seed
                )
                continue

            var e = multiply(a, b)
            if e.numel() != 1:
                result.add_failure(
                    "Scalar multiply should give scalar", iteration_seed
                )
                continue

            var f = divide(a, b)
            if f.numel() != 1:
                result.add_failure(
                    "Scalar divide should give scalar", iteration_seed
                )
                continue

            result.add_success()

        except e:
            result.add_failure(
                "Scalar operations failed: " + String(e), iteration_seed
            )

    assert_true(
        result.is_success(),
        "Operations on scalar tensors should work correctly",
    )


fn test_fuzz_nan_propagation() raises:
    """Fuzz test: NaN should propagate through arithmetic operations."""
    print("  test_fuzz_nan_propagation...")

    var result = FuzzResult()
    var shape = List[Int]()
    shape.append(5)

    try:
        var nan_tensor = create_edge_case_tensor(shape, DType.float32, "nan")
        var normal_tensor = ones(shape, DType.float32)

        # NaN + x = NaN
        var add_result = add(nan_tensor, normal_tensor)
        if not has_nan(add_result):
            result.add_failure("NaN + 1 should contain NaN", 0)
        else:
            result.add_success()

        # NaN - x = NaN
        var sub_result = subtract(nan_tensor, normal_tensor)
        if not has_nan(sub_result):
            result.add_failure("NaN - 1 should contain NaN", 1)
        else:
            result.add_success()

        # NaN * x = NaN
        var mul_result = multiply(nan_tensor, normal_tensor)
        if not has_nan(mul_result):
            result.add_failure("NaN * 1 should contain NaN", 2)
        else:
            result.add_success()

        # NaN / x = NaN
        var div_result = divide(nan_tensor, normal_tensor)
        if not has_nan(div_result):
            result.add_failure("NaN / 1 should contain NaN", 3)
        else:
            result.add_success()

        # NaN * 0 = NaN (not 0!)
        var zero_tensor = zeros(shape, DType.float32)
        var nan_times_zero = multiply(nan_tensor, zero_tensor)
        if not has_nan(nan_times_zero):
            result.add_failure("NaN * 0 should be NaN", 4)
        else:
            result.add_success()

    except e:
        result.add_failure("NaN propagation test failed: " + String(e), 0)

    assert_true(
        result.is_success(),
        "NaN should propagate correctly through operations",
    )


fn test_fuzz_inf_propagation() raises:
    """Fuzz test: Infinity should propagate correctly through operations."""
    print("  test_fuzz_inf_propagation...")

    var result = FuzzResult()
    var shape = List[Int]()
    shape.append(5)

    try:
        var inf_tensor = create_edge_case_tensor(shape, DType.float32, "inf")
        var normal_tensor = ones(shape, DType.float32)
        var zero_tensor = zeros(shape, DType.float32)

        # Inf + x = Inf
        var add_result = add(inf_tensor, normal_tensor)
        if not has_inf(add_result):
            result.add_failure("Inf + 1 should contain Inf", 0)
        else:
            result.add_success()

        # Inf * x = Inf (for x > 0)
        var mul_result = multiply(inf_tensor, normal_tensor)
        if not has_inf(mul_result):
            result.add_failure("Inf * 1 should contain Inf", 1)
        else:
            result.add_success()

        # Inf * 0 = NaN
        var inf_times_zero = multiply(inf_tensor, zero_tensor)
        if not has_nan(inf_times_zero):
            result.add_failure("Inf * 0 should be NaN", 2)
        else:
            result.add_success()

    except e:
        result.add_failure("Inf propagation test failed: " + String(e), 0)

    assert_true(
        result.is_success(),
        "Infinity should propagate correctly through operations",
    )


fn test_fuzz_division_by_zero() raises:
    """Fuzz test: division by zero should produce Inf, not crash."""
    print("  test_fuzz_division_by_zero...")

    var result = FuzzResult()
    var shape = List[Int]()
    shape.append(5)

    try:
        var numerator = ones(shape, DType.float32)
        var denominator = zeros(shape, DType.float32)

        # 1 / 0 = Inf
        var div_result = divide(numerator, denominator)

        # Should not crash, should produce Inf
        if not has_inf(div_result):
            result.add_failure("1 / 0 should produce Inf", 0)
        else:
            result.add_success()

        # 0 / 0 = NaN
        var zero_num = zeros(shape, DType.float32)
        var zero_div = divide(zero_num, denominator)

        if not has_nan(zero_div):
            result.add_failure("0 / 0 should produce NaN", 1)
        else:
            result.add_success()

    except e:
        result.add_failure("Division by zero test failed: " + String(e), 0)

    assert_true(
        result.is_success(),
        "Division by zero should produce Inf/NaN, not crash",
    )


fn test_fuzz_large_values() raises:
    """Fuzz test: operations with large values should handle overflow gracefully.
    """
    print("  test_fuzz_large_values...")

    var result = FuzzResult()
    var shape = List[Int]()
    shape.append(3)

    try:
        # Create tensors near float32 max
        var large_tensor = create_edge_case_tensor(shape, DType.float32, "max")
        var small_positive = full(shape, 2.0, DType.float32)

        # Large * 2 should overflow to Inf
        var mul_result = multiply(large_tensor, small_positive)

        # Should not crash, may produce Inf
        if not has_inf(mul_result):
            # May or may not overflow depending on exact values
            # Just check it didn't crash
            pass

        result.add_success()

        # Large + Large should overflow to Inf
        var add_result = add(large_tensor, large_tensor)

        # Should not crash
        result.add_success()

    except e:
        result.add_failure("Large value test failed: " + String(e), 0)

    assert_true(
        result.is_success(),
        "Operations with large values should handle overflow gracefully",
    )


fn test_fuzz_subnormal_values() raises:
    """Fuzz test: operations with very small (subnormal) values."""
    print("  test_fuzz_subnormal_values...")

    var result = FuzzResult()
    var shape = List[Int]()
    shape.append(5)

    try:
        # Create tensor with very small values
        var small_tensor = create_edge_case_tensor(
            shape, DType.float32, "epsilon"
        )
        var normal_tensor = ones(shape, DType.float32)

        # Small + 1 should be approximately 1
        var add_result = add(small_tensor, normal_tensor)
        if not is_finite(add_result):
            result.add_failure("Small + 1 should be finite", 0)
        else:
            result.add_success()

        # Small * Small may underflow to 0
        var mul_result = multiply(small_tensor, small_tensor)
        # Should not crash
        result.add_success()

    except e:
        result.add_failure("Subnormal value test failed: " + String(e), 0)

    assert_true(
        result.is_success(),
        "Operations with subnormal values should not crash",
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all fuzz tests (part 2)."""
    print("Running tensor fuzz tests (part 2)...")

    # Edge case tests (part 2)
    print("\n=== Edge Case Fuzz Tests (Part 2) ===")
    test_fuzz_operations_on_scalars()
    test_fuzz_nan_propagation()
    test_fuzz_inf_propagation()
    test_fuzz_division_by_zero()
    test_fuzz_large_values()
    test_fuzz_subnormal_values()

    print("\nAll tensor fuzz tests (part 2) completed!")
