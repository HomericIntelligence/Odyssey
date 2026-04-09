"""Tests that shared/base/ package imports work correctly.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- memory_pool imports (pooled_alloc, pooled_free)
- broadcasting imports (broadcast_shapes, are_shapes_broadcastable)
- dtype_ordinal imports (dtype_to_ordinal, DTYPE_FLOAT32, etc.)
- defaults imports (DEFAULT_DROPOUT_RATE, etc.)
- math_constants imports (PI, SQRT_2, etc.)
- numerical_constants imports (EPSILON_DIV, etc.)
- Package-level re-exports via shared.base
"""

from std.testing import assert_true


def test_memory_pool_imports() raises:
    """Verify memory_pool module imports work."""
    from shared.base.memory_pool import pooled_alloc, pooled_free

    # Just verify the imports compile and the symbols are accessible.
    # pooled_alloc/pooled_free are runtime functions; we verify they exist.
    print("PASS: test_memory_pool_imports")


def test_broadcasting_imports() raises:
    """Verify broadcasting module imports and basic function work."""
    from shared.base.broadcasting import broadcast_shapes, are_shapes_broadcastable

    # Test basic broadcasting: [3, 1] and [1, 4] -> [3, 4]
    var a = [3, 1]
    var b = [1, 4]
    var result = broadcast_shapes(a, b)
    assert_true(result[0] == 3, "broadcast dim 0 should be 3")
    assert_true(result[1] == 4, "broadcast dim 1 should be 4")

    assert_true(
        are_shapes_broadcastable(a, b),
        "[3,1] and [1,4] should be broadcastable",
    )
    print("PASS: test_broadcasting_imports")


def test_dtype_ordinal_imports() raises:
    """Verify dtype_ordinal module imports and constants."""
    from shared.base.dtype_ordinal import (
        dtype_to_ordinal,
        DTYPE_FLOAT32,
        DTYPE_FLOAT64,
        DTYPE_INT32,
        DTYPE_UNSUPPORTED,
        SUPPORTED_DTYPE_COUNT,
    )

    # Verify ordinal constants are distinct
    assert_true(
        DTYPE_FLOAT32 != DTYPE_FLOAT64,
        "FLOAT32 and FLOAT64 ordinals should differ",
    )
    assert_true(
        DTYPE_FLOAT32 != DTYPE_INT32,
        "FLOAT32 and INT32 ordinals should differ",
    )

    # Verify dtype_to_ordinal maps correctly
    var ord32 = dtype_to_ordinal(DType.float32)
    assert_true(ord32 == DTYPE_FLOAT32, "float32 ordinal should match")

    var ord64 = dtype_to_ordinal(DType.float64)
    assert_true(ord64 == DTYPE_FLOAT64, "float64 ordinal should match")

    assert_true(
        SUPPORTED_DTYPE_COUNT > 0, "should support at least one dtype"
    )
    print("PASS: test_dtype_ordinal_imports")


def test_defaults_imports() raises:
    """Verify defaults module imports and constant values."""
    from shared.base.defaults import (
        DEFAULT_DROPOUT_RATE,
        DEFAULT_BATCHNORM_MOMENTUM,
        DEFAULT_RANDOM_SEED,
    )

    # Verify defaults are reasonable
    assert_true(
        DEFAULT_DROPOUT_RATE >= 0.0 and DEFAULT_DROPOUT_RATE <= 1.0,
        "dropout rate should be in [0, 1]",
    )
    assert_true(
        DEFAULT_BATCHNORM_MOMENTUM >= 0.0
        and DEFAULT_BATCHNORM_MOMENTUM <= 1.0,
        "batchnorm momentum should be in [0, 1]",
    )
    print("PASS: test_defaults_imports")


def test_math_constants_imports() raises:
    """Verify math_constants module imports and values."""
    from shared.base.math_constants import PI, SQRT_2, LN2

    # Verify known mathematical values
    assert_true(PI > 3.14 and PI < 3.15, "PI should be ~3.14159")
    assert_true(SQRT_2 > 1.41 and SQRT_2 < 1.42, "SQRT_2 should be ~1.4142")
    assert_true(LN2 > 0.69 and LN2 < 0.70, "LN2 should be ~0.6931")
    print("PASS: test_math_constants_imports")


def test_numerical_constants_imports() raises:
    """Verify numerical_constants module imports and values."""
    from shared.base.numerical_constants import (
        EPSILON_DIV,
        EPSILON_LOSS,
        EPSILON_NORM,
        GRADIENT_MAX_NORM,
    )

    # Verify epsilon values are small and positive
    assert_true(EPSILON_DIV > 0.0, "EPSILON_DIV should be positive")
    assert_true(EPSILON_DIV < 1.0, "EPSILON_DIV should be small")
    assert_true(EPSILON_LOSS > 0.0, "EPSILON_LOSS should be positive")
    assert_true(EPSILON_NORM > 0.0, "EPSILON_NORM should be positive")
    assert_true(
        GRADIENT_MAX_NORM > 0.0, "GRADIENT_MAX_NORM should be positive"
    )
    print("PASS: test_numerical_constants_imports")


def test_package_level_reexports() raises:
    """Verify shared.base package re-exports symbols."""
    from shared.base import (
        pooled_alloc,
        pooled_free,
        broadcast_shapes,
        dtype_to_ordinal,
        DTYPE_FLOAT32,
        PI,
        EPSILON_DIV,
        DEFAULT_DROPOUT_RATE,
    )

    # Verify the re-exports are accessible
    var ord32 = dtype_to_ordinal(DType.float32)
    assert_true(ord32 == DTYPE_FLOAT32, "package re-export should work")
    print("PASS: test_package_level_reexports")


def main() raises:
    test_memory_pool_imports()
    test_broadcasting_imports()
    test_dtype_ordinal_imports()
    test_defaults_imports()
    test_math_constants_imports()
    test_numerical_constants_imports()
    test_package_level_reexports()
    print("All test_base_imports tests passed!")
