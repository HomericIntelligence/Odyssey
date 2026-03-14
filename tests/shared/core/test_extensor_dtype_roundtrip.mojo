"""Audit of _set_float64/_get_float64 round-trip support for all dtypes in get_test_dtypes().

Verifies that each dtype in get_test_dtypes() — float16, bfloat16, int8, float32 — correctly
round-trips non-zero values through the float64 I/O path. Any dtype that silently returns zeros
(broken _set_float64 path) is documented with a TODO comment.

Follow-up from issue #3088 (bfloat16 _set_float64/_get_float64 silently did nothing).
Fixes tracked in issue #3301.

Note: Split into a dedicated file following ADR-009 pattern — heap corruption occurs after
~15 cumulative tests in a single file in Mojo 0.26.1.
"""

from shared.core.extensor import ExTensor, zeros
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# float16 — supported, tolerance 1e-3
# ============================================================================


fn test_float16_set_get_float64_roundtrip() raises:
    """Float16: _set_float64(1.5) -> _get_float64 should return ~1.5."""
    var t = zeros([1], DType.float16)
    t._set_float64(0, 1.5)
    var got = t._get_float64(0)
    # Zero-guard: detects silent-write failure (the original bug pattern)
    assert_true(
        got != 0.0, "float16 _get_float64 returned 0 after _set_float64(1.5)"
    )
    assert_almost_equal(got, 1.5, tolerance=1e-3)


fn test_float16_nonzero_after_set() raises:
    """Float16: writing a non-zero value must not silently produce zero."""
    var t = zeros([3], DType.float16)
    t._set_float64(1, 2.0)
    var got = t._get_float64(1)
    assert_true(got != 0.0, "float16 _set_float64 silently wrote zero")


# ============================================================================
# float32 — supported, tolerance 1e-6
# ============================================================================


fn test_float32_set_get_float64_roundtrip() raises:
    """Float32: _set_float64(1.5) -> _get_float64 should return ~1.5."""
    var t = zeros([1], DType.float32)
    t._set_float64(0, 1.5)
    var got = t._get_float64(0)
    assert_true(
        got != 0.0, "float32 _get_float64 returned 0 after _set_float64(1.5)"
    )
    assert_almost_equal(got, 1.5, tolerance=1e-6)


fn test_float32_nonzero_after_set() raises:
    """Float32: writing a non-zero value must not silently produce zero."""
    var t = zeros([3], DType.float32)
    t._set_float64(1, 2.0)
    var got = t._get_float64(1)
    assert_true(got != 0.0, "float32 _set_float64 silently wrote zero")


# ============================================================================
# float64 — supported, tolerance 1e-9
# ============================================================================


fn test_float64_set_get_float64_roundtrip() raises:
    """Float64: _set_float64(1.5) -> _get_float64 should return 1.5 exactly."""
    var t = zeros([1], DType.float64)
    t._set_float64(0, 1.5)
    var got = t._get_float64(0)
    assert_true(
        got != 0.0, "float64 _get_float64 returned 0 after _set_float64(1.5)"
    )
    assert_almost_equal(got, 1.5, tolerance=1e-9)


fn test_float64_nonzero_after_set() raises:
    """Float64: writing a non-zero value must not silently produce zero."""
    var t = zeros([3], DType.float64)
    t._set_float64(1, 2.0)
    var got = t._get_float64(1)
    assert_true(got != 0.0, "float64 _set_float64 silently wrote zero")


# ============================================================================
# bfloat16 — fixed in #3301: added bfloat16 branches to _set_float64/_get_float64
# and fixed _get_dtype_size_static to return 2 bytes (not 4) for bfloat16.
# tolerance 1e-2 (bfloat16 has ~7-bit mantissa precision)
# ============================================================================


fn test_bfloat16_set_get_float64_roundtrip() raises:
    """Bfloat16: _set_float64(1.5) -> _get_float64 should return ~1.5.

    Before fix (#3301): _set_float64 had no bfloat16 branch, so writes were
    silently ignored and _get_float64 misread bits via _get_int64. After fix,
    this round-trip should work correctly.
    """
    var t = zeros([1], DType.bfloat16)
    t._set_float64(0, 1.5)
    var got = t._get_float64(0)
    # Zero-guard: detects the original silent-write bug
    assert_true(
        got != 0.0,
        (
            "bfloat16 _get_float64 returned 0 after _set_float64(1.5) —"
            " bfloat16 branch missing in _set_float64"
        ),
    )
    # bfloat16 has ~2 decimal digits precision (7-bit mantissa), 1.5 is exactly representable
    assert_almost_equal(got, 1.5, tolerance=1e-2)


fn test_bfloat16_nonzero_after_set() raises:
    """Bfloat16: writing a non-zero value must not silently produce zero."""
    var t = zeros([3], DType.bfloat16)
    t._set_float64(1, 2.0)
    var got = t._get_float64(1)
    assert_true(
        got != 0.0,
        "bfloat16 _set_float64 silently wrote zero — bfloat16 branch missing",
    )


fn test_bfloat16_dtype_size_is_2_bytes() raises:
    """Bfloat16 tensor should allocate 2 bytes per element (not 4).

    Before fix (#3301): _get_dtype_size_static had no bfloat16 branch, falling
    through to the default `return 4`. This caused incorrect offset calculations
    and memory overreads when accessing elements by index.
    """
    var t = zeros([4], DType.bfloat16)
    # Write to all 4 elements — with wrong dtype_size (4 instead of 2) the
    # offsets would be wrong and we would read stale memory or overflow
    t._set_float64(0, 1.0)
    t._set_float64(1, 2.0)
    t._set_float64(2, 3.0)
    t._set_float64(3, 4.0)
    assert_almost_equal(t._get_float64(0), 1.0, tolerance=1e-2)
    assert_almost_equal(t._get_float64(1), 2.0, tolerance=1e-2)
    assert_almost_equal(t._get_float64(2), 3.0, tolerance=1e-2)
    assert_almost_equal(t._get_float64(3), 4.0, tolerance=1e-2)


# ============================================================================
# int8 — integer type: _set_float64 has no int8 branch (silent no-op).
# _get_float64 falls through to _get_int64 which works for integer-safe values.
# Only integer-representable values (1.0, -1.0, 0.0) are meaningful; 1.5 truncates.
# TODO(#3301): _set_float64 does not write to int8 tensors (no branch exists).
#              Use _set_int64 for int8 tensors instead.
# ============================================================================


fn test_int8_get_float64_via_int64_path() raises:
    """Int8: _get_float64 correctly reads integer values via the _get_int64 fallback.

    The int8 path in _get_float64 falls through to _get_int64(), which correctly
    reads int8 bits and casts to Float64. Integer-safe values like 1.0 round-trip
    correctly. This tests the read path (set via _set_int64 which works correctly).
    """
    var t = zeros([3], DType.int8)
    # Use _set_int64 (the correct API for int8 tensors)
    t._set_int64(0, 1)
    t._set_int64(1, -1)
    t._set_int64(2, 0)
    # _get_float64 should return the integer value as Float64
    assert_almost_equal(t._get_float64(0), 1.0, tolerance=1e-9)
    assert_almost_equal(t._get_float64(1), -1.0, tolerance=1e-9)
    assert_almost_equal(t._get_float64(2), 0.0, tolerance=1e-9)


fn test_int8_set_float64_is_noop() raises:
    """Int8: _set_float64 silently does nothing (no int8 branch in implementation).

    This documents the known limitation: _set_float64 has no int8 branch, so
    calling it on an int8 tensor is a silent no-op. The value remains unchanged.

    TODO(#3301): Consider adding int8 support to _set_float64 with truncation
    semantics (Float64 -> Int8 cast), or raise an error for unsupported dtypes
    to make this failure mode explicit rather than silent.
    """
    var t = zeros([1], DType.int8)
    t._set_float64(0, 1.0)
    # int8 has no branch in _set_float64 — the write is silently ignored.
    # Value stays 0 (the zero-initialized value).
    var got = t._get_float64(0)
    # Document the current behavior: silent no-op leaves value at 0
    assert_almost_equal(got, 0.0, tolerance=1e-9)


fn test_dtype_sizes() raises:
    """Audit _get_dtype_size_static returns correct byte sizes for each dtype.

    This test catches dtype size bugs (like bfloat16 returning 4 instead of 2)
    at compile time as a regression test.
    """
    assert_equal(
        ExTensor._get_dtype_size_static(DType.float16),
        2,
        "float16 should be 2 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.bfloat16),
        2,
        "bfloat16 should be 2 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.float32),
        4,
        "float32 should be 4 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.float64),
        8,
        "float64 should be 8 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.int8),
        1,
        "int8 should be 1 byte",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.uint8),
        1,
        "uint8 should be 1 byte",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.int16),
        2,
        "int16 should be 2 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.uint16),
        2,
        "uint16 should be 2 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.int32),
        4,
        "int32 should be 4 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.uint32),
        4,
        "uint32 should be 4 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.int64),
        8,
        "int64 should be 8 bytes",
    )
    assert_equal(
        ExTensor._get_dtype_size_static(DType.uint64),
        8,
        "uint64 should be 8 bytes",
    )


fn main() raises:
    """Run all dtype round-trip tests for _set_float64/_get_float64."""
    print("Running dtype size tests...")
    test_dtype_sizes()
    print("✓ test_dtype_sizes")

    print("Running float16 round-trip tests...")
    test_float16_set_get_float64_roundtrip()
    test_float16_nonzero_after_set()

    print("Running float32 round-trip tests...")
    test_float32_set_get_float64_roundtrip()
    test_float32_nonzero_after_set()

    print("Running float64 round-trip tests...")
    test_float64_set_get_float64_roundtrip()
    test_float64_nonzero_after_set()

    print("Running bfloat16 round-trip tests (fixed in #3301)...")
    test_bfloat16_set_get_float64_roundtrip()
    test_bfloat16_nonzero_after_set()
    test_bfloat16_dtype_size_is_2_bytes()

    print("Running int8 tests (documents known limitation of _set_float64)...")
    test_int8_get_float64_via_int64_path()
    test_int8_set_float64_is_noop()

    print("All dtype round-trip tests passed!")
