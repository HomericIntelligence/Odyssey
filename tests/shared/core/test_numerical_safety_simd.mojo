# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for SIMD-vectorized numerical safety functions.

Validates has_nan, has_inf, count_nan, count_inf with SIMD edge cases:
- Tensors smaller than SIMD width (pure scalar tail path)
- NaN/Inf at SIMD boundary positions
- NaN/Inf in scalar tail region
- Large tensors exercising full SIMD loop
- Early exit behavior for has_nan/has_inf
"""

from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    full,
)
from shared.core.numerical_safety import (
    has_nan,
    has_inf,
    count_nan,
    count_inf,
)
from std.math import nan, inf
from std.collections import List


def _shape1(n: Int) -> List[Int]:
    """Create a 1D shape list."""
    var s = List[Int]()
    s.append(n)
    return s^


def _shape2(r: Int, c: Int) -> List[Int]:
    """Create a 2D shape list."""
    var s = List[Int]()
    s.append(r)
    s.append(c)
    return s^


def test_has_nan_all_normal() raises:
    """Verify has_nan returns False for tensor with all normal values."""
    print("Testing has_nan with all normal values...")
    var tensor = full(_shape2(2, 8), 1.5, DType.float32)
    if has_nan(tensor):
        raise Error("has_nan should return False for all-normal tensor")
    print("  PASS")


def test_has_nan_first_element() raises:
    """Detect NaN at index 0 (first SIMD chunk, early exit)."""
    print("Testing has_nan with NaN at first element...")
    var tensor = full(_shape1(32), 1.0, DType.float32)
    var ptr = tensor._data.bitcast[Float32]()
    ptr[0] = nan[DType.float32]()
    if not has_nan(tensor):
        raise Error("has_nan should detect NaN at index 0")
    print("  PASS")


def test_has_nan_tail_element() raises:
    """Detect NaN in scalar tail (last element, size not divisible by SIMD width)."""
    print("Testing has_nan with NaN in scalar tail...")
    # Size 19: with SIMD width 8, tail is elements 16..18
    var tensor = full(_shape1(19), 1.0, DType.float32)
    var ptr = tensor._data.bitcast[Float32]()
    ptr[18] = nan[DType.float32]()
    if not has_nan(tensor):
        raise Error("has_nan should detect NaN in scalar tail")
    print("  PASS")


def test_has_nan_small_tensor() raises:
    """Detect NaN in tensor smaller than SIMD width (pure scalar path)."""
    print("Testing has_nan with small tensor (< SIMD width)...")
    var tensor = full(_shape1(3), 0.5, DType.float32)
    var ptr = tensor._data.bitcast[Float32]()
    ptr[1] = nan[DType.float32]()
    if not has_nan(tensor):
        raise Error("has_nan should detect NaN in small tensor")
    print("  PASS")


def test_has_inf_positive_and_negative() raises:
    """Detect both positive and negative infinity."""
    print("Testing has_inf with +Inf and -Inf...")

    # Test +Inf
    var t1 = full(_shape1(16), 1.0, DType.float32)
    var p1 = t1._data.bitcast[Float32]()
    p1[7] = inf[DType.float32]()
    if not has_inf(t1):
        raise Error("has_inf should detect +Inf")

    # Test -Inf
    var t2 = full(_shape1(16), 1.0, DType.float32)
    var p2 = t2._data.bitcast[Float32]()
    p2[15] = -inf[DType.float32]()
    if not has_inf(t2):
        raise Error("has_inf should detect -Inf")

    # Test clean tensor
    var t3 = full(_shape1(16), 1.0, DType.float32)
    if has_inf(t3):
        raise Error("has_inf should return False for clean tensor")
    print("  PASS")


def test_count_nan_mixed() raises:
    """Count NaNs scattered across SIMD chunks and scalar tail."""
    print("Testing count_nan with scattered NaNs...")
    var tensor = full(_shape1(20), 1.0, DType.float32)
    var ptr = tensor._data.bitcast[Float32]()
    # Place NaNs at indices 0, 5, 10, 19 (across SIMD chunks + tail)
    ptr[0] = nan[DType.float32]()
    ptr[5] = nan[DType.float32]()
    ptr[10] = nan[DType.float32]()
    ptr[19] = nan[DType.float32]()
    var count = count_nan(tensor)
    if count != 4:
        raise Error("count_nan expected 4, got " + String(count))
    print("  PASS")


def test_count_inf_mixed() raises:
    """Count both +Inf and -Inf correctly."""
    print("Testing count_inf with mixed Inf values...")
    var tensor = full(_shape1(20), 0.5, DType.float32)
    var ptr = tensor._data.bitcast[Float32]()
    ptr[3] = inf[DType.float32]()
    ptr[9] = -inf[DType.float32]()
    ptr[17] = inf[DType.float32]()
    var count = count_inf(tensor)
    if count != 3:
        raise Error("count_inf expected 3, got " + String(count))
    print("  PASS")


def test_has_nan_float64() raises:
    """Verify NaN detection works with float64 dtype."""
    print("Testing has_nan with float64...")
    var tensor = full(_shape1(10), 1.0, DType.float64)
    var ptr = tensor._data.bitcast[Float64]()
    ptr[7] = nan[DType.float64]()
    if not has_nan(tensor):
        raise Error("has_nan should detect NaN in float64 tensor")

    # Clean tensor
    var clean = full(_shape1(10), -0.5, DType.float64)
    if has_nan(clean):
        raise Error("has_nan should return False for clean float64 tensor")
    print("  PASS")


def test_count_nan_large_tensor() raises:
    """Validate count_nan on large tensor exercising full SIMD path."""
    print("Testing count_nan with large tensor (1024 elements)...")
    var tensor = full(_shape1(1024), 1.0, DType.float32)
    var ptr = tensor._data.bitcast[Float32]()
    # Place 5 NaNs at various positions
    ptr[0] = nan[DType.float32]()
    ptr[100] = nan[DType.float32]()
    ptr[512] = nan[DType.float32]()
    ptr[1000] = nan[DType.float32]()
    ptr[1023] = nan[DType.float32]()
    var count = count_nan(tensor)
    if count != 5:
        raise Error("count_nan expected 5, got " + String(count))
    print("  PASS")


def main() raises:
    test_has_nan_all_normal()
    test_has_nan_first_element()
    test_has_nan_tail_element()
    test_has_nan_small_tensor()
    test_has_inf_positive_and_negative()
    test_count_nan_mixed()
    test_count_inf_mixed()
    test_has_nan_float64()
    test_count_nan_large_tensor()
    print("All numerical safety SIMD tests passed!")
