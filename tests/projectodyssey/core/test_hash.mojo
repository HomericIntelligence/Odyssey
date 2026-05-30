"""Tests for AnyTensor __hash__ NaN stability (issue #3382).

Verifies that different NaN bit patterns (quiet NaN, signaling NaN, negative NaN,
different NaN payloads) all produce the same hash, ensuring deterministic hash
behavior for tensors containing NaN values.
"""

from std.memory import UnsafePointer
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import arange, full, nan_tensor, ones, zeros
from tests.projectodyssey.conftest import (
    assert_equal_int,
)


# ============================================================================
# Helpers: inject specific NaN bit patterns into a tensor
# ============================================================================


def make_f32_nan_tensor(bits: UInt32) raises -> AnyTensor:
    """Create a scalar float32 tensor whose single element has the given raw bits.

    Args:
        bits: The IEEE 754 bit pattern to store. Must represent a NaN.

    Returns:
        A scalar (0-D) AnyTensor with DType.float32 containing that bit pattern.
    """
    var shape = List[Int]()
    var t = AnyTensor(shape, DType.float32)
    t.set(0, UInt32(bits))
    return t^


def make_f64_nan_tensor(bits: UInt64) raises -> AnyTensor:
    """Create a scalar float64 tensor whose single element has the given raw bits.

    Args:
        bits: The IEEE 754 bit pattern to store. Must represent a NaN.

    Returns:
        A scalar (0-D) AnyTensor with DType.float64 containing that bit pattern.
    """
    var shape = List[Int]()
    var t = AnyTensor(shape, DType.float64)
    t.set(0, UInt64(bits))
    return t^


def make_f16_nan_tensor(bits: UInt16) raises -> AnyTensor:
    """Create a scalar float16 tensor whose single element has the given raw bits.

    Args:
        bits: The IEEE 754 bit pattern to store. Must represent a NaN.

    Returns:
        A scalar (0-D) AnyTensor with DType.float16 containing that bit pattern.
    """
    var shape = List[Int]()
    var t = AnyTensor(shape, DType.float16)
    t.set(0, UInt16(bits))
    return t^


# ============================================================================
# Test: normal values hash consistently
# ============================================================================


def test_hash_normal_values() raises:
    """Equal tensors yield equal hashes."""
    var a = arange(0.0, 4.0, 1.0, DType.float32)
    var b = arange(0.0, 4.0, 1.0, DType.float32)
    assert_equal_int(
        Int(hash(a)), Int(hash(b)), "Equal tensors must hash equal"
    )


def test_hash_different_values() raises:
    """Different-valued tensors should produce different hashes (probabilistic).
    """
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float64)
    var b = full(shape, 2.0, DType.float64)
    if hash(a) == hash(b):
        raise Error("Different-valued tensors should not collide on hash")


# ============================================================================
# Test: NaN canonicalization — float32
# ============================================================================


def test_hash_f32_quiet_nan_equals_negative_nan() raises:
    """Positive and negative quiet NaN (float32) hash identically.

    Positive quiet NaN:  0x7FC00000
    Negative quiet NaN:  0xFFC00000  (sign bit = 1)
    """
    # Positive quiet NaN (canonical)
    var pos_nan = make_f32_nan_tensor(UInt32(0x7FC00000))
    # Negative quiet NaN (sign bit flipped)
    var neg_nan = make_f32_nan_tensor(UInt32(0xFFC00000))
    assert_equal_int(
        Int(hash(pos_nan)),
        Int(hash(neg_nan)),
        "Positive and negative quiet NaN (f32) must hash equal",
    )


def test_hash_f32_nan_payload_irrelevant() raises:
    """Two float32 NaNs with different mantissa payloads hash identically.

    NaN with payload 1:  0x7FC00001
    NaN with payload 2:  0x7FC00002
    """
    var nan_a = make_f32_nan_tensor(UInt32(0x7FC00001))
    var nan_b = make_f32_nan_tensor(UInt32(0x7FC00002))
    assert_equal_int(
        Int(hash(nan_a)),
        Int(hash(nan_b)),
        "NaN payloads must not affect hash (f32)",
    )


def test_hash_f32_signaling_nan_equals_quiet_nan() raises:
    """Signaling NaN (f32) hashes the same as quiet NaN.

    Quiet NaN:     0x7FC00000  (MSB of mantissa = 1)
    Signaling NaN: 0x7F800001  (MSB of mantissa = 0, payload = 1)
    """
    var quiet = make_f32_nan_tensor(UInt32(0x7FC00000))
    var signaling = make_f32_nan_tensor(UInt32(0x7F800001))
    assert_equal_int(
        Int(hash(quiet)),
        Int(hash(signaling)),
        "Signaling and quiet NaN (f32) must hash equal",
    )


# ============================================================================
# Test: NaN canonicalization — float64
# ============================================================================


def test_hash_f64_nan_sign_irrelevant() raises:
    """Positive and negative quiet NaN (float64) hash identically.

    Positive quiet NaN:  0x7FF8000000000000
    Negative quiet NaN:  0xFFF8000000000000
    """
    var pos_nan = make_f64_nan_tensor(UInt64(0x7FF8000000000000))
    var neg_nan = make_f64_nan_tensor(UInt64(0xFFF8000000000000))
    assert_equal_int(
        Int(hash(pos_nan)),
        Int(hash(neg_nan)),
        "Positive and negative quiet NaN (f64) must hash equal",
    )


def test_hash_f64_nan_payload_irrelevant() raises:
    """Two float64 NaNs with different payloads hash identically.

    NaN with payload 1:  0x7FF8000000000001
    NaN with payload 2:  0x7FF8000000000002
    """
    var nan_a = make_f64_nan_tensor(UInt64(0x7FF8000000000001))
    var nan_b = make_f64_nan_tensor(UInt64(0x7FF8000000000002))
    assert_equal_int(
        Int(hash(nan_a)),
        Int(hash(nan_b)),
        "NaN payloads must not affect hash (f64)",
    )


def test_hash_f64_signaling_nan_equals_quiet_nan() raises:
    """Signaling NaN (f64) hashes the same as quiet NaN.

    Quiet NaN:     0x7FF8000000000000
    Signaling NaN: 0x7FF0000000000001
    """
    var quiet = make_f64_nan_tensor(UInt64(0x7FF8000000000000))
    var signaling = make_f64_nan_tensor(UInt64(0x7FF0000000000001))
    assert_equal_int(
        Int(hash(quiet)),
        Int(hash(signaling)),
        "Signaling and quiet NaN (f64) must hash equal",
    )


# ============================================================================
# Test: NaN canonicalization — float16
# ============================================================================


def test_hash_f16_nan_sign_irrelevant() raises:
    """Positive and negative quiet NaN (float16) hash identically.

    Positive quiet NaN (f16):  0x7E00
    Negative quiet NaN (f16):  0xFE00
    """
    var pos_nan = make_f16_nan_tensor(UInt16(0x7E00))
    var neg_nan = make_f16_nan_tensor(UInt16(0xFE00))
    assert_equal_int(
        Int(hash(pos_nan)),
        Int(hash(neg_nan)),
        "Positive and negative quiet NaN (f16) must hash equal",
    )


def test_hash_f16_nan_payload_irrelevant() raises:
    """Two float16 NaNs with different payloads hash identically.

    NaN with payload 1:  0x7E01
    NaN with payload 2:  0x7E02
    """
    var nan_a = make_f16_nan_tensor(UInt16(0x7E01))
    var nan_b = make_f16_nan_tensor(UInt16(0x7E02))
    assert_equal_int(
        Int(hash(nan_a)),
        Int(hash(nan_b)),
        "NaN payloads must not affect hash (f16)",
    )


# ============================================================================
# Test: mixed NaN and normal values hash deterministically
# ============================================================================


def test_hash_mixed_nan_normal_deterministic() raises:
    """A tensor mixing NaN and normal values hashes the same across calls."""
    var shape = List[Int]()
    shape.append(3)
    var a = nan_tensor(shape, DType.float32)
    # Set non-NaN elements
    a.set(1, Float32(Float32(1.0)))
    a.set(2, Float32(Float32(2.0)))

    var b = nan_tensor(shape, DType.float32)
    b.set(1, Float32(Float32(1.0)))
    b.set(2, Float32(Float32(2.0)))

    assert_equal_int(
        Int(hash(a)),
        Int(hash(b)),
        "Mixed NaN/normal tensor must hash consistently",
    )


def test_hash_mixed_nan_different_nan_patterns() raises:
    """Tensors with same logical content but different NaN bit patterns hash equal.
    """
    var shape = List[Int]()
    shape.append(2)

    # First tensor: element 0 = positive quiet NaN, element 1 = 1.0
    var a = AnyTensor(shape, DType.float32)
    a.set(0, UInt32(UInt32(0x7FC00000)))  # +qNaN
    a.set(1, Float32(Float32(1.0)))

    # Second tensor: element 0 = negative quiet NaN, element 1 = 1.0
    var b = AnyTensor(shape, DType.float32)
    b.set(0, UInt32(UInt32(0xFFC00000)))  # -qNaN
    b.set(1, Float32(Float32(1.0)))

    assert_equal_int(
        Int(hash(a)),
        Int(hash(b)),
        "Tensors differing only in NaN bit pattern must hash equal",
    )


# ============================================================================
# Test: shape and dtype sensitivity
# ============================================================================


def test_hash_shape_sensitivity() raises:
    """Tensors with different shapes hash differently."""
    var shape_a = List[Int]()
    shape_a.append(2)
    var shape_b = List[Int]()
    shape_b.append(3)
    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)
    if hash(a) == hash(b):
        raise Error("Tensors with different shapes should not collide on hash")


def test_hash_dtype_sensitivity() raises:
    """Tensors with same shape/values but different dtypes hash differently."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 1.0, DType.float64)
    if hash(a) == hash(b):
        raise Error("Tensors with different dtypes should not collide on hash")


def test_hash_int_vs_float_same_numeric_value() raises:
    """Tensors with same numeric value but different dtype/kind hash differently.

    Tests cross-kind collision: int32(1) vs float32(1.0) must hash differently,
    even though they represent the same numeric value.
    """
    var shape = List[Int]()
    shape.append(1)
    var int_tensor = full(shape, 1.0, DType.int32)  # 1 as int32
    var float_tensor = full(shape, 1.0, DType.float32)  # 1.0 as float32
    if hash(int_tensor) == hash(float_tensor):
        raise Error(
            "int32(1) and float32(1.0) should hash differently (dtype/kind"
            " collision)"
        )


# ============================================================================
# Test: integer types unaffected (no NaN concern)
# ============================================================================


def test_hash_integer_types_consistent() raises:
    """Integer tensors hash consistently (no NaN issue for integer dtypes)."""
    var shape = List[Int]()
    shape.append(4)
    var a = arange(0.0, 4.0, 1.0, DType.int32)
    var b = arange(0.0, 4.0, 1.0, DType.int32)
    assert_equal_int(
        Int(hash(a)), Int(hash(b)), "Equal int32 tensors must hash equal"
    )


def test_hash_integer_dtype_distinct() raises:
    """Integer tensors with different values should produce different hashes.

    Verifies that two int32 tensors with different numeric values produce
    distinct hashes, similar to test_hash_different_values for float types.
    Closes #4062.
    """
    var shape = List[Int]()
    shape.append(4)
    var a = arange(0.0, 4.0, 1.0, DType.int32)  # [0, 1, 2, 3]
    var b = arange(1.0, 5.0, 1.0, DType.int32)  # [1, 2, 3, 4]
    if hash(a) == hash(b):
        raise Error(
            "Integer tensors with different values should not collide on hash"
        )


# ============================================================================
# Test: empty tensor hash behavior (shape and dtype sensitivity)
# ============================================================================


def test_hash_empty_tensor_base() raises:
    """Empty tensor with single 0-dimension hashes consistently."""
    var shape = List[Int]()
    shape.append(0)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    assert_equal_int(
        Int(hash(a)), Int(hash(b)), "Equal empty tensors must hash equal"
    )


def test_hash_empty_tensor_different_shapes() raises:
    """Empty tensors with different shapes produce different hashes. Closes #4067.
    """
    var shape1 = List[Int]()
    shape1.append(0)
    var a = zeros(shape1, DType.float32)

    var shape2 = List[Int]()
    shape2.append(0)
    shape2.append(0)
    var b = zeros(shape2, DType.float32)

    if hash(a) == hash(b):
        raise Error(
            "Empty tensors with different shapes should not collide on hash"
        )


def test_hash_empty_tensor_different_dtypes() raises:
    """Empty tensors with different dtypes produce different hashes. Closes #4068.
    """
    var shape = List[Int]()
    shape.append(0)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float64)

    if hash(a) == hash(b):
        raise Error(
            "Empty tensors with different dtypes should not collide on hash"
        )


def test_hash_empty_tensor_stability() raises:
    """Empty tensor hash is stable across repeated calls. Closes #4069."""
    var shape = List[Int]()
    shape.append(0)
    var a = zeros(shape, DType.float32)

    var hash1 = hash(a)
    var hash2 = hash(a)
    var hash3 = hash(a)

    assert_equal_int(
        Int(hash1), Int(hash2), "Empty tensor hash must be stable (call 1 vs 2)"
    )
    assert_equal_int(
        Int(hash2), Int(hash3), "Empty tensor hash must be stable (call 2 vs 3)"
    )


# ============================================================================
# Test: hash collision with same dtype, different shapes
# ============================================================================


def test_hash_same_dtype_different_shapes() raises:
    """Tensors with same dtype/values but different shapes hash differently."""
    var shape1 = List[Int]()
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(2)
    shape2.append(2)
    # Both have 4 elements with value 1.0, but different shapes
    var a = ones(shape1, DType.float32)
    var b = ones(shape2, DType.float32)
    if hash(a) == hash(b):
        raise Error(
            "Tensors with same dtype/values but different shapes should not"
            " collide on hash"
        )


# ============================================================================
# Test: NaN canonicalization with 0-D scalar tensors
# ============================================================================


def test_hash_nan_canonicalization_scalar() raises:
    """0-D scalar NaN tensors hash the same regardless of bit pattern.

    0-D tensors (scalars) should still canonicalize NaN properly.
    """
    # Positive quiet NaN
    var shape = List[Int]()
    var pos_nan = AnyTensor(shape, DType.float32)
    pos_nan.set(0, UInt32(UInt32(0x7FC00000)))

    # Negative quiet NaN
    var neg_nan = AnyTensor(shape, DType.float32)
    neg_nan.set(0, UInt32(UInt32(0xFFC00000)))

    assert_equal_int(
        Int(hash(pos_nan)),
        Int(hash(neg_nan)),
        "0-D scalar NaN tensors must canonicalize regardless of sign",
    )


# ============================================================================
# Entry point
# ============================================================================


def main() raises:
    print("Running NaN hash stability tests (issue #3382)...")

    print("  test_hash_normal_values...")
    test_hash_normal_values()

    print("  test_hash_different_values...")
    test_hash_different_values()

    print("  test_hash_f32_quiet_nan_equals_negative_nan...")
    test_hash_f32_quiet_nan_equals_negative_nan()

    print("  test_hash_f32_nan_payload_irrelevant...")
    test_hash_f32_nan_payload_irrelevant()

    print("  test_hash_f32_signaling_nan_equals_quiet_nan...")
    test_hash_f32_signaling_nan_equals_quiet_nan()

    print("  test_hash_f64_nan_sign_irrelevant...")
    test_hash_f64_nan_sign_irrelevant()

    print("  test_hash_f64_nan_payload_irrelevant...")
    test_hash_f64_nan_payload_irrelevant()

    print("  test_hash_f64_signaling_nan_equals_quiet_nan...")
    test_hash_f64_signaling_nan_equals_quiet_nan()

    print("  test_hash_f16_nan_sign_irrelevant...")
    test_hash_f16_nan_sign_irrelevant()

    print("  test_hash_f16_nan_payload_irrelevant...")
    test_hash_f16_nan_payload_irrelevant()

    print("  test_hash_mixed_nan_normal_deterministic...")
    test_hash_mixed_nan_normal_deterministic()

    print("  test_hash_mixed_nan_different_nan_patterns...")
    test_hash_mixed_nan_different_nan_patterns()

    print("  test_hash_shape_sensitivity...")
    test_hash_shape_sensitivity()

    print("  test_hash_dtype_sensitivity...")
    test_hash_dtype_sensitivity()

    print("  test_hash_int_vs_float_same_numeric_value...")
    test_hash_int_vs_float_same_numeric_value()

    print("  test_hash_integer_types_consistent...")
    test_hash_integer_types_consistent()

    print("  test_hash_integer_dtype_distinct...")
    test_hash_integer_dtype_distinct()

    print("  test_hash_empty_tensor_base...")
    test_hash_empty_tensor_base()

    print("  test_hash_empty_tensor_different_shapes...")
    test_hash_empty_tensor_different_shapes()

    print("  test_hash_empty_tensor_different_dtypes...")
    test_hash_empty_tensor_different_dtypes()

    print("  test_hash_empty_tensor_stability...")
    test_hash_empty_tensor_stability()

    print("  test_hash_same_dtype_different_shapes...")
    test_hash_same_dtype_different_shapes()

    print("  test_hash_nan_canonicalization_scalar...")
    test_hash_nan_canonicalization_scalar()

    print("All NaN hash stability tests passed!")
