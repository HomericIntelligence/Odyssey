"""Tests for ExTensor __hash__ NaN stability (issue #3382).

Verifies that different NaN bit patterns (quiet NaN, signaling NaN, negative NaN,
different NaN payloads) all produce the same hash, ensuring deterministic hash
behavior for tensors containing NaN values.
"""

from memory import UnsafePointer
from shared.core.extensor import ExTensor, zeros, ones, full, arange, nan_tensor
from tests.shared.conftest import (
    assert_equal_int,
)


# ============================================================================
# Helpers: inject specific NaN bit patterns into a tensor
# ============================================================================


fn make_f32_nan_tensor(bits: UInt32) raises -> ExTensor:
    """Create a scalar float32 tensor whose single element has the given raw bits.

    Args:
        bits: The IEEE 754 bit pattern to store. Must represent a NaN.

    Returns:
        A scalar (0-D) ExTensor with DType.float32 containing that bit pattern.
    """
    var shape = List[Int]()
    var t = ExTensor(shape, DType.float32)
    t._data.bitcast[UInt32]()[] = bits
    return t^


fn make_f64_nan_tensor(bits: UInt64) raises -> ExTensor:
    """Create a scalar float64 tensor whose single element has the given raw bits.

    Args:
        bits: The IEEE 754 bit pattern to store. Must represent a NaN.

    Returns:
        A scalar (0-D) ExTensor with DType.float64 containing that bit pattern.
    """
    var shape = List[Int]()
    var t = ExTensor(shape, DType.float64)
    t._data.bitcast[UInt64]()[] = bits
    return t^


fn make_f16_nan_tensor(bits: UInt16) raises -> ExTensor:
    """Create a scalar float16 tensor whose single element has the given raw bits.

    Args:
        bits: The IEEE 754 bit pattern to store. Must represent a NaN.

    Returns:
        A scalar (0-D) ExTensor with DType.float16 containing that bit pattern.
    """
    var shape = List[Int]()
    var t = ExTensor(shape, DType.float16)
    t._data.bitcast[UInt16]()[] = bits
    return t^


fn assert_0d_tensor_safe_allocation(dtype: DType) raises:
    """Verify that 0-D tensors allocate sufficient memory for bitcast writes.

    A 0-D tensor (empty shape List) still has numel=1, meaning it must
    allocate at least sizeof(dtype) bytes so that bitcast writes and reads
    are both safe. This assertion confirms the constructor does this.

    Args:
        dtype: The data type to verify allocation for.

    Closes #4063.
    """
    var shape = List[Int]()
    var t = ExTensor(shape, dtype)

    # Attempt to write via bitcast. If allocation is insufficient,
    # this would cause a segfault or memory corruption.
    if dtype == DType.float32:
        t._data.bitcast[UInt32]()[] = 0x7FC00000  # quiet NaN bits
        var readback = t._data.bitcast[UInt32]()[0]
        assert_equal_int(
            Int(readback), Int(0x7FC00000), "f32 0-D tensor bitcast write/read mismatch"
        )
    elif dtype == DType.float64:
        t._data.bitcast[UInt64]()[] = 0x7FF8000000000000  # quiet NaN bits
        var readback = t._data.bitcast[UInt64]()[0]
        assert_equal_int(
            Int(readback), Int(0x7FF8000000000000), "f64 0-D tensor bitcast write/read mismatch"
        )
    elif dtype == DType.float16:
        t._data.bitcast[UInt16]()[] = 0x7E00  # quiet NaN bits
        var readback = t._data.bitcast[UInt16]()[0]
        assert_equal_int(
            Int(readback), Int(0x7E00), "f16 0-D tensor bitcast write/read mismatch"
        )


# ============================================================================
# Test: normal values hash consistently
# ============================================================================


fn test_hash_normal_values() raises:
    """Equal tensors yield equal hashes."""
    var a = arange(0.0, 4.0, 1.0, DType.float32)
    var b = arange(0.0, 4.0, 1.0, DType.float32)
    assert_equal_int(
        Int(hash(a)), Int(hash(b)), "Equal tensors must hash equal"
    )


fn test_hash_different_values() raises:
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


fn test_hash_f32_quiet_nan_equals_negative_nan() raises:
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


fn test_hash_f32_nan_payload_irrelevant() raises:
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


fn test_hash_f32_signaling_nan_equals_quiet_nan() raises:
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


fn test_hash_f64_nan_sign_irrelevant() raises:
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


fn test_hash_f64_nan_payload_irrelevant() raises:
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


fn test_hash_f64_signaling_nan_equals_quiet_nan() raises:
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


fn test_hash_f16_nan_sign_irrelevant() raises:
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


fn test_hash_f16_nan_payload_irrelevant() raises:
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
# Test: 0-D tensor safe allocation for bitcast operations
# ============================================================================


fn test_hash_0d_scalar_tensor_allocation() raises:
    """Verify 0-D tensors allocate sufficient memory for NaN bitcast operations.

    0-D tensors (empty shape List) have numel=1, so the ExTensor constructor
    must allocate at least sizeof(dtype) bytes. Otherwise, the bitcast write
    and read in the NaN canonicalization tests and helper functions would
    cause segfaults or memory corruption.

    Closes #4063.
    """
    assert_0d_tensor_safe_allocation(DType.float32)
    assert_0d_tensor_safe_allocation(DType.float64)
    assert_0d_tensor_safe_allocation(DType.float16)


# ============================================================================
# Test: mixed NaN and normal values hash deterministically
# ============================================================================


fn test_hash_mixed_nan_normal_deterministic() raises:
    """A tensor mixing NaN and normal values hashes the same across calls."""
    var shape = List[Int]()
    shape.append(3)
    var a = nan_tensor(shape, DType.float32)
    # Set non-NaN elements
    a._data.bitcast[Float32]()[1] = Float32(1.0)
    a._data.bitcast[Float32]()[2] = Float32(2.0)

    var b = nan_tensor(shape, DType.float32)
    b._data.bitcast[Float32]()[1] = Float32(1.0)
    b._data.bitcast[Float32]()[2] = Float32(2.0)

    assert_equal_int(
        Int(hash(a)),
        Int(hash(b)),
        "Mixed NaN/normal tensor must hash consistently",
    )


fn test_hash_mixed_nan_different_nan_patterns() raises:
    """Tensors with same logical content but different NaN bit patterns hash equal.
    """
    var shape = List[Int]()
    shape.append(2)

    # First tensor: element 0 = positive quiet NaN, element 1 = 1.0
    var a = ExTensor(shape, DType.float32)
    a._data.bitcast[UInt32]()[0] = UInt32(0x7FC00000)  # +qNaN
    a._data.bitcast[Float32]()[1] = Float32(1.0)

    # Second tensor: element 0 = negative quiet NaN, element 1 = 1.0
    var b = ExTensor(shape, DType.float32)
    b._data.bitcast[UInt32]()[0] = UInt32(0xFFC00000)  # -qNaN
    b._data.bitcast[Float32]()[1] = Float32(1.0)

    assert_equal_int(
        Int(hash(a)),
        Int(hash(b)),
        "Tensors differing only in NaN bit pattern must hash equal",
    )


# ============================================================================
# Test: shape and dtype sensitivity
# ============================================================================


fn test_hash_shape_sensitivity() raises:
    """Tensors with different shapes hash differently."""
    var shape_a = List[Int]()
    shape_a.append(2)
    var shape_b = List[Int]()
    shape_b.append(3)
    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)
    if hash(a) == hash(b):
        raise Error("Tensors with different shapes should not collide on hash")


fn test_hash_dtype_sensitivity() raises:
    """Tensors with same shape/values but different dtypes hash differently."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 1.0, DType.float64)
    if hash(a) == hash(b):
        raise Error("Tensors with different dtypes should not collide on hash")


fn test_hash_int_vs_float_same_numeric_value() raises:
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
            "int32(1) and float32(1.0) should hash differently (dtype/kind collision)"
        )


# ============================================================================
# Test: integer types unaffected (no NaN concern)
# ============================================================================


fn test_hash_integer_types_consistent() raises:
    """Integer tensors hash consistently (no NaN issue for integer dtypes)."""
    var shape = List[Int]()
    shape.append(4)
    var a = arange(0.0, 4.0, 1.0, DType.int32)
    var b = arange(0.0, 4.0, 1.0, DType.int32)
    assert_equal_int(
        Int(hash(a)), Int(hash(b)), "Equal int32 tensors must hash equal"
    )


fn test_hash_integer_dtype_distinct() raises:
    """Integer tensors with different values produce different hashes. Closes #4062."""
    var shape = List[Int]()
    shape.append(4)
    var a = arange(0.0, 4.0, 1.0, DType.int32)
    var b = arange(1.0, 5.0, 1.0, DType.int32)
    if hash(a) == hash(b):
        raise Error("Different-valued int32 tensors should not collide on hash")


# ============================================================================
# Entry point
# ============================================================================


fn main() raises:
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

    print("  test_hash_0d_scalar_tensor_allocation...")
    test_hash_0d_scalar_tensor_allocation()

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

    print("All NaN hash stability tests passed!")
