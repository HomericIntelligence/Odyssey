"""Example demonstrating the usage of Mojo's built-in integer types.

This example shows how to use Int8, Int16, Int32, Int64, UInt8, UInt16,
UInt32, and UInt64 built-in types, including:
- Creating integer values
- Converting between types
- Performing arithmetic operations
- Converting tensors to/from integer types
- Handling overflow and type casting
"""

from shared.tensor.any_tensor import AnyTensor, zeros


fn example_basic_signed_integers() raises:
    """Demonstrate basic signed integer operations."""
    print("\n=== Basic Signed Integer Operations ===\n")

    # Create Int8 values
    var i8_1 = Int8(42)
    var i8_2 = Int8(-100)
    print("Int8 values:", i8_1, i8_2)

    # Arithmetic operations
    var sum = i8_1 + Int8(10)
    var diff = i8_1 - Int8(5)
    var prod = i8_1 * Int8(2)
    print("Int8 arithmetic: 42+10=", sum, " 42-5=", diff, " 42*2=", prod)

    # Comparisons
    if i8_1 > i8_2:
        print("42 > -100: True")


fn example_basic_unsigned_integers() raises:
    """Demonstrate basic unsigned integer operations."""
    print("\n=== Basic Unsigned Integer Operations ===\n")

    # Create UInt8 values
    var u8_1 = UInt8(255)
    var u8_2 = UInt8(100)
    print("UInt8 values:", u8_1, u8_2)

    # Arithmetic operations
    var sum = u8_2 + UInt8(50)
    var diff = u8_1 - u8_2
    print("UInt8 arithmetic: 100+50=", sum, " 255-100=", diff)


fn example_type_conversions() raises:
    """Demonstrate conversions between integer types."""
    print("\n=== Type Conversions ===\n")

    # Int8 to larger types (lossless via cast)
    var i8 = Int8(42)
    var i16 = Int16(i8)
    var i32 = Int32(i8)
    var i64 = Int64(i8)
    print("Int8(42) -> Int16:", i16, " Int32:", i32, " Int64:", i64)

    # Larger types to Int8 (truncation may occur)
    var large_i32 = Int32(1000)
    var back_to_i8 = Int8(large_i32)
    print("Int32(1000) -> Int8:", back_to_i8, " (truncated)")

    # Float conversions
    var float_val = Float32(42.7)
    var from_float = Int8(float_val)
    var to_float = Float32(from_float)
    print("Float32(42.7) -> Int8:", from_float, " -> Float32:", to_float)


fn example_tensor_conversions() raises:
    """Demonstrate tensor conversions with integer types."""
    print("\n=== Tensor Conversions ===\n")

    # Create a float32 tensor
    var t = zeros([2, 3], DType.float32)

    # Set some values
    t.set(0, Float32(10.5))
    t.set(1, Float32(-20.3))
    t.set(2, Float32(200.0))
    t.set(3, Float32(-128.5))
    t.set(4, Float32(0.0))
    t.set(5, Float32(42.9))

    print("Original float32 tensor values:")
    print("  [10.5, -20.3, 200.0, -128.5, 0.0, 42.9]")

    # Convert to Int8
    var i8_tensor = t.to_int8()
    print("\nAfter conversion to Int8 (truncated):")
    print(
        " ",
        i8_tensor._data.bitcast[Int8]()[0],
        i8_tensor._data.bitcast[Int8]()[1],
        i8_tensor._data.bitcast[Int8]()[2],
        i8_tensor._data.bitcast[Int8]()[3],
        i8_tensor._data.bitcast[Int8]()[4],
        i8_tensor._data.bitcast[Int8]()[5],
    )

    # Convert to UInt8
    var u8_tensor = t.to_uint8()
    print("\nAfter conversion to UInt8 (clamped to 0-255):")
    print(
        " ",
        u8_tensor._data.bitcast[UInt8]()[0],
        u8_tensor._data.bitcast[UInt8]()[1],
        u8_tensor._data.bitcast[UInt8]()[2],
        u8_tensor._data.bitcast[UInt8]()[3],
        u8_tensor._data.bitcast[UInt8]()[4],
        u8_tensor._data.bitcast[UInt8]()[5],
    )


fn example_16bit_integers() raises:
    """Demonstrate 16-bit integer operations."""
    print("\n=== 16-bit Integer Operations ===\n")

    # Int16
    var i16_1 = Int16(1000)
    var i16_2 = Int16(-500)
    print("Int16 values:", i16_1, i16_2)

    # UInt16
    var u16_1 = UInt16(60000)
    var u16_2 = UInt16(5000)
    print("UInt16 values:", u16_1, u16_2)

    # Arithmetic
    var i16_sum = i16_1 + i16_2
    var u16_sum = u16_1 + u16_2
    print("Int16 sum:", i16_sum, " UInt16 sum:", u16_sum)


fn example_32bit_integers() raises:
    """Demonstrate 32-bit integer operations."""
    print("\n=== 32-bit Integer Operations ===\n")

    # Int32
    var i32 = Int32(1000000)
    print("Int32 value:", i32)

    # UInt32
    var u32 = UInt32(4000000000)
    print("UInt32 value:", u32)

    # Conversions via cast
    var i32_from_i8 = Int32(Int8(42))
    var u32_from_u16 = UInt32(UInt16(1000))
    print("Int32 from Int8(42):", i32_from_i8)
    print("UInt32 from UInt16(1000):", u32_from_u16)


fn example_64bit_integers() raises:
    """Demonstrate 64-bit integer operations."""
    print("\n=== 64-bit Integer Operations ===\n")

    # Int64
    var i64 = Int64(9999999999)
    print("Int64 value:", i64)

    # UInt64
    var u64 = UInt64(18446744073709551615)
    print("UInt64 large value:", u64)

    # Lossless conversions from smaller types
    var i64_from_i32 = Int64(Int32(1000000))
    var u64_from_u32 = UInt64(UInt32(4000000000))
    print("Int64 from Int32:", i64_from_i32)
    print("UInt64 from UInt32:", u64_from_u32)


fn example_practical_use_case() raises:
    """Demonstrate a practical use case: quantization."""
    print("\n=== Practical Use Case: Simple Quantization ===\n")

    print(
        "Scenario: Quantizing a float32 tensor to UInt8 for memory efficiency"
    )

    # Create a float tensor with normalized values (0.0 to 1.0)
    var normalized = zeros([5], DType.float32)
    normalized.set(0, Float32(0.0))
    normalized.set(1, Float32(0.25))
    normalized.set(2, Float32(0.5))
    normalized.set(3, Float32(0.75))
    normalized.set(4, Float32(1.0))

    print("\nOriginal normalized values (0.0 to 1.0):")
    print("  [0.0, 0.25, 0.5, 0.75, 1.0]")

    # Scale to 0-255 range
    var scaled = zeros([5], DType.float32)
    for i in range(5):
        scaled.set(i, Float32(normalized._data.bitcast[Float32]()[i] * 255.0))

    print("\nScaled to 0-255 range:")
    print("  [0.0, 63.75, 127.5, 191.25, 255.0]")

    # Convert to UInt8 (quantized)
    var quantized = scaled.to_uint8()

    print("\nQuantized to UInt8 (truncated):")
    print(
        " ",
        quantized._data.bitcast[UInt8]()[0],
        quantized._data.bitcast[UInt8]()[1],
        quantized._data.bitcast[UInt8]()[2],
        quantized._data.bitcast[UInt8]()[3],
        quantized._data.bitcast[UInt8]()[4],
    )

    # Dequantize back to float
    var dequantized = zeros([5], DType.float32)
    for i in range(5):
        dequantized.set(i, Float32(quantized._data.bitcast[UInt8]()[i]) / 255.0)

    print("\nDequantized back to 0.0-1.0 range:")
    print(
        " ",
        dequantized._data.bitcast[Float32]()[0],
        dequantized._data.bitcast[Float32]()[1],
        dequantized._data.bitcast[Float32]()[2],
        dequantized._data.bitcast[Float32]()[3],
        dequantized._data.bitcast[Float32]()[4],
    )

    print(
        "\nNote: Some precision is lost due to quantization (e.g., 0.25 ->"
        " 0.24901961)"
    )


fn main() raises:
    """Run all integer type examples."""
    print("=" * 60)
    print("Mojo Built-in Integer Types - Usage Examples")
    print("=" * 60)

    example_basic_signed_integers()
    example_basic_unsigned_integers()
    example_type_conversions()
    example_16bit_integers()
    example_32bit_integers()
    example_64bit_integers()
    example_tensor_conversions()
    example_practical_use_case()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")
