# shared/export/protobuf.mojo
"""
Minimal Protocol Buffers encoder for ONNX export.

This module provides low-level protobuf encoding functionality
needed to generate valid ONNX files without external dependencies.

ONNX uses protobuf wire format. This implements the subset needed
for ONNX model serialization.
"""

from memory import bitcast


alias WIRE_VARINT: UInt8 = 0  # int32, int64, uint32, uint64, sint32, sint64, bool, enum
alias WIRE_64BIT: UInt8 = 1  # fixed64, sfixed64, double
alias WIRE_LENGTH_DELIMITED: UInt8 = 2  # string, bytes, embedded messages, packed repeated
alias WIRE_32BIT: UInt8 = 5  # fixed32, sfixed32, float


@fieldwise_init
struct ProtoBuffer(Copyable, Movable):
    """Buffer for building protobuf messages."""

    var data: List[UInt8]

    fn __init__(out self):
        """Initialize empty buffer."""
        self.data = List[UInt8]()

    fn __init__(out self, capacity: Int):
        """Initialize buffer with capacity hint."""
        self.data = List[UInt8]()

    fn size(self) -> Int:
        """Get buffer size in bytes."""
        return len(self.data)

    fn write_byte(mut self, value: UInt8):
        """Write a single byte."""
        self.data.append(value)

    fn write_bytes(mut self, bytes: List[UInt8]):
        """Write multiple bytes."""
        for i in range(len(bytes)):
            self.data.append(bytes[i])

    fn write_varint(mut self, value: UInt64):
        """Write a variable-length integer (unsigned).

        Varints use 7 bits per byte with MSB as continuation flag.
        """
        var v = value
        while v >= 0x80:
            self.write_byte(UInt8((v & 0x7F) | 0x80))
            v >>= 7
        self.write_byte(UInt8(v))

    fn write_signed_varint(mut self, value: Int64):
        """Write a signed varint using ZigZag encoding."""
        # ZigZag encoding: (n << 1) ^ (n >> 63)
        var encoded = UInt64((value << 1) ^ (value >> 63))
        self.write_varint(encoded)

    fn write_fixed32(mut self, value: UInt32):
        """Write a 32-bit value in little-endian."""
        self.write_byte(UInt8(value & 0xFF))
        self.write_byte(UInt8((value >> 8) & 0xFF))
        self.write_byte(UInt8((value >> 16) & 0xFF))
        self.write_byte(UInt8((value >> 24) & 0xFF))

    fn write_fixed64(mut self, value: UInt64):
        """Write a 64-bit value in little-endian."""
        self.write_byte(UInt8(value & 0xFF))
        self.write_byte(UInt8((value >> 8) & 0xFF))
        self.write_byte(UInt8((value >> 16) & 0xFF))
        self.write_byte(UInt8((value >> 24) & 0xFF))
        self.write_byte(UInt8((value >> 32) & 0xFF))
        self.write_byte(UInt8((value >> 40) & 0xFF))
        self.write_byte(UInt8((value >> 48) & 0xFF))
        self.write_byte(UInt8((value >> 56) & 0xFF))

    fn write_float32(mut self, value: Float32):
        """Write a 32-bit float."""
        var float_simd = SIMD[DType.float32, 1](value)
        var bits = bitcast[DType.uint32, 1](float_simd)
        self.write_fixed32(UInt32(bits[0]))

    fn write_float64(mut self, value: Float64):
        """Write a 64-bit float (double)."""
        var float_simd = SIMD[DType.float64, 1](value)
        var bits = bitcast[DType.uint64, 1](float_simd)
        self.write_fixed64(UInt64(bits[0]))

    fn write_tag(mut self, field_number: Int, wire_type: UInt8):
        """Write a field tag.

        Tag = (field_number << 3) | wire_type
        """
        var tag = UInt64((field_number << 3) | Int(wire_type))
        self.write_varint(tag)

    fn write_string(mut self, field_number: Int, value: String):
        """Write a string field."""
        self.write_tag(field_number, WIRE_LENGTH_DELIMITED)
        self.write_varint(UInt64(len(value)))
        for i in range(len(value)):
            self.write_byte(UInt8(ord(value[i])))

    fn write_bytes_field(mut self, field_number: Int, value: List[UInt8]):
        """Write a bytes field."""
        self.write_tag(field_number, WIRE_LENGTH_DELIMITED)
        self.write_varint(UInt64(len(value)))
        self.write_bytes(value)

    fn write_int32(mut self, field_number: Int, value: Int32):
        """Write an int32 field."""
        self.write_tag(field_number, WIRE_VARINT)
        # For negative values, write as 10-byte varint
        if value < 0:
            var int_simd = SIMD[DType.int32, 1](value)
            var uint_simd = bitcast[DType.uint32, 1](int_simd)
            self.write_varint(UInt64(uint_simd[0]))
        else:
            self.write_varint(UInt64(value))

    fn write_int64(mut self, field_number: Int, value: Int64):
        """Write an int64 field."""
        self.write_tag(field_number, WIRE_VARINT)
        if value < 0:
            var int_simd = SIMD[DType.int64, 1](value)
            var uint_simd = bitcast[DType.uint64, 1](int_simd)
            self.write_varint(UInt64(uint_simd[0]))
        else:
            self.write_varint(UInt64(value))

    fn write_uint64(mut self, field_number: Int, value: UInt64):
        """Write a uint64 field."""
        self.write_tag(field_number, WIRE_VARINT)
        self.write_varint(value)

    fn write_float(mut self, field_number: Int, value: Float32):
        """Write a float field."""
        self.write_tag(field_number, WIRE_32BIT)
        self.write_float32(value)

    fn write_double(mut self, field_number: Int, value: Float64):
        """Write a double field."""
        self.write_tag(field_number, WIRE_64BIT)
        self.write_float64(value)

    fn write_embedded_message(
        mut self, field_number: Int, message: ProtoBuffer
    ):
        """Write an embedded message field."""
        self.write_tag(field_number, WIRE_LENGTH_DELIMITED)
        self.write_varint(UInt64(message.size()))
        self.write_bytes(message.data)

    fn write_packed_int64(mut self, field_number: Int, values: List[Int64]):
        """Write a packed repeated int64 field."""
        var packed = ProtoBuffer()
        for i in range(len(values)):
            var val = values[i]
            if val < 0:
                var int_simd = SIMD[DType.int64, 1](val)
                var uint_simd = bitcast[DType.uint64, 1](int_simd)
                packed.write_varint(UInt64(uint_simd[0]))
            else:
                packed.write_varint(UInt64(val))

        self.write_tag(field_number, WIRE_LENGTH_DELIMITED)
        self.write_varint(UInt64(packed.size()))
        self.write_bytes(packed.data)

    fn write_packed_float(mut self, field_number: Int, values: List[Float32]):
        """Write a packed repeated float field."""
        self.write_tag(field_number, WIRE_LENGTH_DELIMITED)
        self.write_varint(UInt64(len(values) * 4))
        for i in range(len(values)):
            self.write_float32(values[i])

    fn write_packed_double(mut self, field_number: Int, values: List[Float64]):
        """Write a packed repeated double field."""
        self.write_tag(field_number, WIRE_LENGTH_DELIMITED)
        self.write_varint(UInt64(len(values) * 8))
        for i in range(len(values)):
            self.write_float64(values[i])

    fn get_bytes(self) -> List[UInt8]:
        """Get the buffer contents as bytes."""
        var result = List[UInt8]()
        for i in range(len(self.data)):
            result.append(self.data[i])
        return result^

    fn clear(mut self):
        """Clear the buffer."""
        self.data.clear()
