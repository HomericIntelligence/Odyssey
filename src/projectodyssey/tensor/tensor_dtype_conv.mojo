"""Tensor dtype conversion helpers for AnyTensor.

Extracted from any_tensor.mojo per issue #5182 (SRP reduction).
Cross-module private-field access is valid in Mojo (package-scoped privacy).
See tensor_io.mojo, tensor_creation.mojo for precedent.
"""

from std.collections import List
from std.memory import bitcast
from .any_tensor import AnyTensor


def _convert_to_fp8_family_impl[
    target_fp8_dtype: DType
](tensor: AnyTensor, method_name: String) raises -> AnyTensor:
    """Shared helper for to_fp8() and to_bf8() (FP8/BF8 SIMD encode family).

    Rejects bfloat16, validates float input, SIMD-encodes each element to
    target_fp8_dtype, and stores as uint8.
    """
    # Explicitly reject bfloat16 at validation time
    if tensor._dtype == DType.bfloat16:
        raise Error(
            method_name
            + " does not support bfloat16: "
            + "the bfloat16 conversion path does not correctly round-trip "
            + "through the Float32 intermediate representation"
        )

    # Verify source is floating point
    if not (
        tensor._dtype == DType.float16
        or tensor._dtype == DType.float32
        or tensor._dtype == DType.float64
    ):
        raise Error(method_name + " requires a floating-point tensor")

    # Create output tensor with uint8 dtype
    var result = AnyTensor(tensor._shape, DType.uint8)

    # Convert each element using native SIMD encode + bitcast, store as uint8
    for i in range(tensor._numel):
        var val: Float32
        # Defensive dtype re-validation (fixes DATA-003)
        if tensor._dtype == DType.float16:
            val = tensor._data.bitcast[Float16]()[i].cast[DType.float32]()
        elif tensor._dtype == DType.float32:
            val = tensor._data.bitcast[Float32]()[i]
        elif tensor._dtype == DType.float64:
            val = tensor._data.bitcast[Float64]()[i].cast[DType.float32]()
        else:
            raise Error("Invalid dtype for " + method_name + " conversion")

        var fp8_val = SIMD[target_fp8_dtype, 1](val)
        var fp8_bits = bitcast[DType.uint8, 1](fp8_val)[0]
        var fp8_ptr = (result._data + i).bitcast[UInt8]()
        fp8_ptr[] = fp8_bits

    return result^


def convert_to_fp8_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor to FP8 E4M3 format."""
    from projectodyssey.core.types.dtype_aliases import FP8

    return _convert_to_fp8_family_impl[FP8](tensor, "to_fp8()")


def from_fp8_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert FP8-encoded tensor (uint8) back to Float32.

    This method interprets a uint8 tensor as FP8 E4M3 encoded values and
    converts them back to Float32 for computation.
    """
    from projectodyssey.core.types.dtype_aliases import FP8

    # Verify source is uint8
    if tensor._dtype != DType.uint8:
        raise Error("from_fp8() requires a uint8 tensor (FP8-encoded)")

    # Create output tensor with float32 dtype
    var result = AnyTensor(tensor._shape, DType.float32)

    # Convert each element from FP8 to Float32 using native SIMD
    for i in range(tensor._numel):
        var fp8_bits = tensor._data.bitcast[UInt8]()[i]
        # Bitcast uint8 to FP8, then convert to float32
        var fp8_val = bitcast[FP8, 1](SIMD[DType.uint8, 1](fp8_bits))
        var float_val = Float32(fp8_val[0])
        result[i] = Float32(float_val)

    return result^


def _convert_to_int_dtype_impl[
    target_dtype: DType
](
    tensor: AnyTensor,
    method_name: String,
    min_val: Int64,
    max_val: Int64,
    do_clamp: Bool,
) raises -> AnyTensor:
    """Shared helper for all 8 to_int*/to_uint* methods (integer-clamp family).

    Handles same-dtype fast-path, full dtype-dispatch read, optional clamp, and
    _set_int64 store.  target_dtype selects the output element type at compile time.
    """
    var result = AnyTensor(tensor._shape, target_dtype)

    for i in range(tensor._numel):
        # Same-dtype fast-path: bitcast directly without going through Float32.
        if tensor._dtype == target_dtype:
            result._set_int64(i, tensor._get_int64(i))
            continue

        # Read source element as Float32 (handles all supported dtypes).
        var val: Float32
        if tensor._dtype == DType.float16:
            val = tensor._data.bitcast[Float16]()[i].cast[DType.float32]()
        elif tensor._dtype == DType.float32:
            val = tensor._data.bitcast[Float32]()[i]
        elif tensor._dtype == DType.float64:
            val = tensor._data.bitcast[Float64]()[i].cast[DType.float32]()
        elif tensor._dtype == DType.int8:
            val = Float32(tensor._data.bitcast[Int8]()[i])
        elif tensor._dtype == DType.int16:
            val = Float32(tensor._data.bitcast[Int16]()[i])
        elif tensor._dtype == DType.int32:
            val = Float32(tensor._data.bitcast[Int32]()[i])
        elif tensor._dtype == DType.int64:
            val = Float32(tensor._data.bitcast[Int64]()[i])
        elif tensor._dtype == DType.uint8:
            val = Float32(tensor._data.bitcast[UInt8]()[i])
        elif tensor._dtype == DType.uint16:
            val = Float32(tensor._data.bitcast[UInt16]()[i])
        elif tensor._dtype == DType.uint32:
            val = Float32(tensor._data.bitcast[UInt32]()[i])
        elif tensor._dtype == DType.uint64:
            val = Float32(tensor._data.bitcast[UInt64]()[i])
        else:
            raise Error("Unsupported dtype for " + method_name + " conversion")

        var int_val = Int64(Int(val))
        if do_clamp:
            if int_val < min_val:
                int_val = min_val
            elif int_val > max_val:
                int_val = max_val
        result._set_int64(i, int_val)

    return result^


def to_int8_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to Int8 format."""
    return _convert_to_int_dtype_impl[DType.int8](
        tensor, "to_int8", Int64(-128), Int64(127), True
    )


def to_int16_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to Int16 format."""
    return _convert_to_int_dtype_impl[DType.int16](
        tensor, "to_int16", Int64(-32768), Int64(32767), True
    )


def to_int32_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to Int32 format."""
    return _convert_to_int_dtype_impl[DType.int32](
        tensor, "to_int32", Int64(0), Int64(0), False
    )


def to_int64_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to Int64 format."""
    return _convert_to_int_dtype_impl[DType.int64](
        tensor, "to_int64", Int64(0), Int64(0), False
    )


def to_uint8_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to UInt8 format."""
    return _convert_to_int_dtype_impl[DType.uint8](
        tensor, "to_uint8", Int64(0), Int64(255), True
    )


def to_uint16_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to UInt16 format."""
    return _convert_to_int_dtype_impl[DType.uint16](
        tensor, "to_uint16", Int64(0), Int64(0), False
    )


def to_uint32_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to UInt32 format."""
    return _convert_to_int_dtype_impl[DType.uint32](
        tensor, "to_uint32", Int64(0), Int64(0), False
    )


def to_uint64_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to UInt64 format."""
    return _convert_to_int_dtype_impl[DType.uint64](
        tensor, "to_uint64", Int64(0), Int64(0), False
    )


def to_bf8_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert tensor values to BF8 E5M2 format."""
    from projectodyssey.core.types.dtype_aliases import BF8

    return _convert_to_fp8_family_impl[BF8](tensor, "to_bf8()")


def from_bf8_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert BF8-encoded tensor (uint8) back to Float32.

    This method interprets a uint8 tensor as BF8 E5M2 encoded values and
    converts them back to Float32 for computation.
    """
    from projectodyssey.core.types.dtype_aliases import BF8

    # Verify source is uint8
    if tensor._dtype != DType.uint8:
        raise Error("from_bf8() requires a uint8 tensor (BF8-encoded)")

    # Create output tensor with float32 dtype
    var result = AnyTensor(tensor._shape, DType.float32)

    # Convert each element from BF8 to Float32 using native SIMD
    for i in range(tensor._numel):
        var bf8_bits = tensor._data.bitcast[UInt8]()[i]
        # Bitcast uint8 to BF8, then convert to float32
        var bf8_val = bitcast[BF8, 1](SIMD[DType.uint8, 1](bf8_bits))
        var float_val = Float32(bf8_val[0])
        result[i] = Float32(float_val)

    return result^


def _convert_to_block_quant_impl[
    is_mxfp4: Bool
](tensor: AnyTensor, fmt_name: String) raises -> AnyTensor:
    """Shared helper for to_mxfp4() and to_nvfp4() (block-quantized family).
    Extracted from AnyTensor._convert_to_block_quant per #5182.

    is_mxfp4=True: 32-elem blocks, 17 bytes each (MXFP4).
    is_mxfp4=False: 16-elem blocks, 9 bytes each (NVFP4).

    Note: bfloat16 passes the outer guard but raises in the inner dispatch.
    This asymmetry is a pre-existing bug preserved verbatim.
    TODO(#5181-followup): fix bfloat16 guard for block-quant methods.
    """
    # Verify source is floating point (bfloat16 outer guard passes; inner raises).
    # TODO(#5181-followup): the bfloat16 outer guard accepts but inner raises;
    # this asymmetry is preserved verbatim as a pre-existing bug.
    if not (
        tensor._dtype == DType.float16
        or tensor._dtype == DType.float32
        or tensor._dtype == DType.float64
        or tensor._dtype == DType.bfloat16
    ):
        raise Error(fmt_name + " requires a floating-point tensor")

    comptime if is_mxfp4:
        from projectodyssey.core.types.mxfp4 import MXFP4Block

        comptime block_size = 32
        comptime bytes_per_block = 17
        comptime data_bytes = 16
        var num_blocks = (tensor._numel + block_size - 1) // block_size
        var result = AnyTensor([num_blocks * bytes_per_block], DType.uint8)
        result._original_numel_quantized = tensor._numel
        for block_idx in range(num_blocks):
            var start_idx = block_idx * block_size
            var values = List[Float32]()
            for i in range(block_size):
                var idx = start_idx + i
                if idx < tensor._numel:
                    if idx >= tensor._numel:
                        raise Error("Index out of bounds during bitcast")
                    var val: Float32
                    if tensor._dtype == DType.float16:
                        val = tensor._data.bitcast[Float16]()[idx].cast[
                            DType.float32
                        ]()
                    elif tensor._dtype == DType.float32:
                        val = tensor._data.bitcast[Float32]()[idx]
                    elif tensor._dtype == DType.float64:
                        val = tensor._data.bitcast[Float64]()[idx].cast[
                            DType.float32
                        ]()
                    else:
                        raise Error("Invalid dtype for MXFP4 quantization")
                    values.append(val)
                else:
                    values.append(Float32(0.0))
            var block = MXFP4Block.from_float32_array(values)
            var block_offset = block_idx * bytes_per_block
            for i in range(data_bytes):
                var ptr = (result._data + block_offset + i).bitcast[UInt8]()
                ptr[] = block.data[i]
            var scale_ptr = (result._data + block_offset + data_bytes).bitcast[
                UInt8
            ]()
            scale_ptr[] = bitcast[DType.uint8, 1](block.scale)[0]
        return result^
    else:
        from projectodyssey.core.types.nvfp4 import NVFP4Block

        comptime block_size = 16
        comptime bytes_per_block = 9
        comptime data_bytes = 8
        var num_blocks = (tensor._numel + block_size - 1) // block_size
        var result = AnyTensor([num_blocks * bytes_per_block], DType.uint8)
        result._original_numel_quantized = tensor._numel
        for block_idx in range(num_blocks):
            var start_idx = block_idx * block_size
            var values = List[Float32]()
            for i in range(block_size):
                var idx = start_idx + i
                if idx < tensor._numel:
                    if idx >= tensor._numel:
                        raise Error("Index out of bounds during bitcast")
                    var val: Float32
                    if tensor._dtype == DType.float16:
                        val = tensor._data.bitcast[Float16]()[idx].cast[
                            DType.float32
                        ]()
                    elif tensor._dtype == DType.float32:
                        val = tensor._data.bitcast[Float32]()[idx]
                    elif tensor._dtype == DType.float64:
                        val = tensor._data.bitcast[Float64]()[idx].cast[
                            DType.float32
                        ]()
                    else:
                        raise Error("Invalid dtype for NVFP4 quantization")
                    values.append(val)
                else:
                    values.append(Float32(0.0))
            var block = NVFP4Block.from_float32_array(values)
            var block_offset = block_idx * bytes_per_block
            for i in range(data_bytes):
                var ptr = (result._data + block_offset + i).bitcast[UInt8]()
                ptr[] = block.data[i]
            var scale_ptr = (result._data + block_offset + data_bytes).bitcast[
                UInt8
            ]()
            scale_ptr[] = bitcast[DType.uint8, 1](block.scale)[0]
        return result^


def from_mxfp4_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert MXFP4-encoded tensor (uint8 blocks) back to Float32.
    Extracted from AnyTensor.from_mxfp4 per #5182.

    This method interprets a uint8 tensor as MXFP4 blocks and converts them
    back to Float32 for computation.
    """
    from projectodyssey.core.types.mxfp4 import MXFP4Block
    from projectodyssey.core.types.dtype_aliases import E8M0

    # Verify source is uint8
    if tensor._dtype != DType.uint8:
        raise Error("from_mxfp4() requires a uint8 tensor (MXFP4-encoded)")

    # Calculate number of blocks and output size
    if tensor._numel % 17 != 0:
        raise Error("MXFP4 tensor size must be multiple of 17 bytes")

    var num_blocks = tensor._numel // 17
    var padded_output_size = num_blocks * 32

    # Check if original size is stored
    var output_size: Int
    if tensor._original_numel_quantized >= 0:
        output_size = tensor._original_numel_quantized
    else:
        output_size = padded_output_size

    # Create output tensor with proper shape
    var output_shape = List[Int]()
    output_shape.append(padded_output_size)
    var result = AnyTensor(output_shape, DType.float32)

    # Decode each block
    for block_idx in range(num_blocks):
        var block_offset = block_idx * 17

        # Reconstruct MXFP4Block
        var data = SIMD[DType.uint8, 16](0)
        for i in range(16):
            data[i] = tensor._data.bitcast[UInt8]()[block_offset + i]
        # Reconstruct E8M0 scale from raw exponent byte
        var scale_byte = tensor._data.bitcast[UInt8]()[block_offset + 16]
        var scale = bitcast[E8M0, 1](SIMD[DType.uint8, 1](scale_byte))

        var block = MXFP4Block(data, scale)

        # Decode block to Float32 values (only decode needed elements)
        var values = block.to_float32_array()
        for i in range(32):
            var output_idx = block_idx * 32 + i
            if output_idx < output_size:
                result[output_idx] = Float32(values[i])

    # Trim result to original size if needed
    if output_size < padded_output_size:
        var trimmed = AnyTensor([output_size], DType.float32)
        for i in range(output_size):
            trimmed[i] = result._data.bitcast[Float32]()[i]
        return trimmed^

    return result^


def from_nvfp4_impl(tensor: AnyTensor) raises -> AnyTensor:
    """Convert NVFP4-encoded tensor (uint8 blocks) back to Float32.
    Extracted from AnyTensor.from_nvfp4 per #5182.

    This method interprets a uint8 tensor as NVFP4 blocks and converts them
    back to Float32 for computation.
    """
    from projectodyssey.core.types.nvfp4 import NVFP4Block
    from projectodyssey.core.types.dtype_aliases import FP8

    # Verify source is uint8
    if tensor._dtype != DType.uint8:
        raise Error("from_nvfp4() requires a uint8 tensor (NVFP4-encoded)")

    # Calculate number of blocks and output size
    if tensor._numel % 9 != 0:
        raise Error("NVFP4 tensor size must be multiple of 9 bytes")

    var num_blocks = tensor._numel // 9
    var padded_output_size = num_blocks * 16

    # Check if original size is stored
    var output_size: Int
    if tensor._original_numel_quantized >= 0:
        output_size = tensor._original_numel_quantized
    else:
        output_size = padded_output_size

    # Create output tensor with proper shape
    var output_shape = List[Int]()
    output_shape.append(padded_output_size)
    var result = AnyTensor(output_shape, DType.float32)

    # Decode each block
    for block_idx in range(num_blocks):
        var block_offset = block_idx * 9

        # Reconstruct NVFP4Block
        var data = SIMD[DType.uint8, 8](0)
        for i in range(8):
            data[i] = tensor._data.bitcast[UInt8]()[block_offset + i]
        # Reconstruct FP8 (E4M3) scale from raw byte
        var scale_byte = tensor._data.bitcast[UInt8]()[block_offset + 8]
        var scale = bitcast[FP8, 1](SIMD[DType.uint8, 1](scale_byte))

        var block = NVFP4Block(data, scale)

        # Decode block to Float32 values (only decode needed elements)
        var values = block.to_float32_array()
        for i in range(16):
            var output_idx = block_idx * 16 + i
            if output_idx < output_size:
                result[output_idx] = Float32(values[i])

    # Trim result to original size if needed
    if output_size < padded_output_size:
        var trimmed = AnyTensor([output_size], DType.float32)
        for i in range(output_size):
            trimmed[i] = result._data.bitcast[Float32]()[i]
        return trimmed^

    return result^
