"""Dtype casting utilities for mixed precision training.

Provides efficient dtype conversion for tensors with optimized paths
for common conversions (FP32 <-> FP16, FP32 <-> BF16).
"""

from .extensor import ExTensor
from shared.core.types.dtype_aliases import BF16
from memory import bitcast


fn cast_tensor(tensor: ExTensor, target_dtype: DType) raises -> ExTensor:
    """Cast tensor to different dtype with optimized conversion paths.

    Provides fast conversion between common dtypes used in mixed precision
    training. Uses SIMD optimization where possible.

    Args:
        tensor: Source tensor to convert.
        target_dtype: Target dtype.

    Returns:
        New tensor with target dtype.

    Raises:
        Error: If tensor is empty or conversion is not supported.

    Example:
        ```mojo
        # Convert FP32 to FP16
        var fp16_tensor = cast_tensor(fp32_tensor, DType.float16)

        # Convert FP16 back to FP32
        var fp32_tensor = cast_tensor(fp16_tensor, DType.float32)
        ```
    """
    # Validate input
    if tensor._numel == 0:
        raise Error("Cannot cast empty tensor")

    # No conversion needed
    if tensor.dtype() == target_dtype:
        return tensor

    var result = ExTensor(tensor.shape(), target_dtype)
    var size = tensor._numel

    # Optimized paths for common conversions
    # FP32 -> FP16
    if tensor.dtype() == DType.float32 and target_dtype == DType.float16:
        var src_ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            result[i] = Float64(src_ptr[i])
        return result

    # FP16 -> FP32
    if tensor.dtype() == DType.float16 and target_dtype == DType.float32:
        var src_ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            result[i] = Float64(src_ptr[i])
        return result

    # FP32 -> FP32 (copy)
    if tensor.dtype() == DType.float32 and target_dtype == DType.float32:
        var src_ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            result[i] = Float64(src_ptr[i])
        return result

    # FP64 -> FP32
    if tensor.dtype() == DType.float64 and target_dtype == DType.float32:
        var src_ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            result[i] = src_ptr[i]
        return result

    # FP32 -> FP64
    if tensor.dtype() == DType.float32 and target_dtype == DType.float64:
        var src_ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            result[i] = Float64(src_ptr[i])
        return result

    # Generic slow path for other conversions
    for i in range(size):
        var val = tensor._get_float64(i)
        result._set_float64(i, val)

    return result


fn cast_to_bfloat16(tensor: ExTensor) raises -> ExTensor:
    """Convert tensor to BF16 storage (stored as uint16).

    Creates new tensor with BF16 values stored as uint16.
    Use this for storing model parameters in BF16 format.

    Args:
        tensor: Source tensor (any floating point dtype).

    Returns:
        Tensor with uint16 storage containing BF16 values.

    Raises:
        Error: If tensor is empty.

    Example:
        ```mojo
        var fp32_params = ExTensor.randn((1000, 1000), DType.float32)
        var bf16_params = cast_to_bfloat16(fp32_params)
        # bf16_params.dtype() == DType.uint16
        ```
    """
    if tensor._numel == 0:
        raise Error("Cannot convert empty tensor to BF16")

    # Create uint16 tensor for BF16 storage
    var result = ExTensor(tensor.shape(), DType.uint16)
    var size = tensor._numel

    # Convert each element using native SIMD conversion
    for i in range(size):
        var f32_val = Float32(tensor._get_float64(i))
        # Convert to bfloat16 using native SIMD, then bitcast to uint16 for storage
        var bf16_val = SIMD[BF16, 1](f32_val)
        var bf16_bits = bitcast[DType.uint16, 1](bf16_val)[0]
        result[i] = Float64(bf16_bits)

    return result


fn cast_from_bfloat16(
    tensor: ExTensor, target_dtype: DType = DType.float32
) raises -> ExTensor:
    """Convert tensor from BF16 storage to floating point.

    Assumes input tensor stores BF16 values as uint16.

    Args:
        tensor: Source tensor with uint16 BF16 storage.
        target_dtype: Target floating point dtype (default: float32).

    Returns:
        Tensor with target dtype.

    Raises:
        Error: If tensor is not uint16 or target is not floating point.

    Example:
        ```mojo
        var bf16_params = cast_to_bfloat16(fp32_params)
        var fp32_params = cast_from_bfloat16(bf16_params)
        ```
    """
    if tensor.dtype() != DType.uint16:
        raise Error(
            "Expected uint16 tensor for BF16 storage, got: "
            + String(tensor.dtype())
        )

    if (
        target_dtype != DType.float32
        and target_dtype != DType.float64
        and target_dtype != DType.float16
    ):
        raise Error("Target dtype must be floating point")

    var result = ExTensor(tensor.shape(), target_dtype)
    var size = tensor._numel

    # Convert each element using native SIMD conversion
    for i in range(size):
        var bf16_bits = tensor._data.bitcast[UInt16]()[i]
        # Bitcast uint16 to bfloat16, then convert to float32
        var bf16_val = bitcast[BF16, 1](SIMD[DType.uint16, 1](bf16_bits))
        var f32_val = Float32(bf16_val[0])

        if target_dtype == DType.float32:
            result[i] = Float64(f32_val)
        elif target_dtype == DType.float64:
            result[i] = Float64(f32_val)
        elif target_dtype == DType.float16:
            result[i] = Float64(f32_val)

    return result


fn get_dtype_size(dtype: DType) -> Int:
    """Get size in bytes for a dtype.

    Args:
        dtype: DType to query.

    Returns:
        Size in bytes.

    Example:
        ```mojo
        var size = get_dtype_size(DType.float16)  # 2
        ```
    """
    if dtype == DType.float16:
        return 2
    elif dtype == DType.float32:
        return 4
    elif dtype == DType.float64:
        return 8
    elif dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
        return 1
    elif dtype == DType.int16 or dtype == DType.uint16:
        return 2
    elif dtype == DType.int32 or dtype == DType.uint32:
        return 4
    elif dtype == DType.int64 or dtype == DType.uint64:
        return 8
    else:
        return 4  # Default


fn is_floating_dtype(dtype: DType) -> Bool:
    """Check if dtype is floating point.

    Args:
        dtype: DType to check.

    Returns:
        True if floating point dtype.

    Example:
        ```mojo
        if is_floating_dtype(tensor.dtype()):
            # Can use floating point operations
            var scaled = tensor * 0.5
        ```
    """
    return (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    )


fn is_integer_dtype(dtype: DType) -> Bool:
    """Check if dtype is integer (signed or unsigned).

    Args:
        dtype: DType to check.

    Returns:
        True if integer dtype.

    Example:
        ```mojo
        if is_integer_dtype(tensor.dtype()):
            # Integer tensor - no fractional values
            pass
        ```
    """
    return (
        dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
        or dtype == DType.uint8
        or dtype == DType.uint16
        or dtype == DType.uint32
        or dtype == DType.uint64
    )
