"""DType ordinal mapping for efficient dispatch.

Provides O(1) dispatch by converting DType to integer ordinal for jump table optimization.
The ordinal values are stable and match the order of supported dtypes.

This module centralizes DType to ordinal mapping, providing a single source of truth
for dtype dispatch operations across the codebase.

Benefits:
- Single source of truth for dtype ordinal mapping
- Compiler can optimize consecutive integer comparisons to jump table
- Easy to add new dtypes (update only this file)
- Consistent ordinal values across all dispatch operations

Supported DTypes:
- Float16 (FP16) - ordinal 0
- Float32 (FP32) - ordinal 1
- Float64 (FP64) - ordinal 2
- Int8 - ordinal 3
- Int16 - ordinal 4
- Int32 - ordinal 5
- Int64 - ordinal 6
- UInt8 - ordinal 7
- UInt16 - ordinal 8
- UInt32 - ordinal 9
- UInt64 - ordinal 10

Usage:
    from .dtype_ordinal import dtype_to_ordinal, DTYPE_FLOAT32

    var ordinal = dtype_to_ordinal(tensor.dtype())
    if ordinal == DTYPE_FLOAT32:
        # Handle float32 case
        pass
"""

# DType ordinals - stable values for dispatch
comptime DTYPE_FLOAT16: Int = 0
comptime DTYPE_FLOAT32: Int = 1
comptime DTYPE_FLOAT64: Int = 2
comptime DTYPE_INT8: Int = 3
comptime DTYPE_INT16: Int = 4
comptime DTYPE_INT32: Int = 5
comptime DTYPE_INT64: Int = 6
comptime DTYPE_UINT8: Int = 7
comptime DTYPE_UINT16: Int = 8
comptime DTYPE_UINT32: Int = 9
comptime DTYPE_UINT64: Int = 10
comptime DTYPE_UNSUPPORTED: Int = -1

comptime SUPPORTED_DTYPE_COUNT: Int = 11


def dtype_to_ordinal(dtype: DType) -> Int:
    """Convert DType to integer ordinal for dispatch lookup.

    Returns DTYPE_UNSUPPORTED (-1) for unknown dtypes.
    The compiler can optimize this to an efficient comparison sequence
    or jump table for performance.

    Args:
        dtype: The DType to convert.

    Returns:
        Integer ordinal for the dtype, or -1 for unsupported dtypes.

    Examples:
        ```mojo
        var ordinal = dtype_to_ordinal(DType.float32)
        if ordinal == DTYPE_FLOAT32:
            # Handle float32
            pass
        elif ordinal == DTYPE_UNSUPPORTED:
            # Handle error
            pass
        ```
    """
    if dtype == DType.float16:
        return DTYPE_FLOAT16
    elif dtype == DType.float32:
        return DTYPE_FLOAT32
    elif dtype == DType.float64:
        return DTYPE_FLOAT64
    elif dtype == DType.int8:
        return DTYPE_INT8
    elif dtype == DType.int16:
        return DTYPE_INT16
    elif dtype == DType.int32:
        return DTYPE_INT32
    elif dtype == DType.int64:
        return DTYPE_INT64
    elif dtype == DType.uint8:
        return DTYPE_UINT8
    elif dtype == DType.uint16:
        return DTYPE_UINT16
    elif dtype == DType.uint32:
        return DTYPE_UINT32
    elif dtype == DType.uint64:
        return DTYPE_UINT64
    else:
        return DTYPE_UNSUPPORTED


def format_dtype_name(dtype: DType) -> String:
    """Format DType for error messages.

    Args:
        dtype: The DType to format.

    Returns:
        String representation of the dtype name.

    Examples:
        ```mojo
        var name = format_dtype_name(DType.float32)
        # Returns "float32"
        ```

    Note:
        Used internally for error messages. Returns "unknown" for
        unrecognized dtypes.
    """
    if dtype == DType.float16:
        return "float16"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    else:
        return "unknown"
