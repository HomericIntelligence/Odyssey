"""Error formatting utilities for consistent, context-rich error messages.

Provides helper functions for creating descriptive error messages that include
operation context, actual vs expected values, and actionable fix suggestions.
"""


def format_shape(shape: List[Int]) -> String:
    """Convert a shape list to a bracketed string representation."""
    if len(shape) == 0:
        return "[]"

    var result = "["
    result += String(shape[0])
    for i in range(1, len(shape)):
        result += ", " + String(shape[i])
    result += "]"

    return result


def format_dtype(dtype: DType) -> String:
    """Convert a DType to a readable string representation."""
    if dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.float16:
        return "float16"
    elif dtype == DType.bfloat16:
        return "bfloat16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unknown"


def format_matmul_error(a_shape: List[Int], b_shape: List[Int]) -> String:
    """Create descriptive error for matmul dimension mismatches."""
    if len(a_shape) != 2 or len(b_shape) != 2:
        return (
            "matmul: requires 2D tensors. Got A with shape "
            + format_shape(a_shape)
            + " and B with shape "
            + format_shape(b_shape)
            + ". Hint: Use reshape() to convert to 2D"
        )

    var a_cols = a_shape[1]
    var b_rows = b_shape[0]

    var msg = "matmul: cannot multiply matrices A" + format_shape(a_shape)
    msg += " and B" + format_shape(b_shape) + ". "
    msg += "Inner dimensions must match: A has " + String(a_cols) + " columns, "
    msg += "B has " + String(b_rows) + " rows. "
    msg += "Hint: B should have shape [" + String(a_cols) + ", N] for any N"

    return msg
