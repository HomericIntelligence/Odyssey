"""Typed Tensor[dtype] matrix dispatch cores.

Internal module -- not part of the public API.
"""

from collections import List, Optional
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor
from shared.base.dtype_ordinal import (
    dtype_to_ordinal,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
)


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] Typed Implementations
# ============================================================================


fn _matmul_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[dt]:
    """Native typed matrix multiplication (Layer 3 core).

    Accepts Tensor[dt] inputs, delegates to existing parametric kernels
    for the actual computation, and returns Tensor[dt] result.

    Args:
        a: First input tensor (typed).
        b: Second input tensor (typed).

    Returns:
        A new Tensor[dt] with the matrix product.
    """
    from shared.core.matrix import matmul

    # Convert to AnyTensor for shape logic and existing kernel infrastructure
    var a_any = a.as_any()
    var b_any = b.as_any()
    # Delegate to existing AnyTensor matmul (which uses parametric _impl kernels)
    var result_any = matmul(a_any, b_any)
    return result_any.as_tensor[dt]()


fn _transpose_typed[dt: DType](
    tensor: Tensor[dt], axes: Optional[List[Int]] = None
) raises -> Tensor[dt]:
    """Native typed transpose (Layer 3 core).

    Args:
        tensor: Input typed tensor.
        axes: Optional permutation of axes. If None, reverses all axes.

    Returns:
        A new Tensor[dt] with permuted dimensions.
    """
    from shared.core.matrix import transpose

    var t_any = tensor.as_any()
    var result_any = transpose(t_any, axes)
    return result_any.as_tensor[dt]()


fn _dot_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[dt]:
    """Native typed dot product (Layer 3 core).

    Args:
        a: First input tensor (typed).
        b: Second input tensor (typed).

    Returns:
        A new Tensor[dt] with the dot product result.
    """
    from shared.core.matrix import dot

    var a_any = a.as_any()
    var b_any = b.as_any()
    var result_any = dot(a_any, b_any)
    return result_any.as_tensor[dt]()


fn _outer_typed[dt: DType](
    a: Tensor[dt], b: Tensor[dt]
) raises -> Tensor[dt]:
    """Native typed outer product (Layer 3 core).

    Args:
        a: First 1D input tensor (typed).
        b: Second 1D input tensor (typed).

    Returns:
        A new Tensor[dt] with the outer product result.
    """
    from shared.core.matrix import outer

    var a_any = a.as_any()
    var b_any = b.as_any()
    var result_any = outer(a_any, b_any)
    return result_any.as_tensor[dt]()


# ============================================================================
# Layer 2: Ordinal-Based Dispatch for Typed Operations
# ============================================================================


fn _dispatch_matmul_typed(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to typed matmul via ordinal-based lookup.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Matrix product result.
    """
    var ordinal = dtype_to_ordinal(a.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _matmul_typed[DType.float16](
            a.as_tensor[DType.float16](), b.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _matmul_typed[DType.float32](
            a.as_tensor[DType.float32](), b.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _matmul_typed[DType.float64](
            a.as_tensor[DType.float64](), b.as_tensor[DType.float64]()
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _matmul_typed[DType.int32](
            a.as_tensor[DType.int32](), b.as_tensor[DType.int32]()
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _matmul_typed[DType.int64](
            a.as_tensor[DType.int64](), b.as_tensor[DType.int64]()
        ).as_any()
    else:
        raise Error("matmul: unsupported dtype")


fn _dispatch_dot_typed(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to typed dot product via ordinal-based lookup.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Dot product result.
    """
    var ordinal = dtype_to_ordinal(a.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _dot_typed[DType.float16](
            a.as_tensor[DType.float16](), b.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _dot_typed[DType.float32](
            a.as_tensor[DType.float32](), b.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _dot_typed[DType.float64](
            a.as_tensor[DType.float64](), b.as_tensor[DType.float64]()
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _dot_typed[DType.int32](
            a.as_tensor[DType.int32](), b.as_tensor[DType.int32]()
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _dot_typed[DType.int64](
            a.as_tensor[DType.int64](), b.as_tensor[DType.int64]()
        ).as_any()
    else:
        raise Error("dot: unsupported dtype")


fn _dispatch_outer_typed(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to typed outer product via ordinal-based lookup.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Outer product result.
    """
    var ordinal = dtype_to_ordinal(a.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _outer_typed[DType.float16](
            a.as_tensor[DType.float16](), b.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _outer_typed[DType.float32](
            a.as_tensor[DType.float32](), b.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _outer_typed[DType.float64](
            a.as_tensor[DType.float64](), b.as_tensor[DType.float64]()
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _outer_typed[DType.int32](
            a.as_tensor[DType.int32](), b.as_tensor[DType.int32]()
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _outer_typed[DType.int64](
            a.as_tensor[DType.int64](), b.as_tensor[DType.int64]()
        ).as_any()
    else:
        raise Error("outer: unsupported dtype")


fn _dispatch_transpose_typed(
    tensor: AnyTensor, axes: Optional[List[Int]] = None
) raises -> AnyTensor:
    """Runtime dispatch to typed transpose via ordinal-based lookup.

    Args:
        tensor: Input tensor.
        axes: Optional permutation of axes.

    Returns:
        Transposed tensor result.
    """
    var ordinal = dtype_to_ordinal(tensor.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _transpose_typed[DType.float16](
            tensor.as_tensor[DType.float16](), axes
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _transpose_typed[DType.float32](
            tensor.as_tensor[DType.float32](), axes
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _transpose_typed[DType.float64](
            tensor.as_tensor[DType.float64](), axes
        ).as_any()
    elif ordinal == DTYPE_INT8:
        return _transpose_typed[DType.int8](
            tensor.as_tensor[DType.int8](), axes
        ).as_any()
    elif ordinal == DTYPE_INT16:
        return _transpose_typed[DType.int16](
            tensor.as_tensor[DType.int16](), axes
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _transpose_typed[DType.int32](
            tensor.as_tensor[DType.int32](), axes
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _transpose_typed[DType.int64](
            tensor.as_tensor[DType.int64](), axes
        ).as_any()
    else:
        raise Error("transpose: unsupported dtype")
