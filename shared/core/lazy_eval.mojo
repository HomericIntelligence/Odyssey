"""Lazy evaluation kernels for tensor expressions.

Implements fused evaluation of expression trees, computing results without
intermediate tensor allocations. Uses recursive tree traversal and
compile-time dtype specialization for efficient execution.

Architecture:
    - _evaluate_at_index[dtype](): Recursively evaluate expression at index
    - _dispatch_evaluate(): Runtime dtype dispatch
    - evaluate(): Main entry point with automatic kernel selection
    - Broadcasting index computation integrated into evaluation
"""

from collections import List
from .any_tensor import AnyTensor, full
from .lazy_expression import (
    TensorExpr,
    ExprNode,
    OpType,
    OP_LEAF,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_NEG,
    OP_SCALAR_MUL,
    OP_SCALAR_DIV,
)
from shared.base.broadcasting import compute_broadcast_strides
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
# Index Computation Helpers
# ============================================================================


fn _flat_to_coords(flat_idx: Int, shape: List[Int]) -> List[Int]:
    """Convert flat index to multi-dimensional coordinates.

    Args:
        flat_idx: Flat index in row-major layout.
        shape: Shape of the tensor.

    Returns:
        Multi-dimensional coordinates.
    """
    var coords = List[Int]()
    var remaining = flat_idx

    # Convert flat index to coordinates (right-to-left)
    for i in range(len(shape) - 1, -1, -1):
        coords.append(remaining % shape[i])
        remaining //= shape[i]

    # Reverse to get left-to-right order
    var result = List[Int]()
    for i in range(len(coords) - 1, -1, -1):
        result.append(coords[i])

    return result^


fn _coords_to_index(coords: List[Int], strides: List[Int]) -> Int:
    """Convert multi-dimensional coordinates to flat index.

    Args:
        coords: Multi-dimensional coordinates.
        strides: Strides for each dimension.

    Returns:
        Flat index in row-major layout.
    """
    var idx = 0
    for i in range(len(coords)):
        idx += coords[i] * strides[i]
    return idx


fn _compute_source_idx(
    flat_idx: Int,
    result_shape: List[Int],
    source_shape: List[Int],
    source_strides: List[Int],
) -> Int:
    """Compute source tensor index for a result index with broadcasting.

    Handles broadcasting by treating dimensions of size 1 as repeating.

    Args:
        flat_idx: Flat index in result tensor.
        result_shape: Result tensor shape.
        source_shape: Source tensor shape.
        source_strides: Source tensor strides.

    Returns:
        Flat index in source tensor.
    """
    var coords = _flat_to_coords(flat_idx, result_shape)

    # Align shapes for broadcasting (pad left with 1s)
    var result_ndim = len(result_shape)
    var source_ndim = len(source_shape)
    var offset = result_ndim - source_ndim

    var source_coords = List[Int]()
    for i in range(len(coords)):
        if i < offset:
            # Result dimension not in source (broadcasted)
            source_coords.append(0)
        else:
            # Get coordinate from source, 0 if dimension is 1
            var dim_idx = i - offset
            if source_shape[dim_idx] == 1:
                source_coords.append(0)
            else:
                source_coords.append(coords[i])

    return _coords_to_index(source_coords, source_strides)


# ============================================================================
# Core Typed Evaluation
# ============================================================================


fn _evaluate_at_index[
    dtype: DType
](
    expr: TensorExpr,
    nodes: List[ExprNode],
    tensors: List[AnyTensor],
    scalars: List[Float64],
    result_shape: List[Int],
    node_idx: Int,
    flat_idx: Int,
) -> Scalar[dtype]:
    """Recursively evaluate expression tree at a single output index.

    Uses compile-time specialization for zero-overhead dtype operations.

    Args:
        expr: Expression being evaluated.
        nodes: Array of expression nodes.
        tensors: Array of referenced tensors.
        scalars: Array of referenced scalars.
        result_shape: Result tensor shape.
        node_idx: Index of node to evaluate.
        flat_idx: Flat index in result tensor.

    Returns:
        Computed scalar value.
    """
    var node = nodes[node_idx]

    if node.op == OP_LEAF:
        # Leaf node: fetch value from tensor
        var tensor = tensors[node.tensor_idx]
        var tensor_shape = tensor.shape()

        # Use compute_broadcast_strides for proper broadcasting
        var broadcast_strides = compute_broadcast_strides(
            tensor_shape, result_shape
        )
        var source_idx = _compute_source_idx(
            flat_idx, result_shape, tensor_shape, broadcast_strides
        )
        var tensor_ptr = tensor._data.bitcast[Scalar[dtype]]()
        return tensor_ptr[source_idx]

    elif node.op == OP_ADD:
        var left_val = _evaluate_at_index[dtype](
            expr, nodes, tensors, scalars, result_shape, node.left_idx, flat_idx
        )
        var right_val = _evaluate_at_index[dtype](
            expr,
            nodes,
            tensors,
            scalars,
            result_shape,
            node.right_idx,
            flat_idx,
        )
        return left_val + right_val

    elif node.op == OP_SUB:
        var left_val = _evaluate_at_index[dtype](
            expr, nodes, tensors, scalars, result_shape, node.left_idx, flat_idx
        )
        var right_val = _evaluate_at_index[dtype](
            expr,
            nodes,
            tensors,
            scalars,
            result_shape,
            node.right_idx,
            flat_idx,
        )
        return left_val - right_val

    elif node.op == OP_MUL:
        var left_val = _evaluate_at_index[dtype](
            expr, nodes, tensors, scalars, result_shape, node.left_idx, flat_idx
        )
        var right_val = _evaluate_at_index[dtype](
            expr,
            nodes,
            tensors,
            scalars,
            result_shape,
            node.right_idx,
            flat_idx,
        )
        return left_val * right_val

    elif node.op == OP_DIV:
        var left_val = _evaluate_at_index[dtype](
            expr, nodes, tensors, scalars, result_shape, node.left_idx, flat_idx
        )
        var right_val = _evaluate_at_index[dtype](
            expr,
            nodes,
            tensors,
            scalars,
            result_shape,
            node.right_idx,
            flat_idx,
        )
        return left_val / right_val

    elif node.op == OP_NEG:
        var val = _evaluate_at_index[dtype](
            expr, nodes, tensors, scalars, result_shape, node.left_idx, flat_idx
        )
        return -val

    elif node.op == OP_SCALAR_MUL:
        var val = _evaluate_at_index[dtype](
            expr, nodes, tensors, scalars, result_shape, node.left_idx, flat_idx
        )
        var scalar = Scalar[dtype](scalars[node.scalar_idx])
        return val * scalar

    elif node.op == OP_SCALAR_DIV:
        var val = _evaluate_at_index[dtype](
            expr, nodes, tensors, scalars, result_shape, node.left_idx, flat_idx
        )
        var scalar = Scalar[dtype](scalars[node.scalar_idx])
        return val / scalar

    else:
        # Fallback (should not happen)
        return Scalar[dtype](0)


# ============================================================================
# Dtype Dispatch
# ============================================================================


fn _dispatch_evaluate(expr: TensorExpr) raises -> AnyTensor:
    """Runtime dispatch to compile-time specialized evaluation.

    Performs dtype dispatch and calls specialized kernel.

    Args:
        expr: Expression to evaluate.

    Returns:
        Computed result tensor.

    Raises:
        Error: If dtype is not supported.
    """
    var dtype_ord = dtype_to_ordinal(expr._dtype)

    if dtype_ord == DTYPE_FLOAT32:
        return _evaluate_typed[DType.float32](expr)
    elif dtype_ord == DTYPE_FLOAT64:
        return _evaluate_typed[DType.float64](expr)
    elif dtype_ord == DTYPE_INT32:
        return _evaluate_typed[DType.int32](expr)
    elif dtype_ord == DTYPE_INT64:
        return _evaluate_typed[DType.int64](expr)
    elif dtype_ord == DTYPE_FLOAT16:
        return _evaluate_typed[DType.float16](expr)
    elif dtype_ord == DTYPE_INT8:
        return _evaluate_typed[DType.int8](expr)
    elif dtype_ord == DTYPE_INT16:
        return _evaluate_typed[DType.int16](expr)
    elif dtype_ord == DTYPE_UINT8:
        return _evaluate_typed[DType.uint8](expr)
    elif dtype_ord == DTYPE_UINT16:
        return _evaluate_typed[DType.uint16](expr)
    elif dtype_ord == DTYPE_UINT32:
        return _evaluate_typed[DType.uint32](expr)
    elif dtype_ord == DTYPE_UINT64:
        return _evaluate_typed[DType.uint64](expr)
    else:
        raise Error("Unsupported dtype for lazy evaluation")


# ============================================================================
# Typed Kernel
# ============================================================================


fn _evaluate_typed[dtype: DType](expr: TensorExpr) raises -> AnyTensor:
    """Compile-time specialized evaluation kernel.

    Args:
        expr: Expression to evaluate.

    Returns:
        Computed result tensor.

    Raises:
        Error: If result tensor allocation fails.
    """
    # Create result tensor with zero fill
    var result = full(expr._result_shape, 0.0, expr._dtype)
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    var total_elems = expr.numel()
    var root_idx = expr._root_idx

    # Evaluate at each output index
    for idx in range(total_elems):
        var value = _evaluate_at_index[dtype](
            expr,
            expr._nodes,
            expr._tensors,
            expr._scalars,
            expr._result_shape,
            root_idx,
            idx,
        )
        result_ptr[idx] = value

    return result


# ============================================================================
# Main Public API
# ============================================================================


fn evaluate(expr: TensorExpr) raises -> AnyTensor:
    """Evaluate lazy expression to produce result tensor.

    Performs fused evaluation of the entire expression tree, producing a
    single output tensor without intermediate allocations. Automatically
    selects specialized kernel based on expression dtype.

    Broadcasting is handled transparently during evaluation.

    Args:
        expr: Expression to evaluate.

    Returns:
        Computed result tensor with shape = broadcast shape of all operands.

    Raises:
        Error: If evaluation fails or dtype unsupported.

    Example:
        ```mojo
        # Build expression lazily (no computation)
        var expression = (expr(a) + expr(b)) * expr(c)

        # Evaluate with fusion (1 allocation, 1 kernel)
        var result = evaluate(expression)
        ```
    """
    return _dispatch_evaluate(expr)
