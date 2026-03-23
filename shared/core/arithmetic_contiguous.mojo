"""Fast path optimizations for contiguous tensor arithmetic operations.

This module provides optimized implementations for element-wise arithmetic
operations when both tensors are contiguous and have the same shape. This
enables SIMD vectorization and eliminates stride calculations, providing
20-40% speedup for contiguous operations.

Architecture: Tensor[dtype] typed implementations are the core (zero bitcasts
for the result tensor). AnyTensor dispatch functions delegate to typed core.

Design:
- Detects when fast path is applicable (same shape, both contiguous)
- Uses SIMD for float32/float64 (via vectorize)
- Falls back to simple scalar loops for other dtypes
- Integrates with broadcast dispatcher for general case

Usage:
    from .arithmetic_contiguous import can_use_fast_path
    if can_use_fast_path(a, b):
        # Fast path selected automatically by dispatcher
        result = add(a, b)  # Uses optimized implementation
    else:
        # Fallback to broadcasting path
        result = add(a, b)  # Uses stride-aware broadcast
"""

from algorithm import vectorize
from sys.info import simd_width_of
from shared.tensor.any_tensor import AnyTensor


# ============================================================================
# Helper Functions
# ============================================================================


fn shapes_match(a: AnyTensor, b: AnyTensor) -> Bool:
    """Check if two tensors have identical shapes.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        True if shapes are identical element-by-element, False otherwise.
    """
    if len(a.shape()) != len(b.shape()):
        return False

    for i in range(len(a.shape())):
        if a.shape()[i] != b.shape()[i]:
            return False

    return True


fn can_use_fast_path(a: AnyTensor, b: AnyTensor) -> Bool:
    """Check if tensors are eligible for contiguous fast path.

    The fast path applies when:
    1. Tensors have identical shapes (not just broadcastable)
    2. Both tensors are contiguous (row-major, no strides)
    3. Tensors have the same dtype

    This enables SIMD vectorization and eliminates stride calculations.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        True if fast path can be used, False otherwise.
    """
    # Check dtype match
    if a.dtype() != b.dtype():
        return False

    # Check shape match (not just broadcastable)
    if not shapes_match(a, b):
        return False

    # Check both tensors are contiguous
    # Non-contiguous tensors (views from slicing/transposing) use strides
    # and cannot use simple pointer arithmetic
    if not a.is_contiguous():
        return False
    if not b.is_contiguous():
        return False

    return True


# ============================================================================
# Layer 2: AnyTensor Dispatch (delegates to typed core)
# ============================================================================


fn _add_contiguous_dispatch(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Dispatch to typed contiguous addition.

    Args:
        a: First tensor (contiguous).
        b: Second tensor (contiguous).

    Returns:
        Result tensor containing a + b.
    """
    from shared.tensor.typed.arithmetic_contiguous import _add_contiguous_typed

    if a.dtype() == DType.float32:
        return _add_contiguous_typed[DType.float32](
            a.as_tensor[DType.float32](), b.as_tensor[DType.float32]()
        ).as_any()
    elif a.dtype() == DType.float64:
        return _add_contiguous_typed[DType.float64](
            a.as_tensor[DType.float64](), b.as_tensor[DType.float64]()
        ).as_any()
    elif a.dtype() == DType.int8:
        return _add_contiguous_typed[DType.int8](
            a.as_tensor[DType.int8](), b.as_tensor[DType.int8]()
        ).as_any()
    elif a.dtype() == DType.int16:
        return _add_contiguous_typed[DType.int16](
            a.as_tensor[DType.int16](), b.as_tensor[DType.int16]()
        ).as_any()
    elif a.dtype() == DType.int32:
        return _add_contiguous_typed[DType.int32](
            a.as_tensor[DType.int32](), b.as_tensor[DType.int32]()
        ).as_any()
    elif a.dtype() == DType.int64:
        return _add_contiguous_typed[DType.int64](
            a.as_tensor[DType.int64](), b.as_tensor[DType.int64]()
        ).as_any()
    elif a.dtype() == DType.uint8:
        return _add_contiguous_typed[DType.uint8](
            a.as_tensor[DType.uint8](), b.as_tensor[DType.uint8]()
        ).as_any()
    elif a.dtype() == DType.uint16:
        return _add_contiguous_typed[DType.uint16](
            a.as_tensor[DType.uint16](), b.as_tensor[DType.uint16]()
        ).as_any()
    elif a.dtype() == DType.uint32:
        return _add_contiguous_typed[DType.uint32](
            a.as_tensor[DType.uint32](), b.as_tensor[DType.uint32]()
        ).as_any()
    elif a.dtype() == DType.uint64:
        return _add_contiguous_typed[DType.uint64](
            a.as_tensor[DType.uint64](), b.as_tensor[DType.uint64]()
        ).as_any()
    else:
        raise Error("Unsupported dtype for contiguous addition")


fn _subtract_contiguous_dispatch(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Dispatch to typed contiguous subtraction.

    Args:
        a: First tensor (contiguous).
        b: Second tensor (contiguous).

    Returns:
        Result tensor containing a - b.
    """
    from shared.tensor.typed.arithmetic_contiguous import _subtract_contiguous_typed

    if a.dtype() == DType.float32:
        return _subtract_contiguous_typed[DType.float32](
            a.as_tensor[DType.float32](), b.as_tensor[DType.float32]()
        ).as_any()
    elif a.dtype() == DType.float64:
        return _subtract_contiguous_typed[DType.float64](
            a.as_tensor[DType.float64](), b.as_tensor[DType.float64]()
        ).as_any()
    elif a.dtype() == DType.int8:
        return _subtract_contiguous_typed[DType.int8](
            a.as_tensor[DType.int8](), b.as_tensor[DType.int8]()
        ).as_any()
    elif a.dtype() == DType.int16:
        return _subtract_contiguous_typed[DType.int16](
            a.as_tensor[DType.int16](), b.as_tensor[DType.int16]()
        ).as_any()
    elif a.dtype() == DType.int32:
        return _subtract_contiguous_typed[DType.int32](
            a.as_tensor[DType.int32](), b.as_tensor[DType.int32]()
        ).as_any()
    elif a.dtype() == DType.int64:
        return _subtract_contiguous_typed[DType.int64](
            a.as_tensor[DType.int64](), b.as_tensor[DType.int64]()
        ).as_any()
    elif a.dtype() == DType.uint8:
        return _subtract_contiguous_typed[DType.uint8](
            a.as_tensor[DType.uint8](), b.as_tensor[DType.uint8]()
        ).as_any()
    elif a.dtype() == DType.uint16:
        return _subtract_contiguous_typed[DType.uint16](
            a.as_tensor[DType.uint16](), b.as_tensor[DType.uint16]()
        ).as_any()
    elif a.dtype() == DType.uint32:
        return _subtract_contiguous_typed[DType.uint32](
            a.as_tensor[DType.uint32](), b.as_tensor[DType.uint32]()
        ).as_any()
    elif a.dtype() == DType.uint64:
        return _subtract_contiguous_typed[DType.uint64](
            a.as_tensor[DType.uint64](), b.as_tensor[DType.uint64]()
        ).as_any()
    else:
        raise Error("Unsupported dtype for contiguous subtraction")


fn _multiply_contiguous_dispatch(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Dispatch to typed contiguous multiplication.

    Args:
        a: First tensor (contiguous).
        b: Second tensor (contiguous).

    Returns:
        Result tensor containing a * b.
    """
    from shared.tensor.typed.arithmetic_contiguous import _multiply_contiguous_typed

    if a.dtype() == DType.float32:
        return _multiply_contiguous_typed[DType.float32](
            a.as_tensor[DType.float32](), b.as_tensor[DType.float32]()
        ).as_any()
    elif a.dtype() == DType.float64:
        return _multiply_contiguous_typed[DType.float64](
            a.as_tensor[DType.float64](), b.as_tensor[DType.float64]()
        ).as_any()
    elif a.dtype() == DType.int8:
        return _multiply_contiguous_typed[DType.int8](
            a.as_tensor[DType.int8](), b.as_tensor[DType.int8]()
        ).as_any()
    elif a.dtype() == DType.int16:
        return _multiply_contiguous_typed[DType.int16](
            a.as_tensor[DType.int16](), b.as_tensor[DType.int16]()
        ).as_any()
    elif a.dtype() == DType.int32:
        return _multiply_contiguous_typed[DType.int32](
            a.as_tensor[DType.int32](), b.as_tensor[DType.int32]()
        ).as_any()
    elif a.dtype() == DType.int64:
        return _multiply_contiguous_typed[DType.int64](
            a.as_tensor[DType.int64](), b.as_tensor[DType.int64]()
        ).as_any()
    elif a.dtype() == DType.uint8:
        return _multiply_contiguous_typed[DType.uint8](
            a.as_tensor[DType.uint8](), b.as_tensor[DType.uint8]()
        ).as_any()
    elif a.dtype() == DType.uint16:
        return _multiply_contiguous_typed[DType.uint16](
            a.as_tensor[DType.uint16](), b.as_tensor[DType.uint16]()
        ).as_any()
    elif a.dtype() == DType.uint32:
        return _multiply_contiguous_typed[DType.uint32](
            a.as_tensor[DType.uint32](), b.as_tensor[DType.uint32]()
        ).as_any()
    elif a.dtype() == DType.uint64:
        return _multiply_contiguous_typed[DType.uint64](
            a.as_tensor[DType.uint64](), b.as_tensor[DType.uint64]()
        ).as_any()
    else:
        raise Error("Unsupported dtype for contiguous multiplication")


fn _divide_contiguous_dispatch(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Dispatch to typed contiguous division.

    Args:
        a: First tensor (contiguous).
        b: Second tensor (contiguous).

    Returns:
        Result tensor containing a / b.
    """
    from shared.tensor.typed.arithmetic_contiguous import _divide_contiguous_typed

    if a.dtype() == DType.float32:
        return _divide_contiguous_typed[DType.float32](
            a.as_tensor[DType.float32](), b.as_tensor[DType.float32]()
        ).as_any()
    elif a.dtype() == DType.float64:
        return _divide_contiguous_typed[DType.float64](
            a.as_tensor[DType.float64](), b.as_tensor[DType.float64]()
        ).as_any()
    elif a.dtype() == DType.int8:
        return _divide_contiguous_typed[DType.int8](
            a.as_tensor[DType.int8](), b.as_tensor[DType.int8]()
        ).as_any()
    elif a.dtype() == DType.int16:
        return _divide_contiguous_typed[DType.int16](
            a.as_tensor[DType.int16](), b.as_tensor[DType.int16]()
        ).as_any()
    elif a.dtype() == DType.int32:
        return _divide_contiguous_typed[DType.int32](
            a.as_tensor[DType.int32](), b.as_tensor[DType.int32]()
        ).as_any()
    elif a.dtype() == DType.int64:
        return _divide_contiguous_typed[DType.int64](
            a.as_tensor[DType.int64](), b.as_tensor[DType.int64]()
        ).as_any()
    elif a.dtype() == DType.uint8:
        return _divide_contiguous_typed[DType.uint8](
            a.as_tensor[DType.uint8](), b.as_tensor[DType.uint8]()
        ).as_any()
    elif a.dtype() == DType.uint16:
        return _divide_contiguous_typed[DType.uint16](
            a.as_tensor[DType.uint16](), b.as_tensor[DType.uint16]()
        ).as_any()
    elif a.dtype() == DType.uint32:
        return _divide_contiguous_typed[DType.uint32](
            a.as_tensor[DType.uint32](), b.as_tensor[DType.uint32]()
        ).as_any()
    elif a.dtype() == DType.uint64:
        return _divide_contiguous_typed[DType.uint64](
            a.as_tensor[DType.uint64](), b.as_tensor[DType.uint64]()
        ).as_any()
    else:
        raise Error("Unsupported dtype for contiguous division")


# ============================================================================
# Legacy AnyTensor wrappers (preserved for backward compatibility)
# ============================================================================


@always_inline
fn _add_contiguous[dtype: DType](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """AnyTensor contiguous addition -- delegates to typed core."""
    from shared.tensor.typed.arithmetic_contiguous import _add_contiguous_typed

    return _add_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


@always_inline
fn _subtract_contiguous[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """AnyTensor contiguous subtraction -- delegates to typed core."""
    from shared.tensor.typed.arithmetic_contiguous import _subtract_contiguous_typed

    return _subtract_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


@always_inline
fn _multiply_contiguous[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """AnyTensor contiguous multiplication -- delegates to typed core."""
    from shared.tensor.typed.arithmetic_contiguous import _multiply_contiguous_typed

    return _multiply_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


@always_inline
fn _divide_contiguous[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """AnyTensor contiguous division -- delegates to typed core."""
    from shared.tensor.typed.arithmetic_contiguous import _divide_contiguous_typed

    return _divide_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()
