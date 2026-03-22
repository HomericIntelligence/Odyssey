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
from .any_tensor import AnyTensor
from shared.tensor.tensor import Tensor


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
# Layer 3 (Core): Native Tensor[dtype] Contiguous Implementations
# ============================================================================


@always_inline
fn _add_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized addition for contiguous same-shape Tensor[dtype].

    Uses SIMD vectorization for float32/float64 and scalar loops for others.
    Zero bitcasts -- result uses native typed pointer directly.

    Args:
        a: First tensor (must be contiguous, same shape as b).
        b: Second tensor (must be contiguous, same shape as a).

    Returns:
        Result Tensor[dtype] containing a + b.
    """
    var result = Tensor[dtype](a.shape())
    var size = a.numel()

    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    # SIMD vectorization for float types
    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_add[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec + b_vec)

        vectorize[simd_width](size, vectorized_add)
    else:
        # Scalar loop for other dtypes
        # Still faster than stride-aware broadcasting due to linear memory access
        for i in range(size):
            result_ptr[i] = a_ptr[i] + b_ptr[i]

    return result^


@always_inline
fn _subtract_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized subtraction for contiguous same-shape Tensor[dtype].

    Args:
        a: First tensor (must be contiguous, same shape as b).
        b: Second tensor (must be contiguous, same shape as a).

    Returns:
        Result Tensor[dtype] containing a - b.
    """
    var result = Tensor[dtype](a.shape())
    var size = a.numel()

    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    # SIMD vectorization for float types
    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_sub[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec - b_vec)

        vectorize[simd_width](size, vectorized_sub)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] - b_ptr[i]

    return result^


@always_inline
fn _multiply_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized multiplication for contiguous same-shape Tensor[dtype].

    Args:
        a: First tensor (must be contiguous, same shape as b).
        b: Second tensor (must be contiguous, same shape as a).

    Returns:
        Result Tensor[dtype] containing a * b.
    """
    var result = Tensor[dtype](a.shape())
    var size = a.numel()

    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    # SIMD vectorization for float types
    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_mul[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec * b_vec)

        vectorize[simd_width](size, vectorized_mul)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] * b_ptr[i]

    return result^


@always_inline
fn _divide_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized division for contiguous same-shape Tensor[dtype].

    Args:
        a: First tensor (must be contiguous, same shape as b).
        b: Second tensor (must be contiguous, same shape as a).

    Returns:
        Result Tensor[dtype] containing a / b.
    """
    var result = Tensor[dtype](a.shape())
    var size = a.numel()

    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    # SIMD vectorization for float types
    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_div[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec / b_vec)

        vectorize[simd_width](size, vectorized_div)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] / b_ptr[i]

    return result^


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
    return _add_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


@always_inline
fn _subtract_contiguous[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """AnyTensor contiguous subtraction -- delegates to typed core."""
    return _subtract_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


@always_inline
fn _multiply_contiguous[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """AnyTensor contiguous multiplication -- delegates to typed core."""
    return _multiply_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


@always_inline
fn _divide_contiguous[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """AnyTensor contiguous division -- delegates to typed core."""
    return _divide_contiguous_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()
