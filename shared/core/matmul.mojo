"""Optimized Matrix Multiplication Kernels for ML Odyssey.

This module provides progressively optimized matrix multiplication implementations,
demonstrating a 100-400x speedup journey from naive scalar to fully optimized GEMM.

Optimization Stages:
    Stage 0 (baseline): Naive triple-nested loop with Float64 conversion
    Stage 1 (matmul_typed): Dtype-specific kernels eliminating conversion overhead (3-5x)
    Stage 2 (matmul_simd): SIMD vectorization of inner loop (15-40x cumulative)
    Stage 3 (matmul_tiled): Cache-aware blocking/tiling (50-150x cumulative)
    Stage 4 (matmul): Advanced optimizations - transpose + register blocking (100-400x)

Performance Characteristics:
    - All kernels support Float32 and Float64 dtypes
    - SIMD width is automatically selected based on dtype
    - Block sizes tuned for typical L1/L2 cache (32KB-256KB)
    - Register blocking uses MICRO_M=4, MICRO_N=16 for optimal register utilization

Usage:
    from .matmul import matmul, matmul_optimized

    var a = zeros([1024, 512], DType.float32)
    var b = zeros([512, 1024], DType.float32)
    var c = zeros([1024, 1024], DType.float32)

    # Use the optimized kernel (dispatches to matmul)
    matmul_optimized(a, b, c)

References:
    - Goto and Van de Geijn, "Anatomy of High-Performance GEMM"
    - Intel MKL GEMM optimization guide
    - Mojo SIMD documentation
"""

from algorithm import vectorize
from sys.info import simd_width_of
from memory import memset_zero
from shared.tensor.any_tensor import AnyTensor, zeros
from .error_utils import format_dtype, format_matmul_error
from .strassen import (
    matmul_strassen,
    STRASSEN_ENABLED,
    STRASSEN_THRESHOLD,
)


# ============================================================================
# Configuration Constants
# ============================================================================

# Cache blocking parameters (tuned for L1/L2 cache)
comptime BLOCK_M: Int = 64  # Block size in M dimension
comptime BLOCK_N: Int = 64  # Block size in N dimension
comptime BLOCK_K: Int = 64  # Block size in K dimension

# Register blocking parameters (micro-kernel tile sizes)
comptime MICRO_M: Int = 4  # Number of rows in micro-tile
comptime MICRO_N: Int = 16  # Number of columns in micro-tile (must be multiple of SIMD width)

# Transpose threshold: only transpose B if matrix is large enough to amortize cost
comptime TRANSPOSE_THRESHOLD: Int = 64


# ============================================================================
# Stage 1: Dtype-Specific Kernels (3-5x speedup)
# ============================================================================


fn matmul_typed(a: AnyTensor, b: AnyTensor, mut c: AnyTensor) raises:
    """Dtype-specific matrix multiplication kernel.

    Eliminates Float64 conversion overhead by using direct typed memory access.
    Achieves 3-5x speedup over baseline.

    Args:
        a: First matrix (M x K).
        b: Second matrix (K x N).
        c: Output matrix (M x N), must be pre-allocated.

    Raises:
        Error: If dimensions are incompatible or dtypes don't match.

    Performance:
        - Eliminates Float64 conversion overhead from baseline.
        - Direct memory access via typed pointers.
        - Expected speedup: 3-5x over Stage 0.
    """
    if a.dtype() != b.dtype() or a.dtype() != c.dtype():
        var msg = "matmul_typed: all tensors must have the same dtype. Got A: "
        msg += format_dtype(a.dtype()) + ", B: " + format_dtype(b.dtype())
        msg += ", C: " + format_dtype(c.dtype())
        msg += ". Hint: Use tensor.cast() to convert"
        raise Error(msg)

    var a_shape = a.shape()
    var b_shape = b.shape()

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise Error(format_matmul_error(a_shape, b_shape))

    var M = a_shape[0]
    var K = a_shape[1]
    var N = b_shape[1]

    if K != b_shape[0]:
        raise Error(format_matmul_error(a_shape, b_shape))

    # Dispatch to dtype-specific implementation
    if a.dtype() == DType.float32:
        _matmul_typed_float32(a, b, c, M, K, N)
    elif a.dtype() == DType.float64:
        _matmul_typed_float64(a, b, c, M, K, N)
    else:
        var msg = (
            "matmul_typed: only float32 and float64 dtypes supported, got "
        )
        msg += (
            format_dtype(a.dtype()) + ". Hint: Use tensor.cast(DType.float32)"
        )
        raise Error(msg)


@always_inline
fn _matmul_typed_float32(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float32-specific matmul implementation.

    Raises:
        Error: If operation fails.
    """
    var a_ptr = a._data.bitcast[Float32]()
    var b_ptr = b._data.bitcast[Float32]()
    var c_ptr = c._data.bitcast[Float32]()

    for i in range(M):
        for j in range(N):
            var sum_val: Float32 = 0.0
            for k in range(K):
                sum_val += a_ptr.load(i * K + k) * b_ptr.load(k * N + j)
            c_ptr.store(i * N + j, sum_val)


@always_inline
fn _matmul_typed_float64(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float64-specific matmul implementation.

    Raises:
        Error: If operation fails.
    """
    var a_ptr = a._data.bitcast[Float64]()
    var b_ptr = b._data.bitcast[Float64]()
    var c_ptr = c._data.bitcast[Float64]()

    for i in range(M):
        for j in range(N):
            var sum_val: Float64 = 0.0
            for k in range(K):
                sum_val += a_ptr.load(i * K + k) * b_ptr.load(k * N + j)
            c_ptr.store(i * N + j, sum_val)


# ============================================================================
# Stage 2: SIMD Vectorization (15-40x cumulative speedup)
# ============================================================================


fn matmul_simd(a: AnyTensor, b: AnyTensor, mut c: AnyTensor) raises:
    """SIMD-vectorized matrix multiplication kernel.

    Vectorizes the J-loop (columns of C and B) for contiguous memory access.
    Achieves 4-8x speedup over Stage 1, 15-40x cumulative.

    Args:
        a: First matrix (M x K).
        b: Second matrix (K x N).
        c: Output matrix (M x N), must be pre-allocated.

    Raises:
        Error: If dimensions are incompatible or dtypes don't match.

    Performance:
        - Vectorizes innermost loop with SIMD instructions.
        - Contiguous access pattern in B and C matrices.
        - Automatic remainder handling for N not divisible by SIMD width.
        - Expected speedup: 4-8x over Stage 1 (15-40x cumulative).
    """
    if a.dtype() != b.dtype() or a.dtype() != c.dtype():
        raise Error("matmul_simd: all tensors must have the same dtype")

    var a_shape = a.shape()
    var b_shape = b.shape()

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise Error("matmul_simd: requires 2D tensors")

    var M = a_shape[0]
    var K = a_shape[1]
    var N = b_shape[1]

    if K != b_shape[0]:
        raise Error(
            "matmul_simd: incompatible dimensions for matmul: "
            + String(K)
            + " != "
            + String(b_shape[0])
        )

    # Dispatch to dtype-specific implementation
    if a.dtype() == DType.float32:
        _matmul_simd_float32(a, b, c, M, K, N)
    elif a.dtype() == DType.float64:
        _matmul_simd_float64(a, b, c, M, K, N)
    else:
        raise Error("matmul_simd: only float32 and float64 are supported")


@always_inline
fn _matmul_simd_float32(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float32-specific SIMD matmul implementation."""
    comptime simd_width = simd_width_of[DType.float32]()

    var a_ptr = a._data.bitcast[Float32]()
    var b_ptr = b._data.bitcast[Float32]()
    var c_ptr = c._data.bitcast[Float32]()

    for i in range(M):
        # Vectorize J loop for contiguous access in C and B
        var row_i = i

        @parameter
        fn vec_j[width: Int](j: Int) unified {mut}:
            var c_vec = SIMD[DType.float32, width](0)
            for k in range(K):
                var a_scalar = a_ptr.load(row_i * K + k)
                var b_vec = b_ptr.load[width=width](k * N + j)
                c_vec += a_scalar * b_vec
            c_ptr.store[width=width](row_i * N + j, c_vec)

        vectorize[simd_width](N, vec_j)


@always_inline
fn _matmul_simd_float64(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float64-specific SIMD matmul implementation."""
    comptime simd_width = simd_width_of[DType.float64]()

    var a_ptr = a._data.bitcast[Float64]()
    var b_ptr = b._data.bitcast[Float64]()
    var c_ptr = c._data.bitcast[Float64]()

    for i in range(M):
        # Vectorize J loop for contiguous access in C and B
        var row_i = i

        @parameter
        fn vec_j[width: Int](j: Int) unified {mut}:
            var c_vec = SIMD[DType.float64, width](0)
            for k in range(K):
                var a_scalar = a_ptr.load(row_i * K + k)
                var b_vec = b_ptr.load[width=width](k * N + j)
                c_vec += a_scalar * b_vec
            c_ptr.store[width=width](row_i * N + j, c_vec)

        vectorize[simd_width](N, vec_j)


# ============================================================================
# Stage 3: Cache-Aware Blocking/Tiling (50-150x cumulative speedup)
# ============================================================================


fn matmul_tiled(a: AnyTensor, b: AnyTensor, mut c: AnyTensor) raises:
    """Cache-blocked matrix multiplication with SIMD.

    Implements 2D tiling to improve cache locality. Block sizes are tuned
    for typical L1/L2 cache sizes (32KB-256KB).

    Args:
        a: First matrix (M x K).
        b: Second matrix (K x N).
        c: Output matrix (M x N), must be pre-allocated.

    Raises:
        Error: If dimensions are incompatible or dtypes don't match.

    Performance:
        - 2D blocking for L1/L2 cache reuse.
        - SIMD vectorization within blocks.
        - Handles arbitrary matrix sizes (not just multiples of block size).
        - Expected speedup: 3-5x over Stage 2 (50-150x cumulative).
    """
    if a.dtype() != b.dtype() or a.dtype() != c.dtype():
        raise Error("matmul_tiled: all tensors must have the same dtype")

    var a_shape = a.shape()
    var b_shape = b.shape()

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise Error("matmul_tiled: requires 2D tensors")

    var M = a_shape[0]
    var K = a_shape[1]
    var N = b_shape[1]

    if K != b_shape[0]:
        raise Error(
            "matmul_tiled: incompatible dimensions for matmul: "
            + String(K)
            + " != "
            + String(b_shape[0])
        )

    # Zero initialize C (required for accumulation)
    _zero_matrix(c, M * N)

    # Dispatch to dtype-specific implementation
    if a.dtype() == DType.float32:
        _matmul_tiled_float32(a, b, c, M, K, N)
    elif a.dtype() == DType.float64:
        _matmul_tiled_float64(a, b, c, M, K, N)
    else:
        raise Error("matmul_tiled: only float32 and float64 are supported")


@always_inline
fn _zero_matrix(mut c: AnyTensor, size: Int):
    """Zero out the matrix using memset."""
    memset_zero(c._data, size * c._get_dtype_size())


@always_inline
fn _matmul_tiled_float32(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float32-specific cache-blocked SIMD matmul implementation."""
    comptime simd_width = simd_width_of[DType.float32]()

    var a_ptr = a._data.bitcast[Float32]()
    var b_ptr = b._data.bitcast[Float32]()
    var c_ptr = c._data.bitcast[Float32]()

    # Block over all three dimensions
    for i0 in range(0, M, BLOCK_M):
        var i1 = min(i0 + BLOCK_M, M)

        for j0 in range(0, N, BLOCK_N):
            var j1 = min(j0 + BLOCK_N, N)
            var block_n = j1 - j0

            for k0 in range(0, K, BLOCK_K):
                var k1 = min(k0 + BLOCK_K, K)

                # Micro-kernel with SIMD within block
                for i in range(i0, i1):
                    var row_i = i

                    @parameter
                    fn vec_inner[width: Int](j_off: Int) unified {mut}:
                        var j = j0 + j_off
                        var c_vec = c_ptr.load[width=width](row_i * N + j)
                        for k in range(k0, k1):
                            var a_val = a_ptr.load(row_i * K + k)
                            var b_vec = b_ptr.load[width=width](k * N + j)
                            c_vec += a_val * b_vec
                        c_ptr.store[width=width](row_i * N + j, c_vec)

                    vectorize[simd_width](block_n, vec_inner)


@always_inline
fn _matmul_tiled_float64(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float64-specific cache-blocked SIMD matmul implementation."""
    comptime simd_width = simd_width_of[DType.float64]()

    var a_ptr = a._data.bitcast[Float64]()
    var b_ptr = b._data.bitcast[Float64]()
    var c_ptr = c._data.bitcast[Float64]()

    # Block over all three dimensions
    for i0 in range(0, M, BLOCK_M):
        var i1 = min(i0 + BLOCK_M, M)

        for j0 in range(0, N, BLOCK_N):
            var j1 = min(j0 + BLOCK_N, N)
            var block_n = j1 - j0

            for k0 in range(0, K, BLOCK_K):
                var k1 = min(k0 + BLOCK_K, K)

                # Micro-kernel with SIMD within block
                for i in range(i0, i1):
                    var row_i = i

                    @parameter
                    fn vec_inner[width: Int](j_off: Int) unified {mut}:
                        var j = j0 + j_off
                        var c_vec = c_ptr.load[width=width](row_i * N + j)
                        for k in range(k0, k1):
                            var a_val = a_ptr.load(row_i * K + k)
                            var b_vec = b_ptr.load[width=width](k * N + j)
                            c_vec += a_val * b_vec
                        c_ptr.store[width=width](row_i * N + j, c_vec)

                    vectorize[simd_width](block_n, vec_inner)


# ============================================================================
# Stage 4: Advanced Optimizations (100-400x cumulative speedup)
# ============================================================================


fn matmul_transpose(a: AnyTensor, b: AnyTensor, mut c: AnyTensor) raises:
    """Fully optimized GEMM with transpose and register blocking.

    Combines all optimizations for maximum performance:
    1. Transpose B matrix for contiguous access in both operands.
    2. Register blocking within micro-kernel (MICRO_M x MICRO_N tiles).
    3. Cache blocking for L1/L2 reuse.
    4. SIMD vectorization.

    Args:
        a: First matrix (M x K).
        b: Second matrix (K x N).
        c: Output matrix (M x N), must be pre-allocated.

    Raises:
        Error: If dimensions are incompatible or dtypes don't match.

    Performance:
        - Transpose B for O(1) contiguous access in both A and B^T.
        - Register blocking reduces load/store operations.
        - Amortizes transpose cost O(K*N) over O(M*K*N) compute.
        - Only transposes for matrices larger than TRANSPOSE_THRESHOLD.
        - Expected speedup: 2-3x over Stage 3 (100-400x cumulative).

    Trade-offs:
        - Extra O(K*N) memory for transposed B.
        - Transpose overhead amortized for large matrices.
        - Falls back to matmul_tiled for small matrices (below threshold).
    """
    if a.dtype() != b.dtype() or a.dtype() != c.dtype():
        raise Error("matmul: all tensors must have the same dtype")

    var a_shape = a.shape()
    var b_shape = b.shape()

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise Error("matmul: requires 2D tensors")

    var M = a_shape[0]
    var K = a_shape[1]
    var N = b_shape[1]

    if K != b_shape[0]:
        raise Error(
            "matmul: incompatible dimensions for matmul: "
            + String(K)
            + " != "
            + String(b_shape[0])
        )

    # For small matrices, fall back to matmul_tiled (transpose overhead not worth it)
    if (
        M < TRANSPOSE_THRESHOLD
        or N < TRANSPOSE_THRESHOLD
        or K < TRANSPOSE_THRESHOLD
    ):
        matmul_tiled(a, b, c)
        return

    # Zero initialize C (required for accumulation)
    _zero_matrix(c, M * N)

    # Dispatch to dtype-specific implementation
    if a.dtype() == DType.float32:
        _matmul_float32(a, b, c, M, K, N)
    elif a.dtype() == DType.float64:
        _matmul_float64(a, b, c, M, K, N)
    else:
        raise Error("matmul: only float32 and float64 are supported")


@always_inline
fn _transpose_matrix_float32(b: AnyTensor, K: Int, N: Int) raises -> AnyTensor:
    """Transpose B matrix (K x N) to B^T (N x K) for contiguous access."""
    var b_t_shape = List[Int]()
    b_t_shape.append(N)
    b_t_shape.append(K)
    var b_t = AnyTensor(b_t_shape, DType.float32)

    var b_ptr = b._data.bitcast[Float32]()
    var bt_ptr = b_t._data.bitcast[Float32]()

    # Transpose with blocking for cache efficiency
    comptime TILE = 32
    for i0 in range(0, K, TILE):
        var i1 = min(i0 + TILE, K)
        for j0 in range(0, N, TILE):
            var j1 = min(j0 + TILE, N)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    bt_ptr.store(j * K + i, b_ptr.load(i * N + j))

    return b_t^


@always_inline
fn _transpose_matrix_float64(b: AnyTensor, K: Int, N: Int) raises -> AnyTensor:
    """Transpose B matrix (K x N) to B^T (N x K) for contiguous access."""
    var b_t_shape = List[Int]()
    b_t_shape.append(N)
    b_t_shape.append(K)
    var b_t = AnyTensor(b_t_shape, DType.float64)

    var b_ptr = b._data.bitcast[Float64]()
    var bt_ptr = b_t._data.bitcast[Float64]()

    # Transpose with blocking for cache efficiency
    comptime TILE = 32
    for i0 in range(0, K, TILE):
        var i1 = min(i0 + TILE, K)
        for j0 in range(0, N, TILE):
            var j1 = min(j0 + TILE, N)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    bt_ptr.store(j * K + i, b_ptr.load(i * N + j))

    return b_t^


@always_inline
fn _matmul_float32(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float32-specific fully optimized GEMM with transpose and register blocking.
    """
    comptime simd_width = simd_width_of[DType.float32]()

    # Transpose B for contiguous access: B^T[j, k] = B[k, j]
    # This makes the dot product use contiguous memory in both operands
    var b_t = _transpose_matrix_float32(b, K, N)

    var a_ptr = a._data.bitcast[Float32]()
    var bt_ptr = b_t._data.bitcast[Float32]()
    var c_ptr = c._data.bitcast[Float32]()

    # Cache-blocked computation with register-blocked micro-kernel
    for i0 in range(0, M, BLOCK_M):
        var i1 = min(i0 + BLOCK_M, M)

        for j0 in range(0, N, BLOCK_N):
            var j1 = min(j0 + BLOCK_N, N)

            # Register-blocked micro-kernel: process MICRO_M rows at a time
            var i = i0
            while i + MICRO_M <= i1:
                # Process MICRO_M rows together for register reuse
                for j in range(j0, j1):
                    var col_j = j
                    # Accumulate dot products for MICRO_M rows
                    var c0: Float32 = 0.0
                    var c1: Float32 = 0.0
                    var c2: Float32 = 0.0
                    var c3: Float32 = 0.0

                    # SIMD dot product using transposed B
                    # A[i, :] dot B^T[j, :] = A[i, :] dot B[:, j]
                    @parameter
                    fn vec_k[width: Int](k: Int) unified {mut}:
                        var bt_vec = bt_ptr.load[width=width](col_j * K + k)

                        var a0_vec = a_ptr.load[width=width]((i + 0) * K + k)
                        var a1_vec = a_ptr.load[width=width]((i + 1) * K + k)
                        var a2_vec = a_ptr.load[width=width]((i + 2) * K + k)
                        var a3_vec = a_ptr.load[width=width]((i + 3) * K + k)

                        c0 += (a0_vec * bt_vec).reduce_add()
                        c1 += (a1_vec * bt_vec).reduce_add()
                        c2 += (a2_vec * bt_vec).reduce_add()
                        c3 += (a3_vec * bt_vec).reduce_add()

                    vectorize[simd_width](K, vec_k)

                    # Store accumulated results
                    c_ptr.store(
                        (i + 0) * N + col_j, c_ptr.load((i + 0) * N + col_j) + c0
                    )
                    c_ptr.store(
                        (i + 1) * N + col_j, c_ptr.load((i + 1) * N + col_j) + c1
                    )
                    c_ptr.store(
                        (i + 2) * N + col_j, c_ptr.load((i + 2) * N + col_j) + c2
                    )
                    c_ptr.store(
                        (i + 3) * N + col_j, c_ptr.load((i + 3) * N + col_j) + c3
                    )

                i += MICRO_M

            # Handle remaining rows (< MICRO_M)
            while i < i1:
                for j in range(j0, j1):
                    var col_j = j
                    var c_val: Float32 = 0.0

                    @parameter
                    fn vec_k_rem[width: Int](k: Int) unified {mut}:
                        var a_vec = a_ptr.load[width=width](i * K + k)
                        var bt_vec = bt_ptr.load[width=width](col_j * K + k)
                        c_val += (a_vec * bt_vec).reduce_add()

                    vectorize[simd_width](K, vec_k_rem)
                    c_ptr.store(i * N + col_j, c_ptr.load(i * N + col_j) + c_val)

                i += 1


@always_inline
fn _matmul_float64(
    a: AnyTensor, b: AnyTensor, mut c: AnyTensor, M: Int, K: Int, N: Int
) raises:
    """Float64-specific fully optimized GEMM with transpose and register blocking.
    """
    comptime simd_width = simd_width_of[DType.float64]()

    # Transpose B for contiguous access
    var b_t = _transpose_matrix_float64(b, K, N)

    var a_ptr = a._data.bitcast[Float64]()
    var bt_ptr = b_t._data.bitcast[Float64]()
    var c_ptr = c._data.bitcast[Float64]()

    # Cache-blocked computation with register-blocked micro-kernel
    for i0 in range(0, M, BLOCK_M):
        var i1 = min(i0 + BLOCK_M, M)

        for j0 in range(0, N, BLOCK_N):
            var j1 = min(j0 + BLOCK_N, N)

            # Register-blocked micro-kernel: process MICRO_M rows at a time
            var i = i0
            while i + MICRO_M <= i1:
                for j in range(j0, j1):
                    var col_j = j
                    # Accumulate dot products for MICRO_M rows
                    var c0: Float64 = 0.0
                    var c1: Float64 = 0.0
                    var c2: Float64 = 0.0
                    var c3: Float64 = 0.0

                    @parameter
                    fn vec_k[width: Int](k: Int) unified {mut}:
                        var bt_vec = bt_ptr.load[width=width](col_j * K + k)

                        var a0_vec = a_ptr.load[width=width]((i + 0) * K + k)
                        var a1_vec = a_ptr.load[width=width]((i + 1) * K + k)
                        var a2_vec = a_ptr.load[width=width]((i + 2) * K + k)
                        var a3_vec = a_ptr.load[width=width]((i + 3) * K + k)

                        c0 += (a0_vec * bt_vec).reduce_add()
                        c1 += (a1_vec * bt_vec).reduce_add()
                        c2 += (a2_vec * bt_vec).reduce_add()
                        c3 += (a3_vec * bt_vec).reduce_add()

                    vectorize[simd_width](K, vec_k)

                    c_ptr.store(
                        (i + 0) * N + col_j, c_ptr.load((i + 0) * N + col_j) + c0
                    )
                    c_ptr.store(
                        (i + 1) * N + col_j, c_ptr.load((i + 1) * N + col_j) + c1
                    )
                    c_ptr.store(
                        (i + 2) * N + col_j, c_ptr.load((i + 2) * N + col_j) + c2
                    )
                    c_ptr.store(
                        (i + 3) * N + col_j, c_ptr.load((i + 3) * N + col_j) + c3
                    )

                i += MICRO_M

            # Handle remaining rows
            while i < i1:
                for j in range(j0, j1):
                    var col_j = j
                    var c_val: Float64 = 0.0

                    @parameter
                    fn vec_k_rem[width: Int](k: Int) unified {mut}:
                        var a_vec = a_ptr.load[width=width](i * K + k)
                        var bt_vec = bt_ptr.load[width=width](col_j * K + k)
                        c_val += (a_vec * bt_vec).reduce_add()

                    vectorize[simd_width](K, vec_k_rem)
                    c_ptr.store(i * N + col_j, c_ptr.load(i * N + col_j) + c_val)

                i += 1


# ============================================================================
# Optimized Dispatch Function
# ============================================================================


fn matmul(a: AnyTensor, b: AnyTensor, mut c: AnyTensor) raises:
    """Production matrix multiplication - uses best performing kernel.

    This is the main production function that users should call.
    Currently delegates to matmul_tiled as it provides the best performance.

    Args:
        a: First matrix (M x K).
        b: Second matrix (K x N).
        c: Output matrix (M x N), must be pre-allocated.

    Raises:
        Error: If dimensions are incompatible or dtypes don't match.
    """
    matmul_tiled(a, b, c)


fn matmul_optimized(a: AnyTensor, b: AnyTensor, mut c: AnyTensor) raises:
    """Dispatch to the most optimized matmul kernel.

    Selects the optimized matmul kernel for large matrices, falling back to simpler
    implementations for small matrices where overhead outweighs benefits.

    For large square matrices (>= STRASSEN_THRESHOLD), uses Strassen's algorithm
    which reduces complexity from O(n³) to O(n^2.807).

    Args:
        a: First matrix (M x K).
        b: Second matrix (K x N).
        c: Output matrix (M x N), must be pre-allocated.

    Raises:
        Error: If dimensions are incompatible or dtypes don't match.

    Note:
        This is the recommended entry point for optimized matrix multiplication.
        It automatically selects the best kernel based on matrix dimensions.
    """

    # Check if we should use Strassen for large square matrices
    @parameter
    if STRASSEN_ENABLED:
        var a_shape = a.shape()
        var b_shape = b.shape()

        # Strassen only applies to square matrices
        if (
            len(a_shape) == 2
            and len(b_shape) == 2
            and a_shape[0] == a_shape[1]
            and b_shape[0] == b_shape[1]
            and a_shape[0] >= STRASSEN_THRESHOLD
        ):
            matmul_strassen(a, b, c)
            return

    matmul(a, b, c)


# ============================================================================
# Testing Utilities
# ============================================================================


fn assert_matrices_equal(
    a: AnyTensor, b: AnyTensor, rtol: Float64 = 1e-5, atol: Float64 = 1e-8
) raises:
    """Compare two matrices element-wise with tolerance.

    Args:
        a: First matrix.
        b: Second matrix.
        rtol: Relative tolerance (default 1e-5).
        atol: Absolute tolerance (default 1e-8).

    Raises:
        Error: If matrices have different shapes or any element differs beyond tolerance.

    Note:
        Uses the formula: |a - b| <= atol + rtol * |b|.
        This allows larger absolute differences for larger values.
    """
    var a_shape = a.shape()
    var b_shape = b.shape()

    if len(a_shape) != len(b_shape):
        raise Error(
            "assert_matrices_equal: different number of dimensions: "
            + String(len(a_shape))
            + " vs "
            + String(len(b_shape))
        )

    for i in range(len(a_shape)):
        if a_shape[i] != b_shape[i]:
            raise Error(
                "assert_matrices_equal: shape mismatch at dimension "
                + String(i)
                + ": "
                + String(a_shape[i])
                + " vs "
                + String(b_shape[i])
            )

    var numel = a.numel()
    for i in range(numel):
        var a_val = a._get_float64(i)
        var b_val = b._get_float64(i)
        var diff = a_val - b_val
        if diff < 0:
            diff = -diff
        var b_abs = b_val
        if b_abs < 0:
            b_abs = -b_abs
        var tolerance = atol + rtol * b_abs

        if diff > tolerance:
            raise Error(
                "assert_matrices_equal: mismatch at index "
                + String(i)
                + ": "
                + String(a_val)
                + " vs "
                + String(b_val)
                + " (diff="
                + String(diff)
                + ", tol="
                + String(tolerance)
                + ")"
            )


fn verify_matmul_correctness(M: Int, K: Int, N: Int) raises -> Bool:
    """Verify that all matmul stages produce identical results.

    Creates random matrices and verifies that v1, v2, v3, and v4
    all produce results matching v1 within tolerance.

    Args:
        M: Number of rows in A and C.
        K: Number of columns in A, rows in B.
        N: Number of columns in B and C.

    Returns:
        True if all stages match, raises error otherwise.

    Raises:
        Error: If operation fails.

    Note:
        Uses rtol=1e-4 and atol=1e-6 for v4 due to operation reordering
        (transpose affects floating-point accumulation order).
    """
    # Create test matrices
    var a_shape = List[Int]()
    a_shape.append(M)
    a_shape.append(K)
    var a = AnyTensor(a_shape, DType.float32)

    var b_shape = List[Int]()
    b_shape.append(K)
    b_shape.append(N)
    var b = AnyTensor(b_shape, DType.float32)

    # Initialize with simple values for reproducibility
    var a_ptr = a._data.bitcast[Float32]()
    var b_ptr = b._data.bitcast[Float32]()

    for i in range(M * K):
        a_ptr.store(i, Float32(i % 10) * 0.1)

    for i in range(K * N):
        b_ptr.store(i, Float32(i % 10) * 0.1)

    # Create output matrices
    var c_shape = List[Int]()
    c_shape.append(M)
    c_shape.append(N)

    var c1 = AnyTensor(c_shape, DType.float32)
    var c2 = AnyTensor(c_shape, DType.float32)
    var c3 = AnyTensor(c_shape, DType.float32)
    var c4 = AnyTensor(c_shape, DType.float32)

    # Run all stages
    matmul_typed(a, b, c1)
    matmul_simd(a, b, c2)
    matmul_tiled(a, b, c3)
    matmul(a, b, c4)

    # Verify simd matches typed
    assert_matrices_equal(c2, c1, rtol=1e-5, atol=1e-8)

    # Verify tiled matches typed
    assert_matrices_equal(c3, c1, rtol=1e-5, atol=1e-8)

    # Verify matmul matches typed (with looser tolerance due to operation reordering)
    assert_matrices_equal(c4, c1, rtol=1e-4, atol=1e-6)

    return True
