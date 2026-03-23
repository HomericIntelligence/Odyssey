"""Strassen's Algorithm for Fast Matrix Multiplication.

Implements Strassen's divide-and-conquer algorithm reducing O(n^3) to O(n^2.807)
by performing 7 multiplications instead of 8.

Typed helper cores live in shared/tensor/typed/strassen.mojo.
"""

from shared.tensor.any_tensor import AnyTensor, zeros
from .arithmetic import add, subtract
from .matmul import matmul_tiled
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

comptime STRASSEN_ENABLED: Bool = True
comptime STRASSEN_THRESHOLD: Int = 512


fn next_power_of_2(n: Int) -> Int:
    """Find the next power of 2 >= n."""
    if n == 0:
        return 1
    var power = 1
    while power < n:
        power *= 2
    return power


fn _strassen_recursive(A: AnyTensor, B: AnyTensor) raises -> AnyTensor:
    """Recursive core of Strassen's algorithm using 7 products."""
    from shared.tensor.typed.strassen import (
        _extract_quadrants_typed,
        _combine_quadrants_typed,
    )

    var shape = A.shape()
    var n = shape[0]

    if n <= STRASSEN_THRESHOLD:
        var c_shape = List[Int]()
        c_shape.append(n)
        c_shape.append(n)
        var C = AnyTensor(c_shape, A.dtype())
        matmul_tiled(A, B, C)
        return C^

    var n_half = n // 2

    # Extract quadrants using typed core (zero bitcasts)
    var ordinal = dtype_to_ordinal(A.dtype())
    var A11: AnyTensor
    var A12: AnyTensor
    var A21: AnyTensor
    var A22: AnyTensor
    var B11: AnyTensor
    var B12: AnyTensor
    var B21: AnyTensor
    var B22: AnyTensor

    if ordinal == DTYPE_FLOAT32:
        var a_typed = A.as_tensor[DType.float32]()
        var b_typed = B.as_tensor[DType.float32]()
        var a_quads = _extract_quadrants_typed[DType.float32](
            a_typed, n, n_half
        )
        var b_quads = _extract_quadrants_typed[DType.float32](
            b_typed, n, n_half
        )
        A11 = a_quads[0].as_any()
        A12 = a_quads[1].as_any()
        A21 = a_quads[2].as_any()
        A22 = a_quads[3].as_any()
        B11 = b_quads[0].as_any()
        B12 = b_quads[1].as_any()
        B21 = b_quads[2].as_any()
        B22 = b_quads[3].as_any()
    elif ordinal == DTYPE_FLOAT64:
        var a_typed = A.as_tensor[DType.float64]()
        var b_typed = B.as_tensor[DType.float64]()
        var a_quads = _extract_quadrants_typed[DType.float64](
            a_typed, n, n_half
        )
        var b_quads = _extract_quadrants_typed[DType.float64](
            b_typed, n, n_half
        )
        A11 = a_quads[0].as_any()
        A12 = a_quads[1].as_any()
        A21 = a_quads[2].as_any()
        A22 = a_quads[3].as_any()
        B11 = b_quads[0].as_any()
        B12 = b_quads[1].as_any()
        B21 = b_quads[2].as_any()
        B22 = b_quads[3].as_any()
    else:
        raise Error("strassen: only float32/float64 supported")

    # Compute 7 products
    var sum_a1 = add(A11, A22)
    var sum_b1 = add(B11, B22)
    var M1 = _strassen_recursive(sum_a1, sum_b1)

    var sum_a2 = add(A21, A22)
    var M2 = _strassen_recursive(sum_a2, B11)

    var diff_b1 = subtract(B12, B22)
    var M3 = _strassen_recursive(A11, diff_b1)

    var diff_b2 = subtract(B21, B11)
    var M4 = _strassen_recursive(A22, diff_b2)

    var sum_a3 = add(A11, A12)
    var M5 = _strassen_recursive(sum_a3, B22)

    var diff_a1 = subtract(A21, A11)
    var sum_b2 = add(B11, B12)
    var M6 = _strassen_recursive(diff_a1, sum_b2)

    var diff_a2 = subtract(A12, A22)
    var sum_b3 = add(B21, B22)
    var M7 = _strassen_recursive(diff_a2, sum_b3)

    # Combine results
    var C11 = add(M1, M4)
    var C11_temp = subtract(C11, M5)
    C11 = add(C11_temp, M7)

    var C12 = add(M3, M5)
    var C21 = add(M2, M4)

    var C22 = subtract(M1, M2)
    var C22_temp = add(C22, M3)
    C22 = add(C22_temp, M6)

    # Combine quadrants using typed core (zero bitcasts)
    if ordinal == DTYPE_FLOAT32:
        return _combine_quadrants_typed[DType.float32](
            C11.as_tensor[DType.float32](),
            C12.as_tensor[DType.float32](),
            C21.as_tensor[DType.float32](),
            C22.as_tensor[DType.float32](),
            n,
            n_half,
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _combine_quadrants_typed[DType.float64](
            C11.as_tensor[DType.float64](),
            C12.as_tensor[DType.float64](),
            C21.as_tensor[DType.float64](),
            C22.as_tensor[DType.float64](),
            n,
            n_half,
        ).as_any()
    else:
        raise Error("strassen: only float32/float64 supported")


fn matmul_strassen(A: AnyTensor, B: AnyTensor, mut C: AnyTensor) raises:
    """Matrix multiplication using Strassen's algorithm."""
    from shared.tensor.typed.strassen import _matmul_strassen_copy_result

    if A.dtype() != B.dtype() or A.dtype() != C.dtype():
        raise Error("matmul_strassen: all tensors must have the same dtype")

    var a_shape = A.shape()
    var b_shape = B.shape()
    var c_shape = C.shape()

    if len(a_shape) != 2 or len(b_shape) != 2 or len(c_shape) != 2:
        raise Error("matmul_strassen: all tensors must be 2D")

    var M = a_shape[0]
    var K = a_shape[1]
    var N = b_shape[1]

    if K != b_shape[0]:
        raise Error(
            "matmul_strassen: dimension mismatch: A.shape[1]="
            + String(K)
            + " != B.shape[0]="
            + String(b_shape[0])
        )

    if c_shape[0] != M or c_shape[1] != N:
        raise Error(
            "matmul_strassen: C must have shape ("
            + String(M)
            + ", "
            + String(N)
            + "), got ("
            + String(c_shape[0])
            + ", "
            + String(c_shape[1])
            + ")"
        )

    # For small matrices or rectangular, use standard GEMM
    var max_dim = M if M > K else K
    max_dim = N if N > max_dim else max_dim

    if max_dim < STRASSEN_THRESHOLD or M != K or K != N:
        matmul_tiled(A, B, C)
        return

    # For square matrices above threshold, use Strassen
    var C_result = _strassen_recursive(A, B)

    # Copy result using typed core (zero bitcasts)
    var ordinal = dtype_to_ordinal(C.dtype())
    if ordinal == DTYPE_FLOAT32:
        _matmul_strassen_copy_result[DType.float32](C_result, C, M, N)
    elif ordinal == DTYPE_FLOAT64:
        _matmul_strassen_copy_result[DType.float64](C_result, C, M, N)
    else:
        raise Error("matmul_strassen: only float32/float64 supported")
