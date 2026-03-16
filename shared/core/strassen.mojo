"""Strassen's Algorithm for Fast Matrix Multiplication.

Implements Strassen's divide-and-conquer algorithm reducing O(n³) to O(n^2.807)
by performing 7 multiplications instead of 8.
"""

from .extensor import ExTensor, zeros
from .arithmetic import add, subtract
from .matmul import matmul_tiled

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


fn _strassen_recursive(A: ExTensor, B: ExTensor) raises -> ExTensor:
    """Recursive core of Strassen's algorithm using 7 products."""
    var shape = A.shape()
    var n = shape[0]

    if n <= STRASSEN_THRESHOLD:
        var c_shape = List[Int]()
        c_shape.append(n)
        c_shape.append(n)
        var C = ExTensor(c_shape, A.dtype())
        matmul_tiled(A, B, C)
        return C^

    var n_half = n // 2

    # Extract quadrants
    var A11_shape = List[Int]()
    A11_shape.append(n_half)
    A11_shape.append(n_half)
    var A11 = ExTensor(A11_shape, A.dtype())

    var A12 = ExTensor(A11_shape, A.dtype())
    var A21 = ExTensor(A11_shape, A.dtype())
    var A22 = ExTensor(A11_shape, A.dtype())

    var B11 = ExTensor(A11_shape, B.dtype())
    var B12 = ExTensor(A11_shape, B.dtype())
    var B21 = ExTensor(A11_shape, B.dtype())
    var B22 = ExTensor(A11_shape, B.dtype())

    # Extract A quadrants
    if A.dtype() == DType.float32:
        var a_ptr = A._data.bitcast[Float32]()
        var a11_ptr = A11._data.bitcast[Float32]()
        var a12_ptr = A12._data.bitcast[Float32]()
        var a21_ptr = A21._data.bitcast[Float32]()
        var a22_ptr = A22._data.bitcast[Float32]()

        for i in range(n_half):
            for j in range(n_half):
                a11_ptr.store(i * n_half + j, a_ptr.load(i * n + j))
                a12_ptr.store(i * n_half + j, a_ptr.load(i * n + (j + n_half)))
                a21_ptr.store(i * n_half + j, a_ptr.load((i + n_half) * n + j))
                a22_ptr.store(
                    i * n_half + j, a_ptr.load((i + n_half) * n + (j + n_half))
                )

        var b_ptr = B._data.bitcast[Float32]()
        var b11_ptr = B11._data.bitcast[Float32]()
        var b12_ptr = B12._data.bitcast[Float32]()
        var b21_ptr = B21._data.bitcast[Float32]()
        var b22_ptr = B22._data.bitcast[Float32]()

        for i in range(n_half):
            for j in range(n_half):
                b11_ptr.store(i * n_half + j, b_ptr.load(i * n + j))
                b12_ptr.store(i * n_half + j, b_ptr.load(i * n + (j + n_half)))
                b21_ptr.store(i * n_half + j, b_ptr.load((i + n_half) * n + j))
                b22_ptr.store(
                    i * n_half + j, b_ptr.load((i + n_half) * n + (j + n_half))
                )
    else:
        var a_ptr = A._data.bitcast[Float64]()
        var a11_ptr = A11._data.bitcast[Float64]()
        var a12_ptr = A12._data.bitcast[Float64]()
        var a21_ptr = A21._data.bitcast[Float64]()
        var a22_ptr = A22._data.bitcast[Float64]()

        for i in range(n_half):
            for j in range(n_half):
                a11_ptr.store(i * n_half + j, a_ptr.load(i * n + j))
                a12_ptr.store(i * n_half + j, a_ptr.load(i * n + (j + n_half)))
                a21_ptr.store(i * n_half + j, a_ptr.load((i + n_half) * n + j))
                a22_ptr.store(
                    i * n_half + j, a_ptr.load((i + n_half) * n + (j + n_half))
                )

        var b_ptr = B._data.bitcast[Float64]()
        var b11_ptr = B11._data.bitcast[Float64]()
        var b12_ptr = B12._data.bitcast[Float64]()
        var b21_ptr = B21._data.bitcast[Float64]()
        var b22_ptr = B22._data.bitcast[Float64]()

        for i in range(n_half):
            for j in range(n_half):
                b11_ptr.store(i * n_half + j, b_ptr.load(i * n + j))
                b12_ptr.store(i * n_half + j, b_ptr.load(i * n + (j + n_half)))
                b21_ptr.store(i * n_half + j, b_ptr.load((i + n_half) * n + j))
                b22_ptr.store(
                    i * n_half + j, b_ptr.load((i + n_half) * n + (j + n_half))
                )

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

    # Combine quadrants
    var c_shape = List[Int]()
    c_shape.append(n)
    c_shape.append(n)
    var C = ExTensor(c_shape, A.dtype())

    if A.dtype() == DType.float32:
        var c_ptr = C._data.bitcast[Float32]()
        var c11_ptr = C11._data.bitcast[Float32]()
        var c12_ptr = C12._data.bitcast[Float32]()
        var c21_ptr = C21._data.bitcast[Float32]()
        var c22_ptr = C22._data.bitcast[Float32]()

        for i in range(n_half):
            for j in range(n_half):
                c_ptr.store(i * n + j, c11_ptr.load(i * n_half + j))
                c_ptr.store(i * n + (j + n_half), c12_ptr.load(i * n_half + j))
                c_ptr.store((i + n_half) * n + j, c21_ptr.load(i * n_half + j))
                c_ptr.store(
                    (i + n_half) * n + (j + n_half),
                    c22_ptr.load(i * n_half + j),
                )
    else:
        var c_ptr = C._data.bitcast[Float64]()
        var c11_ptr = C11._data.bitcast[Float64]()
        var c12_ptr = C12._data.bitcast[Float64]()
        var c21_ptr = C21._data.bitcast[Float64]()
        var c22_ptr = C22._data.bitcast[Float64]()

        for i in range(n_half):
            for j in range(n_half):
                c_ptr.store(i * n + j, c11_ptr.load(i * n_half + j))
                c_ptr.store(i * n + (j + n_half), c12_ptr.load(i * n_half + j))
                c_ptr.store((i + n_half) * n + j, c21_ptr.load(i * n_half + j))
                c_ptr.store(
                    (i + n_half) * n + (j + n_half),
                    c22_ptr.load(i * n_half + j),
                )

    return C^


fn matmul_strassen(A: ExTensor, B: ExTensor, mut C: ExTensor) raises:
    """Matrix multiplication using Strassen's algorithm."""
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

    # Copy result
    if C.dtype() == DType.float32:
        var src_ptr = C_result._data.bitcast[Float32]()
        var dst_ptr = C._data.bitcast[Float32]()
        for i in range(M):
            for j in range(N):
                var idx = i * N + j
                dst_ptr.store(idx, src_ptr.load(idx))
    else:
        var src_ptr = C_result._data.bitcast[Float64]()
        var dst_ptr = C._data.bitcast[Float64]()
        for i in range(M):
            for j in range(N):
                var idx = i * N + j
                dst_ptr.store(idx, src_ptr.load(idx))
