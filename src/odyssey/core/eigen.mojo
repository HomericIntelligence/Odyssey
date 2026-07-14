"""Symmetric eigendecomposition via the cyclic Jacobi method.

Computes the eigenvalues and eigenvectors of a real symmetric matrix `A = Q Λ Qᵀ`
using cyclic Jacobi rotations. Jacobi is chosen over QR-based methods because it is
short, numerically robust for the small symmetric positive-semidefinite matrices that
arise as Kronecker preconditioners (e.g. in SOAP), and easy to validate against
`numpy.linalg.eigh`.

The method repeatedly applies Givens rotations that zero the largest off-diagonal
entry; the product of the rotations converges to the orthogonal eigenvector matrix
`Q`, and the matrix converges to the diagonal matrix of eigenvalues. Eigenvalues are
returned in ASCENDING order (matching `numpy.linalg.eigh` / `torch.linalg.eigh`), with
the corresponding eigenvectors as the COLUMNS of `Q`.
"""

from std.math import sqrt as scalar_sqrt
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros


def symmetric_eigh(
    matrix: AnyTensor,
    max_sweeps: Int = 100,
    tolerance: Float64 = 1e-15,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Eigendecomposition of a real symmetric matrix via cyclic Jacobi.

    Returns `(eigenvalues, Q)` where `eigenvalues` is a length-N vector in ascending
    order and `Q` is the N×N matrix whose COLUMNS are the corresponding orthonormal
    eigenvectors, so that `A = Q diag(eigenvalues) Qᵀ`. Matches the ordering and
    layout of `numpy.linalg.eigh`.

    The input is assumed symmetric; only the values are read (the caller is
    responsible for symmetry). Computation is done in float64 internally regardless
    of the input dtype, and the outputs are float64.

    Args:
        matrix: A rank-2 square symmetric tensor (N×N).
        max_sweeps: Maximum number of Jacobi sweeps (each sweep touches every
            off-diagonal pair once). Default 100 — ample for small matrices.
        tolerance: Convergence threshold on the off-diagonal Frobenius norm; the
            iteration stops early once the off-diagonal mass falls below this.

    Returns:
        Tuple `(eigenvalues, Q)`: a length-N float64 vector and an N×N float64 matrix.

    Raises:
        Error: If the matrix is not rank-2 square.
    """
    var shape = matrix.shape()
    if matrix.ndim() != 2 or shape[0] != shape[1]:
        raise Error("symmetric_eigh requires a square rank-2 matrix")
    var n = shape[0]

    # Working copy A (row-major flat float64) and the accumulating rotation matrix V.
    var a = List[Float64](capacity=n * n)
    for i in range(n * n):
        a.append(matrix.load[DType.float64](i))
    # V starts as the identity.
    var v = List[Float64](capacity=n * n)
    for i in range(n):
        for j in range(n):
            v.append(1.0 if i == j else 0.0)

    # 1×1 matrix: already diagonal.
    if n == 1:
        var vals1 = zeros([1], DType.float64)
        vals1.store[DType.float64](0, a[0])
        var q1 = zeros([1, 1], DType.float64)
        q1.store[DType.float64](0, 1.0)
        return (vals1, q1)

    for _sweep in range(max_sweeps):
        # Off-diagonal Frobenius mass (sum of squares of upper-triangle entries).
        var off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                var apq = a[p * n + q]
                off += apq * apq
        if off < tolerance * tolerance:
            break

        # One sweep over all upper-triangle pairs (p < q).
        for p in range(n):
            for q in range(p + 1, n):
                var apq = a[p * n + q]
                if apq == 0.0:
                    continue
                var app = a[p * n + p]
                var aqq = a[q * n + q]
                # Jacobi rotation angle: theta from cot(2θ) = (aqq - app)/(2 apq).
                var tau = (aqq - app) / (2.0 * apq)
                var t: Float64
                if tau >= 0.0:
                    t = 1.0 / (tau + scalar_sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + scalar_sqrt(1.0 + tau * tau))
                var c = 1.0 / scalar_sqrt(1.0 + t * t)
                var s = t * c

                # Apply the rotation to rows/cols p and q of A: A = Jᵀ A J.
                for k in range(n):
                    var akp = a[k * n + p]
                    var akq = a[k * n + q]
                    a[k * n + p] = c * akp - s * akq
                    a[k * n + q] = s * akp + c * akq
                for k in range(n):
                    var apk = a[p * n + k]
                    var aqk = a[q * n + k]
                    a[p * n + k] = c * apk - s * aqk
                    a[q * n + k] = s * apk + c * aqk

                # Accumulate the rotation into V (eigenvectors): V = V J.
                for k in range(n):
                    var vkp = v[k * n + p]
                    var vkq = v[k * n + q]
                    v[k * n + p] = c * vkp - s * vkq
                    v[k * n + q] = s * vkp + c * vkq

    # Eigenvalues are the diagonal of A; eigenvectors are the columns of V.
    var eigenvalues = List[Float64](capacity=n)
    for i in range(n):
        eigenvalues.append(a[i * n + i])

    # Sort ascending by eigenvalue (selection sort — N is small).
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    for i in range(n):
        var min_idx = i
        for j in range(i + 1, n):
            if eigenvalues[order[j]] < eigenvalues[order[min_idx]]:
                min_idx = j
        var tmp = order[i]
        order[i] = order[min_idx]
        order[min_idx] = tmp

    var vals = zeros([n], DType.float64)
    var q_out = zeros([n, n], DType.float64)
    for new_col in range(n):
        var src = order[new_col]
        vals.store[DType.float64](new_col, eigenvalues[src])
        for row in range(n):
            q_out.store[DType.float64](row * n + new_col, v[row * n + src])

    return (vals, q_out)
