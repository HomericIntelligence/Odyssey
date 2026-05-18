# Linear Algebra Operations

Matrix operations and linear algebra routines.

## Matrix Multiplication

### matmul (@)

Matrix multiplication following NumPy broadcasting rules.

```mojo
fn __matmul__(self, other: AnyTensor) raises -> AnyTensor
fn matmul(a: AnyTensor, b: AnyTensor) raises -> AnyTensor
```

**Parameters:**

- `a`: First tensor (at least 1D)
- `b`: Second tensor (at least 1D)

**Shape Rules:**

- `(n,) @ (n,)` -> `()` scalar (dot product)
- `(m, n) @ (n,)` -> `(m,)` matrix-vector
- `(n,) @ (n, p)` -> `(p,)` vector-matrix
- `(m, n) @ (n, p)` -> `(m, p)` matrix-matrix
- `(b, m, n) @ (b, n, p)` -> `(b, m, p)` batched

**Examples:**

```mojo
from projectodyssey.core import randn, matmul

# Matrix-matrix multiplication
var A = randn[DType.float32](3, 4)
var B = randn[DType.float32](4, 5)
var C = A @ B  # Shape: (3, 5)
var D = matmul(A, B)  # Same as A @ B

# Matrix-vector multiplication
var x = randn[DType.float32](4)
var y = A @ x  # Shape: (3,)

# Batched matrix multiplication
var batch_A = randn[DType.float32](32, 64, 128)
var batch_B = randn[DType.float32](32, 128, 256)
var batch_C = batch_A @ batch_B  # Shape: (32, 64, 256)
```

### dot

Dot product of two 1D tensors.

```mojo
fn dot(a: AnyTensor, b: AnyTensor) raises -> AnyTensor
```

**Parameters:**

- `a`: 1D tensor
- `b`: 1D tensor (same length as a)

**Example:**

```mojo
from projectodyssey.core import randn, dot

var a = randn[DType.float32](100)
var b = randn[DType.float32](100)
var result = dot(a, b)  # Shape: () - scalar
```

### outer

Outer product of two 1D tensors.

```mojo
fn outer(a: AnyTensor, b: AnyTensor) raises -> AnyTensor
```

**Parameters:**

- `a`: 1D tensor of shape (m,)
- `b`: 1D tensor of shape (n,)

**Returns:** 2D tensor of shape (m, n)

**Example:**

```mojo
from projectodyssey.core import randn, outer

var a = randn[DType.float32](3)
var b = randn[DType.float32](4)
var result = outer(a, b)  # Shape: (3, 4)
```

## Transpose

### transpose / T

Transpose the tensor (swap last two dimensions).

```mojo
fn transpose(self) raises -> AnyTensor
fn T(self) raises -> AnyTensor  # Property alias
```

**Example:**

```mojo
var A = randn[DType.float32](3, 4)
var AT = A.T  # Shape: (4, 3)
var AT2 = A.transpose()  # Same as A.T
```

### permute

Permute dimensions in arbitrary order.

```mojo
fn permute(self, dims: List[Int]) raises -> AnyTensor
```

**Parameters:**

- `dims`: New order of dimensions

**Example:**

```mojo
var x = randn[DType.float32](2, 3, 4, 5)
var y = x.permute(List[Int](0, 2, 1, 3))  # Shape: (2, 4, 3, 5)
```

## Norms

### norm

Compute tensor norm.

```mojo
fn norm(self, p: Float = 2.0, axis: Optional[Int] = None) raises -> AnyTensor
```

**Parameters:**

- `p`: Norm order (1, 2, inf, etc.)
- `axis`: Axis to compute norm along (None for all elements)

**Example:**

```mojo
var x = randn[DType.float32](3, 4)

# Frobenius norm (L2 over all elements)
var frob = x.norm()

# L2 norm along axis 1
var row_norms = x.norm(p=2.0, axis=1)  # Shape: (3,)

# L1 norm
var l1 = x.norm(p=1.0)

# Infinity norm
var linf = x.norm(p=Float.inf)
```

## Diagonal Operations

### diag

Extract diagonal or create diagonal matrix.

```mojo
fn diag(self, offset: Int = 0) raises -> AnyTensor
```

**Parameters:**

- `offset`: Diagonal offset (0 = main, positive = above, negative = below)

**Example:**

```mojo
from projectodyssey.core import randn, eye

# Extract diagonal from matrix
var A = randn[DType.float32](4, 4)
var d = A.diag()  # Shape: (4,) - main diagonal

# Create diagonal matrix from vector
var v = randn[DType.float32](3)
var D = v.diag()  # Shape: (3, 3)
```

### trace

Compute the trace (sum of diagonal elements).

```mojo
fn trace(self) raises -> AnyTensor
```

**Example:**

```mojo
var A = eye(4, DType.float32)  # 4x4 identity
var t = A.trace()  # 4.0
```

## Decompositions

### svd

Singular value decomposition.

```mojo
fn svd(self) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]
```

**Returns:** (U, S, Vh) where `A = U @ diag(S) @ Vh`

**Example:**

```mojo
var A = randn[DType.float32](5, 3)
var (U, S, Vh) = A.svd()
# U: (5, 3), S: (3,), Vh: (3, 3)
```

### qr

QR decomposition.

```mojo
fn qr(self) raises -> Tuple[AnyTensor, AnyTensor]
```

**Returns:** (Q, R) where `A = Q @ R`

**Example:**

```mojo
var A = randn[DType.float32](5, 3)
var (Q, R) = A.qr()
# Q: (5, 3), R: (3, 3)
```

## Solving Linear Systems

### solve

Solve linear system Ax = b.

```mojo
fn solve(A: AnyTensor, b: AnyTensor) raises -> AnyTensor
```

**Parameters:**

- `A`: Coefficient matrix (n, n)
- `b`: Right-hand side vector (n,) or matrix (n, m)

**Example:**

```mojo
from projectodyssey.core import randn, solve

var A = randn[DType.float32](3, 3)
var b = randn[DType.float32](3)
var x = solve(A, b)  # Ax = b
```

### inv

Compute matrix inverse.

```mojo
fn inv(self) raises -> AnyTensor
```

**Example:**

```mojo
var A = randn[DType.float32](3, 3)
var A_inv = A.inv()
# A @ A_inv ≈ eye(3)
```

## See Also

- [Arithmetic Operations](arithmetic.md) - Element-wise operations
- [Reduction Operations](reduction.md) - Sum, mean, max, min
- [AnyTensor Reference](../tensor.md) - Core tensor class
