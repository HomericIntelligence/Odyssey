# Reduction Operations

Operations that reduce tensor dimensions by aggregating values.

## Overview

Reduction operations collapse one or more dimensions of a tensor by computing
an aggregate value (sum, mean, max, etc.).

## Sum

Compute the sum of tensor elements.

```mojo
fn sum(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> AnyTensor
```

**Parameters:**

- `axis`: Dimension to reduce (None for all elements)
- `keepdim`: Keep reduced dimension as size 1

**Examples:**

```mojo
from odyssey.core import randn

var x = randn[DType.float32](3, 4)

# Sum all elements
var total = x.sum()  # Shape: () - scalar

# Sum along axis 0
var col_sums = x.sum(axis=0)  # Shape: (4,)

# Sum along axis 1, keep dimension
var row_sums = x.sum(axis=1, keepdim=True)  # Shape: (3, 1)
```

## Mean

Compute the mean of tensor elements.

```mojo
fn mean(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> AnyTensor
```

**Parameters:**

- `axis`: Dimension to reduce (None for all elements)
- `keepdim`: Keep reduced dimension as size 1

**Examples:**

```mojo
var x = randn[DType.float32](3, 4)

# Mean of all elements
var avg = x.mean()  # Shape: ()

# Mean along axis 0
var col_means = x.mean(axis=0)  # Shape: (4,)

# Mean along axis 1, keep dimension
var row_means = x.mean(axis=1, keepdim=True)  # Shape: (3, 1)
```

## Max

Compute the maximum value of tensor elements.

```mojo
fn max(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> AnyTensor
```

**Parameters:**

- `axis`: Dimension to reduce (None for all elements)
- `keepdim`: Keep reduced dimension as size 1

**Examples:**

```mojo
var x = randn[DType.float32](3, 4)

# Maximum of all elements
var maximum = x.max()  # Shape: ()

# Maximum along axis 0
var col_max = x.max(axis=0)  # Shape: (4,)

# Maximum along axis 1, keep dimension
var row_max = x.max(axis=1, keepdim=True)  # Shape: (3, 1)
```

## Min

Compute the minimum value of tensor elements.

```mojo
fn min(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> AnyTensor
```

**Parameters:**

- `axis`: Dimension to reduce (None for all elements)
- `keepdim`: Keep reduced dimension as size 1

**Examples:**

```mojo
var x = randn[DType.float32](3, 4)

# Minimum of all elements
var minimum = x.min()  # Shape: ()

# Minimum along axis 0
var col_min = x.min(axis=0)  # Shape: (4,)

# Minimum along axis 1, keep dimension
var row_min = x.min(axis=1, keepdim=True)  # Shape: (3, 1)
```

## Argmax

Find the index of the maximum value.

```mojo
fn argmax(self, axis: Optional[Int] = None) raises -> AnyTensor
```

**Parameters:**

- `axis`: Dimension to reduce (None for flattened index)

**Examples:**

```mojo
var x = randn[DType.float32](3, 4)

# Index of maximum in flattened tensor
var idx = x.argmax()  # Shape: () - single index

# Indices of maximum along axis 0
var col_argmax = x.argmax(axis=0)  # Shape: (4,) - index per column

# Indices of maximum along axis 1
var row_argmax = x.argmax(axis=1)  # Shape: (3,) - index per row
```

## Argmin

Find the index of the minimum value.

```mojo
fn argmin(self, axis: Optional[Int] = None) raises -> AnyTensor
```

**Parameters:**

- `axis`: Dimension to reduce (None for flattened index)

**Examples:**

```mojo
var x = randn[DType.float32](3, 4)

# Index of minimum in flattened tensor
var idx = x.argmin()  # Shape: ()

# Indices of minimum along axis 1
var row_argmin = x.argmin(axis=1)  # Shape: (3,)
```

## Prod

Compute the product of tensor elements.

```mojo
fn prod(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> AnyTensor
```

**Parameters:**

- `axis`: Dimension to reduce (None for all elements)
- `keepdim`: Keep reduced dimension as size 1

**Examples:**

```mojo
var x = full(List[Int](2, 3), 2.0, DType.float32)  # All 2s
var product = x.prod()  # 64.0 (2^6)
```

## Any / All

Logical reduction operations.

```mojo
fn any(self, axis: Optional[Int] = None) raises -> AnyTensor
fn all(self, axis: Optional[Int] = None) raises -> AnyTensor
```

**Examples:**

```mojo
var x = zeros[DType.float32](3, 4)
x[List[Int](0, 0)] = 1.0  # Set one element to non-zero

var has_nonzero = x.any()  # True
var all_nonzero = x.all()  # False
```

## Keepdim Behavior

The `keepdim` parameter controls output shape:

```text
Input shape: (3, 4, 5)

# Without keepdim (default)
sum(axis=1).shape()  -> (3, 5)     # Dimension removed

# With keepdim=True
sum(axis=1, keepdim=True).shape()  -> (3, 1, 5)  # Dimension kept as 1
```

`keepdim=True` is useful for broadcasting the result back to the original shape.

## Negative Axis

Negative axis values count from the end:

```mojo
var x = randn[DType.float32](3, 4, 5)

x.sum(axis=-1)  # Same as axis=2, shape: (3, 4)
x.sum(axis=-2)  # Same as axis=1, shape: (3, 5)
```

## See Also

- [Arithmetic Operations](arithmetic.md) - Element-wise operations
- [Linear Algebra](linalg.md) - Matrix operations
- [AnyTensor Reference](../tensor.md) - Core tensor class
