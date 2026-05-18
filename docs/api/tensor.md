# AnyTensor

The core tensor class in ML Odyssey. Provides a dynamic, multi-dimensional array with automatic
gradient computation support.

## Overview

```mojo
from projectodyssey.core import AnyTensor
```

`AnyTensor` is a dynamic tensor supporting:

- Arbitrary dimensions (0D scalars to N-D tensors)
- Multiple data types (float32, float16, bfloat16, int8, etc.)
- NumPy-style broadcasting
- Memory-safe reference counting

## Creation Functions

### zeros

Create a tensor filled with zeros.

```mojo
fn zeros(shape: List[Int], dtype: DType = DType.float32) raises -> AnyTensor
fn zeros[dtype: DType](dims: Int...) raises -> AnyTensor  # Variadic version
```

**Parameters:**

- `shape`: List of dimension sizes
- `dtype`: Data type (default: float32)

**Example:**

```mojo
from projectodyssey.core import zeros

var a = zeros[DType.float32](3, 4)       # Shape: (3, 4)
var b = zeros(List[Int](2, 3, 4), DType.float16)  # Shape: (2, 3, 4)
```

### ones

Create a tensor filled with ones.

```mojo
fn ones(shape: List[Int], dtype: DType = DType.float32) raises -> AnyTensor
fn ones[dtype: DType](dims: Int...) raises -> AnyTensor
```

**Example:**

```mojo
from projectodyssey.core import ones

var x = ones[DType.float32](64, 128)
```

### full

Create a tensor filled with a specified value.

```mojo
fn full(shape: List[Int], fill_value: Scalar, dtype: DType) raises -> AnyTensor
```

**Parameters:**

- `shape`: List of dimension sizes
- `fill_value`: Value to fill the tensor with
- `dtype`: Data type

**Example:**

```mojo
from projectodyssey.core import full

var x = full(List[Int](3, 3), 3.14, DType.float32)
```

### randn

Create a tensor with random values from a normal distribution.

```mojo
fn randn(shape: List[Int], dtype: DType = DType.float32, seed: Int = -1) raises -> AnyTensor
fn randn[dtype: DType](dims: Int...) raises -> AnyTensor
```

**Parameters:**

- `shape`: List of dimension sizes
- `dtype`: Data type (default: float32)
- `seed`: Random seed for reproducibility (-1 for random)

**Example:**

```mojo
from projectodyssey.core import randn

var x = randn[DType.float32](32, 784)  # Random input batch
var w = randn[DType.float32](784, 128, seed=42)  # Reproducible weights
```

### arange

Create a 1D tensor with evenly spaced values.

```mojo
fn arange(start: Scalar, stop: Scalar, step: Scalar, dtype: DType) raises -> AnyTensor
```

**Parameters:**

- `start`: Start value (inclusive)
- `stop`: End value (exclusive)
- `step`: Step size between values
- `dtype`: Data type

**Example:**

```mojo
from projectodyssey.core import arange

var x = arange(0.0, 10.0, 1.0, DType.float32)  # [0, 1, 2, ..., 9]
var y = arange(0.0, 1.0, 0.1, DType.float32)   # [0, 0.1, 0.2, ..., 0.9]
```

### eye

Create an identity matrix.

```mojo
fn eye(n: Int, dtype: DType = DType.float32) raises -> AnyTensor
```

**Parameters:**

- `n`: Size of the square identity matrix
- `dtype`: Data type (default: float32)

**Example:**

```mojo
from projectodyssey.core import eye

var I = eye(3)  # 3x3 identity matrix
```

### linspace

Create a 1D tensor with linearly spaced values.

```mojo
fn linspace(start: Scalar, stop: Scalar, num: Int, dtype: DType) raises -> AnyTensor
```

**Parameters:**

- `start`: Start value
- `stop`: End value (inclusive)
- `num`: Number of values to generate
- `dtype`: Data type

**Example:**

```mojo
from projectodyssey.core import linspace

var x = linspace(0.0, 1.0, 11, DType.float32)  # [0, 0.1, 0.2, ..., 1.0]
```

## Properties

### shape

Get the shape of the tensor.

```mojo
fn shape(self) -> List[Int]
```

**Example:**

```mojo
var x = zeros[DType.float32](3, 4, 5)
print(x.shape())  # [3, 4, 5]
```

### dtype

Get the data type of the tensor.

```mojo
fn dtype(self) -> DType
```

**Example:**

```mojo
var x = zeros[DType.float16](3, 4)
print(x.dtype())  # float16
```

### numel

Get the total number of elements.

```mojo
fn numel(self) -> Int
```

**Example:**

```mojo
var x = zeros[DType.float32](3, 4, 5)
print(x.numel())  # 60
```

### ndim

Get the number of dimensions.

```mojo
fn ndim(self) -> Int
```

**Example:**

```mojo
var x = zeros[DType.float32](3, 4, 5)
print(x.ndim())  # 3
```

### size

Get the size of a specific dimension.

```mojo
fn size(self, dim: Int) -> Int
```

**Parameters:**

- `dim`: Dimension index (supports negative indexing)

**Example:**

```mojo
var x = zeros[DType.float32](3, 4, 5)
print(x.size(0))   # 3
print(x.size(-1))  # 5
```

### is_contiguous

Check if the tensor is contiguous in memory.

```mojo
fn is_contiguous(self) -> Bool
```

**Example:**

```mojo
var x = zeros[DType.float32](3, 4)
print(x.is_contiguous())  # True

var y = x.T  # Transpose creates non-contiguous view
print(y.is_contiguous())  # False
```

## Element Access

### item

Get a scalar value from a single-element tensor.

```mojo
fn item[dtype: DType](self) -> Scalar[dtype]
```

**Example:**

```mojo
var x = full(List[Int](1), 3.14, DType.float32)
var value = x.item[DType.float32]()  # 3.14
```

### \_\_getitem\_\_

Access elements using indexing.

```mojo
fn __getitem__(self, indices: List[Int]) raises -> AnyTensor
```

**Example:**

```mojo
var x = randn[DType.float32](3, 4)
var row = x[List[Int](0)]  # First row
var elem = x[List[Int](1, 2)]  # Element at (1, 2)
```

## Shape Operations

### reshape

Reshape tensor to new dimensions.

```mojo
fn reshape(self, new_shape: List[Int]) raises -> AnyTensor
```

**Parameters:**

- `new_shape`: New shape (must have same total elements)

**Example:**

```mojo
var x = arange(0.0, 12.0, 1.0, DType.float32)  # Shape: (12,)
var y = x.reshape(List[Int](3, 4))             # Shape: (3, 4)
var z = y.reshape(List[Int](2, 6))             # Shape: (2, 6)
```

### transpose / T

Transpose the tensor.

```mojo
fn transpose(self) raises -> AnyTensor
fn T(self) raises -> AnyTensor  # Alias
```

**Example:**

```mojo
var x = randn[DType.float32](3, 4)
var y = x.T  # Shape: (4, 3)
```

### squeeze

Remove dimensions of size 1.

```mojo
fn squeeze(self, dim: Optional[Int] = None) raises -> AnyTensor
```

**Parameters:**

- `dim`: Dimension to squeeze (None for all)

**Example:**

```mojo
var x = zeros[DType.float32](1, 3, 1, 4)
var y = x.squeeze()          # Shape: (3, 4)
var z = x.squeeze(dim=0)     # Shape: (3, 1, 4)
```

### unsqueeze

Add a dimension of size 1.

```mojo
fn unsqueeze(self, dim: Int) raises -> AnyTensor
```

**Parameters:**

- `dim`: Position to insert new dimension

**Example:**

```mojo
var x = zeros[DType.float32](3, 4)
var y = x.unsqueeze(0)   # Shape: (1, 3, 4)
var z = x.unsqueeze(-1)  # Shape: (3, 4, 1)
```

### contiguous

Return a contiguous copy of the tensor.

```mojo
fn contiguous(self) raises -> AnyTensor
```

**Example:**

```mojo
var x = randn[DType.float32](4, 3)
var y = x.T  # Non-contiguous view
var z = y.contiguous()  # Contiguous copy
```

## See Also

- [Arithmetic Operations](operations/arithmetic.md)
- [Reduction Operations](operations/reduction.md)
- [Linear Algebra](operations/linalg.md)
- [Indexing and Slicing](operations/indexing.md)
