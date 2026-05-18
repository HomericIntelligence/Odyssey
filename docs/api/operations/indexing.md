# Indexing and Slicing

Access and manipulate tensor elements and subarrays.

## Basic Indexing

### Single Element Access

Access individual elements using a list of indices.

```mojo
fn __getitem__(self, indices: List[Int]) raises -> AnyTensor
```

**Example:**

```mojo
from projectodyssey.core import randn

var x = randn[DType.float32](3, 4, 5)

# Access single element (returns 0D tensor)
var elem = x[List[Int](0, 1, 2)]

# Access row
var row = x[List[Int](0)]  # Shape: (4, 5)

# Access slice along first two dimensions
var slice = x[List[Int](0, 1)]  # Shape: (5,)
```

### Setting Elements

Set individual elements.

```mojo
fn __setitem__(mut self, indices: List[Int], value: Scalar) raises
fn __setitem__(mut self, indices: List[Int], value: AnyTensor) raises
```

**Example:**

```mojo
var x = zeros[DType.float32](3, 4)

# Set single element
x[List[Int](0, 0)] = 1.0

# Set row
x[List[Int](1)] = ones[DType.float32](4)
```

## Slicing

### slice

Extract a contiguous subtensor.

```mojo
fn slice(self, starts: List[Int], ends: List[Int]) raises -> AnyTensor
```

**Parameters:**

- `starts`: Start indices for each dimension
- `ends`: End indices for each dimension (exclusive)

**Example:**

```mojo
var x = randn[DType.float32](10, 20)

# Slice rows 2-5, columns 5-15
var starts = List[Int](2, 5)
var ends = List[Int](5, 15)
var y = x.slice(starts, ends)  # Shape: (3, 10)
```

### slice_along_axis

Slice along a single axis.

```mojo
fn slice_along_axis(self, axis: Int, start: Int, end: Int) raises -> AnyTensor
```

**Example:**

```mojo
var x = randn[DType.float32](32, 64, 128)

# Slice first 16 along axis 0
var y = x.slice_along_axis(axis=0, start=0, end=16)  # Shape: (16, 64, 128)

# Slice middle section along axis 2
var z = x.slice_along_axis(axis=2, start=32, end=96)  # Shape: (32, 64, 64)
```

## Advanced Indexing

### index_select

Select elements along an axis using indices tensor.

```mojo
fn index_select(self, axis: Int, indices: AnyTensor) raises -> AnyTensor
```

**Parameters:**

- `axis`: Axis to select along
- `indices`: 1D tensor of indices to select

**Example:**

```mojo
var x = randn[DType.float32](5, 3)

# Select rows 0, 2, 4
var idx = arange(0.0, 5.0, 2.0, DType.int32)  # [0, 2, 4]
var y = x.index_select(axis=0, idx)  # Shape: (3, 3)
```

### gather

Gather elements using multi-dimensional indices.

```mojo
fn gather(self, axis: Int, indices: AnyTensor) raises -> AnyTensor
```

**Parameters:**

- `axis`: Axis to gather along
- `indices`: Index tensor (same shape as output)

**Example:**

```mojo
var x = randn[DType.float32](3, 4)

# Gather specific elements
var indices = zeros[DType.int32](3, 2)  # 3x2 indices
var y = x.gather(axis=1, indices)  # Shape: (3, 2)
```

### scatter

Scatter values into tensor at specified indices.

```mojo
fn scatter(mut self, axis: Int, indices: AnyTensor, values: AnyTensor) raises
```

**Example:**

```mojo
var x = zeros[DType.float32](3, 4)
var indices = zeros[DType.int32](3, 2)
var values = ones[DType.float32](3, 2)
x.scatter(axis=1, indices, values)
```

## Masking

### masked_select

Select elements where mask is true.

```mojo
fn masked_select(self, mask: AnyTensor) raises -> AnyTensor
```

**Parameters:**

- `mask`: Boolean tensor (same shape as self)

**Returns:** 1D tensor of selected elements

**Example:**

```mojo
var x = randn[DType.float32](3, 4)
var mask = x > 0.0  # Boolean mask
var positive = x.masked_select(mask)  # 1D tensor
```

### masked_fill

Fill elements where mask is true.

```mojo
fn masked_fill(mut self, mask: AnyTensor, value: Scalar) raises
```

**Example:**

```mojo
var x = randn[DType.float32](3, 4)
var mask = x < 0.0
x.masked_fill(mask, 0.0)  # Set negative values to zero
```

### where

Select elements from two tensors based on condition.

```mojo
fn where(condition: AnyTensor, x: AnyTensor, y: AnyTensor) raises -> AnyTensor
```

**Example:**

```mojo
from projectodyssey.core import where, randn, zeros

var condition = randn[DType.float32](3, 4) > 0.0
var x = ones[DType.float32](3, 4)
var y = zeros[DType.float32](3, 4)
var result = where(condition, x, y)  # 1 where positive, 0 otherwise
```

## Concatenation and Stacking

### concatenate

Join tensors along an existing axis.

```mojo
fn concatenate(tensors: List[AnyTensor], axis: Int = 0) raises -> AnyTensor
```

**Example:**

```mojo
from projectodyssey.core import concatenate, randn

var a = randn[DType.float32](2, 3)
var b = randn[DType.float32](3, 3)
var c = concatenate(List[AnyTensor](a, b), axis=0)  # Shape: (5, 3)
```

### stack

Join tensors along a new axis.

```mojo
fn stack(tensors: List[AnyTensor], axis: Int = 0) raises -> AnyTensor
```

**Example:**

```mojo
from projectodyssey.core import stack, randn

var a = randn[DType.float32](3, 4)
var b = randn[DType.float32](3, 4)
var c = stack(List[AnyTensor](a, b), axis=0)  # Shape: (2, 3, 4)
```

### split

Split tensor into chunks.

```mojo
fn split(self, chunks: Int, axis: Int = 0) raises -> List[AnyTensor]
```

**Example:**

```mojo
var x = randn[DType.float32](10, 4)
var chunks = x.split(5, axis=0)  # 5 tensors of shape (2, 4)
```

## View Operations

### view

Create a view with different shape (no copy).

```mojo
fn view(self, new_shape: List[Int]) raises -> AnyTensor
```

**Example:**

```mojo
var x = randn[DType.float32](6, 4)
var y = x.view(List[Int](2, 3, 4))  # Shares data with x
```

### flatten

Flatten to 1D tensor.

```mojo
fn flatten(self) raises -> AnyTensor
```

**Example:**

```mojo
var x = randn[DType.float32](2, 3, 4)
var y = x.flatten()  # Shape: (24,)
```

## See Also

- [Shape Operations](../tensor.md#shape-operations) - reshape, squeeze, unsqueeze
- [Arithmetic Operations](arithmetic.md) - Element-wise operations
- [AnyTensor Reference](../tensor.md) - Core tensor class
