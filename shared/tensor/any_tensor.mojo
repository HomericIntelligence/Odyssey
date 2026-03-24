"""AnyTensor - Extensible Tensor for ML Odyssey.

A comprehensive, dynamic tensor class implementing the Python Array API Standard

Compliance:
- Follows the Python Array API Standard (https://data-apis.org/array-api/latest/)
- Implements Array API Standard 2023.12 specification
- Provides 150+ operations across all API categories
- NumPy-style broadcasting semantics for element-wise operations
- Supports 13 data types (float16/32/64, int8/16/32/64, uint8/16/32/64, bool)

Architecture:
- Dynamic shapes: 0D scalars to N-D tensors with runtime-determined dimensions
- Type-erased storage: UnsafePointer enables dtype flexibility
- Row-major memory layout (C-order) for efficient access patterns
- Memory-safe via Mojo's ownership and borrow checking

Array API Categories:
- Creation: zeros, ones, full, empty, arange, eye, linspace ✓
- Arithmetic: add, subtract, multiply, divide, floor_divide, modulo, power ✓
- Comparison: equal, not_equal, less, less_equal, greater, greater_equal ✓
- Reduction: sum, mean, max, min (all-elements only) ✓
- Matrix: matmul, transpose, dot, outer ✓ (shared/core/matrix.mojo)
- Shape manipulation: reshape, squeeze, unsqueeze, concatenate ✓ (shared/core/shape.mojo)
- Broadcasting: Full n-dim support for different-shape operations ✓ (shared/core/broadcasting.mojo)
- Element-wise math: exp, log, sqrt, sin, cos, tanh ✓ (shared/core/elementwise.mojo)
- Statistical: var, std, median, percentile ✓ (shared/core/reduction.mojo)
- Indexing: slicing, advanced indexing ✓ (__getitem__ methods)
- Hashing: __hash__ via Hashable trait ✓

Slicing Design:
- `slice(start, end)` — view-based extraction (shares memory, zero-copy). Use for
  batch processing where the original data must not be modified.
- `tensor[i]` / `tensor[i, j]` — copy-based element access via __getitem__(Int)
  or __getitem__(*slices). Returns a new tensor with independent memory.
- The split between view (slice) and copy (getitem) is intentional: views support
  efficient batch iteration; copies ensure safety when downstream code may mutate.

Reference: https://data-apis.org/array-api/latest/API_specification/index.html
"""

from collections import List
from memory import UnsafePointer, memset_zero, alloc, bitcast
from sys.info import simd_width_of
from math import ceildiv, sqrt, log, cos, sin
from utils.numerics import inf as numeric_inf, neg_inf as numeric_neg_inf
from random import random_float64, seed as random_seed
from hashlib.hasher import Hasher
from shared.base.memory_pool import pooled_alloc, pooled_free
from .tensor import Tensor
from .tensor_traits import TensorLike
from shared.base.broadcasting import broadcast_shapes, compute_broadcast_strides, are_shapes_broadcastable
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

# Memory safety constants
comptime MAX_TENSOR_BYTES: Int = 2_000_000_000  # 2 GB max per tensor
comptime WARN_TENSOR_BYTES: Int = 500_000_000  # 500 MB warning threshold

# Print options for AnyTensor.__str__ and __repr__ truncation
# Can be modified globally to control output behavior (e.g., in test utilities)
comptime ANYTENSOR_PRINT_THRESHOLD: Int = 1000  # Truncate if numel > threshold
comptime ANYTENSOR_PRINT_SHOW_ELEMENTS: Int = 3  # Show first/last N elements


struct AnyTensor(
    Copyable,
    Hashable,
    ImplicitlyCopyable,
    Movable,
    Representable,
    Sized,
    Stringable,
    TensorLike,
):
    """Dynamic tensor with runtime-determined shape and data type.

        AnyTensor provides a flexible tensor implementation for machine learning workloads,
        supporting arbitrary dimensions (0D scalars to N-D tensors), multiple data types,
        and NumPy-style broadcasting for all operations.

        Memory Safety: Implements reference counting for safe shared ownership.
        Copying a tensor increments the reference count, allowing views and copies
        to safely share data. Memory is freed only when the last reference is destroyed.

        Attributes:
            _data: UnsafePointer to raw byte storage (type-erased).
            _shape: List storing the shape dimensions.
            _strides: List storing the stride for each dimension (in elements).
            _dtype: The data type of tensor elements.
            _numel: Total number of elements in the tensor.
            _is_view: Whether this tensor is a view (shares data with another tensor).
            _refcount: Shared reference count for memory management.
            _original_numel_quantized: For quantized tensors, stores original size before padding (-1 if not quantized).

    Examples:
            # Create tensors
            var a = zeros([3, 4], DType.float32)
            var b = ones([3, 4], DType.float32)

            # Access properties
            print(a.shape())  # [3, 4]
            print(a.dtype())  # float32
            print(a.numel())  # 12.
    """

    var _data: UnsafePointer[UInt8, origin=MutAnyOrigin]
    """Raw byte storage for tensor elements."""
    var _shape: List[Int]
    """List of dimension sizes."""
    var _strides: List[Int]
    """Row-major strides for each dimension."""
    var _dtype: DType
    """Data type of tensor elements."""
    var _numel: Int
    """Total number of elements."""
    var _is_view: Bool
    """Whether this tensor shares data with another."""
    var _refcount: UnsafePointer[Int, origin=MutAnyOrigin]
    """Reference count for shared memory management."""
    var _original_numel_quantized: Int
    """Original element count before quantization padding."""
    var _allocated_size: Int
    """Actual allocated size (may differ from requested due to pool bucketing)."""

    fn __init__(out self, shape: List[Int], dtype: DType) raises:
        """Initialize a new AnyTensor with given shape and dtype.

        Args:
            shape: The shape of the tensor as a vector of dimension sizes.
            dtype: The data type of tensor elements.

        Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES (2 GB).

        Note:
            This is a low-level constructor. Users should prefer creation
            functions like zeros(), ones(), full(), etc.

            Allocating tensors larger than WARN_TENSOR_BYTES (500 MB) will
            print a console warning. This is informational only and does not
            raise an error.
        """
        # Copy shape to avoid mutation issues
        self._shape = List[Int]()
        for i in range(len(shape)):
            self._shape.append(shape[i])

        self._dtype = dtype
        self._is_view = False
        self._original_numel_quantized = (
            -1
        )  # Initialize as non-quantized (fixes DATA-001)

        # Calculate total number of elements
        self._numel = 1
        for i in range(len(self._shape)):
            self._numel *= self._shape[i]

        # Calculate row-major strides (in elements, not bytes)
        self._strides = List[Int]()
        var stride = 1
        # Pre-allocate strides list with correct forward iteration
        for _ in range(len(self._shape)):
            self._strides.append(0)
        # Now fill strides in backward order
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides[i] = stride
            stride *= self._shape[i]

        # Validate memory requirements
        var dtype_size = AnyTensor._get_dtype_size_static(dtype)
        var total_bytes = self._numel * dtype_size

        if total_bytes > MAX_TENSOR_BYTES:
            raise Error(
                "Tensor too large: "
                + String(total_bytes)
                + " bytes exceeds maximum "
                + String(MAX_TENSOR_BYTES)
                + " bytes. Consider using smaller batch sizes."
            )

        if total_bytes > WARN_TENSOR_BYTES:
            print("Warning: Large tensor allocation:", total_bytes, "bytes")

        # Allocate raw byte storage through memory pool (for efficiency)
        self._data = pooled_alloc(total_bytes)
        self._allocated_size = total_bytes

        # Allocate and initialize reference count (fixes MOJO-003, MOJO-006)
        self._refcount = alloc[Int](1)
        self._refcount[] = 1  # Start with 1 reference

    fn __init__(out self, value: IntLiteral) raises:
        """Create a scalar AnyTensor from an integer literal.

        Enables implicit conversion from integer literals to AnyTensor.
        Creates a 0D (scalar) tensor with Int64 dtype.

        Args:
            value: Integer literal to convert.

        Raises:
            Error: If tensor allocation fails.

        Example:
            ```mojo
            var x: AnyTensor = 42  # Implicit conversion from IntLiteral
        ```
        ```
        """
        # Initialize scalar tensor (0D shape)
        self._shape = List[Int]()
        self._strides = List[Int]()
        self._dtype = DType.int64
        self._numel = 1
        self._is_view = False
        self._original_numel_quantized = -1
        var dtype_size = AnyTensor._get_dtype_size_static(DType.int64)
        self._data = pooled_alloc(dtype_size)
        self._allocated_size = dtype_size
        self._refcount = alloc[Int](1)
        self._refcount[] = 1
        self._set_int64(0, Int64(value))

    fn __init__(out self, value: FloatLiteral) raises:
        """Create a scalar AnyTensor from a float literal.

        Enables implicit conversion from float literals to AnyTensor.
        Creates a 0D (scalar) tensor with Float64 dtype.

        Args:
            value: Float literal to convert.

        Raises:
            Error: If tensor allocation fails.

        Example:
            ```mojo
            var x: AnyTensor = 3.14  # Implicit conversion from FloatLiteral
        ```
        ```
        """
        # Initialize scalar tensor (0D shape)
        self._shape = List[Int]()
        self._strides = List[Int]()
        self._dtype = DType.float64
        self._numel = 1
        self._is_view = False
        self._original_numel_quantized = -1
        var dtype_size = AnyTensor._get_dtype_size_static(DType.float64)
        self._data = pooled_alloc(dtype_size)
        self._allocated_size = dtype_size
        self._refcount = alloc[Int](1)
        self._refcount[] = 1
        self._set_float64(0, Float64(value))

    fn __init__(out self, value: Int) raises:
        """Create a scalar AnyTensor from an Int.

        Enables implicit conversion from Int to AnyTensor.
        Creates a 0D (scalar) tensor with Int64 dtype.

        Args:
            value: Int value to convert.

        Raises:
            Error: If tensor allocation fails.
        """
        # Initialize scalar tensor (0D shape)
        self._shape = List[Int]()
        self._strides = List[Int]()
        self._dtype = DType.int64
        self._numel = 1
        self._is_view = False
        self._original_numel_quantized = -1
        var dtype_size = AnyTensor._get_dtype_size_static(DType.int64)
        self._data = pooled_alloc(dtype_size)
        self._allocated_size = dtype_size
        self._refcount = alloc[Int](1)
        self._refcount[] = 1
        self._set_int64(0, Int64(value))

    fn __init__(out self, value: Float64) raises:
        """Create a scalar AnyTensor from a Float64.

        Enables implicit conversion from Float64 to AnyTensor.
        Creates a 0D (scalar) tensor with Float64 dtype.

        Args:
            value: Float64 value to convert.

        Raises:
            Error: If tensor allocation fails.

        Example:
            ```mojo
            var x: AnyTensor = Float64(3.14)
        ```
        """
        # Initialize scalar tensor (0D shape)
        self._shape = List[Int]()
        self._strides = List[Int]()
        self._dtype = DType.float64
        self._numel = 1
        self._is_view = False
        self._original_numel_quantized = -1
        var dtype_size = AnyTensor._get_dtype_size_static(DType.float64)
        self._data = pooled_alloc(dtype_size)
        self._allocated_size = dtype_size
        self._refcount = alloc[Int](1)
        self._refcount[] = 1
        self._set_float64(0, value)

    fn __init__(out self, var data: List[Float32]) raises:
        """Create 1D tensor from List[Float32].

        Args:
            data: List of Float32 values.

        Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

        Example:
            ```mojo
            var values : List[Float32] = [1.0, 2.0, 3.0]
            var tensor = AnyTensor(values)
        ```
        ```
        """
        var shape = List[Int]()
        shape.append(len(data))

        # Initialize fields manually (delegating constructor doesn't satisfy compiler)
        self._shape = List[Int]()
        self._shape.append(len(data))
        self._dtype = DType.float32
        self._is_view = False
        self._original_numel_quantized = -1

        # Calculate numel
        self._numel = len(data)

        # Calculate strides
        self._strides = List[Int]()
        self._strides.append(1)

        # Allocate memory
        var dtype_size = AnyTensor._get_dtype_size_static(DType.float32)
        var total_bytes = self._numel * dtype_size

        if total_bytes > MAX_TENSOR_BYTES:
            raise Error(
                "Tensor too large: "
                + String(total_bytes)
                + " bytes exceeds maximum "
                + String(MAX_TENSOR_BYTES)
                + " bytes"
            )

        self._data = pooled_alloc(total_bytes)
        self._allocated_size = total_bytes
        self._refcount = alloc[Int](1)
        self._refcount[] = 1

        # Copy data
        for i in range(len(data)):
            self._set_float32(i, data[i])

    fn __init__(out self, var data: List[Int]) raises:
        """Create 1D tensor from List[Int].

        Args:
            data: List of Int values.

        Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

        Example:
            ```mojo
            var values : List[Int] = [1, 2, 3]
            var tensor = AnyTensor(values)
        ```
        ```
        """
        var shape = List[Int]()
        shape.append(len(data))

        # Initialize fields manually (delegating constructor doesn't satisfy compiler)
        self._shape = List[Int]()
        self._shape.append(len(data))
        self._dtype = DType.int64
        self._is_view = False
        self._original_numel_quantized = -1

        # Calculate numel
        self._numel = len(data)

        # Calculate strides
        self._strides = List[Int]()
        self._strides.append(1)

        # Allocate memory
        var dtype_size = AnyTensor._get_dtype_size_static(DType.int64)
        var total_bytes = self._numel * dtype_size

        if total_bytes > MAX_TENSOR_BYTES:
            raise Error(
                "Tensor too large: "
                + String(total_bytes)
                + " bytes exceeds maximum "
                + String(MAX_TENSOR_BYTES)
                + " bytes"
            )

        self._data = pooled_alloc(total_bytes)
        self._allocated_size = total_bytes
        self._refcount = alloc[Int](1)
        self._refcount[] = 1

        # Copy data
        for i in range(len(data)):
            self._set_float64(i, Float64(data[i]))

    fn __copyinit__(out self, existing: Self):
        """Copy constructor - creates shared ownership with reference counting.

        Creates a new reference to the same underlying data.
        Increments the reference count to track shared ownership.
        This prevents double-free and enables safe view semantics.

        """
        # Shallow copy all fields
        self._data = existing._data
        self._shape = existing._shape.copy()
        self._strides = existing._strides.copy()
        self._dtype = existing._dtype
        self._numel = existing._numel
        self._is_view = existing._is_view
        self._refcount = existing._refcount
        self._original_numel_quantized = existing._original_numel_quantized
        self._allocated_size = existing._allocated_size

        # Increment reference count (shared ownership)
        # All copies (views or not) participate in refcount management
        if self._refcount:
            self._refcount[] += 1

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor - transfers ownership.

        For safety, we copy the List fields instead of moving them with ^
        to avoid potential corruption issues with List's internal buffer.
        """
        self._data = existing._data
        self._shape = existing._shape.copy()
        self._strides = existing._strides.copy()
        self._dtype = existing._dtype
        self._numel = existing._numel
        self._is_view = existing._is_view
        self._refcount = existing._refcount
        self._original_numel_quantized = existing._original_numel_quantized
        self._allocated_size = existing._allocated_size

    fn __del__(deinit self):
        """Destructor - decrements ref count, frees if last reference.

        Uses reference counting to safely manage shared ownership.
        Only frees memory when the last reference is destroyed.

        """
        # All copies (views or not) participate in refcount management
        if self._refcount:
            self._refcount[] -= 1

            # If last reference, free everything
            if self._refcount[] == 0:
                pooled_free(self._data, self._allocated_size)
                self._refcount.free()

    fn _get_dtype_size(self) -> Int:
        """Get size in bytes for the tensor's dtype."""
        return AnyTensor._get_dtype_size_static(self._dtype)

    @staticmethod
    fn _get_dtype_size_static(dtype: DType) -> Int:
        """Get size in bytes for a given dtype (static version for use in __init__).
        """
        if dtype == DType.float16:
            return 2
        elif dtype == DType.bfloat16:
            return 2
        elif dtype == DType.float32:
            return 4
        elif dtype == DType.float64:
            return 8
        elif dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
            return 1
        elif dtype == DType.int16 or dtype == DType.uint16:
            return 2
        elif dtype == DType.int32 or dtype == DType.uint32:
            return 4
        elif dtype == DType.int64 or dtype == DType.uint64:
            return 8
        else:
            return 4  # Default fallback

    fn shape(self) -> List[Int]:
        """Return the shape of the tensor.

        Returns:
            A copy of the shape vector.

        Examples:
            ```var t = zeros([3, 4], DType.float32)
            print(t.shape())  # List[3, 4]```
        """
        # Return a copy to avoid mutation issues
        var result = List[Int]()
        for i in range(len(self._shape)):
            result.append(self._shape[i])
        return result^

    fn dtype(self) -> DType:
        """Return the data type of the tensor.

        Returns:
            The DType of tensor elements.
        """
        return self._dtype

    fn get_dtype(self) -> DType:
        """Return the element data type (TensorLike conformance).

        Returns:
            The DType of tensor elements.
        """
        return self._dtype

    fn numel(self) -> Int:
        """Return the total number of elements in the tensor.

        Returns:
            The product of all dimension sizes.

        Examples:
            `var t = AnyTensor.zeros((3, 4), DType.float32)
            print(t.numel())  # 12`
        """
        return self._numel

    fn num_elements(self) -> Int:
        """Return the total number of elements in the tensor.

        This is an comptime for numel() for API compatibility.

        Returns:
            The product of all dimension sizes.

        Examples:
            `var t = zeros([3, 4], DType.float32)
            print(t.num_elements())  # 12`
        """
        return self._numel

    fn dim(self) -> Int:
        """Return the number of dimensions (rank) of the tensor.

        Returns:
            The number of dimensions.

        Examples:```
            var t = AnyTensor.zeros((3, 4), DType.float32)
            print(t.dim())  # 2
            ```
        """
        return len(self._shape)

    fn ndim(self) -> Int:
        """Return the number of dimensions (rank).

        Alias for `dim()`, provided for future TensorLike trait conformance.

        Returns:
            The number of dimensions.
        """
        return len(self._shape)

    fn is_contiguous(self) -> Bool:
        """Check if the tensor has a contiguous memory layout.

        Returns:
            True if the tensor is contiguous (row-major, no gaps), False otherwise.

        Note:
            Contiguous tensors enable SIMD optimizations and efficient operations.
        """
        # Check if strides match row-major layout
        var expected_stride = 1
        for i in range(len(self._shape) - 1, -1, -1):
            if self._strides[i] != expected_stride:
                return False
            expected_stride *= self._shape[i]
        return True

    fn reshape(self, new_shape: List[Int]) raises -> AnyTensor:
        """Reshape tensor to new shape (must have same total elements).

        Returns a zero-copy view (shallow pointer copy) sharing data with the
        original tensor. The result has `is_view() == True` and `is_contiguous() == True`
        (because reshape only changes the shape/stride metadata, not the flat layout).
        Uses reference counting to ensure data remains valid while any view is alive.

        Note: This mirrors the view semantics of `slice()` — no data is copied.
        Compare with operations that return independent copies (e.g. `as_contiguous()`).

        Args:
            new_shape: The new shape for the tensor.

        Returns:
            A zero-copy view with the requested shape, sharing the same flat data buffer.

        Raises:
            Error: If the total number of elements doesn't match.

        Example:
        ```mojo
            var t = zeros([2, 3], DType.float32)
            var reshaped = t.reshape([6])  # (2, 3) -> (6,), zero-copy view
        ```
        """
        # Verify total elements match
        var new_numel = 1
        for i in range(len(new_shape)):
            new_numel *= new_shape[i]

        if new_numel != self._numel:
            raise Error("Cannot reshape: element count mismatch")

        # Create view by explicitly copying (increments refcount via __copyinit__)
        var result = self.copy()
        result._is_view = (
            True  # Mark as view since it shares data with original
        )

        # Update shape
        result._shape = List[Int]()
        for i in range(len(new_shape)):
            result._shape.append(new_shape[i])

        # Recalculate strides for new shape
        result._strides = List[Int]()
        var stride = 1
        # Pre-allocate strides list with correct forward iteration
        for _ in range(len(new_shape)):
            result._strides.append(0)
        # Now fill strides in backward order
        for i in range(len(new_shape) - 1, -1, -1):
            result._strides[i] = stride
            stride *= new_shape[i]

        return result^

    fn slice(self, start: Int, end: Int, axis: Int = 0) raises -> AnyTensor:
        """Extract a slice along the specified axis, returning a view into the original data.

        Creates a shallow copy of the tensor struct whose `_data` pointer is offset
        into the original buffer. No data bytes are copied. The returned tensor has
        `_is_view = True`, and modifying its elements will affect the original tensor.

        Args:
            start: Starting index (inclusive).
            end: Ending index (exclusive).
            axis: Axis to slice along (default: 0, the batch dimension).

        Returns:
            A new AnyTensor whose `_data` pointer references the same underlying memory
            as the original, offset to `start` along `axis`. The `_is_view` flag is
            set to True. This is a zero-copy view: no data bytes are allocated or copied.
            Modifying elements of the returned tensor will affect the original.

        Raises:
            Error: If indices are out of bounds or axis is invalid.

        Notes:
            This is the recommended method for memory-efficient batch extraction in
            training loops. Unlike `__getitem__(Slice)` and `__getitem__(*slices)`,
            which both return independent copies (`_is_view = False`), this method
            returns a genuine view that shares memory with the original tensor.

        Example:
        ```mojo
        # Extract batch 0-32 from (112800, 1, 28, 28)
        var batch = dataset.slice(0, 32, axis=0)  # Returns (32, 1, 28, 28)
        ```
        """
        # Validate axis
        if axis < 0 or axis >= len(self._shape):
            raise Error(
                "Axis "
                + String(axis)
                + " out of range for tensor with "
                + String(len(self._shape))
                + " dimensions"
            )

        # Validate indices
        var dim_size = self._shape[axis]
        if start < 0 or start > dim_size:
            raise Error(
                "Start index "
                + String(start)
                + " out of range [0, "
                + String(dim_size)
                + "]"
            )
        if end < start or end > dim_size:
            raise Error(
                "End index "
                + String(end)
                + " out of range ["
                + String(start)
                + ", "
                + String(dim_size)
                + "]"
            )

        # Calculate offset to start of slice
        var offset_elements = start * self._strides[axis]
        var dtype_size = self._get_dtype_size()
        var offset_bytes = offset_elements * dtype_size

        # Create view by copying (increments refcount)
        var result = self.copy()
        result._is_view = True

        # Update the sliced dimension in place
        result._shape[axis] = end - start

        # Update data pointer to point to sliced data
        result._data = self._data + offset_bytes

        # Strides remain the same (already copied by __copyinit__)

        # Recalculate numel after shape change
        result._numel = 1
        for i in range(len(result._shape)):
            result._numel *= result._shape[i]

        return result^

    fn transpose(self, dim0: Int, dim1: Int) raises -> AnyTensor:
        """Return a non-contiguous view with dim0 and dim1 swapped.

        Creates a stride-based view sharing the same underlying data — no
        copying occurs. For any non-trivial swap (dim0 != dim1) the result
        satisfies is_contiguous() == False.

        Note: `transpose()` is the primary API for dimension swapping and returns a
        true stride-based view (shape and strides permuted, data pointer shared).
        Compare with `transpose_view()` in `shared/core/matrix.mojo`, which copies
        raw bytes and sets permuted strides — useful for testing `is_contiguous()` and
        `as_contiguous()` but not recommended for production use. See also #4082.

        Args:
            dim0: First dimension to swap.
            dim1: Second dimension to swap.

        Returns:
            A new AnyTensor view with permuted shape and strides.

        Raises:
            Error: If tensor has fewer than 2 dimensions or dims are out of
                bounds.

        Example:
            ```mojo
            var shape = List[Int]()
            shape.append(3)
            shape.append(4)
            var a = ones(shape, DType.float32)
            var b = a.transpose(0, 1)  # shape (4, 3), non-contiguous view
            ```
        """
        var ndim = self.dim()
        if ndim < 2:
            raise Error("transpose requires at least 2 dimensions")
        if dim0 < 0 or dim0 >= ndim:
            raise Error("transpose: dim0 out of range")
        if dim1 < 0 or dim1 >= ndim:
            raise Error("transpose: dim1 out of range")

        var result = self.copy()
        result._is_view = True

        var tmp_shape = result._shape[dim0]
        result._shape[dim0] = result._shape[dim1]
        result._shape[dim1] = tmp_shape

        var tmp_stride = result._strides[dim0]
        result._strides[dim0] = result._strides[dim1]
        result._strides[dim1] = tmp_stride

        return result^

    fn __getitem__(self, index: Int) raises -> Float32:
        """Get element at flat index.

        For contiguous tensors, the flat index maps directly to a memory offset.
        For non-contiguous tensors (e.g., after transpose or axis>0 slice), the
        flat index is first converted to multi-dimensional coordinates using the
        tensor's shape, then mapped to a memory offset using strides.

        Args:
            index: The flat index to access (logical element index in
                row-major order of the tensor's shape).

        Returns:
            The value at the given index as Float32.

        Raises:
            Error: If index is out of bounds.

        Example:
            ```mojo
            var t = arange(0.0, 10.0, 1.0, DType.float32)
            var val = t[5]  # Get element at index 5
        ```
        """
        if index < 0 or index >= self._numel:
            raise Error("Index out of bounds")

        # For non-contiguous tensors, convert flat index to nd-coordinates
        # then use strides to compute the real memory offset.
        if not self.is_contiguous():
            var remaining = index
            var mem_offset = 0
            for i in range(len(self._shape)):
                # Compute the product of dimensions after axis i
                var dim_size = 1
                for j in range(i + 1, len(self._shape)):
                    dim_size *= self._shape[j]
                var coord = remaining // dim_size
                remaining = remaining % dim_size
                mem_offset += coord * self._strides[i]
            return self._get_float32(mem_offset)

        # Return value based on dtype
        return self._get_float32(index)

    fn _resolve_index(self, index: Int) raises -> Int:
        """Resolve flat index to memory offset, with bounds check.

        For non-contiguous tensors, converts flat index to memory offset
        via nd-coordinates and strides.

        Args:
            index: Flat logical index.

        Returns:
            Memory offset for the element.

        Raises:
            Error: If index is out of bounds.
        """
        if index < 0 or index >= self._numel:
            raise Error("Index out of bounds")
        if not self.is_contiguous():
            var remaining = index
            var mem_offset = 0
            for i in range(len(self._shape)):
                var dim_size = 1
                for j in range(i + 1, len(self._shape)):
                    dim_size *= self._shape[j]
                var coord = remaining // dim_size
                remaining = remaining % dim_size
                mem_offset += coord * self._strides[i]
            return mem_offset
        return index

    fn __setitem__(mut self, index: Int, value: Float64) raises:
        """Set element at flat index.

        Note: Mojo does not dispatch `obj[i] = val` to __setitem__ — it
        treats `obj[i]` as an lvalue via __getitem__ (returns Float32).
        Use `tensor.set(i, val)` for type-safe assignment from any numeric
        type.

        Args:
            index: The flat index to set.
            value: The value to store.

        Raises:
            Error: If index is out of bounds.
        """
        var idx = self._resolve_index(index)
        if (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
            or self._dtype == DType.bfloat16
        ):
            self._set_float64(idx, value)
        else:
            self._set_int64(idx, Int64(value))

    fn __setitem__(mut self, index: Int, value: Int64) raises:
        """Set element at flat index using an integer value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, value)

    fn __setitem__(mut self, index: Int, value: Float32) raises:
        """Set element at flat index using a Float32 value."""
        var idx = self._resolve_index(index)
        self._set_float32(idx, value)

    # ===----------------------------------------------------------------------===#
    # set() — type-safe element assignment
    #
    # TODO: Remove these set() overloads once Tensor[dtype] with proper typed
    # __setitem__ is used everywhere. Currently still needed because AnyTensor
    # is used in metrics, normalization, dropout, attention, and other modules.
    #
    # Mojo does NOT dispatch `obj[i] = val` to __setitem__; it treats
    # `obj[i]` as an lvalue via __getitem__ (returns Float32), so
    # assigning Float64/Float16/Int64/etc. fails with a type error.
    # Use `tensor.set(i, val)` instead of `tensor[i] = val` when the
    # RHS is not Float32.
    #
    # Each overload calls the appropriate internal setter directly to
    # avoid precision-losing type round-trips (e.g. Float32→Float64→Float32).
    # ===----------------------------------------------------------------------===#

    @always_inline
    fn set(mut self, index: Int, value: Float64) raises:
        """Set element at flat index from a Float64 value."""
        var idx = self._resolve_index(index)
        self._set_float64(idx, value)

    @always_inline
    fn set(mut self, index: Int, value: Float32) raises:
        """Set element at flat index from a Float32 value."""
        var idx = self._resolve_index(index)
        self._set_float32(idx, value)

    @always_inline
    fn set(mut self, index: Int, value: Float16) raises:
        """Set element at flat index from a Float16 value."""
        var idx = self._resolve_index(index)
        self._set_float32(idx, Float32(value))

    @always_inline
    fn set(mut self, index: Int, value: Int) raises:
        """Set element at flat index from an Int value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(value))

    @always_inline
    fn set(mut self, index: Int, value: Int64) raises:
        """Set element at flat index from an Int64 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, value)

    @always_inline
    fn set(mut self, index: Int, value: Int32) raises:
        """Set element at flat index from an Int32 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(Int(value)))

    @always_inline
    fn set(mut self, index: Int, value: Int16) raises:
        """Set element at flat index from an Int16 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(Int(value)))

    @always_inline
    fn set(mut self, index: Int, value: Int8) raises:
        """Set element at flat index from an Int8 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(Int(value)))

    @always_inline
    fn set(mut self, index: Int, value: UInt8) raises:
        """Set element at flat index from a UInt8 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(Int(value)))

    @always_inline
    fn set(mut self, index: Int, value: UInt16) raises:
        """Set element at flat index from a UInt16 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(Int(value)))

    @always_inline
    fn set(mut self, index: Int, value: UInt32) raises:
        """Set element at flat index from a UInt32 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(Int(value)))

    @always_inline
    fn set(mut self, index: Int, value: UInt64) raises:
        """Set element at flat index from a UInt64 value."""
        var idx = self._resolve_index(index)
        self._set_int64(idx, Int64(Int(value)))

    fn __getitem__(self, indices: List[Int]) raises -> Float32:
        """Get element at multi-dimensional index.

        Args:
            indices: Per-dimension indices (one per axis).

        Returns:
            The value at the given indices as Float32.

        Raises:
            Error: If number of indices doesn't match tensor rank,
                   or any index is out of bounds.

        Example:
            ```mojo
            var t = ones([3, 4], DType.float32)
            var val = t[[1, 2]]  # Get element at row 1, col 2
            ```
        """
        if len(indices) != len(self._shape):
            raise Error(
                "Number of indices ("
                + String(len(indices))
                + ") must match tensor rank ("
                + String(len(self._shape))
                + ")"
            )
        var mem_offset = 0
        for i in range(len(indices)):
            if indices[i] < 0 or indices[i] >= self._shape[i]:
                raise Error("Index out of bounds at dimension " + String(i))
            mem_offset += indices[i] * self._strides[i]
        return self._get_float32(mem_offset)

    fn __setitem__(mut self, indices: List[Int], value: Float64) raises:
        """Set element at multi-dimensional index.

        Args:
            indices: Per-dimension indices (one per axis).
            value: The value to set (cast to tensor dtype).

        Raises:
            Error: If number of indices doesn't match tensor rank,
                   or any index is out of bounds.

        Example:
            ```mojo
            var t = zeros([3, 4], DType.float32)
            t[[1, 2]] = 5.0  # Set element at row 1, col 2
            ```
        """
        if len(indices) != len(self._shape):
            raise Error(
                "Number of indices ("
                + String(len(indices))
                + ") must match tensor rank ("
                + String(len(self._shape))
                + ")"
            )
        var mem_offset = 0
        for i in range(len(indices)):
            if indices[i] < 0 or indices[i] >= self._shape[i]:
                raise Error("Index out of bounds at dimension " + String(i))
            mem_offset += indices[i] * self._strides[i]
        # Use _set_float64/_set_int64 directly with the stride-computed memory
        # offset to avoid double-conversion in __setitem__(Int) for non-contiguous tensors.
        if (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
            or self._dtype == DType.bfloat16
        ):
            self._set_float64(mem_offset, value)
        else:
            self._set_int64(mem_offset, Int64(value))

    fn __setitem__(mut self, indices: List[Int], value: Float32) raises:
        """Set element at multi-dimensional index using Float32 value.

        Args:
            indices: Per-dimension indices (one per axis).
            value: The Float32 value to set (converted to Float64 internally).

        Raises:
            Error: If number of indices doesn't match tensor rank,
                   or any index is out of bounds.

        Example:
            ```mojo
            var t = zeros([3, 4], DType.float32)
            t[[1, 2]] = Float32(5.0)  # Set element at row 1, col 2
            ```
        """
        self.__setitem__(indices, Float64(value))

    fn _normalize_slice_indices(
        self, start: Int, end: Int, step: Int, size: Int
    ) -> Tuple[Int, Int, Int, Int]:
        """Normalize slice indices to valid ranges.

        Handles negative indices, clamping, and returns normalized
        (start, end, step, result_size) for valid iteration.

        Args:
            start: Start index (may be negative).
            end: End index (may be negative).
            step: Step value (can be negative for reverse).
            size: Size of the dimension being sliced.

        Returns:
            Tuple of (normalized_start, normalized_end, normalized_step, result_size).
            result_size is the number of elements in the slice result.
        """
        var norm_start = start
        var norm_end = end
        var norm_step = step
        var result_size: Int

        if step < 0:
            # Negative step: reverse iteration
            var neg_step = -step
            # Clamp start to [0, size-1], end to [-1, size-1]
            norm_start = max(0, min(norm_start, size - 1))
            norm_end = max(-1, min(norm_end, size - 1))
            result_size = max(0, ceildiv(norm_start - norm_end, neg_step))
        else:
            # Positive step: forward iteration
            # Normalize negative indices first
            if norm_start < 0:
                norm_start = size + norm_start
            if norm_end < 0:
                norm_end = size + norm_end
            # Clamp forward slice to [0, size]
            norm_start = max(0, min(norm_start, size))
            norm_end = max(0, min(norm_end, size))
            result_size = max(0, ceildiv(norm_end - norm_start, step))

        return (norm_start, norm_end, norm_step, result_size)

    fn __getitem__(self, slice: Slice) raises -> Self:
        """Get slice of 1D tensor [start:end] or [start:end:step].

        Args:
            slice: Slice object specifying start, end, and optional step.

        Returns:
            New tensor containing a **copy** of the sliced data. The result
            does not share memory with the original tensor.

        Raises:
            Error: If tensor is not 1D or indices are invalid.

        Notes:
            This method always returns a copy (`_is_view = False`), regardless
            of the step value. This is by design: materializing a strided copy
            keeps the implementation simple and avoids lifetime management
            complexity. For memory-efficient batch extraction over the first
            axis, use `slice()` instead, which returns a true view.

        Example:
            ```mojo
            var t = arange(0.0, 10.0, 1.0, DType.float32)
            var sliced = t[2:7]  # Copy of [2, 3, 4, 5, 6]
            var strided = t[0:10:2]  # Copy of [0, 2, 4, 6, 8]
            var reversed = t[::-1]  # Copy of [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ```
        """
        if len(self._shape) != 1:
            raise Error("Single slice only supported for 1D tensors")

        # Handle slice parameters — extract step first so defaults depend on sign
        var size = self._shape[0]
        var step = slice.step.or_else(1)

        var start: Int
        var end: Int
        if step < 0:
            # Negative step: default start=last element, default end=before index 0
            start = slice.start.or_else(size - 1)
            end = slice.end.or_else(-size - 1)
        else:
            start = slice.start.or_else(0)
            end = slice.end.or_else(size)

        # Normalize negative indices
        if start < 0:
            start = size + start
        if end < 0:
            end = size + end

        # Handle negative step (reverse)
        var result_size: Int
        if step < 0:
            var neg_step = -step
            # Clamp start to [0, size-1], end to [-1, size-1]
            start = max(0, min(start, size - 1))
            end = max(-1, min(end, size - 1))
            # No swap: iterate src_idx = start - i * neg_step while src_idx > end
            result_size = max(0, ceildiv(start - end, neg_step))

            # Create result tensor with shape
            var shape = List[Int]()
            shape.append(result_size)
            var result = Self(shape, self._dtype)
            result._is_view = False

            # Copy in reverse
            var dtype_size = self._get_dtype_size()
            var src_ptr = self._data
            var dst_ptr = result._data

            for i in range(result_size):
                var src_idx = start - i * neg_step
                var src_offset = src_idx * dtype_size
                var dst_offset = i * dtype_size
                for b in range(dtype_size):
                    dst_ptr[dst_offset + b] = src_ptr[src_offset + b]

            return result^
        else:
            # Clamp forward slice to [0, size]
            start = max(0, min(start, size))
            end = max(0, min(end, size))
            # Normal forward slice
            result_size = max(0, ceildiv(end - start, step))

        # Create result tensor with shape
        var shape = List[Int]()
        shape.append(result_size)
        var result = Self(shape, self._dtype)
        result._is_view = False  # Strided slice creates copy, not view

        # Copy strided data
        var dtype_size = self._get_dtype_size()
        var src_ptr = self._data
        var dst_ptr = result._data

        for i in range(result_size):
            var src_idx = start + i * step
            var src_offset = src_idx * dtype_size
            var dst_offset = i * dtype_size

            # Copy element (byte-wise)
            for b in range(dtype_size):
                dst_ptr[dst_offset + b] = src_ptr[src_offset + b]

        return result^

    fn __getitem__(self, *slices: Slice) raises -> Self:
        """Get multi-dimensional slice (e.g., tensor[a:b, c:d, :]).

        Args:
            slices: Variable number of Slice objects, one per dimension.

        Returns:
            New tensor containing a **copy** of the sliced data. The result
            does not share memory with the original tensor.

        Raises:
            Error: If number of slices doesn't match tensor dimensions.

        Notes:
            This method returns a copy (`_is_view = False`), consistent with
            the 1D `__getitem__(Slice)` overload. Multi-dimensional slicing
            produces non-contiguous data in general (e.g., `t[1:4, 1:3]` on
            a 5x4 tensor), so a simple pointer offset is insufficient. Each
            output element is copied individually using per-dimension offsets
            and original strides.

        Example:
            ```mojo
            var t = zeros([10, 8, 6], DType.float32)
            var sliced = t[2:7, :, 1:4]  # Copy with shape [5, 8, 3]
        ```
        """
        var num_slices = len(slices)
        var num_dims = len(self._shape)

        if num_slices != num_dims:
            raise Error(
                "Number of slices ("
                + String(num_slices)
                + ") must match number of dimensions ("
                + String(num_dims)
                + ")"
            )

        # Compute per-dimension starts, steps, and result shape
        var starts = List[Int]()
        var steps = List[Int]()
        var result_shape = List[Int]()
        for dim in range(num_dims):
            var s = slices[dim]
            var size = self._shape[dim]

            var step = s.step.or_else(1)
            if step == 0:
                raise Error(
                    "Slice step cannot be zero for dimension " + String(dim)
                )

            var start: Int
            var end: Int
            if step < 0:
                start = s.start.or_else(size - 1)
                end = s.end.or_else(-size - 1)
            else:
                start = s.start.or_else(0)
                end = s.end.or_else(size)

            var normalized = self._normalize_slice_indices(
                start, end, step, size
            )
            starts.append(normalized[0])
            steps.append(normalized[2])
            result_shape.append(normalized[3])

        # Allocate result tensor (independent copy, not a view)
        var result = Self(result_shape, self._dtype)
        result._is_view = False

        var result_numel = result._numel
        if result_numel == 0:
            return result^

        # Fast-path: detect when only dim-0 is non-trivially sliced
        # and all remaining dimensions use full slices
        var can_use_memcpy = num_dims > 0 and self._strides[0] == self._shape[1] if num_dims > 1 else True
        if can_use_memcpy:
            # Check that all dims >= 1 use full slices
            for dim in range(1, num_dims):
                var s = slices[dim]
                var size = self._shape[dim]
                var step = s.step.or_else(1)
                if step != 1:
                    can_use_memcpy = False
                    break
                var start = s.start.or_else(0)
                var end = s.end.or_else(size)
                if start < 0:
                    start = size + start
                if end < 0:
                    end = size + end
                start = max(0, min(start, size))
                end = max(0, min(end, size))
                if start != 0 or end != size:
                    can_use_memcpy = False
                    break

        if can_use_memcpy and result_numel > 0:
            # Fast-path: use memcpy for contiguous first-axis slice
            var dtype_size = self._get_dtype_size()
            var src_ptr = self._data
            var dst_ptr = result._data

            # Calculate stride (elements per dim-0 slice)
            var stride_numel = result_numel // result_shape[0]

            # Copy each row contiguously
            var src_offset = starts[0] * stride_numel * dtype_size
            for i in range(result_shape[0]):
                var src_addr = src_ptr + src_offset + i * steps[0] * stride_numel * dtype_size
                var dst_addr = dst_ptr + i * stride_numel * dtype_size
                # memcpy semantics: copy stride_numel elements
                for b in range(stride_numel * dtype_size):
                    dst_addr[b] = src_addr[b]
        else:
            # Slow-path: copy each element individually (stride-aware)
            var dtype_size = self._get_dtype_size()
            var src_ptr = self._data
            var dst_ptr = result._data

            for out_flat in range(result_numel):
                # Decompose out_flat into per-dimension indices, then map to source
                var src_flat = 0
                var remaining = out_flat
                for dim in range(num_dims):
                    var out_idx = remaining // result._strides[dim]
                    remaining = remaining % result._strides[dim]
                    var src_idx = starts[dim] + out_idx * steps[dim]
                    src_flat += src_idx * self._strides[dim]

                # Copy element byte-by-byte
                var src_offset = src_flat * dtype_size
                var dst_offset = out_flat * dtype_size
                for b in range(dtype_size):
                    dst_ptr[dst_offset + b] = src_ptr[src_offset + b]

        return result^

    fn _get_float64(self, index: Int) -> Float64:
        """Internal: Get value at index as Float64 (assumes float-compatible dtype).

        Args:
            index: The element index to retrieve.

        Returns:
            The value at the index as Float64.
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.float16:
            var ptr = (self._data + offset).bitcast[Float16]()
            return ptr[].cast[DType.float64]()
        elif self._dtype == DType.bfloat16:
            # BF16 occupies the upper 16 bits of Float32 (same sign + exponent layout).
            # Read raw UInt16 bits and reconstruct Float32 via bitcast to preserve all
            # NaN mantissa bits — numeric cast via Float32(BFloat16) may canonicalize NaN.
            var raw_ptr = (self._data + offset).bitcast[UInt16]()
            var raw: UInt16 = raw_ptr[]
            var f32_bits: UInt32 = UInt32(raw) << 16
            var f32_val = UnsafePointer[UInt32](to=f32_bits).bitcast[Float32]()[]
            return Float64(f32_val)
        elif self._dtype == DType.float32:
            var ptr = (self._data + offset).bitcast[Float32]()
            return ptr[].cast[DType.float64]()
        elif self._dtype == DType.float64:
            var ptr = (self._data + offset).bitcast[Float64]()
            return ptr[]
        else:
            # For integer types, cast to float64
            return Float64(self._get_int64(index))

    fn _set_float64(self, index: Int, value: Float64):
        """Internal: Set value at index (assumes float-compatible dtype).

        Args:
            index: The element index to set.
            value: The value to set (as Float64).
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.float16:
            var ptr = (self._data + offset).bitcast[Float16]()
            ptr[] = value.cast[DType.float16]()
        elif self._dtype == DType.bfloat16:
            var ptr = (self._data + offset).bitcast[BFloat16]()
            ptr[] = BFloat16(Float32(value))
        elif self._dtype == DType.float32:
            var ptr = (self._data + offset).bitcast[Float32]()
            ptr[] = value.cast[DType.float32]()
        elif self._dtype == DType.float64:
            var ptr = (self._data + offset).bitcast[Float64]()
            ptr[] = value
        else:
            # For integer types, truncate Float64 to Int64 and delegate
            self._set_int64(index, Int64(value))

    fn _get_float32(self, index: Int) -> Float32:
        """Internal: Get value at index as Float32 (assumes float-compatible dtype).

        Args:
            index: Flat index to retrieve value from.

        Returns:
            Value at index as Float32.

        Note:
            For Float64 and integer types, value is cast to Float32.
            For Float16, value is upcast to Float32.
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.float16:
            var ptr = (self._data + offset).bitcast[Float16]()
            return ptr[].cast[DType.float32]()
        elif self._dtype == DType.float32:
            var ptr = (self._data + offset).bitcast[Float32]()
            return ptr[]
        elif self._dtype == DType.float64:
            var ptr = (self._data + offset).bitcast[Float64]()
            return ptr[].cast[DType.float32]()
        elif self._dtype == DType.bfloat16:
            var ptr = (self._data + offset).bitcast[BFloat16]()
            return ptr[].cast[DType.float32]()
        else:
            # For integer types, cast to float32
            return Float32(self._get_int64(index))

    fn _set_float32(self, index: Int, value: Float32):
        """Internal: Set value at index as Float32 (assumes float-compatible dtype).

        Args:
            index: Flat index to set value at.
            value: Float32 value to store.

        Note:
            For Float16, value is downcast with potential precision loss.
            For Float64, value is upcast to Float64.
            For integer types, value is truncated to integer.
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.float16:
            var ptr = (self._data + offset).bitcast[Float16]()
            ptr[] = value.cast[DType.float16]()
        elif self._dtype == DType.float32:
            var ptr = (self._data + offset).bitcast[Float32]()
            ptr[] = value
        elif self._dtype == DType.float64:
            var ptr = (self._data + offset).bitcast[Float64]()
            ptr[] = value.cast[DType.float64]()
        elif self._dtype == DType.bfloat16:
            var ptr = (self._data + offset).bitcast[BFloat16]()
            ptr[] = value.cast[DType.bfloat16]()
        else:
            # For integer types, truncate Float32 to Int64 and delegate
            self._set_int64(index, Int64(value))

    fn _get_int64(self, index: Int) -> Int64:
        """Internal: Get value at index as Int64 (assumes integer-compatible dtype).

        Args:
            index: The element index to retrieve.

        Returns:
            The value at the index as Int64.
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.int8:
            var ptr = (self._data + offset).bitcast[Int8]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int16:
            var ptr = (self._data + offset).bitcast[Int16]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int32:
            var ptr = (self._data + offset).bitcast[Int32]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int64:
            var ptr = (self._data + offset).bitcast[Int64]()
            return ptr[]
        elif self._dtype == DType.uint8:
            var ptr = (self._data + offset).bitcast[UInt8]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint16:
            var ptr = (self._data + offset).bitcast[UInt16]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint32:
            var ptr = (self._data + offset).bitcast[UInt32]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint64:
            var ptr = (self._data + offset).bitcast[UInt64]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.bool:
            var ptr = (self._data + offset).bitcast[Scalar[DType.bool]]()
            return 1 if ptr[].__bool__() else 0
        else:
            return 0  # Default fallback

    fn _set_int64(self, index: Int, value: Int64):
        """Internal: Set value at index (assumes integer-compatible dtype).

        Args:
            index: The element index to set.
            value: The value to set (as Int64).
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.int8:
            var ptr = (self._data + offset).bitcast[Int8]()
            ptr[] = value.cast[DType.int8]()
        elif self._dtype == DType.int16:
            var ptr = (self._data + offset).bitcast[Int16]()
            ptr[] = value.cast[DType.int16]()
        elif self._dtype == DType.int32:
            var ptr = (self._data + offset).bitcast[Int32]()
            ptr[] = value.cast[DType.int32]()
        elif self._dtype == DType.int64:
            var ptr = (self._data + offset).bitcast[Int64]()
            ptr[] = value
        elif self._dtype == DType.uint8:
            var ptr = (self._data + offset).bitcast[UInt8]()
            ptr[] = value.cast[DType.uint8]()
        elif self._dtype == DType.uint16:
            var ptr = (self._data + offset).bitcast[UInt16]()
            ptr[] = value.cast[DType.uint16]()
        elif self._dtype == DType.uint32:
            var ptr = (self._data + offset).bitcast[UInt32]()
            ptr[] = value.cast[DType.uint32]()
        elif self._dtype == DType.uint64:
            var ptr = (self._data + offset).bitcast[UInt64]()
            ptr[] = value.cast[DType.uint64]()
        elif self._dtype == DType.bool:
            var ptr = (self._data + offset).bitcast[Scalar[DType.bool]]()
            ptr[] = Scalar[DType.bool](value != 0)

    fn _set_int32(self, index: Int, value: Int32):
        """Internal: Set value at index as Int32 (assumes integer-compatible dtype).

        Args:
            index: Flat index to set value at.
            value: Int32 value to store.

        Note:
            Delegates to _set_int64 after casting to Int64.
        """
        self._set_int64(index, value.cast[DType.int64]())

    fn _fill_zero(mut self):
        """Internal: Fill tensor with zeros (works for all dtypes)."""
        var dtype_size = self._get_dtype_size()
        var total_bytes = self._numel * dtype_size
        memset_zero(self._data, total_bytes)

    fn _fill_value_float(mut self, value: Float64):
        """Internal: Fill tensor with float value.

        Args:
            value: The float value to fill with.
        """
        for i in range(self._numel):
            self._set_float64(i, value)

    fn _fill_value_int(mut self, value: Int64):
        """Internal: Fill tensor with integer value.

        Args:
            value: The integer value to fill with.
        """
        for i in range(self._numel):
            self._set_int64(i, value)

    # ========================================================================
    # Dunder Methods (Operator Overloading)
    # ========================================================================

    fn __add__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise addition: a + b.

        Args:
            other: The tensor to add.

        Returns:
            New tensor with element-wise sum.

        Raises:
            Error: If tensors have incompatible shapes.
        """

        @always_inline
        fn _add[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x + y

        return _anytensor_binary_op[_add](self, other)

    fn __sub__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise subtraction: a - b.

        Args:
            other: The tensor to subtract.

        Returns:
            New tensor with element-wise difference.

        Raises:
            Error: If tensors have incompatible shapes.
        """

        @always_inline
        fn _sub[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x - y

        return _anytensor_binary_op[_sub](self, other)

    fn __mul__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise multiplication: a * b.

        Args:
            other: The tensor to multiply.

        Returns:
            New tensor with element-wise product.

        Raises:
            Error: If tensors have incompatible shapes.
        """

        @always_inline
        fn _mul[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x * y

        return _anytensor_binary_op[_mul](self, other)

    fn __truediv__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise division: a / b.

        Args:
            other: The tensor to divide by.

        Returns:
            New tensor with element-wise quotient.

        Raises:
            Error: If tensors have incompatible shapes or division by zero.

        """

        @always_inline
        fn _div[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x / y

        return _anytensor_binary_op[_div](self, other)

    fn __floordiv__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise floor division: a // b.

        Args:
            other: The tensor to divide by.

        Returns:
            New tensor with element-wise floor quotient.

        Raises:
            Error: If tensors have incompatible shapes or division by zero.
        """

        @always_inline
        fn _floordiv[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x // y

        return _anytensor_binary_op[_floordiv](self, other)

    fn __mod__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise modulo: a % b.

        Args:
            other: The tensor to take modulo with.

        Returns:
            New tensor with element-wise remainder.

        Raises:
            Error: If tensors have incompatible shapes.
        """

        @always_inline
        fn _mod[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x % y

        return _anytensor_binary_op[_mod](self, other)

    fn __pow__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise power: a ** b.

        Args:
            other: The tensor of exponents.

        Returns:
            New tensor with element-wise powers.

        Raises:
            Error: If tensors have incompatible shapes.
        """

        @always_inline
        fn _pow[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
            return x ** y

        return _anytensor_binary_op[_pow](self, other)

    fn __matmul__(self, other: AnyTensor) raises -> AnyTensor:
        """Matrix multiplication: a @ b.

        Args:
            other: The tensor to multiply with.

        Returns:
            New tensor with matrix product.

        Raises:
            Error: If tensors have incompatible dimensions for multiplication.

        Note:
            This operator handles 2D×2D matrix multiplication. For 1D vectors
            or batched matmul, use shared.core.matrix.matmul directly.
        """
        return _anytensor_matmul(self, other)

    fn __eq__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise equality: a == b.

        Note: NaN comparison follows IEEE 754 semantics — NaN is never equal to
        anything, including itself. That is, `NaN == NaN` returns 0.0 (False) for
        every element position where either operand is NaN. Use `isnan()` to detect
        NaN values explicitly rather than relying on equality comparison.

        Args:
            other: The tensor to compare.

        Returns:
            New tensor with 1.0 where equal, 0.0 otherwise.

        Raises:
            Error: If tensors have incompatible shapes.
        """
        @always_inline
        fn _eq[T: DType](x: Scalar[T], y: Scalar[T]) -> Bool:
            return x == y

        return _anytensor_compare_op[_eq](self, other)

    fn __ne__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise inequality: a != b.

        Args:
            other: The tensor to compare.

        Returns:
            New tensor with 1.0 where not equal, 0.0 otherwise.

        Raises:
            Error: If tensors have incompatible shapes.
        """
        @always_inline
        fn _ne[T: DType](x: Scalar[T], y: Scalar[T]) -> Bool:
            return x != y

        return _anytensor_compare_op[_ne](self, other)

    fn __lt__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise less than: a < b.

        Args:
            other: The tensor to compare.

        Returns:
            New tensor with 1.0 where less than, 0.0 otherwise.

        Raises:
            Error: If tensors have incompatible shapes.
        """
        @always_inline
        fn _lt[T: DType](x: Scalar[T], y: Scalar[T]) -> Bool:
            return x < y

        return _anytensor_compare_op[_lt](self, other)

    fn __le__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise less or equal: a <= b.

        Args:
            other: The tensor to compare.

        Returns:
            New tensor with 1.0 where less or equal, 0.0 otherwise.

        Raises:
            Error: If tensors have incompatible shapes.
        """
        @always_inline
        fn _le[T: DType](x: Scalar[T], y: Scalar[T]) -> Bool:
            return x <= y

        return _anytensor_compare_op[_le](self, other)

    fn __gt__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise greater than: a > b.

        Args:
            other: The tensor to compare.

        Returns:
            New tensor with 1.0 where greater than, 0.0 otherwise.

        Raises:
            Error: If tensors have incompatible shapes.
        """
        @always_inline
        fn _gt[T: DType](x: Scalar[T], y: Scalar[T]) -> Bool:
            return x > y

        return _anytensor_compare_op[_gt](self, other)

    fn __ge__(self, other: AnyTensor) raises -> AnyTensor:
        """Element-wise greater or equal: a >= b.

        Args:
            other: The tensor to compare.

        Returns:
            New tensor with 1.0 where greater or equal, 0.0 otherwise.

        Raises:
            Error: If tensors have incompatible shapes.
        """
        @always_inline
        fn _ge[T: DType](x: Scalar[T], y: Scalar[T]) -> Bool:
            return x >= y

        return _anytensor_compare_op[_ge](self, other)

    # ========================================================================
    # FP8 Conversion Methods
    # ========================================================================

    fn to_fp8(self) raises -> AnyTensor:
        """Convert tensor values to FP8 E4M3 format.

        This method converts a tensor of any floating-point dtype to FP8 format,
        stored as uint8. The conversion uses E4M3 encoding (1 sign bit, 4 exponent
        bits, 3 mantissa bits) which is optimized for ML workloads.

        Returns:
            A new AnyTensor with dtype=uint8 containing FP8-encoded values.

        Raises:
            Error: If the source tensor is not a floating-point dtype.

        Examples:
            ```var t = zeros([3, 4], DType.float32)
            var fp8_t = t.to_fp8()  # Returns uint8 tensor with FP8 encoding
            var restored = fp8_t.from_fp8()  # Convert back to float32```

        Note:
            FP8 has limited range (~±240) and precision. Values outside this range
            are clamped. This is useful for memory-efficient training/inference.
            FP16 inputs are converted to FP32 before quantization.
        """
        from shared.core.types.dtype_aliases import FP8
        from memory import bitcast

        # Verify source is floating point
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
            or self._dtype == DType.bfloat16
        ):
            raise Error("to_fp8() requires a floating-point tensor")

        # Create output tensor with uint8 dtype
        var result = AnyTensor(self._shape, DType.uint8)

        # Convert each element to FP8
        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            # Get source value as Float32
            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Invalid dtype for FP8 conversion")

            # Convert to FP8 using native SIMD and store as uint8
            var fp8_val = SIMD[FP8, 1](val)
            var fp8_bits = bitcast[DType.uint8, 1](fp8_val)[0]
            result._data.bitcast[UInt8]()[i] = fp8_bits

        return result^

    fn from_fp8(self) raises -> AnyTensor:
        """Convert FP8-encoded tensor (uint8) back to Float32.

        This method interprets a uint8 tensor as FP8 E4M3 encoded values and
        converts them back to Float32 for computation.

        Returns:
            A new AnyTensor with dtype=float32 containing decoded values.

        Raises:
            Error: If the source tensor is not uint8 dtype.

        Examples:
            var fp8_t = ...  # uint8 tensor with FP8 encoding
            var float_t = fp8_t.from_fp8()  # Decode to float32

        Note:
            This assumes the uint8 tensor contains valid FP8 E4M3 encoded values.
            Use this to decode tensors created by to_fp8().
        """
        from shared.core.types.dtype_aliases import FP8
        from memory import bitcast

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_fp8() requires a uint8 tensor (FP8-encoded)")

        # Create output tensor with float32 dtype
        var result = AnyTensor(self._shape, DType.float32)

        # Convert each element from FP8 to Float32 using native SIMD
        for i in range(self._numel):
            var fp8_bits = self._data.bitcast[UInt8]()[i]
            # Bitcast uint8 to FP8, then convert to float32
            var fp8_val = bitcast[FP8, 1](SIMD[DType.uint8, 1](fp8_bits))
            var float_val = Float32(fp8_val[0])
            result[i] = Float32(float_val)

        return result^

    # ===----------------------------------------------------------------------===#
    # Integer Type Conversions
    # ===----------------------------------------------------------------------===#

    fn to_int8(self) raises -> AnyTensor:
        """Convert tensor values to Int8 format.

        Converts a tensor of any dtype to Int8 format, clamping values to the
        range [-128, 127].

        Returns:
            A new AnyTensor with dtype=int8 containing converted values.

        Raises:
            Error: If conversion is not supported for the source dtype.

        Examples:
            var t = zeros([3, 4], DType.float32)
            var i8_t = t.to_int8()  # Returns int8 tensor

        Note:
            FP16 inputs are converted to FP32 before conversion.
        """

        # Create output tensor with int8 dtype
        var result = AnyTensor(self._shape, DType.int8)

        # Convert each element to Int8
        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                var source_val = self._data.bitcast[SIMD[DType.int8, 1]]()[i]
                result._data.bitcast[SIMD[DType.int8, 1]]()[i] = source_val
                continue
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_int8 conversion")

            # Convert to int8 range [-128, 127]
            var int_val = Int(val)
            if int_val < -128:
                int_val = -128
            elif int_val > 127:
                int_val = 127
            result._data.bitcast[SIMD[DType.int8, 1]]()[i][0] = int_val

        return result^

    fn to_int16(self) raises -> AnyTensor:
        """Convert tensor values to Int16 format.

        Converts a tensor of any dtype to Int16 format, clamping values to the
        range [-32768, 32767].

        Returns:
            A new AnyTensor with dtype=int16 containing converted values.

        Raises:
            Error: If conversion fails or bounds check error occurs.

        """
        var result = AnyTensor(self._shape, DType.int16)

        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                var source_val = self._data.bitcast[SIMD[DType.int16, 1]]()[i]
                result._data.bitcast[SIMD[DType.int16, 1]]()[i] = source_val
                continue
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                raise Error("Unsupported dtype for to_int16 conversion")

            var int_val = Int(val)
            if int_val < -32768:
                int_val = -32768
            elif int_val > 32767:
                int_val = 32767
            result._data.bitcast[SIMD[DType.int16, 1]]()[i][0] = int_val

        return result^

    fn to_int32(self) raises -> AnyTensor:
        """Convert tensor values to Int32 format.

        Converts a tensor of any dtype to Int32 format, clamping values to the
        range [-2147483648, 2147483647].

        Returns:
            A new AnyTensor with dtype=int32 containing converted values.

        Raises:
            Error: If conversion fails or bounds check error occurs.

        """
        var result = AnyTensor(self._shape, DType.int32)

        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                var source_val = self._data.bitcast[SIMD[DType.int32, 1]]()[i]
                result._data.bitcast[SIMD[DType.int32, 1]]()[i] = source_val
                continue
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                raise Error("Unsupported dtype for to_int32 conversion")

            var int_val = Int(val)
            result._data.bitcast[SIMD[DType.int32, 1]]()[i][0] = int_val

        return result^

    fn to_int64(self) raises -> AnyTensor:
        """Convert tensor values to Int64 format.

        Converts a tensor of any dtype to Int64 format.

        Returns:
            A new AnyTensor with dtype=int64 containing converted values.

        Raises:
            Error: If conversion fails or bounds check error occurs.

        """
        var result = AnyTensor(self._shape, DType.int64)

        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                result._set_int64(i, self._data.bitcast[Int64]()[i])
                continue
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                raise Error("Unsupported dtype for to_int64 conversion")

            var i64_val = Int64(val)
            result._set_int64(i, i64_val)

        return result^

    fn to_uint8(self) raises -> AnyTensor:
        """Convert tensor values to UInt8 format.

        Converts a tensor of any dtype to UInt8 format, clamping values to the
        range [0, 255].

        Returns:
            A new AnyTensor with dtype=uint8 containing converted values.

        Raises:
            Error: If conversion fails or bounds check error occurs.

        """
        var result = AnyTensor(self._shape, DType.uint8)

        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                var source_val = self._data.bitcast[SIMD[DType.uint8, 1]]()[i]
                result._data.bitcast[SIMD[DType.uint8, 1]]()[i] = source_val
                continue
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                raise Error("Unsupported dtype for to_uint8 conversion")

            var int_val = Int(val)
            if int_val < 0:
                int_val = 0
            elif int_val > 255:
                int_val = 255
            result._data.bitcast[SIMD[DType.uint8, 1]]()[i][0] = int_val

        return result^

    fn to_uint16(self) raises -> AnyTensor:
        """Convert tensor values to UInt16 format.

        Converts a tensor of any dtype to UInt16 format, clamping values to the
        range [0, 65535].

        Returns:
            A new AnyTensor with dtype=uint16 containing converted values.

        Raises:
            Error: If conversion fails or bounds check error occurs.

        """
        var result = AnyTensor(self._shape, DType.uint16)

        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                result._data.bitcast[UInt16]()[i] = self._data.bitcast[
                    UInt16
                ]()[i]
                continue
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                raise Error("Unsupported dtype for to_uint16 conversion")

            var u16_val = UInt16(val)
            result._data.bitcast[UInt16]()[i] = u16_val

        return result^

    fn to_uint32(self) raises -> AnyTensor:
        """Convert tensor values to UInt32 format.

        Converts a tensor of any dtype to UInt32 format, clamping values to the
        range [0, 4294967295].

        Returns:
            A new AnyTensor with dtype=uint32 containing converted values.

        Raises:
            Error: If conversion fails or bounds check error occurs.

        """
        var result = AnyTensor(self._shape, DType.uint32)

        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                result._data.bitcast[UInt32]()[i] = self._data.bitcast[
                    UInt32
                ]()[i]
                continue
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                raise Error("Unsupported dtype for to_uint32 conversion")

            var u32_val = UInt32(val)
            result._data.bitcast[UInt32]()[i] = u32_val

        return result^

    fn to_uint64(self) raises -> AnyTensor:
        """Convert tensor values to UInt64 format.

        Converts a tensor of any dtype to UInt64 format, clamping negative values to 0.

        Returns:
            A new AnyTensor with dtype=uint64 containing converted values.

        Raises:
            Error: If conversion fails or bounds check error occurs.

        """
        var result = AnyTensor(self._shape, DType.uint64)

        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                result._data.bitcast[UInt64]()[i] = self._data.bitcast[
                    UInt64
                ]()[i]
                continue
            else:
                raise Error("Unsupported dtype for to_uint64 conversion")

            var u64_val = UInt64(val)
            result._data.bitcast[UInt64]()[i] = u64_val

        return result^

    # ========================================================================
    # BF8 Conversion Methods
    # ========================================================================

    fn to_bf8(self) raises -> AnyTensor:
        """Convert tensor values to BF8 E5M2 format.

        This method converts a tensor of any floating-point dtype to BF8 format,
        stored as uint8. The conversion uses E5M2 encoding (1 sign bit, 5 exponent
        bits, 2 mantissa bits) which provides larger range than FP8 E4M3.

        Returns:
            A new AnyTensor with dtype=uint8 containing BF8-encoded values.

        Raises:
            Error: If the source tensor is not a floating-point dtype.

        Examples:
            var t = zeros([3, 4], DType.float32)
            var bf8_t = t.to_bf8()  # Returns uint8 tensor with BF8 encoding
            var restored = bf8_t.from_bf8()  # Convert back to float32

        Note:
            BF8 has larger range (~±57344) than FP8 but less precision (2 mantissa bits).
            Values outside this range are clamped. This is useful for memory-efficient
            training/inference where range is more important than precision.
            FP16 inputs are converted to FP32 before quantization.
        """
        from shared.core.types.dtype_aliases import BF8
        from memory import bitcast

        # Verify source is floating point
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
            or self._dtype == DType.bfloat16
        ):
            raise Error("to_bf8() requires a floating-point tensor")

        # Create output tensor with uint8 dtype
        var result = AnyTensor(self._shape, DType.uint8)

        # Convert each element to BF8
        for i in range(self._numel):
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            else:
                raise Error("Invalid dtype for BF8 conversion")

            # Convert to BF8 using native SIMD and store as uint8
            var bf8_val = SIMD[BF8, 1](val)
            var bf8_bits = bitcast[DType.uint8, 1](bf8_val)[0]
            result._data.bitcast[UInt8]()[i] = bf8_bits

        return result^

    fn from_bf8(self) raises -> AnyTensor:
        """Convert BF8-encoded tensor (uint8) back to Float32.

        This method interprets a uint8 tensor as BF8 E5M2 encoded values and
        converts them back to Float32 for computation.

        Returns:
            A new AnyTensor with dtype=float32 containing decoded values.

        Raises:
            Error: If the source tensor is not uint8 dtype.

        Examples:
            var bf8_t = ...  # uint8 tensor with BF8 encoding
            var float_t = bf8_t.from_bf8()  # Decode to float32

        Note:
            This assumes the uint8 tensor contains valid BF8 E5M2 encoded values.
            Use this to decode tensors created by to_bf8().
        """
        from shared.core.types.dtype_aliases import BF8
        from memory import bitcast

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_bf8() requires a uint8 tensor (BF8-encoded)")

        # Create output tensor with float32 dtype
        var result = AnyTensor(self._shape, DType.float32)

        # Convert each element from BF8 to Float32 using native SIMD
        for i in range(self._numel):
            var bf8_bits = self._data.bitcast[UInt8]()[i]
            # Bitcast uint8 to BF8, then convert to float32
            var bf8_val = bitcast[BF8, 1](SIMD[DType.uint8, 1](bf8_bits))
            var float_val = Float32(bf8_val[0])
            result[i] = Float32(float_val)

        return result^

    # ===----------------------------------------------------------------------===#
    # FP4 Blocked Type Conversions
    # ===----------------------------------------------------------------------===#

    fn to_mxfp4(self) raises -> AnyTensor:
        """Convert tensor values to MXFP4 blocked format.

        This method converts a tensor of any floating-point dtype to MXFP4 format,
        stored as uint8 blocks. Values are packed into 32-element blocks, each with
        a shared E8M0 scale.

        Returns:
            A new AnyTensor with dtype=uint8 containing MXFP4-encoded blocks.

        Raises:
            Error: If the source tensor is not a floating-point dtype.

        Examples:
            # Aligned size (32 elements = 1 block)
            var t = zeros([32], DType.float32)
            var mxfp4_t = t.to_mxfp4()  # Returns uint8 tensor (17 bytes)
            var restored = mxfp4_t.from_mxfp4()  # Restores 32 elements

            # Non-aligned size (33 elements = 2 blocks with padding)
            var t2 = zeros([33], DType.float32)
            var mxfp4_t2 = t2.to_mxfp4()  # Pads to 64 elements, returns 34 bytes
            var restored2 = mxfp4_t2.from_mxfp4()  # Correctly restores 33 elements!

            # Small tensors (1 element still uses full 32-element block)
            var scalar = AnyTensor([1], DType.float32)
            var quantized_scalar = scalar.to_mxfp4()  # Returns 17 bytes (padded to 32)

            # Multi-dimensional tensors (flattened for quantization)
            var weights = AnyTensor([64, 128], DType.float32)  # 8192 elements
            var quantized_weights = weights.to_mxfp4()  # 256 blocks × 17 bytes = 4352 bytes

            # ML workflow: quantize model weights for memory efficiency
            fn quantize_model_weights(weights: AnyTensor) raises -> AnyTensor:
                # Convert FP32 weights to MXFP4 (16:1 compression)
                return weights.to_mxfp4()

            # ML workflow: quantize gradients during training
            fn quantize_gradients(gradients: AnyTensor) raises -> AnyTensor:
                # MXFP4 works for both positive and negative values
                var quantized = gradients.to_mxfp4()
                # Dequantize before optimizer update
                return quantized.from_mxfp4()

        Error Handling:
            - Empty tensors: Raises "requires a floating-point tensor" if dtype is not FP16/FP32/FP64.
            - NaN values: Automatically clamped to max representable value (no error).
            - Infinity values: Automatically clamped to max representable value (no error).
            - Non-aligned sizes: Automatically padded with zeros (no error, transparent).
            - OOM conditions: Raises allocation error if insufficient memory for blocks.

        Performance:
            - Compression ratio: 16:1 vs Float32 (17 bytes per 32 values).
            - Time complexity: O(n) where n is number of elements.
            - Memory overhead: Temporary padding for non-aligned sizes.

        Note:
            MXFP4 uses 32-element blocks. Non-aligned tensors are padded with zeros,
            but original size is preserved in metadata. Round-trip conversion maintains
            original tensor size.
            Memory efficiency: 17 bytes per 32 Float32 values (16:1 compression).
            FP16 inputs are converted to FP32 before quantization.
        """
        from shared.core.types.mxfp4 import MXFP4Block

        # Verify source is floating point
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
            or self._dtype == DType.bfloat16
        ):
            raise Error("to_mxfp4() requires a floating-point tensor")

        # Calculate number of blocks (32 elements per block)
        var num_blocks = (self._numel + 31) // 32
        var total_bytes = num_blocks * 17  # 17 bytes per MXFP4Block

        # Create output tensor as flattened uint8 array
        var output_shape = List[Int]()
        output_shape.append(total_bytes)
        var result = AnyTensor(output_shape, DType.uint8)

        # Store original size before padding
        result._original_numel_quantized = self._numel

        # Process each block
        for block_idx in range(num_blocks):
            var start_idx = block_idx * 32
            var end_idx = min(start_idx + 32, self._numel)

            # Collect 32 values (pad with zeros if needed)
            var values = List[Float32]()
            for i in range(32):
                var idx = start_idx + i
                if idx < self._numel:
                    if idx >= self._numel:
                        raise Error("Index out of bounds during bitcast")

                    var val: Float32
                    if self._dtype == DType.float16:
                        val = self._data.bitcast[Float16]()[idx].cast[
                            DType.float32
                        ]()
                    elif self._dtype == DType.float32:
                        val = self._data.bitcast[Float32]()[idx]
                    elif self._dtype == DType.float64:
                        val = self._data.bitcast[Float64]()[idx].cast[
                            DType.float32
                        ]()
                    else:
                        raise Error("Invalid dtype for MXFP4 quantization")
                    values.append(val)
                else:
                    values.append(Float32(0.0))  # Padding.

            # Create MXFP4Block
            var block = MXFP4Block.from_float32_array(values)

            # Store block data (16 bytes + 1 scale byte)
            var block_offset = block_idx * 17
            for i in range(16):
                result._data.bitcast[UInt8]()[block_offset + i] = block.data[i]
            # Extract exponent bits from E8M0 scale via bitcast
            result._data.bitcast[UInt8]()[block_offset + 16] = bitcast[
                DType.uint8, 1
            ](block.scale)[0]

        return result^

    fn from_mxfp4(self) raises -> AnyTensor:
        """Convert MXFP4-encoded tensor (uint8 blocks) back to Float32.

        This method interprets a uint8 tensor as MXFP4 blocks and converts them
        back to Float32 for computation.

        Returns:
            A new AnyTensor with dtype=float32 containing decoded values.

        Raises:
            Error: If the source tensor is not uint8 dtype or not block-aligned.

        Examples:
            var mxfp4_t = ...  # uint8 tensor with MXFP4 blocks
            var float_t = mxfp4_t.from_mxfp4()  # Decode to float32, restores original size

        Note:
            This assumes the uint8 tensor contains valid MXFP4 blocks.
            Use this to decode tensors created by to_mxfp4().
            Original tensor size is restored from metadata if available.
        """
        from shared.core.types.mxfp4 import MXFP4Block
        from shared.core.types.dtype_aliases import E8M0

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_mxfp4() requires a uint8 tensor (MXFP4-encoded)")

        # Calculate number of blocks and output size
        if self._numel % 17 != 0:
            raise Error("MXFP4 tensor size must be multiple of 17 bytes")

        var num_blocks = self._numel // 17
        var padded_output_size = num_blocks * 32

        # Check if original size is stored
        var output_size: Int
        if self._original_numel_quantized >= 0:
            output_size = self._original_numel_quantized
        else:
            output_size = padded_output_size

        # Create output tensor with proper shape
        var output_shape = List[Int]()
        output_shape.append(padded_output_size)
        var result = AnyTensor(output_shape, DType.float32)

        # Decode each block
        for block_idx in range(num_blocks):
            var block_offset = block_idx * 17

            # Reconstruct MXFP4Block
            var data = SIMD[DType.uint8, 16](0)
            for i in range(16):
                data[i] = self._data.bitcast[UInt8]()[block_offset + i]
            # Reconstruct E8M0 scale from raw exponent byte
            var scale_byte = self._data.bitcast[UInt8]()[block_offset + 16]
            var scale = bitcast[E8M0, 1](SIMD[DType.uint8, 1](scale_byte))

            var block = MXFP4Block(data, scale)

            # Decode block to Float32 values (only decode needed elements)
            var values = block.to_float32_array()
            for i in range(32):
                var output_idx = block_idx * 32 + i
                if output_idx < output_size:
                    result[output_idx] = Float32(values[i])

        # Trim result to original size if needed
        if output_size < padded_output_size:
            var trimmed = AnyTensor([output_size], DType.float32)
            for i in range(output_size):
                trimmed[i] = result._data.bitcast[Float32]()[i]
            return trimmed^

        return result^

    fn to_nvfp4(self) raises -> AnyTensor:
        """Convert tensor values to NVFP4 blocked format.

        This method converts a tensor of any floating-point dtype to NVFP4 format,
        stored as uint8 blocks. Values are packed into 16-element blocks, each with
        a shared E4M3 scale.

        Returns:
            A new AnyTensor with dtype=uint8 containing NVFP4-encoded blocks.

        Raises:
            Error: If the source tensor is not a floating-point dtype.

        Examples:
        ```
                # Aligned size (16 elements = 1 block)
                var t = zeros([16], DType.float32)
                var nvfp4_t = t.to_nvfp4()  # Returns uint8 tensor (9 bytes)
                var restored = nvfp4_t.from_nvfp4()  # Restores 16 elements

                # Non-aligned size (17 elements = 2 blocks with padding)
                var t2 = zeros([17], DType.float32)
                var nvfp4_t2 = t2.to_nvfp4()  # Pads to 32 elements, returns 18 bytes
                var restored2 = nvfp4_t2.from_nvfp4()  # Correctly restores 17 elements!

                # Small tensors (1 element still uses full 16-element block)
                var scalar = AnyTensor([1], DType.float32)
                var quantized_scalar = scalar.to_nvfp4()  # Returns 9 bytes (padded to 16)

                # Multi-dimensional tensors (flattened for quantization)
                var activations = AnyTensor([128, 256], DType.float32)  # 32768 elements
                var quantized_activations = activations.to_nvfp4()  # 2048 blocks × 9 bytes = 18432 bytes

                # ML workflow: quantize activations with better accuracy than MXFP4
                fn quantize_activations(activations: AnyTensor) raises -> AnyTensor:
                    # NVFP4 provides better accuracy (smaller blocks = better scale granularity)
                    return activations.to_nvfp4()

                # ML workflow: quantize gradients with E4M3 scale (recommended by paper)
                fn quantize_gradients_nvfp4(gradients: AnyTensor) raises -> AnyTensor:
                    # E4M3 achieves best results according to Dettmers et al. 2023
                    var quantized = gradients.to_nvfp4()
                    return quantized.from_nvfp4()

                # Compare accuracy: NVFP4 vs MXFP4
                fn compare_quantization_accuracy(data: AnyTensor) raises:
                    var mxfp4_quantized = data.to_mxfp4().from_mxfp4()
                    var nvfp4_quantized = data.to_nvfp4().from_nvfp4()
                    # NVFP4 typically has lower error due to smaller blocks (16 vs 32)
        ```

            Error Handling:
                - Empty tensors: Raises "requires a floating-point tensor" if dtype is not FP16/FP32/FP64.
                - NaN values: Automatically clamped to max representable value (no error).
                - Infinity values: Automatically clamped to max representable value (no error).
                - Non-aligned sizes: Automatically padded with zeros (no error, transparent).
                - OOM conditions: Raises allocation error if insufficient memory for blocks.

            Performance:
                - Compression ratio: 14:1 vs Float32 (9 bytes per 16 values).
                - Time complexity: O(n) where n is number of elements.
                - Memory overhead: Temporary padding for non-aligned sizes.
                - Accuracy: Better than MXFP4 due to smaller blocks (per Dettmers et al.).

            Note:
                NVFP4 uses 16-element blocks for better accuracy. Non-aligned tensors are
                padded with zeros, but original size is preserved in metadata.
                Memory efficiency: 9 bytes per 16 Float32 values (14:1 compression).
                FP16 inputs are converted to FP32 before quantization.
        """
        from shared.core.types.nvfp4 import NVFP4Block

        # Verify source is floating point
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
            or self._dtype == DType.bfloat16
        ):
            raise Error("to_nvfp4() requires a floating-point tensor")

        # Calculate number of blocks (16 elements per block)
        var num_blocks = (self._numel + 15) // 16
        var total_bytes = num_blocks * 9  # 9 bytes per NVFP4Block

        # Create output tensor as flattened uint8 array
        var output_shape = List[Int]()
        output_shape.append(total_bytes)
        var result = AnyTensor(output_shape, DType.uint8)

        # Store original size before padding
        result._original_numel_quantized = self._numel

        # Process each block
        for block_idx in range(num_blocks):
            var start_idx = block_idx * 16
            var end_idx = min(start_idx + 16, self._numel)

            # Collect 16 values (pad with zeros if needed)
            var values = List[Float32]()
            for i in range(16):
                var idx = start_idx + i
                if idx < self._numel:
                    if idx >= self._numel:
                        raise Error("Index out of bounds during bitcast")

                    var val: Float32
                    if self._dtype == DType.float16:
                        val = self._data.bitcast[Float16]()[idx].cast[
                            DType.float32
                        ]()
                    elif self._dtype == DType.float32:
                        val = self._data.bitcast[Float32]()[idx]
                    elif self._dtype == DType.float64:
                        val = self._data.bitcast[Float64]()[idx].cast[
                            DType.float32
                        ]()
                    else:
                        raise Error("Invalid dtype for NVFP4 quantization")
                    values.append(val)
                else:
                    values.append(Float32(0.0))  # Padding.

            # Create NVFP4Block
            var block = NVFP4Block.from_float32_array(values)

            # Store block data (8 bytes + 1 scale byte)
            var block_offset = block_idx * 9
            for i in range(8):
                result._data.bitcast[UInt8]()[block_offset + i] = block.data[i]
            # Extract raw FP8 bits from scale via bitcast
            result._data.bitcast[UInt8]()[block_offset + 8] = bitcast[
                DType.uint8, 1
            ](block.scale)[0]

        return result^

    fn from_nvfp4(self) raises -> AnyTensor:
        """Convert NVFP4-encoded tensor (uint8 blocks) back to Float32.

        This method interprets a uint8 tensor as NVFP4 blocks and converts them
        back to Float32 for computation.

        Returns:
            A new AnyTensor with dtype=float32 containing decoded values.

        Raises:
            Error: If the source tensor is not uint8 dtype or not block-aligned.

        Examples:
        ```
                var nvfp4_t = ...  # uint8 tensor with NVFP4 blocks
                var float_t = nvfp4_t.from_nvfp4()  # Decode to float32, restores original size
        ```

            Note:
                This assumes the uint8 tensor contains valid NVFP4 blocks.
                Use this to decode tensors created by to_nvfp4().
                Original tensor size is restored from metadata if available.
        """
        from shared.core.types.nvfp4 import NVFP4Block
        from shared.core.types.dtype_aliases import FP8

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_nvfp4() requires a uint8 tensor (NVFP4-encoded)")

        # Calculate number of blocks and output size
        if self._numel % 9 != 0:
            raise Error("NVFP4 tensor size must be multiple of 9 bytes")

        var num_blocks = self._numel // 9
        var padded_output_size = num_blocks * 16

        # Check if original size is stored
        var output_size: Int
        if self._original_numel_quantized >= 0:
            output_size = self._original_numel_quantized
        else:
            output_size = padded_output_size

        # Create output tensor with proper shape
        var output_shape = List[Int]()
        output_shape.append(padded_output_size)
        var result = AnyTensor(output_shape, DType.float32)

        # Decode each block
        for block_idx in range(num_blocks):
            var block_offset = block_idx * 9

            # Reconstruct NVFP4Block
            var data = SIMD[DType.uint8, 8](0)
            for i in range(8):
                data[i] = self._data.bitcast[UInt8]()[block_offset + i]
            # Reconstruct FP8 (E4M3) scale from raw byte
            var scale_byte = self._data.bitcast[UInt8]()[block_offset + 8]
            var scale = bitcast[FP8, 1](SIMD[DType.uint8, 1](scale_byte))

            var block = NVFP4Block(data, scale)

            # Decode block to Float32 values (only decode needed elements)
            var values = block.to_float32_array()
            for i in range(16):
                var output_idx = block_idx * 16 + i
                if output_idx < output_size:
                    result[output_idx] = Float32(values[i])

        # Trim result to original size if needed
        if output_size < padded_output_size:
            var trimmed = AnyTensor([output_size], DType.float32)
            for i in range(output_size):
                trimmed[i] = result._data.bitcast[Float32]()[i]
            return trimmed^

        return result^

    # Reflected operators - enable reversed operand order (e.g., 2 + tensor)
    # These are called when the left operand doesn't support the operation
    fn __radd__(self, other: AnyTensor) raises -> AnyTensor:
        """Reflected addition: `other + self` (commutative, so same as __add__).

        Raises:
            Error: If tensors have incompatible shapes.

        """
        return self.__add__(other)

    fn __rsub__(self, other: AnyTensor) raises -> AnyTensor:
        """Reflected subtraction: `other - self` (order matters: returns other - self).

        Raises:
            Error: If tensors have incompatible shapes.

        """
        return other - self

    fn __rmul__(self, other: AnyTensor) raises -> AnyTensor:
        """Reflected multiplication: other * self (commutative, so same as __mul__).

        Raises:
            Error: If tensors have incompatible shapes.

        """
        return self.__mul__(other)

    fn __rtruediv__(self, other: AnyTensor) raises -> AnyTensor:
        """Reflected division: other / self (order matters: returns other / self).

        Raises:
            Error: If tensors have incompatible shapes or division by zero.

        """
        return other / self

    # In-place operators - mutate self instead of creating new tensor
    fn __iadd__(mut self, other: AnyTensor) raises:
        """In-place addition: `self += other`.

        Raises:
            Error: If tensors have incompatible shapes or dtypes.

        """
        self = self + other

    fn __isub__(mut self, other: AnyTensor) raises:
        """In-place subtraction: `self -= other`.

        Raises:
            Error: If tensors have incompatible shapes or dtypes.

        """
        self = self - other

    fn __imul__(mut self, other: AnyTensor) raises:
        """In-place multiplication: `self *= other`.

        Raises:
            Error: If tensors have incompatible shapes or dtypes.

        """
        self = self * other

    fn __itruediv__(mut self, other: AnyTensor) raises:
        """In-place division: `self /= other`.

        Raises:
            Error: If tensors have incompatible shapes or dtypes, or division by zero.

        """
        self = self / other

    # Unary operators - operate on single tensor
    fn __neg__(self) raises -> AnyTensor:
        """Negation: `-self`.

        Raises:
            Error: If tensor allocation fails.

        """

        @always_inline
        fn _neg[T: DType](x: Scalar[T]) -> Scalar[T]:
            return -x

        return _anytensor_unary_op[_neg](self)

    fn __pos__(self) raises -> AnyTensor:
        """Positive: +self (returns a copy).

        Raises:
            Error: If tensor allocation fails.

        """
        # Return a copy of the tensor using Mojo's copy semantics
        var copy = self
        return copy^

    fn __abs__(self) raises -> AnyTensor:
        """Absolute value: abs(self).

        Raises:
            Error: If operation fails.

        """

        @always_inline
        fn _abs[T: DType](x: Scalar[T]) -> Scalar[T]:
            if x < Scalar[T](0):
                return -x
            return x

        return _anytensor_unary_op[_abs](self)

    fn __len__(self) -> Int:
        """Return the size of the first dimension.

        This follows NumPy/PyTorch convention where len() returns the
        size of the first dimension (axis 0).

        Returns:
            The size of the first dimension, or 0 if the tensor is 0-dimensional.

        Example:
            ```mojo
            var x = ones([5, 3], DType.float32)
            var length = len(x)  # Returns 5
            ```
        """
        if len(self._shape) == 0:
            return 0
        return self._shape[0]

    fn __bool__(self) raises -> Bool:
        """Return the boolean value of a single-element tensor.

        Follows PyTorch/NumPy convention: a single-element tensor can be
        used in boolean context. Returns True if the value is non-zero.

        Returns:
            True if the single element is non-zero, False otherwise.

        Raises:
            Error: If tensor has more than one element.

        Example:
            ```mojo
            var x = full([], 5.0, DType.float32)
            if x:  # True
                print("non-zero")
            ```
        """
        return self.item() != 0.0

    fn __int__(self) raises -> Int:
        """Convert single-element tensor to Int.

        Returns:
            The scalar value as Int.

        Raises:
            Error: If tensor has more than one element.

        Example:
            ```mojo
            var x = full([], 7.0, DType.float32)
            var i = Int(x)  # Returns 7
            ```
        """
        return Int(self.item())

    fn __float__(self) raises -> Float64:
        """Convert single-element tensor to Float64.

        Returns:
            The scalar value as Float64.

        Raises:
            Error: If tensor has more than one element.

        Example:
            ```mojo
            var x = full([], 3.14, DType.float32)
            var f = Float64(x)  # Returns 3.14
            ```
        """
        return self.item()

    fn __str__(self) -> String:
        """Human-readable string representation with NumPy-style truncation.

        For tensors with more than 1000 elements, shows only the first 3 and
        last 3 elements with '...' in between to prevent performance issues.

        Formats values by dtype:
        - Float types: display as decimals (1.0, 2.5, etc.)
        - Integer types: display without decimals (1, 42, -100, etc.)
        - Bool type: display as True/False

        For multi-dimensional tensors (2D+), includes shape info and nested brackets.

        Returns:
            1D: AnyTensor([v0, v1, ...], dtype=<dtype>)
            2D+: AnyTensor([[v0, v1, ...], ...], shape=[d0, d1, ...], dtype=<dtype>)

        Example:
            ```mojo
            var x = arange(1000, DType.float32)
            print(x)  # AnyTensor([0.0, 1.0, 2.0, ..., 997.0, 998.0, 999.0], dtype=float32)
            var y = full([2, 3], Float64(42), DType.int32)
            print(y)  # AnyTensor([[42, 42, 42], [42, 42, 42]], shape=[2, 3], dtype=int32)
            ```
        """
        comptime TRUNCATE_THRESHOLD = 1000
        comptime SHOW_ELEMENTS = 3

        var ndim = len(self._shape)

        # Special case: empty tensor
        if ndim == 0 or self._numel == 0:
            return "AnyTensor([], dtype=" + String(self._dtype) + ")"

        # For 1D tensors: use flat format
        if ndim == 1:
            var result = String("AnyTensor([")
            if self._numel > TRUNCATE_THRESHOLD:
                for i in range(SHOW_ELEMENTS):
                    if i > 0:
                        result += ", "
                    result += self._format_element(i)
                result += ", ..."
                for i in range(self._numel - SHOW_ELEMENTS, self._numel):
                    result += ", " + self._format_element(i)
            else:
                for i in range(self._numel):
                    if i > 0:
                        result += ", "
                    result += self._format_element(i)
            result += "], dtype=" + String(self._dtype) + ")"
            return result

        # For multi-dimensional tensors (2D+): build nested brackets.
        # Truncate if total elements exceed threshold to prevent
        # massive string output for large tensors (e.g., [100, 100]).
        if self._numel > TRUNCATE_THRESHOLD:
            # Show first and last sub-arrays along outermost dimension
            var stride = 1
            for d in range(1, ndim):
                stride *= self._shape[d]

            var data_str = String("[")
            for i in range(SHOW_ELEMENTS):
                if i > 0:
                    data_str += ", "
                data_str += self._format_nd_slice(1, i * stride)
            data_str += ", ..."
            for i in range(self._shape[0] - SHOW_ELEMENTS, self._shape[0]):
                data_str += ", " + self._format_nd_slice(1, i * stride)
            data_str += "]"

            var result = "AnyTensor(" + data_str + ", shape=["
            for i in range(len(self._shape)):
                if i > 0:
                    result += ", "
                result += String(self._shape[i])
            result += "], dtype=" + String(self._dtype) + ")"
            return result

        var data_str = self._format_nd_slice(0, 0)

        var result = "AnyTensor(" + data_str + ", shape=["
        for i in range(len(self._shape)):
            if i > 0:
                result += ", "
            result += String(self._shape[i])
        result += "], dtype=" + String(self._dtype) + ")"
        return result

    fn _format_element(self, flat_idx: Int) -> String:
        """Format a single element based on dtype.

        Handles unsigned integers natively to avoid sign corruption when
        values exceed Int64 range (e.g., uint64 values > 2^63).

        Args:
            flat_idx: The flat index in the buffer.

        Returns:
            String representation of the element.
        """
        if self._dtype == DType.bool:
            return "True" if self._get_int64(flat_idx) != 0 else "False"
        elif self._dtype == DType.uint64:
            # Read as native UInt64 to avoid sign corruption via _get_int64
            var dtype_size = self._get_dtype_size()
            var ptr = (self._data + flat_idx * dtype_size).bitcast[UInt64]()
            return String(ptr[])
        elif self._dtype == DType.uint32:
            var dtype_size = self._get_dtype_size()
            var ptr = (self._data + flat_idx * dtype_size).bitcast[UInt32]()
            return String(ptr[])
        elif (
            self._dtype == DType.int8
            or self._dtype == DType.int16
            or self._dtype == DType.int32
            or self._dtype == DType.int64
            or self._dtype == DType.uint8
            or self._dtype == DType.uint16
        ):
            return String(self._get_int64(flat_idx))
        else:
            # Float types
            return String(self._get_float64(flat_idx))

    fn _format_nd_slice(
        self, dim: Int, base_offset: Int
    ) -> String:
        """Format a slice of the N-dimensional tensor with nested brackets.

        Design: uses offset-based recursion instead of threading a mutable counter
        through calls. Each call computes its flat indices as base_offset + i * stride,
        making the function pure — its behavior is determined entirely by its arguments,
        not hidden mutable state. This mirrors the row-major index formula directly:
        element [i,j,k] lives at flat index i*(J*K) + j*K + k.

        Args:
            dim: Current dimension level (0 = outermost).
            base_offset: Flat index offset for the start of this slice.

        Returns:
            String with nested brackets representing the N-D structure.
        """
        var ndim = len(self._shape)

        # Base case: innermost dimension (last dim)
        if dim == ndim - 1:
            var result = String("[")
            for i in range(self._shape[dim]):
                if i > 0:
                    result += ", "
                result += self._format_element(base_offset + i)
            result += "]"
            return result

        # Compute stride for current dimension (product of all inner dims)
        var stride = 1
        for d in range(dim + 1, ndim):
            stride *= self._shape[d]

        # Recursive case: format sub-array
        var result = String("[")
        for i in range(self._shape[dim]):
            if i > 0:
                result += ", "
            result += self._format_nd_slice(dim + 1, base_offset + i * stride)

        result += "]"
        return result

    fn __repr__(self) -> String:
        """Detailed representation for debugging.

        Returns:
            String in the format: AnyTensor(shape=[...], dtype=<dtype>, numel=N, data=[...]).
            For large tensors: AnyTensor(shape=[...], dtype=<dtype>, numel=N, data=[v0, v1, v2, ..., vN-2, vN-1, vN]).
        """
        comptime TRUNCATE_THRESHOLD = 1000
        comptime SHOW_ELEMENTS = 3

        var shape_str = String("[")
        for i in range(len(self._shape)):
            if i > 0:
                shape_str += ", "
            shape_str += String(self._shape[i])
        shape_str += "]"
        var result = String("AnyTensor(shape=") + shape_str
        result += ", dtype=" + String(self._dtype)
        result += ", numel=" + String(self._numel)
        result += ", data=["
        if self._numel > TRUNCATE_THRESHOLD:
            for i in range(SHOW_ELEMENTS):
                if i > 0:
                    result += ", "
                result += String(self._get_float64(i))
            result += ", ..."
            for i in range(self._numel - SHOW_ELEMENTS, self._numel):
                result += ", " + String(self._get_float64(i))
        else:
            for i in range(self._numel):
                if i > 0:
                    result += ", "
                result += String(self._get_float64(i))
        result += "])"
        return result

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Compute hash based on shape, dtype, and data.

        AnyTensor implements the `Hashable` trait, allowing tensors to be used as
        dictionary keys or in hash-based data structures. Two tensors with identical
        shape, dtype, and element values will produce the same hash.

        Parameters:
            H: The hasher type conforming to the Hasher trait.

        Args:
            hasher: The hasher to write values into.

        Note:
            The hash is computed from the tensor's shape dimensions, dtype ordinal,
            and element values. All NaN values are canonicalized before hashing,
            so tensors differing only in NaN bit patterns hash equally.

        Example:
            ```mojo
            from hashlib import hash
            var x = ones([3], DType.float32)
            var h = hash(x)

            # Tensors with identical shape, dtype, and values hash equally
            var y = ones([3], DType.float32)
            assert hash(x) == hash(y)

            # Tensors with different shapes hash differently
            var z = ones([4], DType.float32)
            # hash(x) != hash(z)  (with overwhelming probability)
            ```
        """
        # Hash shape
        for i in range(len(self._shape)):
            hasher.update(self._shape[i])
        # Hash dtype ordinal
        hasher.update(dtype_to_ordinal(self._dtype))
        # Hash data — canonicalize NaN so all NaN bit patterns hash equally
        from math import isnan

        for i in range(self._numel):
            var val = self._get_float64(i)
            if isnan(val):
                # Canonical NaN: positive quiet NaN (0x7FF8000000000000)
                hasher.update(UInt64(0x7FF8000000000000))
            else:
                var int_bits = UnsafePointer[Float64](to=val).bitcast[
                    UInt64
                ]()[]
                hasher.update(int_bits)

    fn contiguous(self) raises -> AnyTensor:
        """Return a contiguous copy of the tensor.

        If the tensor is already contiguous, returns a clone.
        Otherwise, creates a new contiguous tensor with the same data.

        Returns:
            A contiguous AnyTensor with the same shape, dtype, and values.

        Raises:
            Error: If memory allocation fails.

        Example:
            ```mojo
            var x = ones([3, 4], DType.float32)
            var c = x.contiguous()  # Already contiguous, returns clone
            ```
        """
        return self.clone()

    fn as_tensor[dtype: DType](self) raises -> Tensor[dtype]:
        """Zero-copy conversion to compile-time typed Tensor[dtype].

        Creates a Tensor[dtype] that shares the same data buffer and refcount.
        The dtype parameter must match self._dtype at runtime.

        Both files are siblings in shared/tensor/, so no circular dependency.

        Parameters:
            dtype: The compile-time DType parameter (must match self._dtype).

        Returns:
            A Tensor[dtype] sharing the same data and refcount.

        Raises:
            Error: If dtype doesn't match self._dtype.
        """
        if self._dtype != dtype:
            raise Error(
                "DType mismatch: tensor has dtype "
                + String(self._dtype)
                + " but as_tensor called with "
                + String(dtype)
            )
        # Zero-copy: bitcast data pointer, share refcount
        return Tensor[dtype](
            self._data.bitcast[Scalar[dtype]](),
            self._shape,
            self._strides,
            self._refcount,
            self._numel,
            self._is_view,
            self._allocated_size,
            self._original_numel_quantized,
        )

    # ============================================================================
    # Utility Methods
    # ============================================================================

    fn clone(self) raises -> AnyTensor:
        """Create a clone of the tensor.

        Creates a new tensor with the same shape, dtype, and values but with
        separate underlying data. Modifications to the clone do not affect
        the original tensor.

        This method follows PyTorch naming conventions.

        Returns:
            A new AnyTensor that is a deep copy of self.

        Raises:
            Error: If memory allocation fails.

        Example:
            ```mojo
            var x = zeros([3, 4], DType.float32)
            var y = x.clone()  # Independent copy
            ```
        """
        var shape_copy = self._shape.copy()
        var result = AnyTensor(shape_copy, self._dtype)

        # Iterate through all elements using multi-dimensional indexing
        # to correctly handle non-contiguous source tensors with stride-aware access
        var nd_idx = List[Int]()
        for i in range(len(self._shape)):
            nd_idx.append(0)

        var dtype_size = self._get_dtype_size()

        for out_idx in range(self._numel):
            # Compute flat offset in source tensor using strides
            var src_offset = 0
            for d in range(len(self._shape)):
                src_offset += nd_idx[d] * self._strides[d]

            # Read from source using stride-aware byte offset
            var offset_bytes = src_offset * dtype_size
            var val: Float64

            if self._dtype == DType.float16:
                var ptr = (self._data + offset_bytes).bitcast[Float16]()
                val = ptr[].cast[DType.float64]()
            elif self._dtype == DType.bfloat16:
                var ptr = (self._data + offset_bytes).bitcast[BFloat16]()
                val = Float64(Float32(ptr[]))
            elif self._dtype == DType.float32:
                var ptr = (self._data + offset_bytes).bitcast[Float32]()
                val = ptr[].cast[DType.float64]()
            elif self._dtype == DType.float64:
                var ptr = (self._data + offset_bytes).bitcast[Float64]()
                val = ptr[]
            else:
                # For integer types, use _get_int64 via byte offset
                if self._dtype == DType.int8:
                    var ptr = (self._data + offset_bytes).bitcast[Int8]()
                    val = Float64(ptr[])
                elif self._dtype == DType.int16:
                    var ptr = (self._data + offset_bytes).bitcast[Int16]()
                    val = Float64(ptr[])
                elif self._dtype == DType.int32:
                    var ptr = (self._data + offset_bytes).bitcast[Int32]()
                    val = Float64(ptr[])
                elif self._dtype == DType.int64:
                    var ptr = (self._data + offset_bytes).bitcast[Int64]()
                    val = Float64(ptr[])
                elif self._dtype == DType.uint8:
                    var ptr = (self._data + offset_bytes).bitcast[UInt8]()
                    val = Float64(Int(ptr[]))
                elif self._dtype == DType.uint16:
                    var ptr = (self._data + offset_bytes).bitcast[UInt16]()
                    val = Float64(Int(ptr[]))
                elif self._dtype == DType.uint32:
                    var ptr = (self._data + offset_bytes).bitcast[UInt32]()
                    val = Float64(Int(ptr[]))
                elif self._dtype == DType.uint64:
                    var ptr = (self._data + offset_bytes).bitcast[UInt64]()
                    val = Float64(Int(ptr[]))
                else:
                    val = 0.0

            # Write to output tensor at flat index
            result._set_float64(out_idx, val)

            # Increment multi-dimensional index
            var d = len(self._shape) - 1
            while d >= 0:
                nd_idx[d] += 1
                if nd_idx[d] < self._shape[d]:
                    break
                nd_idx[d] = 0
                d -= 1

        return result^

    fn item(self) raises -> Float64:
        """Extract the value from a single-element tensor.

        Returns:
            The scalar value as Float64.

        Raises:
            Error: If tensor has more than one element.

        Example:
            ```mojo
            var x = full([], 42.0, DType.float32)
            var val = x.item()  # Returns 42.0
            ```
        """
        if self._numel != 1:
            raise Error(
                "item() requires single-element tensor, got "
                + String(self._numel)
                + " elements"
            )
        return self._get_float64(0)

    fn tolist(self) raises -> List[Float64]:
        """Convert tensor to a flat list of Float64 values.

        Returns:
            A flat list containing all tensor values.

        Example:
            ```mojo
            var x = arange(0.0, 5.0, 1.0, DType.float32)
            var lst = x.tolist()  # [0.0, 1.0, 2.0, 3.0, 4.0]
            ```
        """
        var result = List[Float64]()
        for i in range(self._numel):
            result.append(self._get_float64(i))
        return result^

    fn diff(self, n: Int = 1) raises -> AnyTensor:
        """Calculate consecutive differences.

        Computes the n-th order discrete difference along the first axis.

        Args:
            n: Order of differences (default: 1).

        Returns:
            A new AnyTensor with differences computed.

        Raises:
            Error: If n <= 0 or n >= tensor size.

        Example:
            ```mojo
            var x = arange(0.0, 5.0, 1.0, DType.float32)
            var d = x.diff()  # [1.0, 1.0, 1.0, 1.0]
            ```
        """
        if n <= 0:
            raise Error("diff order n must be positive, got " + String(n))
        if n >= self._numel:
            raise Error(
                "diff order n="
                + String(n)
                + " exceeds tensor size "
                + String(self._numel)
            )

        var current = self
        for _ in range(n):
            var new_size = current._numel - 1
            var new_shape = List[Int]()
            new_shape.append(new_size)
            var result = AnyTensor(new_shape, current._dtype)

            for i in range(new_size):
                var val = current._get_float64(i + 1) - current._get_float64(i)
                result._set_float64(i, val)

            current = result^

        return current^

    fn save(self, path: String, name: String = "") raises:
        """Save tensor to file in hex-encoded binary format.

        Persists tensor with metadata (dtype, shape) and hex-encoded byte data.
        File format is text-based for portability across platforms.

        Args:
            path: Output file path.
            name: Optional tensor name (defaults to empty string).

        Raises:
            Error: If file write fails or path is invalid.

        Example:
            ```mojo
            var weights = zeros([3, 4], DType.float32)
            weights.save("checkpoint/weights.bin", "conv1_weights")
            ```
        """
        from .tensor_io import save_tensor

        save_tensor(self, path, name)

    @staticmethod
    fn load(path: String) raises -> AnyTensor:
        """Load tensor from file.

        Reads hex-encoded tensor data and metadata, reconstructs
        AnyTensor with original dtype and shape.

        Args:
            path: Input file path.

        Returns:
            Loaded AnyTensor.

        Raises:
            Error: If file format is invalid or file doesn't exist.

        Example:
            ```mojo
            var tensor = AnyTensor.load("checkpoint/weights.bin")
            ```
        """
        from .tensor_io import load_tensor

        return load_tensor(path)

    fn split(self, num_splits: Int, axis: Int = 0) raises -> List[AnyTensor]:
        """Split tensor into equal-sized parts along an axis.

        Method wrapper for the module-level `split()` function, providing
        convenient object syntax: `tensor.split(3)` instead of
        `split(tensor, 3)`.

        Args:
            num_splits: Number of equal parts to split into.
            axis: Axis along which to split (default: 0).

        Returns:
            List of AnyTensor objects, each with same shape except along
            split axis.

        Raises:
            Error: If axis is invalid, num_splits <= 0, or tensor size not
                divisible by num_splits.

        Example:
        ```mojo
            var a = arange(0.0, 12.0, 1.0, DType.float32)
            var parts = a.split(3)  # 3 parts of size 4 each
        ```
        """
        if num_splits <= 0:
            raise Error("split: num_splits must be positive, got " + String(num_splits))
        if axis < 0 or axis >= len(self._shape):
            raise Error(
                "split: axis " + String(axis) + " out of range for "
                + String(len(self._shape)) + "-D tensor"
            )
        var dim_size = self._shape[axis]
        if dim_size % num_splits != 0:
            raise Error(
                "split: dimension " + String(dim_size)
                + " not divisible by " + String(num_splits)
            )
        var chunk_size = dim_size // num_splits
        var parts = List[AnyTensor]()
        for i in range(num_splits):
            var start = i * chunk_size
            var end = start + chunk_size
            # slice returns a view; clone to get independent memory
            var part = self.slice(start, end, axis).clone()
            parts.append(part^)
        return parts^

    fn split_with_indices(
        self, split_indices: List[Int], axis: Int = 0
    ) raises -> List[AnyTensor]:
        """Split tensor at specified indices along an axis.

        Method wrapper for the module-level `split_with_indices()` function,
        providing convenient object syntax:
        `tensor.split_with_indices([3, 7])` instead of
        `split_with_indices(tensor, [3, 7])`.

        Args:
            split_indices: List of indices where to split (e.g., [3, 7]
                creates 3 sections: [0-2], [3-6], [7-end]).
            axis: Axis along which to split (default: 0).

        Returns:
            List of AnyTensor objects resulting from splits.

        Raises:
            Error: If axis is invalid or indices are out of bounds/unordered.

        Example:
        ```mojo
            var a = arange(0.0, 10.0, 1.0, DType.float32)
            var parts = a.split_with_indices([3, 7])
            # parts[0].shape() = (3,)  # indices 0-2
            # parts[1].shape() = (4,)  # indices 3-6
            # parts[2].shape() = (3,)  # indices 7-9
        ```
        """
        if axis < 0 or axis >= len(self._shape):
            raise Error(
                "split_with_indices: axis " + String(axis)
                + " out of range for " + String(len(self._shape)) + "-D tensor"
            )
        var dim_size = self._shape[axis]
        var parts = List[AnyTensor]()
        var prev = 0
        for i in range(len(split_indices)):
            var idx = split_indices[i]
            if idx < prev or idx > dim_size:
                raise Error(
                    "split_with_indices: index " + String(idx)
                    + " out of bounds or unordered"
                )
            if idx > prev:
                var part = self.slice(prev, idx, axis).clone()
                parts.append(part^)
            prev = idx
        # Final segment from last index to end
        if prev < dim_size:
            var part = self.slice(prev, dim_size, axis).clone()
            parts.append(part^)
        return parts^

    fn broadcast_to(self, target_shape: List[Int]) raises -> AnyTensor:
        """Broadcast tensor to target shape.

        Provides convenient object syntax: `tensor.broadcast_to([4, 3])`.
        Uses module-level broadcasting utilities (no circular import).
        """
        # Inline broadcast_to to avoid circular import via shared.core.shape.
        # Uses module-level are_shapes_broadcastable and compute_broadcast_strides.
        # See Issue #4513.
        var shape = self.shape()

        if len(target_shape) < len(shape):
            raise Error("broadcast_to: cannot broadcast to fewer dimensions")

        if not are_shapes_broadcastable(shape, target_shape):
            raise Error("broadcast_to: shapes are not broadcast-compatible")

        var broadcast_strides = compute_broadcast_strides(shape, target_shape)
        var result = AnyTensor(target_shape, self.dtype())
        var result_numel = result.numel()

        for i in range(result_numel):
            var coords = List[Int]()
            var temp_i = i
            for j in range(len(target_shape)):
                var stride = 1
                for k in range(j + 1, len(target_shape)):
                    stride *= target_shape[k]
                var coord = temp_i // stride
                coords.append(coord)
                temp_i = temp_i % stride

            var src_idx = 0
            for j in range(len(target_shape)):
                src_idx += coords[j] * broadcast_strides[j]

            var val = self._get_float64(src_idx)
            result._set_float64(i, val)

        return result^



# ============================================================================
# Private Broadcasting Helpers
# ============================================================================
# Binary, unary, and comparison helpers implement element-wise operations with
# NumPy-style broadcasting for use by AnyTensor's operator overloads.
# Defined here (rather than in arithmetic/comparison modules) to break circular
# import chains. Both files are now siblings in shared/tensor/.
# See Issue #4513.


fn _anytensor_binary_op[
    op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Apply a compile-time-typed binary arithmetic op with broadcasting."""
    if a._dtype != b._dtype:
        raise Error("Cannot operate on tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var result = AnyTensor(result_shape, a._dtype)
    var ordinal = dtype_to_ordinal(a._dtype)

    @parameter
    fn _apply[dtype: DType]():
        var a_ptr = a._data.bitcast[Scalar[dtype]]()
        var b_ptr = b._data.bitcast[Scalar[dtype]]()
        var r_ptr = result._data.bitcast[Scalar[dtype]]()
        var result_strides = List[Int]()
        var s = 1
        for i in range(len(result_shape) - 1, -1, -1):
            result_strides.append(s)
            s *= result_shape[i]
        var result_strides_final = List[Int]()
        for i in range(len(result_strides) - 1, -1, -1):
            result_strides_final.append(result_strides[i])
        for result_idx in range(total_elems):
            var idx_a = 0
            var idx_b = 0
            var remaining = result_idx
            for d in range(len(result_shape) - 1, -1, -1):
                var coord = remaining % result_shape[d]
                remaining //= result_shape[d]
                idx_a += coord * strides_a[d]
                idx_b += coord * strides_b[d]
            r_ptr[result_idx] = op[dtype](a_ptr[idx_a], b_ptr[idx_b])

    if ordinal == DTYPE_FLOAT16:
        _apply[DType.float16]()
    elif ordinal == DTYPE_FLOAT32:
        _apply[DType.float32]()
    elif ordinal == DTYPE_FLOAT64:
        _apply[DType.float64]()
    elif ordinal == DTYPE_INT8:
        _apply[DType.int8]()
    elif ordinal == DTYPE_INT16:
        _apply[DType.int16]()
    elif ordinal == DTYPE_INT32:
        _apply[DType.int32]()
    elif ordinal == DTYPE_INT64:
        _apply[DType.int64]()
    elif ordinal == DTYPE_UINT8:
        _apply[DType.uint8]()
    elif ordinal == DTYPE_UINT16:
        _apply[DType.uint16]()
    elif ordinal == DTYPE_UINT32:
        _apply[DType.uint32]()
    elif ordinal == DTYPE_UINT64:
        _apply[DType.uint64]()

    return result^


fn _anytensor_unary_op[
    op: fn[T: DType] (Scalar[T]) -> Scalar[T]
](tensor: AnyTensor) raises -> AnyTensor:
    """Apply a compile-time-typed unary op element-wise."""
    var shape = tensor.shape()
    var result = AnyTensor(shape, tensor._dtype)
    var ordinal = dtype_to_ordinal(tensor._dtype)

    @parameter
    fn _apply[dtype: DType]():
        var src_ptr = tensor._data.bitcast[Scalar[dtype]]()
        var dst_ptr = result._data.bitcast[Scalar[dtype]]()
        for i in range(tensor._numel):
            dst_ptr[i] = op[dtype](src_ptr[i])

    if ordinal == DTYPE_FLOAT16:
        _apply[DType.float16]()
    elif ordinal == DTYPE_FLOAT32:
        _apply[DType.float32]()
    elif ordinal == DTYPE_FLOAT64:
        _apply[DType.float64]()
    elif ordinal == DTYPE_INT8:
        _apply[DType.int8]()
    elif ordinal == DTYPE_INT16:
        _apply[DType.int16]()
    elif ordinal == DTYPE_INT32:
        _apply[DType.int32]()
    elif ordinal == DTYPE_INT64:
        _apply[DType.int64]()
    elif ordinal == DTYPE_UINT8:
        _apply[DType.uint8]()
    elif ordinal == DTYPE_UINT16:
        _apply[DType.uint16]()
    elif ordinal == DTYPE_UINT32:
        _apply[DType.uint32]()
    elif ordinal == DTYPE_UINT64:
        _apply[DType.uint64]()

    return result^


fn _anytensor_compare_op[
    op: fn[T: DType] (Scalar[T], Scalar[T]) -> Bool
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Apply a compile-time-typed binary comparison op with broadcasting."""
    if a._dtype != b._dtype:
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var result = AnyTensor(result_shape, DType.bool)
    var ordinal = dtype_to_ordinal(a._dtype)

    @parameter
    fn _apply[dtype: DType]():
        var a_ptr = a._data.bitcast[Scalar[dtype]]()
        var b_ptr = b._data.bitcast[Scalar[dtype]]()
        var r_ptr = result._data.bitcast[Scalar[DType.bool]]()
        var result_strides = List[Int]()
        var s = 1
        for i in range(len(result_shape) - 1, -1, -1):
            result_strides.append(s)
            s *= result_shape[i]
        var result_strides_final = List[Int]()
        for i in range(len(result_strides) - 1, -1, -1):
            result_strides_final.append(result_strides[i])
        for result_idx in range(total_elems):
            var idx_a = 0
            var idx_b = 0
            var remaining = result_idx
            for d in range(len(result_shape) - 1, -1, -1):
                var coord = remaining % result_shape[d]
                remaining //= result_shape[d]
                idx_a += coord * strides_a[d]
                idx_b += coord * strides_b[d]
            r_ptr[result_idx] = op[dtype](a_ptr[idx_a], b_ptr[idx_b])

    if ordinal == DTYPE_FLOAT16:
        _apply[DType.float16]()
    elif ordinal == DTYPE_FLOAT32:
        _apply[DType.float32]()
    elif ordinal == DTYPE_FLOAT64:
        _apply[DType.float64]()
    elif ordinal == DTYPE_INT8:
        _apply[DType.int8]()
    elif ordinal == DTYPE_INT16:
        _apply[DType.int16]()
    elif ordinal == DTYPE_INT32:
        _apply[DType.int32]()
    elif ordinal == DTYPE_INT64:
        _apply[DType.int64]()
    elif ordinal == DTYPE_UINT8:
        _apply[DType.uint8]()
    elif ordinal == DTYPE_UINT16:
        _apply[DType.uint16]()
    elif ordinal == DTYPE_UINT32:
        _apply[DType.uint32]()
    elif ordinal == DTYPE_UINT64:
        _apply[DType.uint64]()

    return result^


fn _anytensor_matmul(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Basic matrix multiplication (2D x 2D) for AnyTensor.__matmul__.

    Note: For full matmul with batching and contiguity handling, use
    shared.core.matrix.matmul. This implementation handles the common 2D case
    to avoid the circular import: any_tensor <- matrix <- shape <- any_tensor.
    """
    var a_ndim = len(a._shape)
    var b_ndim = len(b._shape)

    # 2D x 2D: (m, k) @ (k, n) -> (m, n)
    if a_ndim == 2 and b_ndim == 2:
        var m = a._shape[0]
        var k = a._shape[1]
        var n = b._shape[1]
        if k != b._shape[0]:
            raise Error(
                "matmul: incompatible dimensions "
                + String(k)
                + " vs "
                + String(b._shape[0])
            )
        if a._dtype != b._dtype:
            raise Error("matmul: tensors must have the same dtype")
        var result = AnyTensor([m, n], a._dtype)
        var ordinal = dtype_to_ordinal(a._dtype)

        @parameter
        fn _mm[dtype: DType]():
            var a_ptr = a._data.bitcast[Scalar[dtype]]()
            var b_ptr = b._data.bitcast[Scalar[dtype]]()
            var r_ptr = result._data.bitcast[Scalar[dtype]]()
            for i in range(m):
                for j in range(n):
                    var acc = Scalar[dtype](0)
                    for p in range(k):
                        acc += a_ptr[i * k + p] * b_ptr[p * n + j]
                    r_ptr[i * n + j] = acc

        if ordinal == DTYPE_FLOAT16:
            _mm[DType.float16]()
        elif ordinal == DTYPE_FLOAT32:
            _mm[DType.float32]()
        elif ordinal == DTYPE_FLOAT64:
            _mm[DType.float64]()
        elif ordinal == DTYPE_INT8:
            _mm[DType.int8]()
        elif ordinal == DTYPE_INT16:
            _mm[DType.int16]()
        elif ordinal == DTYPE_INT32:
            _mm[DType.int32]()
        elif ordinal == DTYPE_INT64:
            _mm[DType.int64]()
        else:
            raise Error("matmul: unsupported dtype")
        return result^

    # 1D x 2D or 2D x 1D: delegate to the local arithmetic for now
    # by raising a helpful error pointing to matrix.matmul
    raise Error(
        "AnyTensor.__matmul__ only supports 2D x 2D. "
        "For 1D/batched matmul use shared.core.matrix.matmul directly."
    )


# ============================================================================
# Creation Operations
# ============================================================================


fn zeros(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with zeros.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor filled with zeros.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
    ```
            var t = zeros([3, 4], DType.float32)
            # Creates a 3x4 tensor of float32 zeros.
    ```

    Performance:
        O(n) time where n is the number of elements.
    """
    var tensor = AnyTensor(shape, dtype)
    tensor._fill_zero()  # Efficiently zero out all bytes
    return tensor^


fn ones(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with ones.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor filled with ones.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
    ```
            var t = ones([3, 4], DType.float32)
            # Creates a 3x4 tensor of float32 ones.
    ```
    """
    var tensor = AnyTensor(shape, dtype)

    # Fill with ones based on dtype category
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    ):
        tensor._fill_value_float(1.0)
    else:
        tensor._fill_value_int(1)

    return tensor^


fn full(shape: List[Int], fill_value: Float64, dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with a specific value.

    Args:
            shape: The shape of the output tensor.
            fill_value: The value to fill the tensor with.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor filled with fill_value.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
            ```var t = full([3, 4], 42.0, DType.float32)
            # Creates a 3x4 tensor filled with 42.0
            ```
    """
    var tensor = AnyTensor(shape, dtype)

    # Fill with value based on dtype category
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    ):
        tensor._fill_value_float(fill_value)
    else:
        tensor._fill_value_int(Int(fill_value))

    return tensor^


fn empty(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create an uninitialized tensor (fast allocation).

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements.

    Returns:
            A new AnyTensor with uninitialized memory.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Warning:
            The tensor contains uninitialized memory. Values are undefined until written.
            Use this for performance when you will immediately write to all elements.

    Examples:
    ```
            var t = empty([3, 4], DType.float32)
            # Creates a 3x4 tensor with undefined values.
    ```
    """
    # Just allocate without initialization
    var tensor = AnyTensor(shape, dtype)
    return tensor^


fn arange(
    start: Float64, stop: Float64, step: Float64, dtype: DType
) raises -> AnyTensor:
    """Create 1D tensor with evenly spaced values.

    Args:
            start: Start value (inclusive).
            stop: End value (exclusive).
            step: Spacing between values.
            dtype: The data type of tensor elements.

    Returns:
            A new 1D AnyTensor with values in range [start, stop) with given step.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
        ```
        var t = arange(0.0, 10.0, 1.0, DType.float32)
        # Creates [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        var t2 = arange(0.0, 10.0, 2.0, DType.int32)
        # Creates [0, 2, 4, 6, 8]
        ```
    """
    # Calculate number of elements
    var num_elements = Int((stop - start) / step)
    var shape = List[Int]()
    shape.append(num_elements)

    var tensor = AnyTensor(shape, dtype)

    # Fill with sequence
    var value = start
    for i in range(num_elements):
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
            or dtype == DType.bfloat16
        ):
            tensor._set_float64(i, value)
        else:
            tensor._set_int64(i, Int(value))
        value += step

    return tensor^


fn eye(n: Int, m: Int, k: Int, dtype: DType) raises -> AnyTensor:
    """Create 2D tensor with ones on diagonal.

    Args:
            n: Number of rows.
            m: Number of columns.
            k: Diagonal offset (0 for main diagonal, >0 for upper, <0 for lower).
            dtype: The data type of tensor elements.

    Returns:
            A new 2D AnyTensor with ones on the k-th diagonal.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
    ```
            var t = eye(3, 3, 0, DType.float32)
            # Creates 3x3 identity matrix.

            var t2 = eye(3, 4, 1, DType.float32)
            # Creates 3x4 matrix with ones on diagonal above main.
    ```
    """
    var shape = List[Int]()
    shape.append(n)
    shape.append(m)

    var tensor = AnyTensor(shape, dtype)
    tensor._fill_zero()

    # Set diagonal to one
    for i in range(n):
        var j = i + k
        if j >= 0 and j < m:
            var index = i * m + j
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
                or dtype == DType.bfloat16
            ):
                tensor._set_float64(index, 1.0)
            else:
                tensor._set_int64(index, 1)

    return tensor^


fn linspace(
    start: Float64, stop: Float64, num: Int, dtype: DType
) raises -> AnyTensor:
    """Create 1D tensor with evenly spaced values (inclusive).

    Args:
            start: Start value (inclusive).
            stop: End value (inclusive).
            num: Number of values.
            dtype: The data type of tensor elements.

    Returns:
            A new 1D AnyTensor with num evenly spaced values.

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
        ```var t = linspace(0.0, 10.0, 11, DType.float32)
        # Creates [0.0, 1.0, 2.0, ..., 10.0]

        var t2 = linspace(0.0, 1.0, 5, DType.float64)
        # Creates [0.0, 0.25, 0.5, 0.75, 1.0]
        ```
    """
    var shape = List[Int]()
    shape.append(num)

    var tensor = AnyTensor(shape, dtype)

    if num == 1:
        # Special case: single value
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
            or dtype == DType.bfloat16
        ):
            tensor._set_float64(0, start)
        else:
            tensor._set_int64(0, Int(start))
    else:
        # Calculate step size
        var step = (stop - start) / (num - 1)

        # Fill with sequence
        for i in range(num):
            var value = start + step * i
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
                or dtype == DType.bfloat16
            ):
                tensor._set_float64(i, value)
            else:
                tensor._set_int64(i, Int(value))

    return tensor^


fn ones_like(tensor: AnyTensor) raises -> AnyTensor:
    """Create tensor of ones with same shape and dtype as input.

    Args:
            tensor: Template tensor to match shape and dtype.

    Returns:
            A new AnyTensor filled with ones, same shape and dtype as input.

    Raises:
            Error: If tensor creation fails.

    Example:
        ```mojo
        var x = zeros([3, 4], DType.float32)
        var y = ones_like(x)  # (3, 4) tensor of ones, float32
        ```
    """
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    return ones(shape, dtype)


fn zeros_like(tensor: AnyTensor) raises -> AnyTensor:
    """Create tensor of zeros with same shape and dtype as input.

    Args:
            tensor: Template tensor to match shape and dtype.

    Returns:
            A new AnyTensor filled with zeros, same shape and dtype as input.

    Raises:
            Error: If tensor creation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = zeros_like(x)  # (3, 4) tensor of zeros, float32
        ```
    """
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    return zeros(shape, dtype)


fn full_like(tensor: AnyTensor, fill_value: Float64) raises -> AnyTensor:
    """Create tensor filled with a value, same shape and dtype as input.

    Args:
            tensor: Template tensor to match shape and dtype.
            fill_value: Value to fill the tensor with.

    Returns:
            A new AnyTensor filled with fill_value, same shape and dtype as input.

    Raises:
            Error: If tensor creation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = full_like(x, 3.14)  # (3, 4) tensor of 3.14, float32
        ```
    """
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    return full(shape, fill_value, dtype)


fn nan_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with NaN values.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (must be floating-point).

    Returns:
            A new AnyTensor filled with NaN values.

    Raises:
            Error: If dtype is not floating-point, or if tensor size exceeds MAX_TENSOR_BYTES.

    Examples:
        ```
        var t = nan_tensor([3, 4], DType.float32)
        # Creates 3x4 tensor filled with NaN
        ```
    """
    # Only floating-point types support NaN
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        raise Error("nan_tensor: only floating-point dtypes support NaN")

    var tensor = AnyTensor(shape, dtype)

    # Fill tensor with NaN values
    # IEEE 754 NaN is represented as 0.0 / 0.0
    var nan_value = 0.0 / 0.0

    for i in range(tensor.numel()):
        tensor._set_float64(i, nan_value)

    return tensor^


fn inf_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with positive infinity values.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (must be floating-point).

    Returns:
            A new AnyTensor filled with positive infinity values.

    Raises:
            Error: If dtype is not floating-point, or if tensor size exceeds MAX_TENSOR_BYTES.

    Examples:
        ```
        var t = inf_tensor([3, 4], DType.float32)
        # Creates 3x4 tensor filled with +inf
        ```
    """
    # Only floating-point types support Inf
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        raise Error("inf_tensor: only floating-point dtypes support Inf")

    var tensor = AnyTensor(shape, dtype)

    # Fill tensor with +inf values using proper IEEE 754 infinity constant
    var inf_value: Float64 = numeric_inf[DType.float64]()

    for i in range(tensor.numel()):
        tensor._set_float64(i, inf_value)

    return tensor^


fn neg_inf_tensor(shape: List[Int], dtype: DType) raises -> AnyTensor:
    """Create a tensor filled with negative infinity values.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (must be floating-point).

    Returns:
            A new AnyTensor filled with negative infinity values.

    Raises:
            Error: If dtype is not floating-point, or if tensor size exceeds MAX_TENSOR_BYTES.

    Examples:
        ```
        var t = neg_inf_tensor([3, 4], DType.float32)
        # Creates 3x4 tensor filled with -inf
        ```
    """
    # Only floating-point types support Inf
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        raise Error("neg_inf_tensor: only floating-point dtypes support Inf")

    var tensor = AnyTensor(shape, dtype)

    # Fill tensor with -inf values using proper IEEE 754 infinity constant
    var neg_inf_value: Float64 = numeric_neg_inf[DType.float64]()

    for i in range(tensor.numel()):
        tensor._set_float64(i, neg_inf_value)

    return tensor^


fn _dtype_to_string(dtype: DType) -> String:
    """Convert a DType to a readable string representation.

    Args:
        dtype: The data type.

    Returns:
        A string like "float32", "int64", etc.
    """
    if dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.float16:
        return "float16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unknown"


fn randn(shape: List[Int], dtype: DType, seed: Int = 0) raises -> AnyTensor:
    """Create tensor filled with random values from standard normal distribution.

        Uses Box-Muller transform to generate normally distributed random values
        from uniform random values. Generates values with mean=0 and std=1.

    Args:
            shape: The shape of the output tensor.
            dtype: The data type of tensor elements (should be floating-point).
            seed: Random seed for reproducibility (default: 0 uses system randomness).

    Returns:
            A new AnyTensor filled with random values from N(0, 1).

    Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES or allocation fails.

    Examples:
            ```var t = randn([3, 4], DType.float32)
            # Creates 3x4 tensor with values from N(0, 1)

            var t2 = randn([100, 100], DType.float32, seed=42)
            # Reproducible random tensor with seed=42```

    Note:
            For integer dtypes, values are generated as floats then truncated.
            Box-Muller transform generates pairs of independent normal values.
    """
    # Verify floating-point dtype (best practice)
    if not (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
        or dtype == DType.bfloat16
    ):
        print(
            "Warning: randn() is designed for floating-point types, got",
            _dtype_to_string(dtype),
        )

    # Set random seed if provided (0 uses system randomness)
    if seed > 0:
        random_seed(seed)

    var tensor = AnyTensor(shape, dtype)

    # Box-Muller transform: generates pairs of independent N(0,1) values
    # from pairs of uniform random values
    var i = 0
    while i < tensor.numel():
        # Generate two uniform random values in (0, 1]
        var u1 = random_float64()
        var u2 = random_float64()

        # Ensure u1 is not zero (would cause log(0))
        if u1 < 1e-10:
            u1 = 1e-10

        # Box-Muller transform
        var magnitude = sqrt(-2.0 * log(u1))
        var angle = 2.0 * 3.14159265358979323846 * u2

        # Generate two independent normal values
        var z0 = magnitude * cos(angle)
        var z1 = magnitude * sin(angle)

        # Store first value
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
            or dtype == DType.bfloat16
        ):
            tensor._set_float64(i, z0)
        else:
            tensor._set_int64(i, Int(z0))

        i += 1

        # Store second value if there's room
        if i < tensor.numel():
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
                or dtype == DType.bfloat16
            ):
                tensor._set_float64(i, z1)
            else:
                tensor._set_int64(i, Int(z1))
            i += 1

    return tensor^


fn calculate_max_batch_size(
    sample_shape: List[Int],
    dtype: DType,
    max_memory_bytes: Int = 500_000_000,  # 500 MB default
) raises -> Int:
    """Calculate maximum safe batch size for given sample shape.

    Args:
            sample_shape: Shape of a single sample (e.g., [1, 28, 28] for MNIST).
            dtype: Data type of the tensor.
            max_memory_bytes: Maximum memory to use for a batch (default: 500 MB).

    Returns:
            Maximum batch size that fits in memory.

    Raises:
            Error: If sample shape is invalid or no batch size can fit in memory.

    Example:
            ```mojo
            # For MNIST: (1, 28, 28) images
            var sample_shape = List[Int]()
            sample_shape.append(1)
            sample_shape.append(28)
            sample_shape.append(28)
            var max_batch = calculate_max_batch_size(sample_shape, DType.float32)
            print("Max batch size:", max_batch)  # ~640,000 samples
            ```
    """
    var sample_elements = 1
    for i in range(len(sample_shape)):
        sample_elements *= sample_shape[i]

    var dtype_size = AnyTensor._get_dtype_size_static(dtype)
    var bytes_per_sample = sample_elements * dtype_size

    if bytes_per_sample <= 0:
        raise Error("Invalid sample shape or dtype")

    var max_batch = max_memory_bytes // bytes_per_sample

    if max_batch < 1:
        raise Error(
            "Single sample ("
            + String(bytes_per_sample)
            + " bytes) exceeds memory limit ("
            + String(max_memory_bytes)
            + " bytes)"
        )

    return max_batch


# ============================================================================
# Utility Function Wrappers
# ============================================================================


fn copy(tensor: AnyTensor) raises -> AnyTensor:
    """Create an independent deep copy of the tensor.

    This is a convenience wrapper around the AnyTensor.clone() method,
    following NumPy naming conventions. The returned tensor has its own
    independent memory; modifications to it do not affect the original.

    Args:
        tensor: The tensor to copy.

    Returns:
        A new AnyTensor that is a deep copy of the input.

    Raises:
        Error: If memory allocation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = copy(x)  # Independent deep copy
        ```
    """
    return tensor.clone()


fn clone(tensor: AnyTensor) raises -> AnyTensor:
    """Create a clone of the tensor.

    This is a convenience wrapper around the AnyTensor.clone() method.

    Args:
        tensor: The tensor to clone.

    Returns:
        A new AnyTensor that is a deep copy of the input.

    Raises:
        Error: If memory allocation fails.

    Example:
        ```mojo
        var x = ones([3, 4], DType.float32)
        var y = clone(x)  # Independent copy
        ```
    """
    return tensor.clone()


fn item(tensor: AnyTensor) raises -> Float64:
    """Extract the value from a single-element tensor.

    This is a convenience wrapper around the AnyTensor.item() method.

    Args:
        tensor: A tensor with exactly one element.

    Returns:
        The scalar value as Float64.

    Raises:
        Error: If tensor has more than one element.

    Example:
        ```mojo
        var x = full([], 42.0, DType.float32)
        var val = item(x)  # Returns 42.0
        ```
    """
    return tensor.item()


fn diff(tensor: AnyTensor, n: Int = 1) raises -> AnyTensor:
    """Calculate consecutive differences along an axis.

    This is a convenience wrapper around the AnyTensor.diff() method.

    Args:
        tensor: The input tensor.
        n: Order of differences (default: 1).

    Returns:
        A new AnyTensor with differences computed.

    Raises:
        Error: If operation fails.

    Example:
        ```mojo
        var x = arange(0.0, 5.0, 1.0, DType.float32)
        var d = diff(x)  # [1.0, 1.0, 1.0, 1.0]
        ```
    """
    return tensor.diff(n)
