"""Tensor[dtype] — compile-time typed parametric tensor.

Provides a tensor type parametric on DType, enabling typed element access via
`__getitem__` that returns `Scalar[Self.dtype]` (no runtime dtype branching).

Uses typed `UnsafePointer[Scalar[dtype], MutAnyOrigin]` storage instead of type-erased
`UnsafePointer[UInt8, MutAnyOrigin]`. Pointer arithmetic auto-scales by element size (H1),
so no manual `* dtype_size` is needed for element offsets.

Zero-copy conversion to AnyTensor via `as_any()` with shared
refcount (B4). Both types share a `TensorLike` trait interface.

Example:
    ```mojo
    from shared.tensor import Tensor

    var t = Tensor[DType.float32]([3, 4])  # 3x4 float32 tensor
    var val = t[0]  # Returns Scalar[DType.float32]
    ```
"""

from std.collections import List
from std.memory import UnsafePointer, memset_zero, alloc
from std.io import Writer
from shared.base.memory_pool import pooled_alloc, pooled_free
from shared.tensor.tensor_traits import TensorLike
from shared.tensor.any_tensor import AnyTensor


# Memory safety constants (match AnyTensor limits)
comptime MAX_TENSOR_BYTES: Int = 2_000_000_000  # 2 GB max per tensor
comptime WARN_TENSOR_BYTES: Int = 500_000_000  # 500 MB warning threshold

# Print options for truncation (match AnyTensor behavior)
comptime TENSOR_PRINT_THRESHOLD: Int = 1000  # Truncate if numel > threshold
comptime TENSOR_PRINT_SHOW_ELEMENTS: Int = 3  # Show first/last N elements


struct Tensor[dtype: DType = DType.float32](
    Copyable,
    ImplicitlyCopyable,
    Movable,
    Sized,
    TensorLike,
    Writable,
):
    """Compile-time typed tensor with SIMD-like element access.

    Parametric on DType, enabling `__getitem__` to return `Scalar[Self.dtype]`
    without runtime dtype branching. Uses typed `UnsafePointer[Scalar[dtype], MutAnyOrigin]`
    storage where pointer arithmetic auto-scales by element size.

    Memory Safety: Implements reference counting for safe shared ownership.
    Copying increments the refcount; memory is freed only when the last
    reference is destroyed. Compatible with AnyTensor's refcount protocol
    for zero-copy conversion via `as_any()`.

    Parameters:
        dtype: The compile-time data type of tensor elements (default: float32).

    Attributes:
        _data: Typed pointer to element storage (auto-scales pointer arithmetic).
        _shape: List storing the shape dimensions.
        _strides: List storing the stride for each dimension (in elements).
        _numel: Total number of elements in the tensor.
        _is_view: Whether this tensor is a view (shares data with another tensor).
        _refcount: Shared reference count for memory management.
        _allocated_size: Actual allocated size in bytes (may differ due to pool bucketing).
        _original_numel_quantized: For quantized tensors, original size before padding.
    """

    var _data: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    """Typed pointer to element storage."""
    var _shape: List[Int]
    """List of dimension sizes."""
    var _strides: List[Int]
    """Row-major strides for each dimension (in elements)."""
    var _numel: Int
    """Total number of elements."""
    var _is_view: Bool
    """Whether this tensor shares data with another."""
    var _refcount: UnsafePointer[Int, origin=MutAnyOrigin]
    """Reference count for shared memory management."""
    var _allocated_size: Int
    """Actual allocated size in bytes."""
    var _original_numel_quantized: Int
    """Original element count before quantization padding."""

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(out self, shape: List[Int]) raises:
        """Initialize a new Tensor with given shape.

        Allocates memory via the pooled allocator and initializes all
        elements to zero.

        Args:
            shape: The shape of the tensor as a list of dimension sizes.

        Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES (2 GB).
        """
        # Copy shape to avoid mutation issues
        self._shape = List[Int]()
        for i in range(len(shape)):
            self._shape.append(shape[i])

        self._is_view = False
        self._original_numel_quantized = -1  # Not quantized

        # Calculate total number of elements
        self._numel = 1
        for i in range(len(self._shape)):
            self._numel *= self._shape[i]

        # Calculate row-major strides (in elements, not bytes)
        self._strides = List[Int]()
        var stride = 1
        for _ in range(len(self._shape)):
            self._strides.append(0)
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides[i] = stride
            stride *= self._shape[i]

        # Calculate total bytes needed
        # Use compile-time element size via dtype_sizeof
        var element_size = Self._element_size()
        var total_bytes = self._numel * element_size

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

        # Allocate through memory pool (returns UInt8 pointer), then bitcast
        # to typed pointer. _allocated_size is in BYTES.
        self._data = pooled_alloc(total_bytes).bitcast[Scalar[Self.dtype]]()
        self._allocated_size = total_bytes

        # Zero-initialize the memory
        memset_zero(self._data.bitcast[UInt8](), total_bytes)

        # Allocate and initialize reference count
        self._refcount = alloc[Int](1)
        self._refcount[] = 1

    def __init__(
        out self,
        data: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        shape: List[Int],
        strides: List[Int],
        refcount: UnsafePointer[Int, origin=MutAnyOrigin],
        numel: Int,
        is_view: Bool,
        allocated_size: Int,
        original_numel_quantized: Int,
    ):
        """Internal constructor for zero-copy conversion (NOT public API).

        Creates a Tensor sharing ownership of existing data via a shared
        refcount pointer. Increments refcount to establish shared ownership
        (B4 critical — prevents dangling pointer from ASAP destruction).

        Args:
            data: Typed pointer to existing element storage.
            shape: Shape dimensions.
            strides: Stride for each dimension (in elements).
            refcount: Shared refcount pointer (same pointer, not a copy).
            numel: Total number of elements.
            is_view: Whether this tensor is a view.
            allocated_size: Allocated size in bytes.
            original_numel_quantized: Original element count before quantization.
        """
        self._data = data
        self._shape = List[Int]()
        for i in range(len(shape)):
            self._shape.append(shape[i])
        self._strides = List[Int]()
        for i in range(len(strides)):
            self._strides.append(strides[i])
        self._refcount = refcount
        self._refcount[] += 1  # CRITICAL: shared ownership (B4)
        self._numel = numel
        self._is_view = is_view
        self._allocated_size = allocated_size
        self._original_numel_quantized = original_numel_quantized

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(out self, *, copy: Self):
        """Copy constructor - creates a shared-ownership copy with reference counting.
        """
        self._data = copy._data
        self._shape = copy._shape.copy()
        self._strides = copy._strides.copy()
        self._numel = copy._numel
        self._is_view = copy._is_view
        self._refcount = copy._refcount
        self._original_numel_quantized = copy._original_numel_quantized
        self._allocated_size = copy._allocated_size
        if self._refcount:
            self._refcount[] += 1

    def __init__(out self, *, deinit take: Self):
        """Move constructor - transfers ownership without refcount change."""
        self._data = take._data
        self._shape = take._shape^
        self._strides = take._strides^
        self._numel = take._numel
        self._is_view = take._is_view
        self._refcount = take._refcount
        self._original_numel_quantized = take._original_numel_quantized
        self._allocated_size = take._allocated_size

    def copy(self) -> Self:
        """Copy constructor — shared ownership with reference counting.

        Creates a new reference to the same underlying data. Increments
        the reference count to track shared ownership.
        """
        from std.memory import alloc as mem_alloc

        var ptr = mem_alloc[Tensor[Self.dtype]](1)
        ptr[0]._data = self._data
        ptr[0]._shape = self._shape.copy()
        ptr[0]._strides = self._strides.copy()
        ptr[0]._numel = self._numel
        ptr[0]._is_view = self._is_view
        ptr[0]._refcount = self._refcount
        ptr[0]._original_numel_quantized = self._original_numel_quantized
        ptr[0]._allocated_size = self._allocated_size
        if self._refcount:
            self._refcount[] += 1
        return ptr.take_pointee()

    def __del__(deinit self):
        """Destructor — decrements refcount, frees if last reference.

        Uses reference counting to safely manage shared ownership. Only
        frees memory when the last reference is destroyed. Frees via
        pooled_free with the byte-sized _allocated_size.
        """
        if self._refcount:
            self._refcount[] -= 1

            if self._refcount[] == 0:
                # Free data via pool — bitcast back to UInt8 since pool
                # works with byte pointers
                pooled_free(self._data.bitcast[UInt8](), self._allocated_size)
                self._refcount.free()

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
        """Get element at flat index — returns typed Scalar[Self.dtype].

        For contiguous tensors, the flat index maps directly to a memory
        offset. For non-contiguous tensors (e.g., after transpose), the
        flat index is converted to multi-dimensional coordinates using the
        shape, then mapped to a memory offset using strides.

        No runtime dtype branching — pointer arithmetic auto-scales (H1).

        Args:
            index: The flat index to access (logical element index in
                row-major order of the tensor's shape).

        Returns:
            The value at the given index as Scalar[Self.dtype].

        Raises:
            Error: If index is out of bounds.
        """
        if index < 0 or index >= self._numel:
            raise Error("Index out of bounds")

        # For non-contiguous tensors, convert flat index to nd-coordinates
        # then use strides to compute the real memory offset.
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
            # Typed pointer — auto-scales by element size (H1)
            return self._data[mem_offset]

        # Contiguous — direct indexed access (auto-scales, no * dtype_size)
        return self._data[index]

    def __setitem__(mut self, index: Int, value: Scalar[Self.dtype]) raises:
        """Set element at flat index — accepts typed Scalar[Self.dtype].

        Args:
            index: The flat index to set.
            value: The value to store (must be same dtype).

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
            self._data[mem_offset] = value
            return

        self._data[index] = value

    # ------------------------------------------------------------------
    # TensorLike conformance
    # ------------------------------------------------------------------

    def numel(self) -> Int:
        """Return total number of elements."""
        return self._numel

    def shape(self) -> List[Int]:
        """Return shape as list of dimension sizes (returns a copy)."""
        return self._shape.copy()

    def get_dtype(self) -> DType:
        """Return the element data type (compile-time constant)."""
        return Self.dtype

    def ndim(self) -> Int:
        """Return the number of dimensions (rank)."""
        return len(self._shape)

    # ------------------------------------------------------------------
    # Sized conformance
    # ------------------------------------------------------------------

    def __len__(self) -> Int:
        """Return the size of the first dimension.

        Follows NumPy/PyTorch convention where len() returns the size
        of the first dimension (axis 0).

        Returns:
            The size of the first dimension, or 0 if 0-dimensional.
        """
        if len(self._shape) == 0:
            return 0
        return self._shape[0]

    # ------------------------------------------------------------------
    # Scalar extraction
    # ------------------------------------------------------------------

    def item(self) raises -> Scalar[Self.dtype]:
        """Extract the value from a single-element tensor.

        Returns:
            The scalar value as Scalar[Self.dtype].

        Raises:
            Error: If tensor has more than one element.

        Example:
            ```mojo
            var t = Tensor[DType.float32]([1])
            t[0] = 42.0
            var val = t.item()  # Returns Scalar[DType.float32](42.0)
            ```
        """
        if self._numel != 1:
            raise Error(
                "item() requires single-element tensor, got "
                + String(self._numel)
                + " elements"
            )
        return self._data[0]

    def __bool__(self) raises -> Bool:
        """Return the boolean value of a single-element tensor.

        Follows PyTorch/NumPy convention: a single-element tensor can be
        used in boolean context. Returns True if the value is non-zero.

        Returns:
            True if the single element is non-zero, False otherwise.

        Raises:
            Error: If tensor has more than one element.

        Example:
            ```mojo
            var t = Tensor[DType.float32]([1])
            t[0] = 5.0
            if t:  # True
                print("non-zero")
            ```
        """
        return self.item().__bool__()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_contiguous(self) -> Bool:
        """Check if the tensor has a contiguous memory layout.

        Returns:
            True if strides match row-major layout, False otherwise.
        """
        var expected_stride = 1
        for i in range(len(self._shape) - 1, -1, -1):
            if self._strides[i] != expected_stride:
                return False
            expected_stride *= self._shape[i]
        return True

    def is_view(self) -> Bool:
        """Return whether this tensor is a view (shares data).

        Returns:
            True if this tensor shares data with another tensor.
        """
        return self._is_view

    def dim(self) -> Int:
        """Return the number of dimensions (alias for ndim).

        Returns:
            The rank of the tensor.
        """
        return len(self._shape)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def as_any(self) raises -> AnyTensor:
        """Zero-copy conversion to runtime-typed AnyTensor (future AnyTensor).

        Creates an AnyTensor sharing the same underlying data via a shared
        refcount pointer. The AnyTensor's `_data` is the byte-level view of
        this tensor's typed pointer. Both objects share the same refcount,
        so ASAP destruction of either only decrements (not frees) the count
        until the last reference goes away (B4).

        Safety: This method performs manual field replacement on an AnyTensor
        because AnyTensor lacks an internal constructor accepting raw pointers.
        The temporary AnyTensor's independent allocation is fully torn down
        (data freed, refcount freed) before fields are replaced with shared
        pointers from this Tensor.

        TODO(#5005): Replace with AnyTensor internal constructor once available,
        eliminating the allocate-then-free overhead and fragile field surgery.

        Returns:
            An AnyTensor sharing the same data and refcount.

        Raises:
            Error: If AnyTensor construction fails.
        """
        # Create an AnyTensor with same shape/dtype — this allocates its own
        # independent data buffer and refcount pointer.
        var shape_copy = List[Int]()
        for i in range(len(self._shape)):
            shape_copy.append(self._shape[i])
        var result = AnyTensor(shape_copy, Self.dtype)

        # --- Tear down the temporary AnyTensor's independent allocation ---
        # Free the independently allocated data buffer (safe: result is the
        # sole owner at this point, refcount == 1).
        var tmp_data = result._data
        var tmp_alloc_size = result._allocated_size
        var tmp_refcount = result._refcount
        pooled_free(tmp_data, tmp_alloc_size)
        # Free the independent refcount pointer (was allocated with alloc[Int](1))
        tmp_refcount.free()

        # --- Replace fields with shared pointers from this Tensor ---
        # Share our data pointer (bitcast typed -> UInt8 for AnyTensor)
        result._data = self._data.bitcast[UInt8]()
        result._allocated_size = self._allocated_size
        result._shape = self._shape.copy()
        result._strides = self._strides.copy()
        result._numel = self._numel
        result._is_view = self._is_view
        result._original_numel_quantized = self._original_numel_quantized

        # Share our refcount and increment for shared ownership (B4 critical).
        # After this, both `self` and `result` point to the same refcount.
        result._refcount = self._refcount
        result._refcount[] += 1

        return result^

    # ------------------------------------------------------------------
    # String representation (H4 fix: typed access, no _get_float64)
    # ------------------------------------------------------------------

    def write_to[W: Writer](self, mut writer: W):
        """Write the string representation to a Writer (required for Writable trait).

        Primary implementation of string formatting. __str__() delegates to
        this method via String.write(self).

        Parameters:
            W: The writer type conforming to the Writer trait.

        Args:
            writer: The writer to write the string representation to.
        """
        writer.write("Tensor([")
        if self._numel > TENSOR_PRINT_THRESHOLD:
            for i in range(TENSOR_PRINT_SHOW_ELEMENTS):
                if i > 0:
                    writer.write(", ")
                try:
                    writer.write(String(self[i]))
                except:
                    writer.write("?")
            writer.write(", ...")
            for i in range(
                self._numel - TENSOR_PRINT_SHOW_ELEMENTS, self._numel
            ):
                writer.write(", ")
                try:
                    writer.write(String(self[i]))
                except:
                    writer.write("?")
        else:
            for i in range(self._numel):
                if i > 0:
                    writer.write(", ")
                try:
                    writer.write(String(self[i]))
                except:
                    writer.write("?")
        writer.write("], dtype=" + String(Self.dtype) + ")")

    def __str__(self) -> String:
        """Human-readable string representation with NumPy-style truncation.

        For tensors with more than 1000 elements, shows only the first 3 and
        last 3 elements with '...' in between.

        Uses stride-aware element access via `self[i]` so non-contiguous
        tensors (e.g., after transpose) display correct logical values.

        Returns:
            String in the format: ``Tensor([v0, v1, ...], dtype=<dtype>)``.
        """
        return String.write(self)

    def write_repr_to[W: Writer](self, mut writer: W):
        """Write the repr representation to a Writer (required for Writable trait).

        This method is called when using `repr(tensor)` to get the detailed
        debugging representation. It delegates to `__repr__()` for the actual
        formatting logic.

        Parameters:
            W: The writer type conforming to the Writer trait.

        Args:
            writer: The writer to write the repr representation to.
        """
        var s = self.__repr__()
        writer.write(s)

    def __repr__(self) -> String:
        """Detailed representation for debugging.

        Uses stride-aware element access via `self[i]` so non-contiguous
        tensors (e.g., after transpose) display correct logical values.

        Returns:
            String with shape, dtype, numel, and data.
        """
        var shape_str = String("[")
        for i in range(len(self._shape)):
            if i > 0:
                shape_str += ", "
            shape_str += String(self._shape[i])
        shape_str += "]"
        var result = String("Tensor(shape=") + shape_str
        result += ", dtype=" + String(Self.dtype)
        result += ", numel=" + String(self._numel)
        result += ", data=["
        if self._numel > TENSOR_PRINT_THRESHOLD:
            for i in range(TENSOR_PRINT_SHOW_ELEMENTS):
                if i > 0:
                    result += ", "
                try:
                    result += String(self[i])
                except:
                    result += "?"
            result += ", ..."
            for i in range(
                self._numel - TENSOR_PRINT_SHOW_ELEMENTS, self._numel
            ):
                result += ", "
                try:
                    result += String(self[i])
                except:
                    result += "?"
        else:
            for i in range(self._numel):
                if i > 0:
                    result += ", "
                try:
                    result += String(self[i])
                except:
                    result += "?"
        result += "])"
        return result

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _element_size() -> Int:
        """Return the size in bytes of a single element.

        Uses compile-time dtype knowledge — no runtime branching needed.
        Matches AnyTensor._get_dtype_size_static() output for the same dtype.
        """
        # DType size lookup — same logic as AnyTensor but resolved at
        # compile time since Self.dtype is a parameter.
        comptime if Self.dtype == DType.float16 or Self.dtype == DType.bfloat16:
            return 2
        elif Self.dtype == DType.float32:
            return 4
        elif Self.dtype == DType.float64:
            return 8
        elif (
            Self.dtype == DType.int8
            or Self.dtype == DType.uint8
            or Self.dtype == DType.bool
        ):
            return 1
        elif Self.dtype == DType.int16 or Self.dtype == DType.uint16:
            return 2
        elif Self.dtype == DType.int32 or Self.dtype == DType.uint32:
            return 4
        elif Self.dtype == DType.int64 or Self.dtype == DType.uint64:
            return 8
        else:
            return 4  # Default fallback
