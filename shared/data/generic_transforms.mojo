"""Generic data transformation utilities.

This module provides domain-agnostic transformations that work across
modalities (images, tensors, arrays). Includes composition patterns,
utility transforms, batch processing, and type conversions.

Key Features:
- Identity transform (passthrough)
- Lambda transforms (inline functions)
- Conditional transforms (predicate-based application)
- Clamp transforms (value limiting)
- Debug transforms (inspection/logging)
- Sequential composition (chaining transforms)
- Batch transforms (apply to lists)
- Type conversions (Float32, Int32)

Example:
    >>> # Create preprocessing pipeline
    >>> fn scale(x: Float32) -> Float32:
    ...     return x / 255.0
    >>>
    >>> var pipeline = SequentialTransform()
    >>> pipeline.append(LambdaTransform(scale))
    >>> pipeline.append(ClampTransform(0.0, 1.0))
    >>>
    >>> var result = pipeline(data)
    ```
"""

from shared.tensor.any_tensor import AnyTensor
from shared.data.transforms import Transform


# ============================================================================
# Identity Transform
# ============================================================================


struct IdentityTransform(Copyable, Movable, Transform):
    """Identity transform - returns input unchanged.

    Useful as a placeholder or for conditional pipelines where
    no transformation should be applied under certain conditions.

    Time Complexity: O(1) - just returns reference to input.
    Space Complexity: O(1) - no allocation.

    Example:
        ```mojo
        >> var identity = IdentityTransform()
        >>> var result = identity(data)  # result == data
        ```
    """

    def __init__(out self):
        """Create identity transform."""
        pass

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply identity transform (passthrough).

        Args:
            data: Input tensor.

        Returns:
            Input tensor unchanged.

        Raises:
            Error: If operation fails.
        """
        return data


# ============================================================================
# Lambda Transform
# ============================================================================


struct LambdaTransform(Copyable, Movable, Transform):
    """Apply a function element-wise to tensor values.

    Provides flexible inline transformations without defining
    a full transform struct. The function is applied to each
    element independently.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        ```mojo
        >> fn double(x: Float32) -> Float32:
        ...     return x * 2.0
        >>>
        >>> var transform = LambdaTransform(double)
        >>> var result = transform(data)
        ```
    """

    # TODO(mojo-1.0): Mojo 1.0 rejects dynamic `def(...)` struct fields with
    # "dynamic traits not supported yet, please use a compile time generic
    # instead". The proper fix is to promote `LambdaTransform` to a parametric
    # struct (`LambdaTransform[F: SomeCallableTrait]`) and rewrite all
    # call sites + `AnyTransform` to thread the parameter through. That cascade
    # is large (one shared/ file plus the test file) and is deferred to
    # Phase E along with the test-file migration. For now this is a stub: the
    # constructor accepts a function but does not store it, and `__call__`
    # returns input unchanged. See migration recipe Recipe 7 (DYNAMIC_TRAIT).
    var _stub: Int
    """TODO(mojo-1.0): placeholder until lambda-fn field is re-introduced."""

    def __init__(out self, func: def(Float32) -> Float32):
        """Create lambda transform.

        Args:
            func: Function to apply element-wise. Currently ignored; see
                  TODO(mojo-1.0) above.
        """
        # TODO(mojo-1.0): store `func` once struct is parametric.
        _ = func
        self._stub = 0

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply function to each element.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor with function applied to each element.

        Raises:
            Error: If tensor creation fails.
        """
        # TODO(mojo-1.0): once the func field is restored, copy each element
        # through `self.func`. For now this is a passthrough stub so the
        # package compiles; tests for this struct are also being deferred to
        # Phase E.
        raise Error(
            "LambdaTransform: temporarily disabled in Mojo 1.0 migration"
            " (see TODO(mojo-1.0) and migration Recipe 7 DYNAMIC_TRAIT)"
        )


# ============================================================================
# Conditional Transform
# ============================================================================


struct ConditionalTransform[T: Transform & Copyable & Movable](
    Copyable, Movable, Transform
):
    """Apply transform only if predicate is true.

    Evaluates a predicate function on the input tensor. If true,
    applies the transform. If false, returns input unchanged.

    Time Complexity: O(p + t) where p is predicate cost, t is transform cost.
    Space Complexity: O(n) if transform applied, O(1) otherwise.

    Example:
        ```mojo
        >> fn is_large(tensor: AnyTensor) -> Bool:
        ...     return tensor.num_elements() > 100
        >>>
        >>> var transform = ConditionalTransform(is_large, augment)
        >>> var result = transform(data)  # Only augments large tensors
        ```
    """

    # TODO(mojo-1.0): same DYNAMIC_TRAIT cascade as LambdaTransform — see
    # Recipe 7 in docs/dev/mojo-1.0-migration-recipe.md. The predicate field
    # below is a stubbed Int placeholder until ConditionalTransform is
    # promoted to a parametric struct in Phase E.
    var _stub: Int
    """TODO(mojo-1.0): placeholder until predicate field is re-introduced."""
    var transform: Self.T
    """Transform to apply if predicate is true."""

    def __init__(
        out self,
        predicate: def(AnyTensor) raises -> Bool,
        var transform: Self.T,
    ):
        """Create conditional transform.

        Args:
            predicate: Function to evaluate on tensor. Currently ignored; see
                       TODO(mojo-1.0) above.
            transform: Transform to apply if predicate is true.
        """
        # TODO(mojo-1.0): store `predicate` once struct is parametric on it.
        _ = predicate
        self._stub = 0
        self.transform = transform^

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply transform if predicate is true.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor if predicate true, otherwise original.

        Raises:
            Error: If predicate evaluation or transform fails.
        """
        # TODO(mojo-1.0): once the predicate field is restored, evaluate it
        # and conditionally call self.transform. For now this is a stub that
        # always raises — tests are being deferred to Phase E.
        raise Error(
            "ConditionalTransform: temporarily disabled in Mojo 1.0 migration"
            " (see TODO(mojo-1.0) and migration Recipe 7 DYNAMIC_TRAIT)"
        )


# ============================================================================
# Clamp Transform
# ============================================================================


struct ClampTransform(Copyable, Movable, Transform):
    """Clamp tensor values to specified range [min_val, max_val].

    Limits all values to be within the specified range. Values below
    min_val are set to min_val, values above max_val are set to max_val.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        ```mojo
        >> var clamp = ClampTransform(0.0, 1.0)
        >>> var result = clamp(data)  # All values in [0, 1]
        ```
    """

    var min_val: Float32
    """Minimum allowed value for clamping."""
    var max_val: Float32
    """Maximum allowed value for clamping."""

    def __init__(out self, min_val: Float32, max_val: Float32) raises:
        """Create clamp transform.

        Args:
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Raises:
            Error: If min_val > max_val.
        """
        if min_val > max_val:
            raise Error("min_val must be <= max_val")

        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Clamp all values to [min_val, max_val].

        Args:
            data: Input tensor.

        Returns:
            AnyTensor with all values clamped to range.

        Raises:
            Error: If tensor creation fails.
        """
        var result_values = List[Float32](capacity=data.num_elements())

        for i in range(data.num_elements()):
            var value = Float32(data[i])

            # Clamp to range
            if value < self.min_val:
                result_values.append(self.min_val)
            elif value > self.max_val:
                result_values.append(self.max_val)
            else:
                result_values.append(value)

        return AnyTensor(result_values^)


# ============================================================================
# Debug Transform
# ============================================================================


struct DebugTransform(Copyable, Movable, Transform):
    """Debug transform for logging/inspection.

    Prints tensor information (shape, statistics) for debugging
    purposes, then returns the tensor unchanged. Useful for
    inspecting intermediate results in transform pipelines.

    Time Complexity: O(n) for statistics computation.
    Space Complexity: O(1) - no allocation.

    Example:
        ```mojo
        >> var debug = DebugTransform("layer1_output")
        >>> var result = debug(data)  # Prints info, returns data
        ```
    """

    var name: String
    """Name to display in debug output."""

    def __init__(out self, name: String):
        """Create debug transform.

        Args:
            name: Name to display in debug output.
        """
        self.name = name

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Print tensor info and return unchanged.

        Args:
            data: Input tensor.

        Returns:
            Input tensor unchanged.

        Raises:
            Error: If operation fails.
        """
        print("[DEBUG: " + self.name + "]")
        print("  Elements:", data.num_elements())

        # Compute basic statistics if tensor is non-empty
        if data.num_elements() > 0:
            var min_val = Float32(data[0])
            var max_val = Float32(data[0])
            var sum_val: Float32 = 0.0

            for i in range(data.num_elements()):
                var val = Float32(data[i])
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
                sum_val += val

            var mean_val = sum_val / Float32(Int(data.num_elements()))

            print("  Min:", min_val)
            print("  Max:", max_val)
            print("  Mean:", mean_val)

        return data


# ============================================================================
# Type-Erased Transform Wrapper
# ============================================================================


struct AnyTransform(Copyable, Movable, Transform):
    """Type-erased wrapper for any Transform type.

    Allows storing different transform types in the same list.
    Uses trait object pattern to enable runtime polymorphism.
    """

    # Internal storage using trait object pattern
    # We store the transform as a variant that can hold different types
    var _lambda: Optional[LambdaTransform]
    """Wrapped LambdaTransform if set."""
    var _clamp: Optional[ClampTransform]
    """Wrapped ClampTransform if set."""
    var _identity: Optional[IdentityTransform]
    """Wrapped IdentityTransform if set."""
    var _debug: Optional[DebugTransform]
    """Wrapped DebugTransform if set."""
    var _to_float32: Optional[ToFloat32]
    """Wrapped ToFloat32 if set."""
    var _to_int32: Optional[ToInt32]
    """Wrapped ToInt32 if set."""
    var _sequential: Optional[SequentialTransform]
    """Wrapped SequentialTransform if set."""

    def __init__(out self, var transform: LambdaTransform):
        """Create from LambdaTransform.

        Args:
            transform: LambdaTransform to wrap.
        """
        self._lambda = transform^
        self._clamp = None
        self._identity = None
        self._debug = None
        self._to_float32 = None
        self._to_int32 = None
        self._sequential = None

    def __init__(out self, var transform: ClampTransform) raises:
        """Create from ClampTransform.

        Args:
            transform: ClampTransform to wrap.

        Raises:
            Error: If operation fails.
        """
        self._lambda = None
        self._clamp = transform^
        self._identity = None
        self._debug = None
        self._to_float32 = None
        self._to_int32 = None
        self._sequential = None

    def __init__(out self, var transform: IdentityTransform):
        """Create from IdentityTransform.

        Args:
            transform: IdentityTransform to wrap.
        """
        self._lambda = None
        self._clamp = None
        self._identity = transform^
        self._debug = None
        self._to_float32 = None
        self._to_int32 = None
        self._sequential = None

    def __init__(out self, var transform: DebugTransform):
        """Create from DebugTransform.

        Args:
            transform: DebugTransform to wrap.
        """
        self._lambda = None
        self._clamp = None
        self._identity = None
        self._debug = transform^
        self._to_float32 = None
        self._to_int32 = None
        self._sequential = None

    def __init__(out self, var transform: ToFloat32):
        """Create from ToFloat32.

        Args:
            transform: ToFloat32 to wrap.
        """
        self._lambda = None
        self._clamp = None
        self._identity = None
        self._debug = None
        self._to_float32 = transform^
        self._to_int32 = None
        self._sequential = None

    def __init__(out self, var transform: ToInt32):
        """Create from ToInt32.

        Args:
            transform: ToInt32 to wrap.
        """
        self._lambda = None
        self._clamp = None
        self._identity = None
        self._debug = None
        self._to_float32 = None
        self._to_int32 = transform^
        self._sequential = None

    def __init__(out self, var transform: SequentialTransform):
        """Create from SequentialTransform.

        Args:
            transform: SequentialTransform to wrap.
        """
        self._lambda = None
        self._clamp = None
        self._identity = None
        self._debug = None
        self._to_float32 = None
        self._to_int32 = None
        self._sequential = transform^

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply the wrapped transform.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor.

        Raises:
            Error: If no transform is set or wrapped transform fails.
        """
        if self._lambda:
            return self._lambda.value()(data)
        if self._clamp:
            return self._clamp.value()(data)
        if self._identity:
            return self._identity.value()(data)
        if self._debug:
            return self._debug.value()(data)
        if self._to_float32:
            return self._to_float32.value()(data)
        if self._to_int32:
            return self._to_int32.value()(data)
        if self._sequential:
            return self._sequential.value()(data)
        raise Error("AnyTransform: No transform set")


# ============================================================================
# Sequential Transform
# ============================================================================


struct SequentialTransform(Copyable, Movable, Transform):
    """Apply transforms sequentially in order.

    Chains multiple transforms together, applying them in sequence.
    The output of each transform becomes the input to the next.

    Time Complexity: O(sum of all transform costs).
    Space Complexity: O(n) for intermediate results.

    Example:
        ```mojo
        >> var transforms : List[AnyTransform] = []
        >>> transforms.append(AnyTransform(normalize))
        >>> transforms.append(AnyTransform(clamp))
        >>>
        >>> var pipeline = SequentialTransform(transforms^)
        >>> var result = pipeline(data)
        ```
    """

    var transforms: List[AnyTransform]
    """List of transforms to apply in sequence."""

    def __init__(out self):
        """Create empty sequential transform."""
        self.transforms = List[AnyTransform]()

    def __init__(out self, var transforms: List[AnyTransform]):
        """Create sequential composition.

        Args:
            transforms: List of transforms to apply in order.
        """
        self.transforms = transforms^

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply all transforms sequentially.

        Args:
            data: Input tensor.

        Returns:
            AnyTensor after all transforms applied.

        Raises:
            Error: If any transform fails.
        """
        var result = data

        # Apply each transform in sequence
        for i in range(len(self.transforms)):
            result = self.transforms[i](result)

        return result

    def __len__(self) -> Int:
        """Return number of transforms in sequence.

        Returns:
            Number of transforms in this sequential transform.
        """
        return len(self.transforms)

    def append(mut self, var transform: AnyTransform):
        """Add a transform to the pipeline.

        Args:
            transform: Transform to add to the sequence.
        """
        self.transforms.append(transform^)


# ============================================================================
# Batch Transform
# ============================================================================


struct BatchTransform(Copyable, Movable):
    """Apply transform to a batch of tensors.

    Applies the same transform to each tensor in a list,
    useful for batch processing in data pipelines.

    Time Complexity: O(b * t) where b is batch size, t is transform cost.
    Space Complexity: O(b * n) for output batch.

    Example:
        ```mojo
        >> var batch : List[AnyTensor] = []
        >>> # ... fill batch ...
        >>>
        >>> var transform = BatchTransform(AnyTransform(normalize))
        >>> var results = transform(batch)
        ```
    """

    var transform: AnyTransform
    """Transform to apply to each tensor in the batch."""

    def __init__(out self, var transform: AnyTransform):
        """Create batch transform.

        Args:
            transform: Transform to apply to each tensor in batch.
        """
        self.transform = transform^

    def __call__(self, batch: List[AnyTensor]) raises -> List[AnyTensor]:
        """Apply transform to each tensor in batch.

        Args:
            batch: List of input tensors.

        Returns:
            List of transformed tensors (same order as input).

        Raises:
            Error: If any transform fails.
        """
        var results = List[AnyTensor](capacity=len(batch))

        for i in range(len(batch)):
            var transformed = self.transform(batch[i])
            results.append(transformed)

        return results^

    def __len__(self) -> Int:
        """Return number of tensors in batch.

        Returns:
            Batch size.
        """
        return 0  # Placeholder - batch size not tracked in current design


# ============================================================================
# Type Conversion Transforms
# ============================================================================


struct ToFloat32(Copyable, Movable, Transform):
    """Convert tensor to Float32 dtype.

    Converts all elements to Float32. If already Float32,
    returns a copy. Preserves values exactly for compatible types.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        ```mojo
        >> var converter = ToFloat32()
        >>> var result = converter(int_tensor)
        ```
    """

    def __init__(out self):
        """Create ToFloat32 converter."""
        pass

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Convert to Float32.

        Args:
            data: Input tensor.

        Returns:
            AnyTensor with all values as Float32.

        Raises:
            Error: If tensor creation fails.
        """
        # AnyTensor is already Float32 in current implementation
        # Just create a copy with Float32 values
        var result_values = List[Float32](capacity=data.num_elements())

        for i in range(data.num_elements()):
            result_values.append(Float32(data[i]))

        return AnyTensor(result_values^)


struct ToInt32(Copyable, Movable, Transform):
    """Convert tensor to Int32 dtype (truncation).

    Converts all elements to Int32 by truncating decimal places.
    Positive values round toward zero: 2.9 -> 2.
    Negative values round toward zero: -2.9 -> -2.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        ```mojo
        >> var converter = ToInt32()
        >>> var result = converter(float_tensor)  # Truncates decimals
        ```
    """

    def __init__(out self):
        """Create ToInt32 converter."""
        pass

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Convert to Int32 (truncate).

        Args:
            data: Input tensor.

        Returns:
            AnyTensor with all values truncated to Int32.

        Raises:
            Error: If tensor creation fails.

        Note:
            Truncates toward zero: 2.9 -> 2, -2.9 -> -2.
        """
        var result_values = List[Float32](capacity=data.num_elements())

        for i in range(data.num_elements()):
            var value = data[i]
            # Truncate to int and convert back to float for storage
            var int_value = Int(value)
            result_values.append(Float32(int_value))

        return AnyTensor(result_values^)


# ============================================================================
# Helper Functions
# ============================================================================


def apply_to_tensor(
    data: AnyTensor, func: def(Float32) -> Float32
) raises -> AnyTensor:
    """Apply function element-wise to tensor.

    Helper function for creating ad-hoc transforms without
    defining a transform struct.

    Args:
        data: Input tensor.
        func: Function to apply to each element.

    Returns:
        AnyTensor with function applied element-wise to all values.

    Raises:
        Error: If tensor creation fails.

    Example:
        ```mojo
        >> fn square(x: Float32) -> Float32:
        ...     return x * x
        >>>
        >>> var result = apply_to_tensor(data, square)
        ```
    """
    var transform = LambdaTransform(func)
    return transform(data)


def compose_transforms(
    var transforms: List[AnyTransform],
) raises -> SequentialTransform:
    """Create sequential composition of transforms.

    Convenience function for building transform pipelines.

    Args:
        transforms: List of transforms to compose in order.

    Returns:
        SequentialTransform that applies all transforms sequentially.

    Raises:
        Error: If composition fails.

    Example:
        ```mojo
        >> var pipeline = compose_transforms(List(AnyTransform(norm), AnyTransform(clamp)))
        >>> var result = pipeline(data)
        ```
    """
    return SequentialTransform(transforms^)
