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

Mojo 1.0 design note (E3):
    ``LambdaTransform[F]`` and ``ConditionalTransform[Pred, T]`` are now
    compile-time parametric structs (Recipe 7 DYNAMIC_TRAIT).  The function
    ``F`` and predicate ``Pred`` must be **thin** (non-capturing) defs —
    i.e., they must not close over runtime variables.  Local ``def``
    functions that do not capture are automatically thin in Mojo 1.0.

    ``AnyTransform`` stores a ``def(AnyTensor) raises thin -> AnyTensor``
    function-pointer field for lambda-originated entries; use the static
    ``AnyTransform.from_lambda[F]()`` factory instead of the old
    ``AnyTransform(LambdaTransform(f))`` syntax.

Example:
    ```mojo
    def scale(x: Float32) -> Float32:
        return x / 255.0

    var pipeline = SequentialTransform()
    pipeline.append(AnyTransform.from_lambda[scale]())
    pipeline.append(AnyTransform(ClampTransform(0.0, 1.0)))

    var result = pipeline(data)
    ```
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.data.transforms import Transform
from std.collections import Optional


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
# Lambda Transform helpers (module-level thin functions)
# ============================================================================


def _apply_f32_to_tensor[
    F: def(Float32) thin -> Float32
](data: AnyTensor,) raises -> AnyTensor:
    """Apply a thin Float32->Float32 function element-wise.

    This module-level function is used both by ``LambdaTransform[F].__call__``
    and by ``AnyTransform.from_lambda[F]()`` to produce a thin function
    pointer that can be stored in ``AnyTransform._lambda_fn``.

    Parameters:
        F: Thin (non-capturing) def function from Float32 to Float32.

    Args:
        data: Input tensor.

    Returns:
        New AnyTensor with F applied to every element.

    Raises:
        Error: If tensor creation fails.
    """
    var result = List[Float32](capacity=data.num_elements())
    for i in range(data.num_elements()):
        result.append(F(Float32(data[i])))
    return AnyTensor(result^)


# ============================================================================
# Lambda Transform
# ============================================================================


struct LambdaTransform[F: def(Float32) thin -> Float32](
    Copyable, Movable, Transform
):
    """Apply a compile-time thin function element-wise to tensor values.

    Mojo 1.0 design (E3 — Recipe 7 DYNAMIC_TRAIT): ``F`` is a compile-time
    **thin** (non-capturing) function parameter, not a runtime field.  The
    struct carries no runtime state; ``__call__`` is non-capturing and
    therefore conforms to the ``Transform`` trait.

    Use ``LambdaTransform[my_fn]()`` to instantiate:

    ```mojo
    def scale(x: Float32) -> Float32:
        return x / 255.0

    var t = LambdaTransform[scale]()
    var result = t(tensor)
    ```

    To wrap in ``AnyTransform`` (e.g. for ``SequentialTransform``), use
    ``AnyTransform.from_lambda[scale]()`` instead.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.
    """

    var _dummy: Int
    """Placeholder field required so the synthesised copy constructor works."""

    def __init__(out self):
        """Create lambda transform.

        The function ``F`` is a compile-time parameter; no runtime argument
        is needed.
        """
        self._dummy = 0

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply F to each element.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor with F applied element-wise.

        Raises:
            Error: If tensor creation fails.
        """
        return _apply_f32_to_tensor[Self.F](data)


# ============================================================================
# Conditional Transform
# ============================================================================


struct ConditionalTransform[
    Pred: def(AnyTensor) raises thin -> Bool,
    T: Transform & Copyable & Movable,
](Copyable, Movable, Transform):
    """Apply transform only if a compile-time thin predicate is true.

    Mojo 1.0 design (E3 — Recipe 7 DYNAMIC_TRAIT): ``Pred`` is a compile-time
    **thin** (non-capturing) predicate function; it is called via
    ``Self.Pred(data)`` so ``__call__`` remains non-capturing and conforms to
    the ``Transform`` trait.  The inner transform ``T`` is stored as a regular
    runtime field.

    ```mojo
    def is_large(tensor: AnyTensor) raises -> Bool:
        return tensor.num_elements() > 100

    var augment = LambdaTransform[my_fn]()
    var conditional = ConditionalTransform[is_large, LambdaTransform[my_fn]](augment^)
    var result = conditional(tensor)  # Only augments large tensors
    ```

    Time Complexity: O(p + t) where p is predicate cost, t is transform cost.
    Space Complexity: O(n) if transform applied, O(1) otherwise.
    """

    var transform: Self.T
    """Transform to apply if predicate is true."""

    def __init__(out self, var transform: Self.T):
        """Create conditional transform.

        Args:
            transform: Transform to apply when ``Pred`` evaluates to True.
        """
        self.transform = transform^

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply transform if predicate is true.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor if predicate true, otherwise input unchanged.

        Raises:
            Error: If predicate evaluation or transform fails.
        """
        if Self.Pred(data):
            return self.transform(data)
        return data


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
    Uses a manual union-of-optionals pattern to enable runtime polymorphism.

    Mojo 1.0 note (E3): ``LambdaTransform[F]`` is now parametric and cannot
    be stored directly in a typed Optional field.  Use the static factory
    ``AnyTransform.from_lambda[F]()`` instead — it stores a thin function
    pointer in ``_lambda_fn`` that applies ``F`` element-wise without
    capturing runtime state.
    """

    var _lambda_fn: Optional[def(AnyTensor) raises thin -> AnyTensor]
    """Thin function pointer for lambda-originated transforms."""
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

    @staticmethod
    def from_lambda[F: def(Float32) thin -> Float32]() -> AnyTransform:
        """Create an AnyTransform that applies a thin Float32→Float32 function.

        This is the Mojo-1.0-compatible replacement for the old
        ``AnyTransform(LambdaTransform(f))`` syntax.

        Parameters:
            F: Thin (non-capturing) def function from Float32 to Float32.

        Returns:
            AnyTransform that applies F element-wise.

        Example:
            ```mojo
            def scale(x: Float32) -> Float32:
                return x / 255.0

            transforms.append(AnyTransform.from_lambda[scale]())
            ```
        """
        # _apply_f32_to_tensor[F] is specialised here where F is statically
        # known; the resulting function pointer is thin and storable as a
        # def(AnyTensor) raises thin -> AnyTensor field.
        return AnyTransform(_apply_f32_to_tensor[F])

    def __init__(out self, `fn`: def(AnyTensor) raises thin -> AnyTensor):
        """Create from a thin AnyTensor→AnyTensor function (internal use).

        Args:
            fn: Thin function to apply.  Callers should prefer
                ``AnyTransform.from_lambda[F]()`` for lambda transforms.
        """
        self._lambda_fn = `fn`
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
        self._lambda_fn = None
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
        self._lambda_fn = None
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
        self._lambda_fn = None
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
        self._lambda_fn = None
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
        self._lambda_fn = None
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
        self._lambda_fn = None
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
        if self._lambda_fn:
            return self._lambda_fn.value()(data)
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


def apply_to_tensor[
    F: def(Float32) thin -> Float32
](data: AnyTensor,) raises -> AnyTensor:
    """Apply a thin function element-wise to a tensor.

    Helper for ad-hoc transforms without defining a transform struct.
    Mojo 1.0 note (E3): the function must be a compile-time thin parameter
    rather than a runtime argument.

    Parameters:
        F: Thin (non-capturing) def function from Float32 to Float32.

    Args:
        data: Input tensor.

    Returns:
        AnyTensor with F applied element-wise to all values.

    Raises:
        Error: If tensor creation fails.

    Example:
        ```mojo
        def square(x: Float32) -> Float32:
            return x * x

        var result = apply_to_tensor[square](data)
        ```
    """
    return _apply_f32_to_tensor[F](data)


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
