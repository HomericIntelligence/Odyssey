"""Trait-based reduction operation definitions for AnyTensor.

Provides trait-based abstraction for reduction operations (sum, mean, max, min),
eliminating code duplication across dtype-specialized implementations.

Benefits:
- Trait-based polymorphism for reduction semantics
- Single source of truth for each reduction operation
- Reusable forward and backward implementations
- Easy to add new reductions (prod, std, var, etc.)
- Zero runtime overhead (compile-time specialization)

Architecture:
- ReduceOp trait: Defines forward reduction semantics
- ReduceBackwardOp trait: Defines backward pass semantics
- Predefined operation structs: SumOp, MeanOp, MaxOp, MinOp
- apply_reduce[Op](): Generic dispatcher for reduction operations

Example usage:
    # Reduce tensor to scalar using sum operation
    var sum_result = reduce_all[SumOp](input_tensor)

    # Reduce along specific axis using mean operation
    var mean_result = reduce_axis[MeanOp](input_tensor, axis=0)

    # Compute backward gradient for sum reduction
    var grad = reduce_backward[SumBackwardOp](grad_output, input_tensor, axis=1)

    # Define custom reduction operation
    struct ProductOp(ReduceOp):
        fn __init__(out self): pass
        fn init_value(self) -> Float64: return 1.0
        fn apply(self, acc: Float64, val: Float64) -> Float64: return acc * val
        fn finalize(self, acc: Float64, count: Int) -> Float64: return acc
        fn is_extremum(self) -> Bool: return False
        fn compare(self, val: Float64, best: Float64) -> Bool: return False

    var product_result = reduce_all[ProductOp](tensor)

See docs/dev/reduction-template-design.md for complete design.
"""

from collections import List


# ============================================================================
# Forward Reduction Trait Definitions
# ============================================================================


trait ReduceOp:
    """Trait for reduction operations.

    Implement this trait to define custom reduction operations that can be
    used with reduce_all[Op]() and reduce_axis[Op](). The operation defines:
    - Initial accumulator value (identity element)
    - How to accumulate values
    - Final value transformation
    - Whether this is an extremum operation (max/min)

    Required Methods:
        `__init__`: Default constructor
        `init_value`: Returns identity element for accumulation
        `apply`: Accumulate next value with current accumulator
        `finalize`: Transform accumulated result (e.g., divide for mean)
        `is_extremum`: True for max/min, False for sum/mean
        `compare`: For extremum ops, determines if value replaces current best

    Contract:
        - Must have a default constructor (no arguments)
        - init_value() returns the identity element (0.0 for sum, etc.)
        - apply(acc, val) is associative: apply(apply(a, b), c) == apply(a, apply(b, c))
        - finalize() may perform post-processing (e.g., division for mean)
        - is_extremum() must match compare() implementation
        - For extremum ops, compare() must be consistent with the extremum type

    Example:
        struct SumOp(ReduceOp):
            '''Sum all elements.'''
            fn __init__(out self): pass
            fn init_value(self) -> Float64: return 0.0
            fn apply(self, acc: Float64, val: Float64) -> Float64: return acc + val
            fn finalize(self, acc: Float64, count: Int) -> Float64: return acc
            fn is_extremum(self) -> Bool: return False
            fn compare(self, val: Float64, best: Float64) -> Bool: return False

        var sum_result = reduce_all[SumOp](tensor)
    """

    fn __init__(out self):
        """Default constructor required for generic instantiation."""
        ...

    fn init_value(self) -> Float64:
        """Return the identity element for this reduction.

        The identity element should be a value such that:
            apply(init_value(), x) == x for all x

        Common identity values:
        - Sum: 0.0
        - Product: 1.0
        - Max: -Float64.MAX
        - Min: Float64.MAX

        Returns:
            Identity element as Float64.
        """
        ...

    fn apply(self, acc: Float64, val: Float64) -> Float64:
        """Apply reduction operation to accumulate values.

        This function must be associative and commutative:
        - apply(apply(a, b), c) == apply(apply(a, c), b)
        - apply(apply(a, b), c) == apply(a, apply(b, c))

        Args:
            acc: Current accumulator value.
            val: Next value to accumulate.

        Returns:
            Updated accumulator value.
        """
        ...

    fn finalize(self, acc: Float64, count: Int) -> Float64:
        """Finalize accumulated result after all values processed.

        This function may transform the accumulated value (e.g., divide by
        count for mean). It receives:
        - The final accumulated value
        - The number of values that were accumulated

        Args:
            acc: Final accumulated value.
            count: Number of values that were accumulated.

        Returns:
            Final result value.
        """
        ...

    fn is_extremum(self) -> Bool:
        """Return True if this is a max/min operation.

        Extremum operations (max, min) require special handling during
        backward pass because gradients only flow to the extremum elements.

        Returns:
            True if this is max/min, False otherwise.
        """
        ...

    fn compare(self, val: Float64, current_best: Float64) -> Bool:
        """
        For extremum ops: returns True if val should replace current_best.

        This method is only meaningful for extremum operations (max/min).
        For non-extremum operations, this is never called.

        Args:
            val: Candidate value.
            current_best: Current best value.

        Returns:
            True if val should replace current_best, False otherwise.

        Examples:
        ```
            # For max operation
            return val > current_best

            # For min operation
            return val < current_best
        ```
        """
        ...


# ============================================================================
# Backward Reduction Trait Definitions
# ============================================================================


trait ReduceBackwardOp:
    """Trait for reduction backward operations.

    Implement this trait to define backward passes for reduction operations.
    The backward operation computes gradients for each input element given
    the upstream gradient and the reduction operation semantics.

    Required Methods:
        `__init__`: Default constructor
        `compute_gradient`: Compute gradient for a single input element

    Contract:
        - Must have a default constructor (no arguments)
        - compute_gradient() must account for the reduction semantics
        - For extremum ops, must only allocate gradient to extremum elements

    Example:
    ```
        struct SumBackwardOp(ReduceBackwardOp):
            '''Backward for sum: gradient flows equally to all inputs.'''
            fn __init__(out self): pass
            fn compute_gradient(self, grad_output_val: Float64, input_val: Float64,
                               axis_values: List[Float64], count: Int) -> Float64:
                return grad_output_val  # Gradient unchanged

        var grad = reduce_backward[SumBackwardOp](grad_output, input, axis)
    ```
    """

    fn __init__(out self):
        """Default constructor required for generic instantiation."""
        ...

    fn compute_gradient(
        self,
        grad_output_val: Float64,
        input_val: Float64,
        axis_values: List[Float64],
        count: Int,
    ) -> Float64:
        """Compute gradient for a single input element.

        This function must implement the backward pass for the corresponding
        forward reduction operation. It receives:
        - The upstream gradient (from grad_output).
        - The original input value at this position.
        - All values along the reduction axis (for extremum operations).
        - The count of values in the reduction.

        For sum/mean operations:
            Simply scale the gradient according to the reduction semantics.

        For extremum operations (max/min):
            - Find the extremum value in axis_values.
            - Count how many elements have the extremum value.
            - Only allocate gradient to extremum elements (split equally).
            - Return 0.0 for non-extremum elements.

        Args:
            grad_output_val: Upstream gradient for this reduction result.
            input_val: Original input value at this position.
            axis_values: All values along the reduction axis.
            count: Number of elements in the reduction.

        Returns:
            Gradient for this input element.
        """
        ...


# ============================================================================
# Predefined Reduction Operations
# ============================================================================


struct SumOp(ReduceOp):
    """Sum reduction: accumulate by addition.

    Forward: result = sum(x)
    Backward: grad_x = grad_output (unchanged)

    Identity: 0.0
    Associative: Yes (a + b + c = (a + b) + c = a + (b + c))
    """

    fn __init__(out self):
        pass

    fn init_value(self) -> Float64:
        return 0.0

    fn apply(self, acc: Float64, val: Float64) -> Float64:
        return acc + val

    fn finalize(self, acc: Float64, count: Int) -> Float64:
        return acc

    fn is_extremum(self) -> Bool:
        return False

    fn compare(self, val: Float64, current_best: Float64) -> Bool:
        return False


struct MeanOp(ReduceOp):
    """Mean reduction: sum then divide by count.

    Forward: result = sum(x) / count
    Backward: grad_x = grad_output / count

    Identity: 0.0 (for accumulation), then divide by count
    Associative: Yes (for accumulation phase)
    """

    fn __init__(out self):
        pass

    fn init_value(self) -> Float64:
        return 0.0

    fn apply(self, acc: Float64, val: Float64) -> Float64:
        return acc + val

    fn finalize(self, acc: Float64, count: Int) -> Float64:
        return acc / Float64(count)

    fn is_extremum(self) -> Bool:
        return False

    fn compare(self, val: Float64, current_best: Float64) -> Bool:
        return False


struct MaxOp(ReduceOp):
    """Max reduction: find maximum value.

    Forward: result = max(x)
    Backward: grad_x = grad_output if x == max, else 0.0 (split if tied)

    Identity: -Float64.MAX
    Associative: Yes (max(max(a, b), c) = max(a, max(b, c)))
    """

    fn __init__(out self):
        pass

    fn init_value(self) -> Float64:
        return -1e308  # -Float64.MAX

    fn apply(self, acc: Float64, val: Float64) -> Float64:
        if val > acc:
            return val
        return acc

    fn finalize(self, acc: Float64, count: Int) -> Float64:
        return acc

    fn is_extremum(self) -> Bool:
        return True

    fn compare(self, val: Float64, current_best: Float64) -> Bool:
        return val > current_best


struct MinOp(ReduceOp):
    """Min reduction: find minimum value.

    Forward: result = min(x)
    Backward: grad_x = grad_output if x == min, else 0.0 (split if tied)

    Identity: Float64.MAX
    Associative: Yes (min(min(a, b), c) = min(a, min(b, c)))
    """

    fn __init__(out self):
        pass

    fn init_value(self) -> Float64:
        return 1e308  # Float64.MAX

    fn apply(self, acc: Float64, val: Float64) -> Float64:
        if val < acc:
            return val
        return acc

    fn finalize(self, acc: Float64, count: Int) -> Float64:
        return acc

    fn is_extremum(self) -> Bool:
        return True

    fn compare(self, val: Float64, current_best: Float64) -> Bool:
        return val < current_best


# ============================================================================
# Predefined Backward Reduction Operations
# ============================================================================


struct SumBackwardOp(ReduceBackwardOp):
    """Backward for sum reduction: gradient flows equally to all inputs.

    Forward: result = sum(x)
    Backward: grad_x = grad_output

    Since sum is linear, the gradient for each element is just the
    upstream gradient (unchanged).
    """

    fn __init__(out self):
        pass

    fn compute_gradient(
        self,
        grad_output_val: Float64,
        input_val: Float64,
        axis_values: List[Float64],
        count: Int,
    ) -> Float64:
        return grad_output_val


struct MeanBackwardOp(ReduceBackwardOp):
    """Backward for mean reduction: gradient divided by count.

    Forward: result = sum(x) / count
    Backward: grad_x = grad_output / count

    Since mean is sum divided by a constant, the gradient is scaled
    by 1/count.
    """

    fn __init__(out self):
        pass

    fn compute_gradient(
        self,
        grad_output_val: Float64,
        input_val: Float64,
        axis_values: List[Float64],
        count: Int,
    ) -> Float64:
        return grad_output_val / Float64(count)


struct MaxBackwardOp(ReduceBackwardOp):
    """Backward for max reduction: gradient flows only to maximum elements.

    Forward: result = max(x)
    Backward: grad_x = grad_output / num_max if x == max(x), else 0.0

    Since max is non-differentiable at ties, we split the gradient equally
    among all elements that achieve the maximum value. This handles the
    case where multiple elements have the same maximum value.
    """

    fn __init__(out self):
        pass

    fn compute_gradient(
        self,
        grad_output_val: Float64,
        input_val: Float64,
        axis_values: List[Float64],
        count: Int,
    ) -> Float64:
        # Find max value in the axis
        var max_val = axis_values[0]
        for i in range(1, len(axis_values)):
            if axis_values[i] > max_val:
                max_val = axis_values[i]

        # Count how many elements achieve the max value
        var max_count = 0
        for i in range(len(axis_values)):
            if axis_values[i] == max_val:
                max_count += 1

        # Gradient flows only to maximum elements (split equally)
        if input_val == max_val:
            return grad_output_val / Float64(max_count)
        return 0.0


struct MinBackwardOp(ReduceBackwardOp):
    """Backward for min reduction: gradient flows only to minimum elements.

    Forward: result = min(x)
    Backward: grad_x = grad_output / num_min if x == min(x), else 0.0

    Since min is non-differentiable at ties, we split the gradient equally
    among all elements that achieve the minimum value. This handles the
    case where multiple elements have the same minimum value.
    """

    fn __init__(out self):
        pass

    fn compute_gradient(
        self,
        grad_output_val: Float64,
        input_val: Float64,
        axis_values: List[Float64],
        count: Int,
    ) -> Float64:
        # Find min value in the axis
        var min_val = axis_values[0]
        for i in range(1, len(axis_values)):
            if axis_values[i] < min_val:
                min_val = axis_values[i]

        # Count how many elements achieve the min value
        var min_count = 0
        for i in range(len(axis_values)):
            if axis_values[i] == min_val:
                min_count += 1

        # Gradient flows only to minimum elements (split equally)
        if input_val == min_val:
            return grad_output_val / Float64(min_count)
        return 0.0
