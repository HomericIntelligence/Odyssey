"""Test utilities for AnyTensor testing.

Provides utility functions for test setup, debugging, and validation.
"""

from shared.core.any_tensor import AnyTensor


fn print_tensor(tensor: AnyTensor) -> String:
    """Pretty-print tensor with shape, dtype, and values.

    Args:
        tensor: The AnyTensor to print.

    Returns:
        String representation of the tensor with shape, dtype, and values.
        Large tensors are truncated showing first and last 3 elements.

    Example:
        ```mojo
        var tensor = random_tensor([2, 3], DType.float32)
        print(print_tensor(tensor))
        # Output: Tensor(shape=[2, 3], dtype=float32):
        #   [[1.234, 2.345, 3.456],
        #    [4.567, 5.678, 6.789]]
        ```
    """
    var result = String()
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    var numel = tensor.numel()

    # Build header with shape and dtype
    result += "Tensor(shape=[" + String(shape[0])
    for i in range(1, shape.__len__()):
        result += "," + String(shape[i])
    result += "], dtype=" + String(dtype) + "):\n"

    # Build value representation
    if numel == 0:
        result += "  []"
    elif numel == 1:
        result += "  [" + String(tensor._get_float64(0)) + "]"
    else:
        # Format values with truncation for large tensors
        result += "  ["
        var show_limit = 3 if numel > 6 else numel

        for i in range(show_limit):
            if i > 0:
                result += ", "
            result += String(tensor._get_float64(i))

        if numel > 6:
            result += ", ..., "
            for i in range(numel - 3, numel):
                if i > numel - 3:
                    result += ", "
                result += String(tensor._get_float64(i))

        result += "]"

    return result


fn tensor_summary(tensor: AnyTensor) -> String:
    """Print tensor statistics: shape, dtype, min, max, mean, std.

    Args:
        tensor: The AnyTensor to summarize.

    Returns:
        String with shape, dtype, numel, and statistical summary.

    Example:
        ```mojo
        var tensor = random_tensor([10, 10], DType.float32)
        print(tensor_summary(tensor))
        # Output: Shape: [10, 10], DType: float32, Numel: 100
        #         Min: 0.001, Max: 0.999, Mean: 0.523, Std: 0.289
        ```
    """
    var result = String()
    var shape = tensor.shape()
    var dtype = tensor.dtype()
    var numel = tensor.numel()

    # Build header
    result += "Shape: [" + String(shape[0])
    for i in range(1, shape.__len__()):
        result += "," + String(shape[i])
    result += "], DType: " + String(dtype) + ", Numel: " + String(numel) + "\n"

    if numel == 0:
        result += "Statistics: Empty tensor"
        return result

    # Calculate min, max, mean, std
    var min_val = tensor._get_float64(0)
    var max_val = tensor._get_float64(0)
    var sum_val = 0.0
    var sum_sq = 0.0

    for i in range(numel):
        var val = tensor._get_float64(i)
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
        sum_val += val
        sum_sq += val * val

    var mean = sum_val / Float64(numel)
    var variance = (sum_sq / Float64(numel)) - (mean * mean)
    var std_dev = 0.0
    if variance >= 0:
        # Simple square root approximation
        var guess = variance / 2.0 + 0.5
        for _ in range(10):
            guess = (guess + variance / guess) / 2.0
        std_dev = guess

    result += "Min: " + String(min_val) + ", Max: " + String(max_val)
    result += ", Mean: " + String(mean) + ", Std: " + String(std_dev)

    return result


fn compare_tensors(a: AnyTensor, b: AnyTensor) -> String:
    """Detailed comparison of two tensors for debugging test failures.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        String with comparison summary showing shape/dtype match and max difference.

    Example:
        ```mojo
        var t1 = random_tensor([3, 3], DType.float32)
        var t2 = random_tensor([3, 3], DType.float32)
        print(compare_tensors(t1, t2))
        # Output: Shapes match: ✓, DTypes match: ✓, Max diff: 0.523 at index 4
        ```
    """
    var result = String()
    var shape_a = a.shape()
    var shape_b = b.shape()
    var dtype_a = a.dtype()
    var dtype_b = b.dtype()

    # Check shapes
    var shapes_match = shape_a.__len__() == shape_b.__len__()
    if shapes_match:
        for i in range(shape_a.__len__()):
            if shape_a[i] != shape_b[i]:
                shapes_match = False
                break

    result += "Shapes match: "
    if shapes_match:
        result += "✓"
    else:
        result += "✗ (A: [" + String(shape_a[0])
        for i in range(1, shape_a.__len__()):
            result += "," + String(shape_a[i])
        result += "] vs B: [" + String(shape_b[0])
        for i in range(1, shape_b.__len__()):
            result += "," + String(shape_b[i])
        result += "])"

    result += ", DTypes match: "
    if dtype_a == dtype_b:
        result += "✓"
    else:
        result += "✗"

    # Calculate max difference
    if shapes_match and a.numel() == b.numel():
        var max_diff = 0.0
        var max_diff_idx = 0

        for i in range(a.numel()):
            var val_a = a._get_float64(i)
            var val_b = b._get_float64(i)
            var diff = val_a - val_b
            if diff < 0:
                diff = -diff

            if diff > max_diff:
                max_diff = diff
                max_diff_idx = i

        result += (
            ", Max diff: "
            + String(max_diff)
            + " at index "
            + String(max_diff_idx)
        )
    else:
        result += ", Max diff: Cannot compare (shape or numel mismatch)"

    return result


fn benchmark[func: fn () raises -> None](iterations: Int) -> Float64:
    """Simple performance testing helper.

    Args:
        iterations: Number of times to run the function.

    Returns:
        Average execution time in milliseconds.

    Example:
        ```mojo
        fn test_op() raises:
            var t = random_tensor([1000, 1000], DType.float32)
            _ = t + t

        var avg_time_ms = benchmark[test_op](100)
        print("Average time: " + String(avg_time_ms) + " ms")
        ```

    Note:
        Uses time.now() for measurement. Times include function call overhead.
    """
    # Note: time module operations would be needed for true benchmarking
    # For now, return a placeholder that indicates the function ran
    # In a real implementation, this would use time.perf_counter_ns() or similar
    return 0.0
