"""Statistical utilities for benchmark analysis.

Provides functions for computing mean, standard deviation, percentiles,
min/max values from benchmark timing data.
"""

from std.math import sqrt


def compute_mean(values: List[Float64]) -> Float64:
    """Compute arithmetic mean of values.

    Args:
        values: List of Float64 values.

    Returns:
        Arithmetic mean of input values.
    """
    if len(values) == 0:
        return 0.0

    var sum: Float64 = 0.0
    for i in range(len(values)):
        sum += values[i]

    return sum / Float64(len(values))


def compute_std_dev(values: List[Float64], mean: Float64) -> Float64:
    """Compute standard deviation.

    Computes sample standard deviation (n-1 divisor for unbiased estimate).

    Args:
        values: List of Float64 values.
        mean: Pre-computed mean (avoids recomputation).

    Returns:
        Sample standard deviation.
    """
    if len(values) <= 1:
        return 0.0

    var sum_sq_diff: Float64 = 0.0
    for i in range(len(values)):
        var diff = values[i] - mean
        sum_sq_diff += diff * diff

    return sqrt(sum_sq_diff / Float64(len(values) - 1))


def compute_min(values: List[Float64]) -> Float64:
    """Compute minimum value.

    Args:
        values: List of Float64 values.

    Returns:
        Minimum value from input.
    """
    if len(values) == 0:
        return 0.0

    var min_val = values[0]
    for i in range(1, len(values)):
        if values[i] < min_val:
            min_val = values[i]

    return min_val


def compute_max(values: List[Float64]) -> Float64:
    """Compute maximum value.

    Args:
        values: List of Float64 values.

    Returns:
        Maximum value from input.
    """
    if len(values) == 0:
        return 0.0

    var max_val = values[0]
    for i in range(1, len(values)):
        if values[i] > max_val:
            max_val = values[i]

    return max_val


def _bubble_sort(mut values: List[Float64]):
    """Simple bubble sort for small arrays.

    Args:
        values: List to sort in-place
    """
    var n = len(values)
    for i in range(n):
        for j in range(n - i - 1):
            if values[j] > values[j + 1]:
                var temp = values[j]
                values[j] = values[j + 1]
                values[j + 1] = temp


def compute_percentile(values: List[Float64], percentile: Int) -> Float64:
    """Compute percentile value.

    Uses linear interpolation between values for percentiles that don't
    align exactly with data points.

    Args:
        values: List of Float64 values.
        percentile: Percentile to compute (0-100).

    Returns:
        Percentile value.
    """
    if len(values) == 0:
        return 0.0

    if len(values) == 1:
        return values[0]

    # Create copy and sort
    var sorted_vals = List[Float64](capacity=len(values))
    for i in range(len(values)):
        sorted_vals.append(values[i])

    _bubble_sort(sorted_vals)

    # Compute index for percentile
    var p = Float64(percentile) / 100.0
    var index = p * Float64(len(sorted_vals) - 1)

    # Linear interpolation for non-integer indices
    var lower_idx = Int(index)
    var upper_idx = lower_idx + 1

    if upper_idx >= len(sorted_vals):
        return sorted_vals[lower_idx]

    var fraction = index - Float64(lower_idx)
    var lower_val = sorted_vals[lower_idx]
    var upper_val = sorted_vals[upper_idx]

    return lower_val + fraction * (upper_val - lower_val)
