"""Consolidated benchmark result tracking with statistics computation.

This module provides the BenchmarkResult struct for tracking individual
iteration timings and computing statistical metrics (mean, standard deviation,
min, max) for benchmarked operations.

The struct is designed for low-level result accumulation where individual
iteration times are recorded in nanoseconds, allowing flexible conversion to
other time units and statistical analysis.

Features:
    - Record individual iteration times in nanoseconds.
    - Compute mean execution time using Welford's algorithm (numerically stable).
    - Compute standard deviation with sample variance (N-1).
    - Track min/max times across iterations.
    - Support for up to 2^63-1 iterations.
    - Efficient memory usage via Welford's online algorithm.

Example:
   ```mojo
    var result = BenchmarkResult("relu_forward", iterations=1000)

    # Record individual iteration times
    for _ in range(1000):
        var start = now()
        relu(tensor)
        var end = now()
        result.record(end - start)

    # Query statistics
    var mean_ns = result.mean()  # Mean in nanoseconds as Float64
    var mean_us = result.mean() / 1000.0  # Convert to microseconds
    var std_ns = result.std()  # Standard deviation
    var min_ns = result.min_time()  # Minimum time
    var max_ns = result.max_time()  # Maximum time
    print(result)  # Formatted summary
    ```
"""

from std.math import sqrt


struct BenchmarkResult(Copyable, Movable, Writable):
    """Results from a benchmark run with iteration-level timing data.

    Uses Welford's algorithm for numerically stable online computation of
    mean and variance as iterations are recorded.

    Attributes:
        name: Descriptive name for the benchmarked operation.
        iterations: Total number of iterations recorded.
        total_time_ns: Sum of all iteration times in nanoseconds.
        min_time_ns: Minimum iteration time in nanoseconds.
        max_time_ns: Maximum iteration time in nanoseconds.
        mean_time_ns: Mean (average) iteration time in nanoseconds (computed via Welford's).
        _M2: Second moment accumulator for variance (Welford's algorithm).

    Notes:
        - Welford's algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        - `Variance = _M2 / (iterations - 1) for sample variance (N-1 denominator)`
        - `Standard deviation = sqrt(variance)`
        - Mean is computed exactly via accumulated sum.
    """

    var name: String
    """Descriptive name for the benchmarked operation."""
    var iterations: Int
    """Total number of iterations recorded."""
    var total_time_ns: Int
    """Sum of all iteration times in nanoseconds."""
    var min_time_ns: Int
    """Minimum iteration time in nanoseconds."""
    var max_time_ns: Int
    """Maximum iteration time in nanoseconds."""
    var _mean: Float64
    """Mean iteration time computed via Welford's algorithm."""
    var _M2: Float64
    """Second moment accumulator for variance (Welford's algorithm)."""

    def __init__(out self, name: String, iterations: Int):
        """Initialize a benchmark result tracker.

        Args:
            name: Descriptive name for the operation (e.g., "relu_forward").
            iterations: Expected number of iterations (for documentation).

        Notes:
            - iterations parameter is stored for reference but does not limit
              the number of records() calls.
            - All timing fields initialized to 0.
            - First call to record() will initialize min/max to that time.
        """
        self.name = name
        self.iterations = iterations
        self.total_time_ns = 0
        self.min_time_ns = 0
        self.max_time_ns = 0
        self._mean = 0.0
        self._M2 = 0.0

    def record(mut self, time_ns: Int):
        """Record a single iteration's execution time in nanoseconds.

        Uses Welford's online algorithm to compute mean and variance
        incrementally without storing all measurements.

        Args:
            time_ns: Execution time of one iteration in nanoseconds (must be >= 0).

        Notes:
            - First call sets min and max to this time.
            - Subsequent calls update min/max if needed.
            - Mean and variance are updated via Welford's algorithm.
            - Total time is accumulated for reference.
        """
        var n = self.iterations + 1
        self.iterations = n
        self.total_time_ns += time_ns

        # Initialize min/max on first call
        if n == 1:
            self.min_time_ns = time_ns
            self.max_time_ns = time_ns
        else:
            if time_ns < self.min_time_ns:
                self.min_time_ns = time_ns
            if time_ns > self.max_time_ns:
                self.max_time_ns = time_ns

        # Welford's algorithm for mean and variance
        # delta = time_ns - old_mean
        var delta = Float64(time_ns) - self._mean
        # mean = old_mean + delta / n
        self._mean = self._mean + delta / Float64(n)
        # delta2 = time_ns - new_mean
        var delta2 = Float64(time_ns) - self._mean
        # M2 = M2 + delta * delta2
        self._M2 = self._M2 + delta * delta2

    def mean(self) -> Float64:
        """Compute mean (average) iteration time in nanoseconds.

        Returns:
            Mean execution time as Float64 in nanoseconds.
            Returns 0.0 if no iterations recorded.

        Notes:
            - Result is exact (computed from accumulated mean via Welford's).
            - Safe even for very large iteration counts.
        """
        if self.iterations == 0:
            return 0.0
        return self._mean

    def std(self) -> Float64:
        """Compute standard deviation of iteration times.

        Uses sample variance (N-1 denominator) for unbiased estimation,
        following standard statistical practice for sample data.

        Returns:
            Standard deviation in nanoseconds as Float64.
            Returns 0.0 if fewer than 2 iterations recorded.

        Notes:
            - Requires at least 2 iterations to compute meaningful std dev.
            - Uses Welford's _M2 accumulated value.
            - Formula: std = sqrt(_M2 / (n - 1)).
            - Returns 0.0 if iterations <= 1.
        """
        if self.iterations <= 1:
            return 0.0
        var variance = self._M2 / Float64(self.iterations - 1)
        return sqrt(variance)

    def min_time(self) -> Float64:
        """Get minimum iteration time in nanoseconds.

        Returns:
            Minimum execution time as Float64 in nanoseconds.
            Returns 0.0 if no iterations recorded.

        Notes:
            - Tracks actual minimum across all recorded iterations.
            - Useful for identifying best-case performance.
        """
        if self.iterations == 0:
            return 0.0
        return Float64(self.min_time_ns)

    def max_time(self) -> Float64:
        """Get maximum iteration time in nanoseconds.

        Returns:
            Maximum execution time as Float64 in nanoseconds.
            Returns 0.0 if no iterations recorded.

        Notes:
            - Tracks actual maximum across all recorded iterations.
            - Useful for identifying worst-case performance.
        """
        if self.iterations == 0:
            return 0.0
        return Float64(self.max_time_ns)

    def write_to(self, mut writer: Some[Writer]):
        """Write benchmark results to a writer.

        Args:
            writer: Target writer to write the formatted results to.

        Notes:
            Primary implementation of string formatting for the Writable trait.
            __str__() delegates to this method via String.write(self).
        """
        var mean_ns = self.mean()
        var mean_us = mean_ns / 1000.0
        var std_ns = self.std()
        var std_us = std_ns / 1000.0
        var min_us = self.min_time() / 1000.0
        var max_us = self.max_time() / 1000.0

        writer.write("BenchmarkResult: " + self.name + "\n")
        writer.write("  Iterations: " + String(self.iterations) + "\n")
        writer.write(
            "  Mean: " + String(mean_us) + " us (" + String(mean_ns) + " ns)\n"
        )
        writer.write("  Std Dev: " + String(std_us) + " us\n")
        writer.write("  Min: " + String(min_us) + " us\n")
        writer.write("  Max: " + String(max_us) + " us")

    def __str__(self) -> String:
        """Format benchmark results as a human-readable string.

        Returns:
            Formatted summary of results with timing in microseconds.

        Example output:
        ```
            BenchmarkResult: relu_forward
                Iterations: 1000
                Mean: 1234.56 us (1234567.89 ns)
                Std Dev: 45.67 us
                Min: 1000.00 us
                Max: 2000.00 us
        ```
        """
        var mean_ns = self.mean()
        var mean_us = mean_ns / 1000.0
        var std_ns = self.std()
        var std_us = std_ns / 1000.0
        var min_us = self.min_time() / 1000.0
        var max_us = self.max_time() / 1000.0

        var result = String("")
        result += "BenchmarkResult: " + self.name + "\n"
        result += "  Iterations: " + String(self.iterations) + "\n"
        result += (
            "  Mean: " + String(mean_us) + " us (" + String(mean_ns) + " ns)\n"
        )
        result += "  Std Dev: " + String(std_us) + " us\n"
        result += "  Min: " + String(min_us) + " us\n"
        result += "  Max: " + String(max_us) + " us"

        return result

    def write_to(self, mut writer: Some[Writer]):
        """Write benchmark results to a writer.

        Args:
            writer: Target writer to write the formatted results to.

        Notes:
            Implements the Writable trait to replace deprecated __str__.
            Writes the same formatted output as __str__() method.
        """
        writer.write(self.__str__())
