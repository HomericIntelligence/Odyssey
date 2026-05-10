"""Loss tracking and aggregation for training monitoring.

Provides numerically stable loss tracking with moving averages, statistics,
and multi-component support for complex loss functions.

Features:
- Moving average with circular buffer (configurable window)
- Welford's algorithm for numerically stable variance
- Multi-component loss tracking (e.g., total, reconstruction, regularization)
- Statistical summaries (mean, std, min, max)

"""

from std.collections import List
from std.math import sqrt
from shared.training.metrics.base import Metric
from shared.tensor.any_tensor import AnyTensor

# min and max are now builtins in Mojo - no import needed


# ============================================================================
# Statistics Struct (#283-287)
# ============================================================================


struct Statistics(Copyable, Movable):
    """Statistical summary of loss values.

    Provides mean, standard deviation, min, max, and count of values

    """

    var mean: Float32
    """Mean loss value."""
    var std: Float32
    """Standard deviation of loss values."""
    var min: Float32
    """Minimum loss value seen."""
    var max: Float32
    """Maximum loss value seen."""
    var count: Int
    """Total number of values tracked."""

    def __init__(out self):
        """Initialize with zero values."""
        self.mean = 0.0
        self.std = 0.0
        self.min = 0.0
        self.max = 0.0
        self.count = 0

    def __init__(
        out self,
        mean: Float32,
        std: Float32,
        min_val: Float32,
        max_val: Float32,
        count: Int,
    ):
        """Initialize with specific values.

        Args:
            mean: Mean loss value.
            std: Standard deviation of loss values.
            min_val: Minimum loss value.
            max_val: Maximum loss value.
            count: Total number of values tracked.
        """
        self.mean = mean
        self.std = std
        self.min = min_val
        self.max = max_val
        self.count = count


# ============================================================================
# Component Tracker (#283-287)
# ============================================================================


struct ComponentTracker(Copyable, Movable):
    """Tracks a single loss component with statistics and moving average.

    Uses Welford's algorithm for numerically stable variance computation.
    and a circular buffer for moving average

    """

    var window_size: Int
    """Size of moving average window."""
    var buffer: List[Float32]
    """Circular buffer for moving average."""
    var buffer_idx: Int
    """Current index in circular buffer."""
    var buffer_full: Bool
    """Whether buffer has been filled at least once."""

    # Welford's algorithm state
    var count: Int
    """Total number of values added."""
    var mean: Float64
    """Running mean value."""
    var m2: Float64
    """Sum of squared differences from mean."""

    # Min/max tracking
    var min_value: Float32
    """Minimum value seen."""
    var max_value: Float32
    """Maximum value seen."""
    var last_value: Float32
    """Most recent value."""

    def __init__(out self, window_size: Int):
        """Initialize tracker with specified window size.

        Args:
            window_size: Number of values to keep for moving average.
        """
        self.window_size = window_size
        self.buffer = List[Float32]()
        for _ in range(window_size):
            self.buffer.append(0.0)

        self.buffer_idx = 0
        self.buffer_full = False

        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

        self.min_value = Float32(1e9)  # Start with large value
        self.max_value = Float32(-1e9)  # Start with small value
        self.last_value = 0.0

    def update(mut self, value: Float32):
        """Add new loss value and update statistics.

        Uses Welford's algorithm for numerically stable online variance computation.

        Args:
            value: New loss value to add.

        Reference:
            Welford, B. P. (1962). "Note on a method for calculating corrected sums.
            of squares and products". Technometrics. 4 (3): 419–420.
        """
        self.last_value = value

        # Update circular buffer for moving average
        self.buffer[self.buffer_idx] = value
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size

        if self.buffer_idx == 0:
            self.buffer_full = True

        # Welford's algorithm for running statistics
        self.count += 1
        var delta = Float64(value) - self.mean
        self.mean += delta / Float64(self.count)
        var delta2 = Float64(value) - self.mean
        self.m2 += delta * delta2

        # Update min/max
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

    def get_current(self) -> Float32:
        """Get most recent loss value.

        Returns:
            Most recent loss value.
        """
        return self.last_value

    def get_average(self) -> Float32:
        """Get moving average over window.

        Returns:
            Moving average of values in the circular buffer.
        """
        if self.count == 0:
            return 0.0

        var sum: Float32 = 0.0
        var n = self.window_size if self.buffer_full else self.buffer_idx

        for i in range(n):
            sum += self.buffer[i]

        return sum / Float32(n)

    def get_statistics(self) -> Statistics:
        """Get statistical summary (mean, std, min, max, count).

        Returns:
            Statistics struct with overall statistics (not just window).
        """
        var stats = Statistics()

        if self.count == 0:
            return stats^

        stats.mean = Float32(self.mean)
        stats.count = self.count
        stats.min = self.min_value
        stats.max = self.max_value

        # Standard deviation from Welford's m2
        if self.count > 1:
            var variance = self.m2 / Float64(self.count)
            stats.std = Float32(sqrt(variance))
        else:
            stats.std = 0.0

        return stats^

    def reset(mut self):
        """Reset all statistics and buffer."""
        for i in range(self.window_size):
            self.buffer[i] = 0.0

        self.buffer_idx = 0
        self.buffer_full = False
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_value = Float32(1e9)
        self.max_value = Float32(-1e9)
        self.last_value = 0.0


# ============================================================================
# Loss Tracker (#283-287)
# ============================================================================


struct LossTracker(Metric):
    """Track loss values with statistics and moving averages.

    Supports multi-component loss tracking (e.g., total, reconstruction, regularization)
    with separate statistics for each component

    Features:
    - Numerically stable variance via Welford's algorithm
    - Moving average with configurable window size
    - Multi-component tracking with independent statistics
    - Min/max tracking for each component

    Usage:
        var tracker = LossTracker(window_size=100)

        # Track total loss
        tracker.update(loss_value, component="total")

        # Track multiple components
        tracker.update(recon_loss, component="reconstruction")
        tracker.update(reg_loss, component="regularization")

        # Get statistics
        var stats = tracker.get_statistics(component="total")
        var avg = tracker.get_average(component="total")

    """

    var window_size: Int
    """Size of circular buffer for moving average."""
    var components: List[String]
    """Names of tracked loss components."""
    var trackers: List[ComponentTracker]
    """Tracker for each component."""

    def __init__(out self, window_size: Int = 100):
        """Initialize loss tracker.

        Args:
            window_size: Number of values to keep for moving average (default: 100).
        """
        self.window_size = window_size
        self.components = List[String]()
        self.trackers = []

    def _get_or_create_component(mut self, component: String) -> Int:
        """Get index of component tracker, creating if needed.

        Args:
            component: Component name.

        Returns:
            Index of component tracker in trackers list.
        """
        # Search for existing component
        for i in range(len(self.components)):
            if self.components[i] == component:
                return i

        # Create new component
        self.components.append(component)
        self.trackers.append(ComponentTracker(self.window_size))
        return len(self.components) - 1

    def update(mut self, loss: Float32, component: String = "total") raises:
        """Add new loss value for specified component.

        Args:
            loss: Loss value to track.
            component: Component name (default: "total").

        Raises:
            Error: If operation fails.
        """
        var idx = self._get_or_create_component(component)
        self.trackers[idx].update(loss)

    def get_current(self, component: String = "total") raises -> Float32:
        """Get most recent loss value for component.

        Args:
            component: Component name (default: "total").

        Returns:
            Most recent loss value, or 0.0 if component doesn't exist.

        Raises:
            Error: If operation fails.
        """
        for i in range(len(self.components)):
            if self.components[i] == component:
                return self.trackers[i].get_current()

        return 0.0

    def get_average(self, component: String = "total") raises -> Float32:
        """Get moving average for component.

        Args:
            component: Component name (default: "total").

        Returns:
            Moving average over window, or 0.0 if component doesn't exist.

        Raises:
            Error: If operation fails.
        """
        for i in range(len(self.components)):
            if self.components[i] == component:
                return self.trackers[i].get_average()

        return 0.0

    def get_statistics(self, component: String = "total") raises -> Statistics:
        """Get statistical summary for component.

        Args:
            component: Component name (default: "total").

        Returns:
            Statistics struct with mean, std, min, max, count.

        Raises:
            Error: If operation fails.
        """
        for i in range(len(self.components)):
            if self.components[i] == component:
                return self.trackers[i].get_statistics()

        return Statistics()

    def reset(mut self, component: String = ""):
        """Reset statistics for component(s).

        Args:
            component: Component name to reset, or "" to reset all (default: "").
        """
        if component == "":
            # Reset all components
            for i in range(len(self.trackers)):
                self.trackers[i].reset()
        else:
            # Reset specific component
            for i in range(len(self.components)):
                if self.components[i] == component:
                    self.trackers[i].reset()
                    return

    def list_components(self) -> List[String]:
        """Get list of all tracked components.

        Returns:
            Vector of component names (copy).
        """
        # Create a copy of the components list
        return List[String](self.components)

    # Metric trait implementation (for coordination interface)
    def update(mut self, predictions: AnyTensor, labels: AnyTensor) raises:
        """Update metric with predictions and labels (Metric trait).

        Note: LossTracker doesn't use predictions/labels directly.
        This method exists for trait compliance but should not be called
        Use update(loss: Float32, component: String) instead.

        Args:
            predictions: Unused.
            labels: Unused.

        Raises:
            Error: Indicating this method should not be used.
        """
        raise Error(
            "LossTracker.update(predictions, labels) not applicable - use"
            " update(loss, component) instead"
        )

    def reset(mut self):
        """Reset all components (Metric trait version).

        Resets statistics for all tracked components.
        """
        # Reset all components
        for i in range(len(self.trackers)):
            self.trackers[i].reset()
