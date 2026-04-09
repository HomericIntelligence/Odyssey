"""Progress bar utilities for training visualization.

This module provides visual feedback on training progress with support for:
- Simple progress bars with percentage and counts
- Metrics display (loss, accuracy, etc.)
- ETA estimation based on elapsed time
- Nested progress bars for epoch/batch tracking

The progress bars update in-place using carriage return (\r) to provide
real-time feedback without scrolling output.

Example:
    ```mojo
    from shared.utils import ProgressBar, ProgressBarWithMetrics

    var progress = ProgressBar(total=100, description="Training")
    for i in range(100):
        # ... training step ...
        progress.update()

    # With metrics
    var progress = ProgressBarWithMetrics(total=938, description="Epoch 1")
    progress.set_metric("loss", 0.342)
    progress.set_metric("acc", 0.891)
    progress.update()
    ```
"""

from std.collections import Dict
from std.time import perf_counter_ns


# ============================================================================
# ANSI Constants
# ============================================================================


comptime CLEAR_LINE = "\033[2K"
comptime CURSOR_HOME = "\r"
comptime RESET = "\033[0m"


# ============================================================================
# Helper Functions
# ============================================================================


def format_duration(seconds: Float64) -> String:
    """Format seconds as human-readable duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "2h 15m 30s", "15m 30s", or "30s".
    """
    var total_secs = Int(seconds)
    var hours = total_secs // 3600
    var minutes = (total_secs % 3600) // 60
    var secs = total_secs % 60

    var result = ""
    if hours > 0:
        result = result + String(hours) + "h "
    if minutes > 0:
        result = result + String(minutes) + "m "
    if secs >= 0:
        result = result + String(secs) + "s"

    # Remove trailing whitespace
    while len(result) > 0 and result.as_bytes()[-1] == UInt8(ord(" ")):
        result = String(result[byte=0:len(result)-1])

    return result


# ============================================================================
# ProgressBar
# ============================================================================


@fieldwise_init
struct ProgressBar(Copyable, Movable):
    """Simple progress bar with percentage and counter.

    Displays progress in format: "[=====>     ] 45% 450/1000"
    Updates in-place using carriage return for no-scroll output.
    """

    var total: Int
    var current: Int
    var width: Int
    var description: String
    var start_time_ns: Int

    def __init__(
        out self,
        total: Int,
        description: String = "",
        width: Int = 50,
    ):
        """Initialize progress bar.

        Args:
            total: Total number of items to process.
            description: Optional description to display.
            width: Width of the bar in characters.
        """
        self.total = total
        self.current = 0
        self.width = width
        self.description = description
        self.start_time_ns = Int(perf_counter_ns())

    def update(mut self, amount: Int = 1):
        """Update progress by given amount and render.

        Args:
            amount: Number of items to increment (default 1).
        """
        self.current = self.current + amount
        if self.current > self.total:
            self.current = self.total
        self._render()

    def set_total(mut self, total: Int):
        """Update total items and re-render.

        Args:
            total: New total count.
        """
        self.total = total
        self._render()

    def reset(mut self):
        """Reset progress bar to initial state."""
        self.current = 0
        self.start_time_ns = Int(perf_counter_ns())

    def _render(self):
        """Render progress bar to stdout with carriage return."""
        var output = CURSOR_HOME + self._format_bar()
        print(output, end="")

    def _format_bar(self) -> String:
        """Format the complete progress bar string."""
        var bar = self._format_bar_visual()
        var percent = self._format_percent()
        var progress = self._format_progress()

        var line = ""
        if self.description != "":
            line = line + self.description + " "

        line = line + bar + " " + percent + " " + progress
        return line

    def _format_bar_visual(self) -> String:
        """Create visual bar like [=====>     ]."""
        if self.total == 0:
            return "[]"

        var filled = (self.current * self.width) // self.total
        var empty = self.width - filled

        var bar = "["
        var i = 0
        while i < filled - 1:
            bar = bar + "="
            i = i + 1

        if filled > 0:
            bar = bar + ">"

        i = 0
        while i < empty:
            bar = bar + " "
            i = i + 1

        bar = bar + "]"
        return bar

    def _format_percent(self) -> String:
        """Format percentage string."""
        if self.total == 0:
            return "0%"

        var percent = (self.current * 100) // self.total
        return String(percent) + "%"

    def _format_progress(self) -> String:
        """Format progress counter string."""
        return String(self.current) + "/" + String(self.total)


# ============================================================================
# ProgressBarWithMetrics
# ============================================================================


@fieldwise_init
struct ProgressBarWithMetrics(Copyable, Movable):
    """Progress bar with real-time metric display.

    Extends ProgressBar to show metrics like loss and accuracy:
    "[=====>     ] 45% 450/1000 | loss=0.342 acc=0.891"
    """

    var progress: ProgressBar
    var metrics: Dict[String, Float32]

    def __init__(
        out self,
        total: Int,
        description: String = "",
        width: Int = 50,
    ):
        """Initialize progress bar with metrics support.

        Args:
            total: Total number of items to process.
            description: Optional description to display.
            width: Width of the bar in characters.
        """
        self.progress = ProgressBar(total, description, width)
        self.metrics = Dict[String, Float32]()

    def set_metric(mut self, name: String, value: Float32):
        """Set or update a metric value.

        Args:
            name: Metric name (e.g., "loss", "accuracy").
            value: Metric value.
        """
        self.metrics[name] = value

    def clear_metrics(mut self):
        """Clear all stored metrics."""
        self.metrics = Dict[String, Float32]()

    def update(mut self, amount: Int = 1):
        """Update progress and re-render with metrics.

        Args:
            amount: Number of items to increment (default 1).
        """
        self.progress.update(amount)
        self._render()

    def reset(mut self):
        """Reset progress bar and clear metrics."""
        self.progress.reset()
        self.clear_metrics()

    def set_total(mut self, total: Int):
        """Update total items.

        Args:
            total: New total count.
        """
        self.progress.set_total(total)

    def _render(self):
        """Render progress bar with metrics."""
        var output = CURSOR_HOME + self._format_bar()
        print(output, end="")

    def _format_bar(self) -> String:
        """Format progress bar with metrics appended."""
        var bar = self.progress._format_bar()

        # Append metrics if any exist
        if self.metrics.__len__() > 0:
            bar = bar + " |"
            var is_first = True
            for key in self.metrics.keys():
                var value = self.metrics.get(key, 0.0)

                if not is_first:
                    bar = bar + " "
                else:
                    bar = bar + " "
                    is_first = False

                bar = bar + key + "=" + self._format_metric_value(value)

        return bar

    def _format_metric_value(self, value: Float32) -> String:
        """Format metric value with 3 decimal places."""
        # Simple formatting: convert to Int for basic display
        var scaled = Int(value * 1000)
        var whole = scaled // 1000
        var decimals = scaled % 1000

        if decimals < 10:
            return String(whole) + ".00" + String(decimals)
        elif decimals < 100:
            return String(whole) + ".0" + String(decimals)
        else:
            return String(whole) + "." + String(decimals)


# ============================================================================
# ProgressBarWithETA
# ============================================================================


@fieldwise_init
struct ProgressBarWithETA(Copyable, Movable):
    """Progress bar with time estimation.

    Extends metrics bar with elapsed time and ETA:
    "[=====>     ] 45% | loss=0.342 | 2m 15s < 5m 30s ETA"
    """

    var progress_with_metrics: ProgressBarWithMetrics

    def __init__(
        out self,
        total: Int,
        description: String = "",
        width: Int = 50,
    ):
        """Initialize progress bar with ETA support.

        Args:
            total: Total number of items to process.
            description: Optional description to display.
            width: Width of the bar in characters.
        """
        self.progress_with_metrics = ProgressBarWithMetrics(
            total, description, width
        )

    def set_metric(mut self, name: String, value: Float32):
        """Set or update a metric value.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self.progress_with_metrics.set_metric(name, value)

    def clear_metrics(mut self):
        """Clear all stored metrics."""
        self.progress_with_metrics.clear_metrics()

    def update(mut self, amount: Int = 1):
        """Update progress and re-render with ETA.

        Args:
            amount: Number of items to increment (default 1).
        """
        self.progress_with_metrics.update(amount)
        self._render()

    def reset(mut self):
        """Reset progress bar, clear metrics, and reset timer."""
        self.progress_with_metrics.reset()

    def set_total(mut self, total: Int):
        """Update total items.

        Args:
            total: New total count.
        """
        self.progress_with_metrics.set_total(total)

    def _render(self):
        """Render progress bar with metrics and ETA."""
        var output = CURSOR_HOME + self._format_bar()
        print(output, end="")

    def _format_bar(self) -> String:
        """Format progress bar with metrics and ETA."""
        var base_bar = self.progress_with_metrics._format_bar()

        # Add elapsed and ETA times
        var elapsed = self._format_elapsed()
        var eta = self._format_eta()

        if elapsed != "" or eta != "":
            base_bar = base_bar + " |"
            if elapsed != "":
                base_bar = base_bar + " " + elapsed
            if eta != "":
                base_bar = base_bar + " < " + eta + " ETA"

        return base_bar

    def _format_elapsed(self) -> String:
        """Format elapsed time since start."""
        var current_time_ns = Int(perf_counter_ns())
        var elapsed_ns = (
            current_time_ns - self.progress_with_metrics.progress.start_time_ns
        )
        var elapsed_secs = Float64(elapsed_ns) / 1_000_000_000.0

        if elapsed_secs < 0.1:
            return ""

        return format_duration(elapsed_secs)

    def _format_eta(self) -> String:
        """Format estimated time to completion."""
        var current = self.progress_with_metrics.progress.current
        var total = self.progress_with_metrics.progress.total
        var start_time_ns = self.progress_with_metrics.progress.start_time_ns

        if current == 0 or total == 0:
            return ""

        var current_time_ns = Int(perf_counter_ns())
        var elapsed_ns = current_time_ns - start_time_ns
        var elapsed_secs = Float64(elapsed_ns) / 1_000_000_000.0

        var rate = Float64(current) / elapsed_secs
        var remaining = Float64(total - current)
        var eta_secs = remaining / rate

        if eta_secs < 0.0:
            return ""

        return format_duration(eta_secs)


# ============================================================================
# Factory Functions
# ============================================================================


def create_progress_bar(total: Int, description: String = "") -> ProgressBar:
    """Create a simple progress bar.

    Args:
        total: Total number of items.
        description: Optional description.

    Returns:
        New ProgressBar instance.
    """
    return ProgressBar(total, description)


def create_progress_bar_with_metrics(
    total: Int, description: String = ""
) -> ProgressBarWithMetrics:
    """Create a progress bar with metrics support.

    Args:
        total: Total number of items.
        description: Optional description.

    Returns:
        New ProgressBarWithMetrics instance.
    """
    return ProgressBarWithMetrics(total, description)


def create_progress_bar_with_eta(
    total: Int, description: String = ""
) -> ProgressBarWithETA:
    """Create a progress bar with metrics and ETA.

    Args:
        total: Total number of items.
        description: Optional description.

    Returns:
        New ProgressBarWithETA instance.
    """
    return ProgressBarWithETA(total, description)
