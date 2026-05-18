"""CSV-based metrics logger for training visualization.

Logs training metrics to CSV files for offline analysis and plotting.
Implements the Callback trait for seamless training loop integration.

Example:
    var logger = CSVMetricsLogger("logs/lenet5_run1")

    # In training loop (manual usage)
    logger.log_scalar("train_loss", 0.5)
    logger.log_scalar("train_accuracy", 0.85)
    logger.step()

    # At end
    _ = logger.save()

    # Or use as callback
    var state = TrainingState(...)
    state.metrics["train_loss"] = 0.5
    _ = logger.on_epoch_end(state)  # Auto-logs all metrics

Features:
- Simple CSV format (step,value) compatible with pandas/Excel
- Per-run directory for isolating training runs
- Incremental saves at each epoch
- Automatic integration via Callback trait
"""

from std.collections import Dict, List
from projectodyssey.training.base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    TrainingState,
)
from projectodyssey.utils.file_io import (
    create_directory,
    safe_write_file,
    file_exists,
)


struct CSVMetricsLogger(Callback, Copyable, Movable):
    """Logger for CSV-based training metric persistence.

    Attributes:
        log_dir: Directory for storing CSV files.
        metrics: Dictionary mapping metric names to value history.
        step_counter: Global step counter for batch-level logging.
        epoch_counter: Epoch counter for epoch-level logging.
        initialized: Whether logger has been initialized.
    """

    var log_dir: String
    var metrics: Dict[String, List[Float64]]
    var step_counter: Int
    var epoch_counter: Int
    var initialized: Bool

    def __init__(out self, log_dir: String):
        """Initialize metrics logger with output directory.

        Args:
            log_dir: Directory path for CSV output files.
        """
        self.log_dir = log_dir
        self.metrics = Dict[String, List[Float64]]()
        self.step_counter = 0
        self.epoch_counter = 0
        self.initialized = False

    def log_scalar(mut self, name: String, value: Float64) raises:
        """Log a single scalar metric value.

        Args:
            name: Name of the metric (e.g., "train_loss").
            value: Scalar value to log.

        Raises:
            Error: If dictionary operation fails.
        """
        if name not in self.metrics:
            self.metrics[name] = List[Float64]()
        self.metrics[name].append(value)

    def log_from_state(mut self, state: TrainingState) raises:
        """Log all metrics from TrainingState.

        Args:
            state: Training state containing metrics dictionary.

        Raises:
            Error: If logging fails.
        """
        for ref item in state.metrics.items():
            self.log_scalar(item.key, item.value)

    def step(mut self):
        """Increment step counter."""
        self.step_counter += 1

    def save(self) -> Bool:
        """Save all metrics to CSV files.

        Returns:
            True if all saves succeeded, False otherwise.
        """
        # Ensure directory exists
        if not create_directory(self.log_dir):
            return False

        var all_success = True

        for ref item in self.metrics.items():
            var filename = self.log_dir + "/" + item.key + ".csv"
            var content = self._build_csv(item.key, item.value)
            if not safe_write_file(filename, content):
                all_success = False

        return all_success

    def _build_csv(self, name: String, values: List[Float64]) -> String:
        """Build CSV content string for a metric.

        Args:
            name: Metric name (for comments).
            values: List of metric values.

        Returns:
            CSV formatted string with header and values.
        """
        var result = "step,value\n"
        for i in range(len(values)):
            result += String(i) + "," + String(values[i]) + "\n"
        return result

    # Callback trait implementation
    def on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Initialize logger at training start."""
        self.initialized = create_directory(self.log_dir)
        return CONTINUE

    def on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Save all metrics at training end."""
        _ = self.save()
        return CONTINUE

    def on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin."""
        return CONTINUE

    def on_epoch_end(
        mut self, mut state: TrainingState
    ) raises -> CallbackSignal:
        """Log metrics at epoch end and save."""
        self.log_from_state(state)
        self.epoch_counter += 1
        _ = self.save()  # Incremental save
        return CONTINUE

    def on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch begin."""
        return CONTINUE

    def on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Optionally log batch-level metrics."""
        self.step_counter += 1
        return CONTINUE
