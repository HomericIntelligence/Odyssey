"""Training interruption and timeout management.

Provides wall-clock timeout tracking and graceful shutdown support for training
operations. Allows training loops to:
- Set maximum training duration (wall-clock timeout)
- Interrupt cleanly mid-epoch on SIGINT/SIGTERM while saving a checkpoint
- Track the reason for training termination

Design principles:
- Check timeout and shutdown state at epoch boundaries (no mid-batch checks)
- Atomic checkpoint saves before exiting
- Distinguish between timeout, signal, and normal completion

Note: Mojo 1.0 does not support global variables. Shutdown flag is managed
through function state. For proper signal handling in production, use OS-level
mechanisms (atexit, signal handlers in C).
"""

from std.time import perf_counter_ns


struct ShutdownReason(TrivialRegisterPassable):
    """Enumeration of reasons training was terminated.

    Variants:
        COMPLETED: Training finished all epochs normally.
        TIMEOUT: Training exceeded max_wall_time_seconds.
        SIGNAL: Training interrupted by SIGINT/SIGTERM.
        MAX_EPOCHS: Stopped after reaching configured epoch count.
    """

    var _value: Int

    def __init__(out self, value: Int = 0):
        self._value = value

    @staticmethod
    def completed() -> ShutdownReason:
        return ShutdownReason(0)

    @staticmethod
    def timeout() -> ShutdownReason:
        return ShutdownReason(1)

    @staticmethod
    def signal() -> ShutdownReason:
        return ShutdownReason(2)

    @staticmethod
    def max_epochs() -> ShutdownReason:
        return ShutdownReason(3)

    def __eq__(self, other: ShutdownReason) -> Bool:
        return self._value == other._value

    def to_string(self) -> String:
        if self._value == 0:
            return "COMPLETED"
        elif self._value == 1:
            return "TIMEOUT"
        elif self._value == 2:
            return "SIGNAL"
        elif self._value == 3:
            return "MAX_EPOCHS"
        return "UNKNOWN"


struct WallClockTimer:
    """Tracks elapsed wall-clock time since training started.

    Provides monotonic time-based timeout checking suitable for long-running
    training operations. Uses perf_counter_ns() for stable measurements.

    Attributes:
        start_ns: Nanoseconds since start of timer.
    """

    var start_ns: UInt

    def __init__(out self):
        """Initialize timer to current time."""
        self.start_ns = perf_counter_ns()

    def elapsed_seconds(self) -> Float32:
        """Get elapsed time in seconds since timer start.

        Returns:
            Elapsed seconds as float.
        """
        var now_ns = perf_counter_ns()
        var elapsed_ns = now_ns - self.start_ns
        return Float32(Int(elapsed_ns)) / 1_000_000_000.0

    def has_elapsed(self, limit_seconds: Int) -> Bool:
        """Check if elapsed time exceeds limit.

        Args:
            limit_seconds: Maximum allowed seconds (0 = no limit).

        Returns:
            True if elapsed time >= limit_seconds, False otherwise.
            Always returns False if limit_seconds is 0 (no timeout).
        """
        if limit_seconds <= 0:
            return False
        return Int(self.elapsed_seconds()) >= limit_seconds


struct TrainingResult:
    """Result of a training run, including termination reason and state.

    Captures the final state of a training operation to allow callers to
    determine why training stopped and what state can be resumed from.

    Attributes:
        stopped_epoch: The last epoch that completed (0-indexed). Can be used
                      to resume from stopped_epoch + 1.
        reason: The ShutdownReason explaining why training stopped.
        checkpoint_path: File path to the saved checkpoint, if any.
        elapsed_seconds: Total wall-clock seconds spent training.
    """

    var stopped_epoch: Int
    var reason: ShutdownReason
    var checkpoint_path: String
    var elapsed_seconds: Float32

    def __init__(
        out self,
        stopped_epoch: Int,
        reason: ShutdownReason,
        checkpoint_path: String = "",
        elapsed_seconds: Float32 = 0.0,
    ):
        """Initialize training result.

        Args:
            stopped_epoch: Last completed epoch (0-indexed).
            reason: The ShutdownReason explaining termination.
            checkpoint_path: Path to saved checkpoint (empty if none).
            elapsed_seconds: Total training time in seconds.
        """
        self.stopped_epoch = stopped_epoch
        self.reason = reason
        self.checkpoint_path = checkpoint_path
        self.elapsed_seconds = elapsed_seconds

    def to_string(self) -> String:
        var result = String()
        result += "TrainingResult:\n"
        result += "  stopped_epoch: " + String(self.stopped_epoch) + "\n"
        result += "  reason: " + self.reason.to_string() + "\n"
        result += "  checkpoint_path: " + self.checkpoint_path + "\n"
        result += "  elapsed_seconds: " + String(self.elapsed_seconds)
        return result


struct ShutdownFlag:
    """Global shutdown request flag.

    Used to communicate shutdown signals from signal handlers to the training
    loop. This is a wrapper around module-level mutable state to enable
    controlled access.

    Note: In Mojo 1.0, global variables are not supported. This struct
    provides an interface for future signal handling integration.
    """

    var _requested: Bool

    def __init__(out self, requested: Bool = False):
        self._requested = requested

    def is_set(self) -> Bool:
        return self._requested

    def set(mut self):
        self._requested = True

    def reset(mut self):
        self._requested = False


# Placeholder functions for shutdown flag management
# In a production system, these would integrate with OS signal handlers
def request_shutdown() -> None:
    """Signal that training should gracefully shutdown.

    Sets the shutdown flag to True, which training loops will check
    at epoch boundaries. Safe to call multiple times.

    Note: In Mojo 1.0, global state is not directly supported.
    For production use, integrate with OS signal handlers (SIGINT/SIGTERM).
    """
    pass


def is_shutdown_requested() -> Bool:
    """Check if a shutdown request has been signaled.

    Returns:
        True if request_shutdown() has been called, False otherwise.

    Note: In Mojo 1.0, this always returns False. For production use,
    integrate with OS signal handlers.
    """
    return False


def reset_shutdown_flag() -> None:
    """Reset the shutdown flag to False.

    Used primarily for test isolation. Should not be called during normal
    training operations.

    Note: In Mojo 1.0, this is a no-op since global state is not supported.
    """
    pass
