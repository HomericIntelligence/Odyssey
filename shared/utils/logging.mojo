"""Logging utilities for ML Odyssey training and debugging.

This module provides structured logging capabilities with multiple handlers
and formatters. It supports file and console output with configurable
log levels.

Example:
    from shared.utils import Logger, StreamHandler, FileHandler

    var logger = Logger("training")
    logger.add_handler(StreamHandler())
    logger.add_handler(FileHandler("training.log"))

    logger.info("Training started")
    logger.warning("High loss detected")
    logger.error("Validation failed")
    ```
"""

from os.env import getenv


# ============================================================================
# Log Level Enumeration
# ============================================================================


struct LogLevel(Copyable, Movable):
    """Log level enumeration with numeric values for comparison.

    Levels are ordered from least (DEBUG) to most (CRITICAL) severe
    Log filtering uses numeric comparison: logger only outputs
    messages with level >= logger's configured level
    """

    comptime DEBUG = 10
    comptime INFO = 20
    comptime WARNING = 30
    comptime ERROR = 40
    comptime CRITICAL = 50


# ============================================================================
# Log Record
# ============================================================================


struct LogRecord(Copyable, Movable):
    """Record of a single log message.

    Contains all information needed by handlers to format and output
    a log message including the logger name, level, message, and timestamp
    """

    var logger_name: String
    var level: Int
    var message: String
    var timestamp: String

    fn __init__(
        out self,
        logger_name: String,
        level: Int,
        message: String,
        timestamp: String = "",
    ):
        """Initialize a log record.

        Args:
            logger_name: Name of the logger that created this record.
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: The log message.
            timestamp: Optional timestamp string.
        """
        self.logger_name = logger_name
        self.level = level
        self.message = message
        # Use provided timestamp or empty string (Mojo lacks stdlib time support)
        self.timestamp = timestamp if timestamp else ""

    fn level_name(self) -> String:
        """Get human-readable level name."""
        if self.level == LogLevel.DEBUG:
            return "DEBUG"
        elif self.level == LogLevel.INFO:
            return "INFO"
        elif self.level == LogLevel.WARNING:
            return "WARNING"
        elif self.level == LogLevel.ERROR:
            return "ERROR"
        elif self.level == LogLevel.CRITICAL:
            return "CRITICAL"
        else:
            return "UNKNOWN"


# ============================================================================
# Formatter Trait and Implementations
# ============================================================================


trait Formatter:
    """Base formatter interface for log messages."""

    fn format(self, record: LogRecord) -> String:
        """Format a log record into a string."""
        ...


@fieldwise_init
struct SimpleFormatter(Copyable, Formatter, ImplicitlyCopyable, Movable):
    """Simple formatter: [LEVEL] message."""

    fn format(self, record: LogRecord) -> String:
        """Format log record as: [LEVEL] message."""
        return "[" + record.level_name() + "] " + record.message


@fieldwise_init
struct TimestampFormatter(Copyable, Formatter, ImplicitlyCopyable, Movable):
    """Timestamp formatter: YYYY-MM-DD HH:MM:SS [LEVEL] message."""

    fn format(self, record: LogRecord) -> String:
        """Format log record with timestamp."""
        return (
            record.timestamp
            + " ["
            + record.level_name()
            + "] "
            + record.message
        )


@fieldwise_init
struct DetailedFormatter(Copyable, Formatter, ImplicitlyCopyable, Movable):
    """Detailed formatter: [LEVEL] logger_name - message."""

    fn format(self, record: LogRecord) -> String:
        """Format log record with logger name."""
        return (
            "["
            + record.level_name()
            + "] "
            + record.logger_name
            + " - "
            + record.message
        )


@fieldwise_init
struct ColoredFormatter(Copyable, Formatter, ImplicitlyCopyable, Movable):
    """Colored formatter using ANSI escape codes."""

    # ANSI color codes
    comptime RED = "\033[91m"
    comptime YELLOW = "\033[93m"
    comptime GREEN = "\033[92m"
    comptime BLUE = "\033[94m"
    comptime RESET = "\033[0m"

    fn format(self, record: LogRecord) -> String:
        """Format log record with ANSI color codes."""
        var color = self._get_color(record.level)
        return (
            color
            + "["
            + record.level_name()
            + "]"
            + self.RESET
            + " "
            + record.message
        )

    fn _get_color(self, level: Int) -> String:
        """Get ANSI color for level."""
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            return self.RED
        elif level == LogLevel.WARNING:
            return self.YELLOW
        elif level == LogLevel.INFO:
            return self.GREEN
        else:  # DEBUG.
            return self.BLUE


# ============================================================================
# Handler Type and Wrapper (for polymorphic handler support)
# ============================================================================


struct HandlerType:
    """Enumeration of handler types."""

    comptime STREAM = 0
    comptime FILE = 1


struct HandlerWrapper(Copyable, Movable):
    """Wrapper to support multiple handler types in a list.

    Since Mojo doesn't support trait objects in lists, this wrapper
    holds either a StreamHandler or FileHandler and dispatches to
    the appropriate one.
    """

    var handler_type: Int
    var stream_handler: StreamHandler
    var file_handler: FileHandler

    fn __init__(out self, handler: StreamHandler):
        """Create wrapper for StreamHandler.

        Args:
            handler: StreamHandler to wrap.
        """
        self.handler_type = HandlerType.STREAM
        self.stream_handler = handler
        self.file_handler = FileHandler("")

    fn __init__(out self, handler: FileHandler):
        """Create wrapper for FileHandler.

        Args:
            handler: FileHandler to wrap.
        """
        self.handler_type = HandlerType.FILE
        self.stream_handler = StreamHandler()
        self.file_handler = handler

    fn emit(self, record: LogRecord):
        """Dispatch emit to appropriate handler."""
        if self.handler_type == HandlerType.STREAM:
            self.stream_handler.emit(record)
        else:
            self.file_handler.emit(record)


# ============================================================================
# Handler Trait and Implementations
# ============================================================================


trait Handler:
    """Base handler interface for log output."""

    fn emit(self, record: LogRecord):
        """Output a log record."""
        ...


struct StreamHandler(Copyable, Handler, ImplicitlyCopyable, Movable):
    """Write log messages to stdout/stderr."""

    var formatter: SimpleFormatter

    fn __init__(out self):
        """Create stream handler with default formatter."""
        self.formatter = SimpleFormatter()

    fn emit(self, record: LogRecord):
        """Write formatted log record to stdout."""
        var formatted = self.formatter.format(record)
        print(formatted)


struct FileHandler(Copyable, Handler, ImplicitlyCopyable, Movable):
    """Write log messages to a file."""

    var filepath: String
    var formatter: TimestampFormatter

    fn __init__(out self, filepath: String):
        """Create file handler that writes to given file.

        Args:
            filepath: Path to log file to write to.
        """
        self.filepath = filepath
        self.formatter = TimestampFormatter()

    fn emit(self, record: LogRecord):
        """Write formatted log record to file."""
        var formatted = self.formatter.format(record)
        self._write_to_file(formatted)

    fn _write_to_file(self, message: String):
        """Write message to log file (append mode)

        Opens file in append mode, writes the message with newline,
        and closes the file. If file cannot be opened, falls back
        to print.

        Args:
            message: Formatted log message to write.
        """
        try:
            # Open file in append mode
            with open(self.filepath, "a") as file:
                _ = file.write(message + "\n")
        except e:
            # Fallback to print if file write fails
            print("[LOG ERROR] Failed to write to " + self.filepath)
            print(message)


# ============================================================================
# Logger Class
# ============================================================================


struct Logger:
    """Structured logger with multiple output handlers.

    Supports configurable log levels, multiple handlers, and various
    formatters for different output formats. Log messages are filtered
    by the configured level threshold
    """

    var name: String
    var level: Int
    var handlers: List[HandlerWrapper]

    fn __init__(out self, name: String, level: Int = LogLevel.INFO):
        """Create logger with name and optional level.

        Args:
            name: Logger name (e.g., "training", "evaluation").
            level: Minimum log level to output (default: INFO).
        """
        self.name = name
        self.level = level
        self.handlers: List[HandlerWrapper] = []

    fn add_handler(mut self, handler: StreamHandler):
        """Add a stream handler to this logger.

        Handlers receive all log records that pass the level filter.

        Args:
            handler: StreamHandler to add.
        """
        self.handlers.append(HandlerWrapper(handler))

    fn add_handler(mut self, handler: FileHandler):
        """Add a file handler to this logger.

        Handlers receive all log records that pass the level filter.

        Args:
            handler: FileHandler to add.
        """
        self.handlers.append(HandlerWrapper(handler))

    fn debug(self, message: String):
        """Log a debug message (lowest priority).

        Args:
            message: Message to log.
        """
        if self.level <= LogLevel.DEBUG:
            self._log(LogLevel.DEBUG, message)

    fn info(self, message: String):
        """Log an info message (normal priority).

        Args:
            message: Message to log.
        """
        if self.level <= LogLevel.INFO:
            self._log(LogLevel.INFO, message)

    fn warning(self, message: String):
        """Log a warning message (medium priority).

        Args:
            message: Message to log.
        """
        if self.level <= LogLevel.WARNING:
            self._log(LogLevel.WARNING, message)

    fn error(self, message: String):
        """Log an error message (high priority).

        Args:
            message: Message to log.
        """
        if self.level <= LogLevel.ERROR:
            self._log(LogLevel.ERROR, message)

    fn critical(self, message: String):
        """Log a critical message (highest priority).

        Args:
            message: Message to log.
        """
        if self.level <= LogLevel.CRITICAL:
            self._log(LogLevel.CRITICAL, message)

    fn _log(self, level: Int, message: String):
        """Internal method to create and emit log record.

        Args:
            level: Log level for this message.
            message: Message to log.
        """
        var record = LogRecord(self.name, level, message)
        for handler in self.handlers:
            handler.emit(record)

    fn set_level(mut self, level: Int):
        """Change the log level for this logger.

        Args:
            level: New log level threshold.
        """
        self.level = level


# ============================================================================
# Module-level Functions
# ============================================================================


fn get_log_level_from_env() -> Int:
    """Get log level from ML_ODYSSEY_LOG_LEVEL environment variable.

    Parses the ML_ODYSSEY_LOG_LEVEL environment variable to determine
    the global log level. Defaults to INFO if not set or invalid.

    Supported values (case-insensitive):
    - "DEBUG" → LogLevel.DEBUG (10)
    - "INFO" → LogLevel.INFO (20)
    - "WARNING" → LogLevel.WARNING (30)
    - "ERROR" → LogLevel.ERROR (40)
    - "CRITICAL" → LogLevel.CRITICAL (50)

    Returns:
        Log level integer (default: INFO if not set).

    Note:
        Only works on macOS and Linux. Returns INFO on other platforms.
    """
    # Get environment variable value (empty string if not set)
    var log_level_str = getenv("ML_ODYSSEY_LOG_LEVEL", "")

    # Handle empty/unset case
    if len(log_level_str) == 0:
        return LogLevel.INFO

    # Convert to uppercase for case-insensitive matching
    var upper_str = log_level_str.upper()

    # Parse log level string
    if upper_str == "DEBUG":
        return LogLevel.DEBUG
    elif upper_str == "INFO":
        return LogLevel.INFO
    elif upper_str == "WARNING" or upper_str == "WARN":
        return LogLevel.WARNING
    elif upper_str == "ERROR":
        return LogLevel.ERROR
    elif upper_str == "CRITICAL":
        return LogLevel.CRITICAL
    else:
        # Invalid value, default to INFO
        return LogLevel.INFO


fn get_logger(name: String, level: Int = LogLevel.INFO) -> Logger:
    """Get or create a named logger.

    Creates a new logger with the specified name and level.
    Note: In this implementation, each call creates a new logger
    instance. For logger caching, users should store the logger
    reference instead of calling get_logger multiple times.

    Args:
        name: Logger name (e.g., "training", "data").
        level: Log level threshold (default: INFO).

    Returns:
        Logger with specified name and level.

    Example:
        ```mojo
        var logger = get_logger("training")
        logger.info("Training started")
        ```
    """
    # Create new logger with provided level
    return Logger(name, level)


fn set_global_log_level(level: Int):
    """Set global log level for all registered loggers.

    Note: This function is a placeholder. In the current implementation
    without global state, you should set levels directly on individual
    logger instances using logger.set_level(level).

    Args:
        level: New global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Placeholder for future global configuration
    pass
