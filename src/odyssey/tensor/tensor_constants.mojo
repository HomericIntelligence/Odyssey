"""Tensor constants shared across tensor modules.

Provides memory limits and size constants used by tensor creation, validation,
and utility functions.
"""

# Maximum number of bytes a single tensor may allocate (2 GB)
comptime MAX_TENSOR_BYTES: Int = 2_000_000_000

# Warning threshold for tensor size (500 MB) - used in logging/diagnostics
comptime WARN_TENSOR_BYTES: Int = 500_000_000
