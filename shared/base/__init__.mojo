"""
Base Library - Zero-dependency modules for ML Odyssey.

This package contains modules with no tensor dependencies, providing
foundational utilities used by both shared/tensor/ and shared/core/.
Extracting these breaks the circular dependency between those packages.

Modules:
    memory_pool: Memory pool for small tensor allocations (stdlib only)
    broadcasting: NumPy-style broadcasting utilities (pure functions)
    dtype_ordinal: DType-to-ordinal mapping for dispatch (constants only)
    defaults: Default hyperparameters (constants only)
    math_constants: Mathematical constants (constants only)
    numerical_constants: Numerical stability constants (constants only)

Dependency graph:
    shared/base/ <- shared/tensor/ <- shared/core/  (clean DAG)
"""

# ============================================================================
# Memory Pool
# ============================================================================

from shared.base.memory_pool import (
    TensorMemoryPool,
    PoolConfig,
    PoolStats,
    FreeList,
    get_global_pool,
    pooled_alloc,
    pooled_free,
)

# ============================================================================
# Broadcasting Utilities
# ============================================================================

from shared.base.broadcasting import (
    broadcast_shapes,
    are_shapes_broadcastable,
    compute_broadcast_strides,
    BroadcastIterator,
)

# ============================================================================
# DType Ordinal Mapping
# ============================================================================

from shared.base.dtype_ordinal import (
    dtype_to_ordinal,
    format_dtype_name,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
    DTYPE_UNSUPPORTED,
    SUPPORTED_DTYPE_COUNT,
)

# ============================================================================
# Default Hyperparameters
# ============================================================================

from shared.base.defaults import (
    DEFAULT_LEAKY_RELU_ALPHA,
    DEFAULT_ELU_ALPHA,
    DEFAULT_HARD_TANH_MIN,
    DEFAULT_HARD_TANH_MAX,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_BATCHNORM_MOMENTUM,
    DEFAULT_UNIFORM_LOW,
    DEFAULT_UNIFORM_HIGH,
    DEFAULT_AUGMENTATION_PROB,
    DEFAULT_TEXT_AUGMENTATION_PROB,
    DEFAULT_RANDOM_SEED,
)

# ============================================================================
# Mathematical Constants
# ============================================================================

from shared.base.math_constants import (
    PI,
    SQRT_2,
    SQRT_2_OVER_PI,
    INV_SQRT_2PI,
    GELU_COEFF,
    LN2,
    LN10,
)

# ============================================================================
# Numerical Stability Constants
# ============================================================================

from shared.base.numerical_constants import (
    EPSILON_DIV,
    EPSILON_LOSS,
    EPSILON_NORM,
    GRADIENT_MAX_NORM,
    GRADIENT_MIN_NORM,
    EPSILON_OPTIMIZER_ADAM,
    EPSILON_OPTIMIZER_ADAGRAD,
    EPSILON_OPTIMIZER_RMSPROP,
    EPSILON_NUMERICAL_GRAD,
    EPSILON_RELATIVE_ERROR,
)
