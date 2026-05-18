"""
Core Library - Pure functional operations for ML Odyssey.

This package provides pure functional operations for neural networks and mathematical
operations. All functions are stateless - they process inputs to produce outputs
without maintaining internal state.

Architecture:
    - Pure functional design - no classes, no internal state
    - All functions work with AnyTensor
    - Caller manages all state (weights, biases, momentum, etc.)
    - Functions return new values, never mutate inputs

Modules:
    any_tensor: Core tensor type (AnyTensor, implements Hashable via __hash__) and creation functions
    types: Custom data types (type aliases for FP8/BF8/BF16/FP4, MXFP4/NVFP4 blocked formats)
    arithmetic: Element-wise arithmetic operations (add, subtract, multiply, divide)
    matrix: Matrix operations (matmul, transpose, dot, outer)
    any_tensor methods: AnyTensor method operations (transpose method via AnyTensor.transpose())
    activation: Activation functions (relu, sigmoid, tanh, softmax, gelu)
    activation_ops: Activation operation utilities (scalar exp functions)
    linear: Linear transformations
    conv: Convolutional operations
    pooling: Pooling operations
    elementwise: Element-wise math functions (exp, log, sqrt, abs, clip)
    comparison: Comparison operations (equal, less, greater)
    broadcasting: Broadcasting utilities
    initializers: Weight initialization functions
    loss: Loss functions
    loss_utils: Utility functions for loss computation (clipping, epsilon handling, blending)
    numerical_safety: NaN/Inf detection, gradient monitoring, numerical stability checks
    dtype_dispatch: Generic dtype dispatch helpers for eliminating dtype branching
    reduction: Reduction operations (sum, mean, max, min) with forward and backward passes
    reduction_utils: Utility functions for reduction operations (coordinate/stride computation)
    utils: Utility functions (argmax, top_k_indices, top_k, argsort)
    scalar_ops: Scalar mathematical operations (sqrt, pow for float32 and float64)
    normalize_ops: Normalization operations for data preprocessing (RGB normalization)

Note:
    Mojo v0.26.1+ automatically exports all imported symbols to package consumers.
    No ``__all__`` equivalent is needed — any symbol imported in this file is
    automatically available to users of ``projectodyssey.core``. See issue #3751.

    Re-exports from ``projectodyssey.core`` work cleanly with no chain limitation.
    Callers may import directly from the parent package:

    ```mojo
    from projectodyssey.core import AnyTensor, linear, relu
    ```

    The chain limitation described in #3210 only applies to importing from the
    ``shared`` top-level package (e.g. ``from projectodyssey import AnyTensor``), not from
    ``projectodyssey.core`` or its submodules directly.

Example:
   ```mojo
    from projectodyssey.tensor.any_tensor import AnyTensor, zeros
    from projectodyssey.core.linear import linear
    from projectodyssey.core.activation import relu
    from projectodyssey.core.matrix import matmul, transpose

    # Create tensors
    var x = zeros([32, 784])
    var weights = zeros([128, 784])
    var bias = zeros([128])

    # Forward pass (pure functional)
    var h1 = linear(x, weights, bias)
    var a1 = relu(h1)
    ```

Import tests in tests/shared/test_imports.mojo are implemented and passing.
See Issue #3033: 4 tests for core module imports — all tests pass.
"""

# Package version
from projectodyssey.version import VERSION

# ============================================================================
# Default Hyperparameters
# ============================================================================

from projectodyssey.base.defaults import (
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

from projectodyssey.base.math_constants import (
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

from projectodyssey.base.numerical_constants import (
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

# ============================================================================
# Activation Function Constants
# ============================================================================

from projectodyssey.core.activation_constants import (
    RELU6_UPPER_BOUND,
    SIGMOID_CLIP_THRESHOLD,
    HARD_SIGMOID_OFFSET,
    HARD_SIGMOID_SCALE,
    HARD_TANH_LOWER_BOUND,
    HARD_TANH_UPPER_BOUND,
)

# ============================================================================
# Optimizer Default Hyperparameters
# ============================================================================

from projectodyssey.core.optimizer_constants import (
    DEFAULT_LEARNING_RATE_SGD,
    DEFAULT_LEARNING_RATE_ADAM,
    DEFAULT_MOMENTUM,
    DEFAULT_ADAM_BETA1,
    DEFAULT_ADAM_BETA2,
    DEFAULT_ADAM_EPSILON,
    DEFAULT_RMSPROP_ALPHA,
    DEFAULT_RMSPROP_EPSILON,
    DEFAULT_ADAGRAD_EPSILON,
)

# ============================================================================
# Core Tensor Type and Creation Functions
# ============================================================================
# AnyTensor and factory functions (zeros, ones, etc.) are now in projectodyssey.tensor.any_tensor.
# Import directly: from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
# NOT re-exported here to avoid circular imports between projectodyssey.core and projectodyssey.tensor.

# ============================================================================
# Shape Manipulation Operations
# ============================================================================

from projectodyssey.core.shape import (
    reshape,
    squeeze,
    unsqueeze,
    expand_dims,
    flatten,
    ravel,
    concatenate,
    stack,
    split,
    split_with_indices,
    tile,
    repeat,
    permute,
    is_contiguous,
    as_contiguous,
    view,
    conv2d_output_shape,
    pool_output_shape,
    flatten_size,
    flatten_to_2d,
    transposed_conv2d_output_shape,
    global_avgpool_output_shape,
    linear_output_shape,
    broadcast_to,
)

# ============================================================================
# Custom Data Types (Type Aliases)
# ============================================================================

from projectodyssey.core.types.dtype_aliases import BF16, FP8, BF8, FP4, E8M0

# ============================================================================
# Gradient Container Types
# ============================================================================

from projectodyssey.core.gradient_types import (
    GradientPair,
    GradientTriple,
    GradientQuad,
    Conv2dNoBiasGradient,
    DepthwiseConv2dNoBiasGradient,
)

# ============================================================================
# Arithmetic Operations
# ============================================================================

from projectodyssey.core.arithmetic import (
    add,
    subtract,
    multiply,
    divide,
    floor_divide,
    modulo,
    power,
    multiply_scalar,
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)

# ============================================================================
# Matrix Operations
# ============================================================================

from projectodyssey.core.matrix import (
    matmul,
    transpose,
    transpose_view,
    dot,
    outer,
    matmul_backward,
    transpose_backward,
)

# ============================================================================
# Optimized Matrix Multiplication Kernels
# ============================================================================

from projectodyssey.core.matmul import (
    matmul_optimized,
    matmul_tiled,
    matmul_simd,
    matmul_typed,
)

from projectodyssey.core.strassen import (
    matmul_strassen,
    STRASSEN_ENABLED,
    STRASSEN_THRESHOLD,
    next_power_of_2,
)

# ============================================================================
# Activation Functions
# ============================================================================

from projectodyssey.core.activation import (
    relu,
    leaky_relu,
    prelu,
    sigmoid,
    tanh,
    softmax,
    gelu,
    swish,
    mish,
    elu,
    selu,
    hard_sigmoid,
    hard_swish,
    hard_tanh,
    relu_backward,
    leaky_relu_backward,
    prelu_backward,
    sigmoid_backward,
    tanh_backward,
    gelu_backward,
    softmax_backward,
    swish_backward,
    mish_backward,
    elu_backward,
    selu_backward,
    hard_sigmoid_backward,
    hard_swish_backward,
    hard_tanh_backward,
)

from projectodyssey.core.activation_ops import (
    exp_scalar_f32,
    exp_scalar_f64,
)

from projectodyssey.core.activation_simd import (
    relu_simd,
    leaky_relu_simd,
    relu6_simd,
    elu_simd,
    selu_simd,
    swish_simd,
)

# ============================================================================
# Neural Network Operations
# ============================================================================

from projectodyssey.core.linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
)

from projectodyssey.core.conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
    depthwise_conv2d,
    depthwise_conv2d_no_bias,
    depthwise_conv2d_backward,
    depthwise_conv2d_no_bias_backward,
    depthwise_separable_conv2d,
    depthwise_separable_conv2d_no_bias,
    depthwise_separable_conv2d_backward,
    depthwise_separable_conv2d_no_bias_backward,
)

from projectodyssey.core.pooling import (
    maxpool2d,
    avgpool2d,
    global_avgpool2d,
    maxpool2d_backward,
    avgpool2d_backward,
    global_avgpool2d_backward,
)

from projectodyssey.core.dropout import (
    dropout,
    dropout2d,
    dropout_backward,
    dropout2d_backward,
)

from projectodyssey.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
    layer_norm,
    layer_norm_backward,
    group_norm,
    group_norm_backward,
    instance_norm,
    instance_norm_backward,
)

from projectodyssey.core.normalization_simd import (
    batch_norm2d_fused,
    batch_norm2d_fused_inference,
)

from projectodyssey.core.normalize_ops import normalize_rgb

from projectodyssey.core.scalar_ops import (
    sqrt_scalar_f32,
    sqrt_scalar_f64,
    pow_scalar_f32,
    pow_scalar_f64,
)

# ============================================================================
# Attention Mechanisms
# ============================================================================

from projectodyssey.core.attention import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_masked,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_backward_masked,
    ScaledDotProductAttentionBackwardResult,
    create_causal_mask,
    multi_head_attention,
    multi_head_attention_masked,
    multi_head_attention_backward,
    MultiHeadAttentionWeights,
    MultiHeadAttentionResult,
    MultiHeadAttentionBackwardResult,
)

# ============================================================================
# Element-wise Operations
# ============================================================================

from projectodyssey.core.elementwise import (
    abs,
    sign,
    exp,
    log,
    sqrt,
    sin,
    cos,
    clip,
    ceil,
    floor,
    round,
    trunc,
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    log10,
    log2,
    exp_backward,
    log_backward,
    sqrt_backward,
    sin_backward,
    cos_backward,
    abs_backward,
    clip_backward,
    log10_backward,
    log2_backward,
)

# ============================================================================
# Comparison Operations
# ============================================================================

from projectodyssey.core.comparison import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)

# ============================================================================
# Broadcasting Utilities
# ============================================================================

from projectodyssey.base.broadcasting import (
    broadcast_shapes,
    are_shapes_broadcastable,
    compute_broadcast_strides,
    BroadcastIterator,
)

# ============================================================================
# Initialization Functions
# ============================================================================

from projectodyssey.core.initializers import (
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    he_uniform,
    he_normal,
    uniform,
    normal,
    constant,
)

# ============================================================================
# Loss Functions
# ============================================================================

from projectodyssey.core.loss import (
    binary_cross_entropy,
    mean_squared_error,
    cross_entropy,
    smooth_l1_loss,
    hinge_loss,
    focal_loss,
    kl_divergence,
    binary_cross_entropy_backward,
    mean_squared_error_backward,
    cross_entropy_backward,
    smooth_l1_loss_backward,
    hinge_loss_backward,
    focal_loss_backward,
    kl_divergence_backward,
)

from projectodyssey.core.loss_utils import (
    clip_predictions,
    create_epsilon_tensor,
    validate_tensor_shapes,
    validate_tensor_dtypes,
    compute_one_minus_tensor,
    compute_sign_tensor,
    blend_tensors,
    compute_max_stable,
    compute_difference,
    compute_product,
    compute_ratio,
    negate_tensor,
)

from projectodyssey.core.numerical_safety import (
    has_nan,
    has_inf,
    count_nan,
    count_inf,
    check_tensor_safety,
    tensor_min,
    tensor_max,
    check_tensor_range,
    compute_tensor_l2_norm,
    check_gradient_norm,
    check_gradient_vanishing,
    check_gradient_safety,
    clip_grad_value_,
    clip_grad_norm_,
    clip_grad_global_norm_,
)

# ============================================================================
# Dtype Dispatch Helpers
# ============================================================================

from projectodyssey.core.dtype_dispatch import (
    dispatch_unary,
    dispatch_binary,
    dispatch_scalar,
    dispatch_float_unary,
    dispatch_float_binary,
    dispatch_float_scalar,
    dispatch_softmax,
    dispatch_softmax_backward,
    dispatch_gelu,
    dispatch_gelu_backward,
    dispatch_hard_sigmoid,
    dispatch_hard_sigmoid_backward,
    dispatch_hard_swish,
    dispatch_hard_swish_backward,
    dispatch_hard_tanh,
    dispatch_hard_tanh_backward,
)

# ============================================================================
# Reduction Operations
# ============================================================================

from projectodyssey.core.reduction import (
    sum,
    mean,
    max_reduce,
    min_reduce,
    variance,
    std_reduce,
    median,
    percentile,
    sum_backward,
    mean_backward,
    max_reduce_backward,
    min_reduce_backward,
    variance_backward,
    std_backward,
    median_backward,
    percentile_backward,
)

from projectodyssey.core.reduction_ops import (
    ReduceOp,
    ReduceBackwardOp,
    SumOp,
    MeanOp,
    MaxOp,
    MinOp,
    SumBackwardOp,
    MeanBackwardOp,
    MaxBackwardOp,
    MinBackwardOp,
)

from projectodyssey.core.reduction_utils import (
    compute_strides,
    linear_to_coords,
    coords_to_linear,
    map_result_to_input_coords,
    create_result_coords,
)

# ============================================================================
# Utility Functions
# ============================================================================

from projectodyssey.core.utils import (
    argmax,
    top_k_indices,
    top_k,
    argsort,
)

# ============================================================================
# Tensor Validation Functions
# ============================================================================

from projectodyssey.core.validation import (
    validate_tensor_shape,
    validate_tensor_dtype,
    validate_matching_tensors,
    validate_2d_input,
    validate_4d_input,
    validate_1d_input,
    validate_3d_input,
    validate_axis,
    validate_slice_range,
    validate_float_dtype,
    validate_positive_shape,
    validate_matmul_dims,
    validate_broadcast_compatible,
    validate_non_empty,
    validate_matching_dtype,
)

# ============================================================================
# Parallel Processing Utilities
# ============================================================================

from projectodyssey.core.parallel_utils import (
    PARALLEL_BATCH_THRESHOLD,
    DEFAULT_NUM_WORKERS,
    should_parallelize,
    parallel_for_batch,
)

# ============================================================================
# Memory Pool for Efficient Small Allocations
# ============================================================================

from projectodyssey.base.memory_pool import (
    TensorMemoryPool,
    PoolConfig,
    PoolStats,
    get_global_pool,
    pooled_alloc,
    pooled_free,
)

# ============================================================================
# Module Interface for Layer Composition
# ============================================================================

from projectodyssey.core.module import Module
from projectodyssey.core.sequential import Sequential2, Sequential3

# ============================================================================
# Lazy Expression Evaluation
# ============================================================================

from projectodyssey.core.lazy_expression import (
    expr,
    TensorExpr,
    ExprNode,
    OpType,
)

from projectodyssey.core.lazy_eval import (
    evaluate,
)

# ============================================================================
# Gradient Clipping Utilities
# ============================================================================
# Defined here (in projectodyssey.core) rather than projectodyssey.autograd to avoid a
# circular type identity issue in Mojo v0.26.1: cross-package imports of
# AnyTensor from projectodyssey.autograd cause any_tensor.mojo to be compiled twice
# with distinct type identities, breaking operator overloads.

from projectodyssey.core.grad_utils import (
    clip_grad_value_,
    clip_grad_norm_,
    clip_grad_global_norm_,
)

# ============================================================================
# Tensor I/O (Save/Load) Utilities
# ============================================================================
# Now in projectodyssey.tensor.tensor_io. Import directly:
#   from projectodyssey.tensor.tensor_io import save_tensor, load_tensor
# NOT re-exported here to avoid circular imports between projectodyssey.core and projectodyssey.tensor.
