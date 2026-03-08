"""
Core Library - Pure functional operations for ML Odyssey.

This package provides pure functional operations for neural networks and mathematical
operations. All functions are stateless - they process inputs to produce outputs
without maintaining internal state.

Architecture:
    - Pure functional design - no classes, no internal state
    - All functions work with ExTensor (no Tensor alias)
    - Caller manages all state (weights, biases, momentum, etc.)
    - Functions return new values, never mutate inputs

Modules:
    extensor: Core tensor type (ExTensor, implements Hashable via __hash__) and creation functions
    types: Custom data types (type aliases for FP8/BF8/BF16/FP4, MXFP4/NVFP4 blocked formats)
    arithmetic: Element-wise arithmetic operations (add, subtract, multiply, divide)
    matrix: Matrix operations (matmul, transpose, dot, outer)
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

Example:
   ```mojo
    from shared.core.extensor import ExTensor, zeros
    from shared.core.linear import linear
    from shared.core.activation import relu
    from shared.core.matrix import matmul, transpose

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
from shared.version import VERSION

# ============================================================================
# Default Hyperparameters
# ============================================================================

from shared.core.defaults import (
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

from shared.core.math_constants import (
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

from shared.core.numerical_constants import (
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

from shared.core.activation_constants import (
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

from shared.core.optimizer_constants import (
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
# ExTensor implements the Hashable trait (__hash__), allowing tensors to be
# used as dictionary keys or in hash-based data structures via hash(tensor).

from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    full,
    empty,
    arange,
    eye,
    linspace,
    ones_like,
    zeros_like,
    full_like,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
    clone,
    item,
    diff,
    randn,
)

# ============================================================================
# Shape Manipulation Operations
# ============================================================================

from shared.core.shape import (
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

from shared.core.types.dtype_aliases import BF16, FP8, BF8, FP4, E8M0

# ============================================================================
# Gradient Container Types
# ============================================================================

from shared.core.gradient_types import (
    GradientPair,
    GradientTriple,
    GradientQuad,
    Conv2dNoBiasGradient,
    DepthwiseConv2dNoBiasGradient,
)

# ============================================================================
# Arithmetic Operations
# ============================================================================

from shared.core.arithmetic import (
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

from shared.core.matrix import (
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

from shared.core.matmul import (
    matmul_optimized,
    matmul_tiled,
    matmul_simd,
    matmul_typed,
)

from shared.core.strassen import (
    matmul_strassen,
    STRASSEN_ENABLED,
    STRASSEN_THRESHOLD,
    next_power_of_2,
)

# ============================================================================
# Activation Functions
# ============================================================================

from shared.core.activation import (
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

from shared.core.activation_ops import (
    exp_scalar_f32,
    exp_scalar_f64,
)

from shared.core.activation_simd import (
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

from shared.core.linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
)

from shared.core.conv import (
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

from shared.core.pooling import (
    maxpool2d,
    avgpool2d,
    global_avgpool2d,
    maxpool2d_backward,
    avgpool2d_backward,
    global_avgpool2d_backward,
)

from shared.core.dropout import (
    dropout,
    dropout2d,
    dropout_backward,
    dropout2d_backward,
)

from shared.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
    layer_norm,
    layer_norm_backward,
    group_norm,
    group_norm_backward,
    instance_norm,
    instance_norm_backward,
)

from shared.core.normalization_simd import (
    batch_norm2d_fused,
    batch_norm2d_fused_inference,
)

from shared.core.normalize_ops import normalize_rgb

from shared.core.scalar_ops import (
    sqrt_scalar_f32,
    sqrt_scalar_f64,
    pow_scalar_f32,
    pow_scalar_f64,
)

# ============================================================================
# Attention Mechanisms
# ============================================================================

from shared.core.attention import (
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

from shared.core.elementwise import (
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

from shared.core.comparison import (
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

from shared.core.broadcasting import (
    broadcast_shapes,
    are_shapes_broadcastable,
    compute_broadcast_strides,
    BroadcastIterator,
)

# ============================================================================
# Initialization Functions
# ============================================================================

from shared.core.initializers import (
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

from shared.core.loss import (
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

from shared.core.loss_utils import (
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

from shared.core.numerical_safety import (
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
)

# ============================================================================
# Dtype Dispatch Helpers
# ============================================================================

from shared.core.dtype_dispatch import (
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

from shared.core.reduction import (
    sum,
    mean,
    max_reduce,
    min_reduce,
    variance,
    std,
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

from shared.core.reduction_ops import (
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

from shared.core.reduction_utils import (
    compute_strides,
    linear_to_coords,
    coords_to_linear,
    map_result_to_input_coords,
    create_result_coords,
)

# ============================================================================
# Utility Functions
# ============================================================================

from shared.core.utils import (
    argmax,
    top_k_indices,
    top_k,
    argsort,
)

# ============================================================================
# Tensor Validation Functions
# ============================================================================

from shared.core.validation import (
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

from shared.core.parallel_utils import (
    PARALLEL_BATCH_THRESHOLD,
    DEFAULT_NUM_WORKERS,
    should_parallelize,
    parallel_for_batch,
)

# ============================================================================
# Memory Pool for Efficient Small Allocations
# ============================================================================

from shared.core.memory_pool import (
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

from shared.core.module import Module

# ============================================================================
# Lazy Expression Evaluation
# ============================================================================

from shared.core.lazy_expression import (
    expr,
    TensorExpr,
    ExprNode,
    OpType,
)

from shared.core.lazy_eval import (
    evaluate,
)

# NOTE: Mojo does not support Python's __all__ mechanism.
# All imported symbols are automatically available to package consumers.
