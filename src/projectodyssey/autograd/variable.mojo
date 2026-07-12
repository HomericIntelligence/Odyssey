"""Variable - Autograd-enabled tensor wrapper.

Provides automatic differentiation capabilities by wrapping AnyTensor with
gradient tracking and computation graph recording.

This module implements a tape-based autograd system similar to PyTorch's eager
mode execution, where operations are recorded during the forward pass and
replayed in reverse during backward propagation.

Key Concepts:
- Variable wraps an AnyTensor and adds requires_grad flag and grad storage
- Operations on Variables are recorded in a gradient tape
- Calling .backward(tape) triggers automatic gradient computation via chain rule
- Gradients accumulate across multiple backward passes (call tape.clear() to reset)

Examples:
    from projectodyssey.autograd import Variable, GradientTape

    # Create gradient tape
    var tape = GradientTape()
    tape.enable()

    # Create variables with gradient tracking
    var x = Variable(zeros(shape, dtype), requires_grad=True, tape)
    var y = Variable(ones(shape, dtype), requires_grad=True, tape)

    # Perform operations (recorded in tape)
    var z = variable_add(x, y, tape)
    var loss = variable_sum(z, tape)

    # Compute gradients
    loss.backward(tape)

    # Access gradients
    print(tape.get_grad(x.id))  # dLoss/dx
    print(tape.get_grad(y.id))  # dLoss/dy
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import ones_like, zeros_like
from projectodyssey.core.arithmetic import add, subtract, multiply, divide
from projectodyssey.core.activation import relu, sigmoid, tanh
from projectodyssey.core.reduction import sum, mean
from projectodyssey.core.matrix import matmul
from projectodyssey.core.linear import linear as _linear
from projectodyssey.core.conv import conv2d as _conv2d
from projectodyssey.core.conv import depthwise_conv2d as _depthwise_conv2d
from projectodyssey.core.shape import concatenate as _concatenate
from projectodyssey.core.pooling import maxpool2d as _maxpool2d
from projectodyssey.core.loss import cross_entropy as _cross_entropy
from projectodyssey.core.normalization import batch_norm2d as _batch_norm2d

comptime tensor_sum = sum
comptime tensor_mean = mean

from projectodyssey.autograd.tape import (
    GradientTape,
    SavedTensors,
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE,
    OP_SUM,
    OP_MEAN,
    OP_MATMUL,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_NEG,
    OP_FLATTEN,
    OP_LINEAR,
    OP_CONV2D,
    OP_MAXPOOL2D,
    OP_CROSS_ENTROPY,
    OP_BATCH_NORM2D,
    OP_DEPTHWISE_CONV2D,
    OP_CONCAT,
)


struct Variable(Copyable, Movable):
    """Tensor wrapper with automatic differentiation support.

        Variable extends AnyTensor with gradient tracking capabilities. Each Variable
        maintains:
        - data: The actual tensor values (AnyTensor)
        - id: Unique identifier for tape tracking
        - requires_grad: Whether to track operations for this variable

        Gradients are stored in the GradientTape's registry, not in the Variable
        itself. This allows for gradient accumulation and cleanup via the tape.

        Attributes:
            data: The underlying AnyTensor containing values.
            id: Unique identifier for gradient tracking.
            requires_grad: Flag indicating whether this Variable participates in autograd.

    Note:
            Operations on Variables create new Variables. The tape records which
            operations were performed and on which variables, enabling automatic
            gradient computation.
    """

    var data: AnyTensor
    var id: Int
    var requires_grad: Bool

    def __init__(
        out self,
        var data: AnyTensor,
        requires_grad: Bool,
        mut tape: GradientTape,
    ) raises:
        """Initialize a Variable and register it with the tape.

        Args:
            data: The tensor values to wrap (ownership transferred).
            requires_grad: Whether to track gradients for this variable.
            tape: The gradient tape to register with.

        Examples:
            var tape = GradientTape()
            var x = Variable(zeros(shape, dtype), True, tape)

        Raises:
            Error: If operation fails.
        """
        self.data = data^
        self.requires_grad = requires_grad
        self.id = tape.register_variable(requires_grad)

    def __init__(
        out self,
        var data: AnyTensor,
        requires_grad: Bool,
        id: Int,
    ):
        """Initialize a Variable with explicit ID (internal use).

        Args:
            data: The tensor values to wrap (ownership transferred).
            requires_grad: Whether to track gradients for this variable.
            id: Pre-assigned variable ID.

        Note:
            This constructor is primarily for internal use when creating
            output Variables from operations.
        """
        self.data = data^
        self.requires_grad = requires_grad
        self.id = id

    def backward(self, mut tape: GradientTape) raises:
        """Compute gradients via automatic differentiation.

        Triggers backward pass through the computation graph, computing gradients
        for all Variables with requires_grad=True that were used to compute this
        Variable.

        The gradient of this Variable with respect to itself is initialized to
        ones (d_self/d_self = 1), then gradients are propagated backward through
        the graph using the chain rule.

        Args:
            tape: The gradient tape that recorded operations.

        Examples:
        ```
                var x = Variable(data, True, tape)
                var loss = compute_loss(x, tape)
                loss.backward(tape)  # Computes gradients for all inputs
                print(tape.get_grad(x.id))  # dLoss/dx
        ```

        Raises:
            Error: If operation fails.
        """
        # Initialize gradient of output to ones
        var grad = ones_like(self.data)
        tape.backward(self.id, grad^)

    def detach(self) -> AnyTensor:
        """Get the underlying tensor without gradient tracking.

        Useful for breaking the computation graph when you want to use values
        without tracking gradients.

        Returns:
            The underlying AnyTensor (copy).

        Examples:
            var x = Variable(data, True, tape).
            var y = x.detach()  # y is just an AnyTensor, no gradient tracking.
        """
        return self.data

    def shape(self) -> List[Int]:
        """Get the shape of the underlying tensor.

        Returns:
            List of dimension sizes.
        """
        return self.data.shape()

    def numel(self) -> Int:
        """Get the number of elements in the tensor.

        Returns:
            Total number of elements.
        """
        return self.data.numel()

    def dtype(self) -> DType:
        """Get the data type of the underlying tensor.

        Returns:
            The DType of the tensor.
        """
        return self.data.dtype()


# ============================================================================
# Variable Operations
# ============================================================================
# These functions perform operations on Variables and record them in the tape.
# They follow the functional API pattern - each operation creates a new Variable.


def variable_add(
    a: Variable,
    b: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Add two Variables element-wise.

    Args:
            a: First input `Variable`.
            b: Second input `Variable`.
            tape: Gradient tape for recording.

    Returns:
            New `Variable` containing `a + b`.

    Raises:
            Error: If operation fails.
    """
    var result_data = add(a.data, b.data)
    var result_id = tape.register_variable(a.requires_grad or b.requires_grad)

    # Record operation for backward pass
    if tape.enabled and (a.requires_grad or b.requires_grad):
        var input_ids = List[Int]()
        input_ids.append(a.id)
        input_ids.append(b.id)

        # Save inputs for backward pass (needed for broadcast reduction)
        var saved = SavedTensors()
        saved.add_tensor(a.data)
        saved.add_tensor(b.data)
        tape.record(OP_ADD, input_ids^, result_id, saved^)

    return Variable(result_data^, a.requires_grad or b.requires_grad, result_id)


def variable_subtract(
    a: Variable,
    b: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Subtract two Variables element-wise.

    Args:
            a: First input `Variable`.
            b: Second input `Variable`.
            tape: Gradient tape for recording.

    Returns:
            New `Variable` containing `a - b`.

    Raises:
            Error: If operation fails.
    """
    var result_data = subtract(a.data, b.data)
    var result_id = tape.register_variable(a.requires_grad or b.requires_grad)

    if tape.enabled and (a.requires_grad or b.requires_grad):
        var input_ids = List[Int]()
        input_ids.append(a.id)
        input_ids.append(b.id)

        # Save inputs for backward pass (needed for broadcast reduction)
        var saved = SavedTensors()
        saved.add_tensor(a.data)
        saved.add_tensor(b.data)
        tape.record(OP_SUBTRACT, input_ids^, result_id, saved^)

    return Variable(result_data^, a.requires_grad or b.requires_grad, result_id)


def variable_multiply(
    a: Variable,
    b: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Multiply two Variables element-wise.

    Args:
            a: First input `Variable`.
            b: Second input `Variable`.
            tape: Gradient tape for recording.

    Returns:
            New `Variable` containing `a * b`.

    Raises:
            Error: If operation fails.
    """
    var result_data = multiply(a.data, b.data)
    var result_id = tape.register_variable(a.requires_grad or b.requires_grad)

    if tape.enabled and (a.requires_grad or b.requires_grad):
        var input_ids = List[Int]()
        input_ids.append(a.id)
        input_ids.append(b.id)

        # Save inputs for backward pass
        var saved = SavedTensors()
        saved.add_tensor(a.data)
        saved.add_tensor(b.data)
        tape.record(OP_MULTIPLY, input_ids^, result_id, saved^)

    return Variable(result_data^, a.requires_grad or b.requires_grad, result_id)


def variable_divide(
    a: Variable,
    b: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Divide two Variables element-wise.

    Args:
            a: Numerator `Variable`.
            b: Denominator `Variable`.
            tape: Gradient tape for recording.

    Returns:
            New `Variable` containing `a / b`.

    Raises:
            Error: If operation fails.
    """
    var result_data = divide(a.data, b.data)
    var result_id = tape.register_variable(a.requires_grad or b.requires_grad)

    if tape.enabled and (a.requires_grad or b.requires_grad):
        var input_ids = List[Int]()
        input_ids.append(a.id)
        input_ids.append(b.id)

        # Save inputs for backward pass
        var saved = SavedTensors()
        saved.add_tensor(a.data)
        saved.add_tensor(b.data)
        tape.record(OP_DIVIDE, input_ids^, result_id, saved^)

    return Variable(result_data^, a.requires_grad or b.requires_grad, result_id)


def variable_matmul(
    a: Variable,
    b: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Matrix multiply two Variables.

    Args:
            a: First matrix `Variable`.
            b: Second matrix `Variable`.
            tape: Gradient tape for recording.

    Returns:
            New `Variable` containing `a @ b`.

    Raises:
            Error: If operation fails.
    """
    var result_data = matmul(a.data, b.data)
    var result_id = tape.register_variable(a.requires_grad or b.requires_grad)

    if tape.enabled and (a.requires_grad or b.requires_grad):
        var input_ids = List[Int]()
        input_ids.append(a.id)
        input_ids.append(b.id)

        # Save inputs for backward pass
        var saved = SavedTensors()
        saved.add_tensor(a.data)
        saved.add_tensor(b.data)
        tape.record(OP_MATMUL, input_ids^, result_id, saved^)

    return Variable(result_data^, a.requires_grad or b.requires_grad, result_id)


def variable_sum(
    x: Variable,
    mut tape: GradientTape,
    axis: Int = -1,
) raises -> Variable:
    """Sum a Variable along an axis (or all elements if axis=-1).

    Args:
            x: Input Variable.
            tape: Gradient tape for recording.
            axis: Axis to sum along (-1 for full reduction).

    Returns:
            New Variable containing the sum.

    Raises:
            Error: If operation fails.
    """
    var result_data = tensor_sum(x.data, axis)
    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)

        # Save input tensor and axis for backward pass
        var saved = SavedTensors()
        saved.add_tensor(x.data)
        saved.add_scalar(Float64(axis))
        tape.record(OP_SUM, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


def variable_mean(
    x: Variable,
    mut tape: GradientTape,
    axis: Int = -1,
) raises -> Variable:
    """Mean of a Variable along an axis (or all elements if axis=-1).

    Args:
            x: Input Variable.
            tape: Gradient tape for recording.
            axis: Axis to average along (-1 for full reduction).

    Returns:
            New Variable containing the mean.

    Raises:
            Error: If operation fails.
    """
    var result_data = tensor_mean(x.data, axis)
    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)

        # Save input tensor and axis for backward pass
        var saved = SavedTensors()
        saved.add_tensor(x.data)
        saved.add_scalar(Float64(axis))
        tape.record(OP_MEAN, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


def variable_relu(
    x: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Apply ReLU activation to a Variable.

    Args:
            x: Input Variable.
            tape: Gradient tape for recording.

    Returns:
            New Variable containing `ReLU(x)`.

    Raises:
            Error: If operation fails.
    """
    var result_data = relu(x.data)
    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)

        # Save input for backward pass
        var saved = SavedTensors()
        saved.add_tensor(x.data)
        tape.record(OP_RELU, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


def variable_sigmoid(
    x: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Apply sigmoid activation to a Variable.

    Args:
            x: Input Variable.
            tape: Gradient tape for recording.

    Returns:
            New Variable containing `sigmoid(x)`.

    Raises:
            Error: If operation fails.
    """
    var result_data = sigmoid(x.data)
    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)

        # Save output for backward pass (sigmoid_backward uses output)
        var saved = SavedTensors()
        saved.add_tensor(result_data)
        tape.record(OP_SIGMOID, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


def variable_tanh(
    x: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Apply tanh activation to a Variable.

    Args:
            x: Input Variable.
            tape: Gradient tape for recording.

    Returns:
            New Variable containing `tanh(x)`.

    Raises:
            Error: If operation fails.
    """
    var result_data = tanh(x.data)
    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)

        # Save output for backward pass (tanh_backward uses output)
        var saved = SavedTensors()
        saved.add_tensor(result_data)
        tape.record(OP_TANH, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


def variable_neg(
    x: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Negate a Variable element-wise.

    Args:
            x: Input Variable.
            tape: Gradient tape for recording.

    Returns:
            New Variable containing `-x`.

    Raises:
            Error: If operation fails.
    """
    # Create negated tensor
    var result_data = zeros_like(x.data)
    var size = x.data.numel()
    for i in range(size):
        result_data._data[i] = -x.data._data[i]

    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)
        var saved = SavedTensors()
        tape.record(OP_NEG, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


# ============================================================================
# Phase 2 substrate ops (convnet primitives)
# ============================================================================


def variable_flatten(
    x: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Flatten a Variable from rank-N to rank-2 (batch, features).

    For input shape [B, d1, d2, ..., dN] returns shape [B, d1*d2*...*dN].

    Saved layout for backward:
        shapes[0] = original input shape.

    Args:
        x: Input Variable (rank >= 2).
        tape: Gradient tape for recording.

    Returns:
        New Variable of shape (batch_size, flattened_features).

    Raises:
        Error: If operation fails or input rank < 1.
    """
    var in_shape = x.data.shape()
    if len(in_shape) < 1:
        raise Error("variable_flatten: input must have at least 1 dimension")

    var batch = in_shape[0] if len(in_shape) > 0 else 1
    var feat = 1
    for i in range(1, len(in_shape)):
        feat *= in_shape[i]

    var new_shape = List[Int]()
    new_shape.append(batch)
    new_shape.append(feat)
    var result_data = x.data.reshape(new_shape)

    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)
        var saved = SavedTensors()
        saved.add_shape(in_shape)
        tape.record(OP_FLATTEN, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


def variable_linear(
    x: Variable,
    weights: Variable,
    bias: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Apply a fully connected (linear) layer: y = x @ W^T + b.

    Saved layout for backward:
        tensors[0] = input x      (batch, in_features)
        tensors[1] = weights      (out_features, in_features)

    Args:
        x: Input activations Variable (batch, in_features).
        weights: Weights Variable (out_features, in_features).
        bias: Bias Variable (out_features,).
        tape: Gradient tape for recording.

    Returns:
        Output Variable of shape (batch, out_features).
    """
    var result_data = _linear(x.data, weights.data, bias.data)
    var needs_grad = (
        x.requires_grad or weights.requires_grad or bias.requires_grad
    )
    var result_id = tape.register_variable(needs_grad)

    if tape.enabled and needs_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)
        input_ids.append(weights.id)
        input_ids.append(bias.id)

        var saved = SavedTensors()
        saved.add_tensor(x.data)
        saved.add_tensor(weights.data)
        tape.record(OP_LINEAR, input_ids^, result_id, saved^)

    return Variable(result_data^, needs_grad, result_id)


def variable_conv2d(
    x: Variable,
    weights: Variable,
    bias: Variable,
    mut tape: GradientTape,
    stride: Int = 1,
    padding: Int = 0,
) raises -> Variable:
    """Apply 2D convolution with bias: y = conv2d(x, weights, stride, padding) + bias.

    Saved layout for backward:
        tensors[0] = input x      (batch, in_C, H, W)
        tensors[1] = weights      (out_C, in_C, kH, kW)
        scalars[0] = stride
        scalars[1] = padding

    Args:
        x: Input activations Variable (batch, in_C, H, W).
        weights: Conv kernel Variable (out_C, in_C, kH, kW).
        bias: Bias Variable (out_C,).
        tape: Gradient tape for recording.
        stride: Convolution stride (default 1).
        padding: Zero-padding (default 0).

    Returns:
        Output Variable of shape (batch, out_C, out_H, out_W).
    """
    var result_data = _conv2d(x.data, weights.data, bias.data, stride, padding)
    var needs_grad = (
        x.requires_grad or weights.requires_grad or bias.requires_grad
    )
    var result_id = tape.register_variable(needs_grad)

    if tape.enabled and needs_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)
        input_ids.append(weights.id)
        input_ids.append(bias.id)

        var saved = SavedTensors()
        saved.add_tensor(x.data)
        saved.add_tensor(weights.data)
        saved.add_scalar(Float64(stride))
        saved.add_scalar(Float64(padding))
        tape.record(OP_CONV2D, input_ids^, result_id, saved^)

    return Variable(result_data^, needs_grad, result_id)


def variable_depthwise_conv2d(
    x: Variable,
    weights: Variable,
    bias: Variable,
    mut tape: GradientTape,
    stride: Int = 1,
    padding: Int = 0,
) raises -> Variable:
    """Apply depthwise 2D convolution with bias: one filter per input channel.

    Unlike `variable_conv2d` (which mixes all input channels), depthwise conv
    convolves each input channel with its own filter — the building block of
    MobileNet / EfficientNet depthwise-separable blocks. Wraps the core
    `depthwise_conv2d` / `depthwise_conv2d_backward`.

    Saved layout for backward (mirrors OP_CONV2D):
        tensors[0] = input x      (batch, C, H, W)
        tensors[1] = weights      (C, 1, kH, kW)
        scalars[0] = stride
        scalars[1] = padding

    Args:
        x: Input activations Variable (batch, C, H, W).
        weights: Depthwise kernel Variable (C, 1, kH, kW) — one filter per channel.
        bias: Bias Variable (C,).
        tape: Gradient tape for recording.
        stride: Convolution stride (default 1).
        padding: Zero-padding (default 0).

    Returns:
        Output Variable of shape (batch, C, out_H, out_W).
    """
    var result_data = _depthwise_conv2d(
        x.data, weights.data, bias.data, stride, padding
    )
    var needs_grad = (
        x.requires_grad or weights.requires_grad or bias.requires_grad
    )
    var result_id = tape.register_variable(needs_grad)

    if tape.enabled and needs_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)
        input_ids.append(weights.id)
        input_ids.append(bias.id)

        var saved = SavedTensors()
        saved.add_tensor(x.data)
        saved.add_tensor(weights.data)
        saved.add_scalar(Float64(stride))
        saved.add_scalar(Float64(padding))
        tape.record(OP_DEPTHWISE_CONV2D, input_ids^, result_id, saved^)

    return Variable(result_data^, needs_grad, result_id)


def variable_concat(
    inputs: List[Variable],
    mut tape: GradientTape,
    axis: Int,
) raises -> Variable:
    """Concatenate Variables along an existing axis (e.g. channel-axis for
    Inception depth-concat).

    Concat is a pure data-routing op: forward stacks the inputs along `axis`;
    backward slices the output gradient back into each input's contiguous range.
    Wraps the core `concatenate`. Result requires grad if ANY input does.

    Saved layout for backward:
        scalars[0]      = axis
        scalars[1]      = num_inputs (N)
        scalars[2..2+N] = each input's size along `axis`, in order (for the split)

    Args:
        inputs: Variables to concatenate (all matching except along `axis`).
        tape: Gradient tape for recording.
        axis: Axis along which to concatenate (channel axis is 1 for NCHW).

    Returns:
        Concatenated Variable.
    """
    if len(inputs) == 0:
        raise Error("variable_concat: need at least one input")

    var data_list = List[AnyTensor]()
    var needs_grad = False
    for i in range(len(inputs)):
        data_list.append(inputs[i].data)
        if inputs[i].requires_grad:
            needs_grad = True

    var result_data = _concatenate(data_list, axis)
    var result_id = tape.register_variable(needs_grad)

    if tape.enabled and needs_grad:
        var input_ids = List[Int]()
        var saved = SavedTensors()
        # Resolve a negative axis against the (shared) input rank so the saved
        # split sizes AND the backward slice use the same positive axis.
        var rank = len(inputs[0].data.shape())
        var actual_axis = axis if axis >= 0 else rank + axis
        saved.add_scalar(Float64(actual_axis))
        saved.add_scalar(Float64(len(inputs)))
        for i in range(len(inputs)):
            input_ids.append(inputs[i].id)
            saved.add_scalar(Float64(inputs[i].data.shape()[actual_axis]))
        tape.record(OP_CONCAT, input_ids^, result_id, saved^)

    return Variable(result_data^, needs_grad, result_id)


def variable_batch_norm(
    x: Variable,
    gamma: Variable,
    beta: Variable,
    running_mean: AnyTensor,
    running_var: AnyTensor,
    mut tape: GradientTape,
    training: Bool,
    momentum: Float64 = 0.1,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[Variable, AnyTensor, AnyTensor]:
    """Apply 2D batch normalization: y = gamma * norm(x) + beta.

    `x`, `gamma`, `beta` are trainable Variables (they receive gradients).
    `running_mean` / `running_var` are non-trainable buffers (plain
    `AnyTensor`, no gradient); the core op is functional, so the *updated*
    running statistics are returned and the caller must thread them back into
    its per-layer storage for the next batch. In training mode the forward and
    backward derive batch statistics from `x` and ignore the incoming running
    stats, so a caller that drops the returned stats still trains correctly
    (only later inference accuracy would suffer).

    Saved layout for backward:
        tensors[0] = input x            (batch, C, H, W)
        tensors[1] = gamma              (C,)
        tensors[2] = running_mean       (C,) — consumed only when training=False
        tensors[3] = running_var        (C,) — consumed only when training=False
        scalars[0] = training (1.0/0.0)
        scalars[1] = epsilon
        (beta is intentionally NOT saved — batch_norm2d_backward does not take
        it; grad_beta = sum(grad_output) needs no saved beta.)

    Args:
        x: Input activations Variable (batch, C, H, W).
        gamma: Scale-parameter Variable (C,).
        beta: Shift-parameter Variable (C,).
        running_mean: Running mean buffer (C,), not a Variable.
        running_var: Running variance buffer (C,), not a Variable.
        tape: Gradient tape for recording.
        training: If True, use batch statistics and update running stats.
        momentum: Momentum for the running-stats update (default 0.1).
        epsilon: Numerical-stability constant (default 1e-5).

    Returns:
        A tuple of (output Variable (batch, C, H, W), updated running_mean,
        updated running_var).
    """
    var bn = _batch_norm2d(
        x.data,
        gamma.data,
        beta.data,
        running_mean,
        running_var,
        training,
        momentum,
        epsilon,
    )
    var result_data = bn[0]
    var new_running_mean = bn[1]
    var new_running_var = bn[2]

    var needs_grad = (
        x.requires_grad or gamma.requires_grad or beta.requires_grad
    )
    var result_id = tape.register_variable(needs_grad)

    if tape.enabled and needs_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)
        input_ids.append(gamma.id)
        input_ids.append(beta.id)

        var saved = SavedTensors()
        saved.add_tensor(x.data)
        saved.add_tensor(gamma.data)
        saved.add_tensor(running_mean)
        saved.add_tensor(running_var)
        saved.add_scalar(1.0 if training else 0.0)
        saved.add_scalar(epsilon)
        tape.record(OP_BATCH_NORM2D, input_ids^, result_id, saved^)

    return (
        Variable(result_data^, needs_grad, result_id),
        new_running_mean^,
        new_running_var^,
    )


def variable_maxpool2d(
    x: Variable,
    mut tape: GradientTape,
    kernel_size: Int,
    stride: Int = 0,
    padding: Int = 0,
) raises -> Variable:
    """Apply 2D max pooling.

    Saved layout for backward (maxpool needs forward INPUT to recompute argmax):
        tensors[0] = input x      (batch, channels, H, W)
        scalars[0] = kernel_size
        scalars[1] = stride
        scalars[2] = padding

    Args:
        x: Input activations Variable (batch, channels, H, W).
        tape: Gradient tape for recording.
        kernel_size: Pooling window size.
        stride: Pool stride (default 0 => kernel_size).
        padding: Zero-padding (default 0).

    Returns:
        Output Variable of shape (batch, channels, out_H, out_W).
    """
    var result_data = _maxpool2d(x.data, kernel_size, stride, padding)
    var result_id = tape.register_variable(x.requires_grad)

    if tape.enabled and x.requires_grad:
        var input_ids = List[Int]()
        input_ids.append(x.id)

        var saved = SavedTensors()
        saved.add_tensor(x.data)
        saved.add_scalar(Float64(kernel_size))
        saved.add_scalar(Float64(stride))
        saved.add_scalar(Float64(padding))
        tape.record(OP_MAXPOOL2D, input_ids^, result_id, saved^)

    return Variable(result_data^, x.requires_grad, result_id)


def variable_cross_entropy(
    logits: Variable,
    targets: Variable,
    mut tape: GradientTape,
) raises -> Variable:
    """Compute cross-entropy loss (with mean-over-batch reduction).

    The targets Variable is treated as non-trainable; only logits receive
    a gradient in the backward pass.

    Saved layout for backward:
        tensors[0] = logits  (batch, num_classes)
        tensors[1] = targets (batch, num_classes)

    Args:
        logits: Raw model outputs (before softmax), Variable (batch, num_classes).
        targets: One-hot ground-truth Variable (batch, num_classes), requires_grad=False.
        tape: Gradient tape for recording.

    Returns:
        Scalar loss Variable.
    """
    var result_data = _cross_entropy(logits.data, targets.data)
    var needs_grad = logits.requires_grad
    var result_id = tape.register_variable(needs_grad)

    if tape.enabled and needs_grad:
        var input_ids = List[Int]()
        input_ids.append(logits.id)
        # NB: do NOT add targets.id — targets are non-trainable and
        # backward_cross_entropy only routes to input_ids[0].

        var saved = SavedTensors()
        saved.add_tensor(logits.data)
        saved.add_tensor(targets.data)
        tape.record(OP_CROSS_ENTROPY, input_ids^, result_id, saved^)

    return Variable(result_data^, needs_grad, result_id)
