"""LeNet-5 Model for EMNIST Classification.

Classic LeNet-5 architecture adapted for EMNIST dataset (28x28 grayscale images).

Architecture:
    Input (28x28x1) ->
    Conv2D(6 filters, 5x5) -> ReLU -> MaxPool(2x2, stride=2) ->
    Conv2D(16 filters, 5x5) -> ReLU -> MaxPool(2x2, stride=2) ->
    Flatten ->
    Linear(120) -> ReLU ->
    Linear(84) -> ReLU ->
    Linear(num_classes)

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
      Proceedings of the IEEE, 86(11), 2278-2324.
    - EMNIST Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from projectodyssey.tensor.any_tensor import AnyTensor, zeros, zeros_like
from projectodyssey.training.optimizers.adamw import adamw_step
from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.core.pooling import maxpool2d, maxpool2d_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.initializers import kaiming_uniform, xavier_uniform
from projectodyssey.core.shape import conv2d_output_shape, pool_output_shape
from projectodyssey.core.traits import Model
from projectodyssey.training.model_utils import (
    save_model_weights,
    load_model_weights,
    get_model_parameter_names,
)
from std.collections import List


# ============================================================================
# Architecture Hyperparameters
# ============================================================================
# All architecture dimensions are defined here for easy modification.
# Change these values to experiment with different model sizes.

# Input dimensions
comptime INPUT_HEIGHT = 28
comptime INPUT_WIDTH = 28
comptime INPUT_CHANNELS = 1

# Conv layer 1 hyperparameters
comptime CONV1_OUT_CHANNELS = 6
comptime CONV1_KERNEL_SIZE = 5
comptime CONV1_STRIDE = 1
comptime CONV1_PADDING = 0

# Pool layer 1 hyperparameters
comptime POOL1_KERNEL_SIZE = 2
comptime POOL1_STRIDE = 2
comptime POOL1_PADDING = 0

# Conv layer 2 hyperparameters
comptime CONV2_OUT_CHANNELS = 16
comptime CONV2_KERNEL_SIZE = 5
comptime CONV2_STRIDE = 1
comptime CONV2_PADDING = 0

# Pool layer 2 hyperparameters
comptime POOL2_KERNEL_SIZE = 2
comptime POOL2_STRIDE = 2
comptime POOL2_PADDING = 0

# Fully connected layer sizes
comptime FC1_OUT_FEATURES = 120
comptime FC2_OUT_FEATURES = 84


def compute_flattened_size() -> Int:
    """Compute the flattened feature size after all conv/pool layers.

    This derives the FC1 input dimension from the architecture hyperparameters.

    Returns:
        Number of features after flattening (channels * height * width).
    """
    # After conv1: Use shared conv2d_output_shape
    var h1, w1 = conv2d_output_shape(
        INPUT_HEIGHT,
        INPUT_WIDTH,
        CONV1_KERNEL_SIZE,
        CONV1_KERNEL_SIZE,
        CONV1_STRIDE,
        CONV1_PADDING,
    )

    # After pool1: Use shared pool_output_shape
    var h2, w2 = pool_output_shape(
        h1, w1, POOL1_KERNEL_SIZE, POOL1_STRIDE, POOL1_PADDING
    )

    # After conv2
    var h3, w3 = conv2d_output_shape(
        h2,
        w2,
        CONV2_KERNEL_SIZE,
        CONV2_KERNEL_SIZE,
        CONV2_STRIDE,
        CONV2_PADDING,
    )

    # After pool2
    var h4, w4 = pool_output_shape(
        h3, w3, POOL2_KERNEL_SIZE, POOL2_STRIDE, POOL2_PADDING
    )

    return CONV2_OUT_CHANNELS * h4 * w4


struct LeNet5(Model, Movable):
    """LeNet-5 model for EMNIST classification.

    Architecture is defined by module-level constants (CONV1_*, POOL1_*, etc.)
    for easy experimentation. FC layer input sizes are automatically derived
    from the conv/pool layer dimensions.

    Attributes:
        num_classes: Number of output classes (47 for EMNIST Balanced)
        conv1_kernel: First conv layer weights (CONV1_OUT_CHANNELS, INPUT_CHANNELS, k, k)
        conv1_bias: First conv layer bias (CONV1_OUT_CHANNELS,)
        conv2_kernel: Second conv layer weights (CONV2_OUT_CHANNELS, CONV1_OUT_CHANNELS, k, k)
        conv2_bias: Second conv layer bias (CONV2_OUT_CHANNELS,)
        fc1_weights: First FC layer weights (FC1_OUT_FEATURES, flattened_size)
        fc1_bias: First FC layer bias (FC1_OUT_FEATURES,)
        fc2_weights: Second FC layer weights (FC2_OUT_FEATURES, FC1_OUT_FEATURES)
        fc2_bias: Second FC layer bias (FC2_OUT_FEATURES,)
        fc3_weights: Third FC layer weights (num_classes, FC2_OUT_FEATURES)
        fc3_bias: Third FC layer bias (num_classes,).
    """

    var num_classes: Int

    # Layer parameters
    var conv1_kernel: AnyTensor
    var conv1_bias: AnyTensor
    var conv2_kernel: AnyTensor
    var conv2_bias: AnyTensor
    var fc1_weights: AnyTensor
    var fc1_bias: AnyTensor
    var fc2_weights: AnyTensor
    var fc2_bias: AnyTensor
    var fc3_weights: AnyTensor
    var fc3_bias: AnyTensor

    def __init__(out self, num_classes: Int = 47) raises:
        """Initialize LeNet-5 model with random weights.

        Args:
            num_classes: Number of output classes (default: 47 for EMNIST Balanced).
        """
        self.num_classes = num_classes

        # Compute derived dimensions
        var flattened_size = compute_flattened_size()

        # Conv1: INPUT_CHANNELS -> CONV1_OUT_CHANNELS, kernel CONV1_KERNEL_SIZE x CONV1_KERNEL_SIZE
        var conv1_shape: List[Int] = [
            CONV1_OUT_CHANNELS,
            INPUT_CHANNELS,
            CONV1_KERNEL_SIZE,
            CONV1_KERNEL_SIZE,
        ]
        var conv1_fan_in = (
            INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE
        )
        var conv1_fan_out = (
            CONV1_OUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE
        )
        self.conv1_kernel = kaiming_uniform(
            conv1_fan_in, conv1_fan_out, conv1_shape, dtype=DType.float32
        )
        var conv1_bias_shape: List[Int] = [CONV1_OUT_CHANNELS]
        self.conv1_bias = zeros(conv1_bias_shape, DType.float32)

        # Conv2: CONV1_OUT_CHANNELS -> CONV2_OUT_CHANNELS, kernel CONV2_KERNEL_SIZE x CONV2_KERNEL_SIZE
        var conv2_shape: List[Int] = [
            CONV2_OUT_CHANNELS,
            CONV1_OUT_CHANNELS,
            CONV2_KERNEL_SIZE,
            CONV2_KERNEL_SIZE,
        ]
        var conv2_fan_in = (
            CONV1_OUT_CHANNELS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE
        )
        var conv2_fan_out = (
            CONV2_OUT_CHANNELS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE
        )
        self.conv2_kernel = kaiming_uniform(
            conv2_fan_in, conv2_fan_out, conv2_shape, dtype=DType.float32
        )
        var conv2_bias_shape: List[Int] = [CONV2_OUT_CHANNELS]
        self.conv2_bias = zeros(conv2_bias_shape, DType.float32)

        # FC1: flattened_size -> FC1_OUT_FEATURES (derived from conv/pool layers)
        var fc1_shape: List[Int] = [FC1_OUT_FEATURES, flattened_size]
        self.fc1_weights = kaiming_uniform(
            flattened_size, FC1_OUT_FEATURES, fc1_shape, dtype=DType.float32
        )
        var fc1_bias_shape: List[Int] = [FC1_OUT_FEATURES]
        self.fc1_bias = zeros(fc1_bias_shape, DType.float32)

        # FC2: FC1_OUT_FEATURES -> FC2_OUT_FEATURES
        var fc2_shape: List[Int] = [FC2_OUT_FEATURES, FC1_OUT_FEATURES]
        self.fc2_weights = kaiming_uniform(
            FC1_OUT_FEATURES, FC2_OUT_FEATURES, fc2_shape, dtype=DType.float32
        )
        var fc2_bias_shape: List[Int] = [FC2_OUT_FEATURES]
        self.fc2_bias = zeros(fc2_bias_shape, DType.float32)

        # FC3: FC2_OUT_FEATURES -> num_classes
        var fc3_shape: List[Int] = [num_classes, FC2_OUT_FEATURES]
        self.fc3_weights = kaiming_uniform(
            FC2_OUT_FEATURES, num_classes, fc3_shape, dtype=DType.float32
        )
        var fc3_bias_shape: List[Int] = [num_classes]
        self.fc3_bias = zeros(fc3_bias_shape, DType.float32)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass through LeNet-5.

        Args:
            input: Input tensor of shape (batch, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH).

        Returns:
            Output logits of shape (batch, num_classes).
        """
        # Conv1 + ReLU + MaxPool
        var conv1_out = conv2d(
            input,
            self.conv1_kernel,
            self.conv1_bias,
            stride=CONV1_STRIDE,
            padding=CONV1_PADDING,
        )
        var relu1_out = relu(conv1_out)
        var pool1_out = maxpool2d(
            relu1_out,
            kernel_size=POOL1_KERNEL_SIZE,
            stride=POOL1_STRIDE,
            padding=POOL1_PADDING,
        )

        # Conv2 + ReLU + MaxPool
        var conv2_out = conv2d(
            pool1_out,
            self.conv2_kernel,
            self.conv2_bias,
            stride=CONV2_STRIDE,
            padding=CONV2_PADDING,
        )
        var relu2_out = relu(conv2_out)
        var pool2_out = maxpool2d(
            relu2_out,
            kernel_size=POOL2_KERNEL_SIZE,
            stride=POOL2_STRIDE,
            padding=POOL2_PADDING,
        )

        # Flatten: (batch, CONV2_OUT_CHANNELS, h, w) -> (batch, flattened_size)
        var pool2_shape = pool2_out.shape()
        var batch_size = pool2_shape[0]
        var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]

        var flatten_shape: List[Int] = [batch_size, flattened_size]
        var flattened = pool2_out.reshape(flatten_shape)

        # FC1 + ReLU
        var fc1_out = linear(flattened, self.fc1_weights, self.fc1_bias)
        var relu3_out = relu(fc1_out)

        # FC2 + ReLU
        var fc2_out = linear(relu3_out, self.fc2_weights, self.fc2_bias)
        var relu4_out = relu(fc2_out)

        # FC3 (output logits)
        var output = linear(relu4_out, self.fc3_weights, self.fc3_bias)

        return output^

    def predict(mut self, input: AnyTensor) raises -> Int:
        """Predict class for a single input.

        Args:
            input: Input tensor of shape (1, 1, 28, 28).

        Returns:
            Predicted class index (0 to num_classes-1).
        """
        var logits = self.forward(input)

        # Find argmax
        var logits_shape = logits.shape()
        var max_idx = 0
        var max_val = logits[0]

        for i in range(1, logits_shape[1]):
            if logits[i] > max_val:
                max_val = logits[i]
                max_idx = i

        return max_idx

    def save_weights(self, weights_dir: String) raises:
        """Save model weights to directory.

        Args:
            weights_dir: Directory to save weight files (one file per parameter).

        Note:
            Creates directory if it doesn't exist. Each parameter saved as:
            - conv1_kernel.weights
            - conv1_bias.weights
            - etc.
        """
        # Collect all parameters
        var parameters: List[AnyTensor] = []
        parameters.append(self.conv1_kernel)
        parameters.append(self.conv1_bias)
        parameters.append(self.conv2_kernel)
        parameters.append(self.conv2_bias)
        parameters.append(self.fc1_weights)
        parameters.append(self.fc1_bias)
        parameters.append(self.fc2_weights)
        parameters.append(self.fc2_bias)
        parameters.append(self.fc3_weights)
        parameters.append(self.fc3_bias)

        # Get standard parameter names
        var param_names = get_model_parameter_names("lenet5")

        # Save using shared utility
        save_model_weights(parameters, weights_dir, param_names)

    def load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing weight files.

        Raises:
            Error: If weight files are missing or have incompatible shapes.
        """
        # Get standard parameter names
        var param_names = get_model_parameter_names("lenet5")

        # Create empty list for loaded parameters
        var loaded_params: List[AnyTensor] = []

        # Load using shared utility
        load_model_weights(loaded_params, weights_dir, param_names)

        # Assign loaded parameters to model fields
        self.conv1_kernel = loaded_params[0]
        self.conv1_bias = loaded_params[1]
        self.conv2_kernel = loaded_params[2]
        self.conv2_bias = loaded_params[3]
        self.fc1_weights = loaded_params[4]
        self.fc1_bias = loaded_params[5]
        self.fc2_weights = loaded_params[6]
        self.fc2_bias = loaded_params[7]
        self.fc3_weights = loaded_params[8]
        self.fc3_bias = loaded_params[9]

    def update_parameters(
        mut self,
        learning_rate: Float32,
        grad_conv1_kernel: AnyTensor,
        grad_conv1_bias: AnyTensor,
        grad_conv2_kernel: AnyTensor,
        grad_conv2_bias: AnyTensor,
        grad_fc1_weights: AnyTensor,
        grad_fc1_bias: AnyTensor,
        grad_fc2_weights: AnyTensor,
        grad_fc2_bias: AnyTensor,
        grad_fc3_weights: AnyTensor,
        grad_fc3_bias: AnyTensor,
    ) raises:
        """Update parameters using SGD.

        Args:
            learning_rate: Learning rate for gradient descent.
            grad_conv1_kernel: Gradient for first convolution kernel.
            grad_conv1_bias: Gradient for first convolution bias.
            grad_conv2_kernel: Gradient for second convolution kernel.
            grad_conv2_bias: Gradient for second convolution bias.
            grad_fc1_weights: Gradient for first fully connected layer weights.
            grad_fc1_bias: Gradient for first fully connected layer bias.
            grad_fc2_weights: Gradient for second fully connected layer weights.
            grad_fc2_bias: Gradient for second fully connected layer bias.
            grad_fc3_weights: Gradient for third fully connected layer weights.
            grad_fc3_bias: Gradient for third fully connected layer bias.
        """
        # SGD update: param = param - lr * grad
        _sgd_update(self.conv1_kernel, grad_conv1_kernel, learning_rate)
        _sgd_update(self.conv1_bias, grad_conv1_bias, learning_rate)
        _sgd_update(self.conv2_kernel, grad_conv2_kernel, learning_rate)
        _sgd_update(self.conv2_bias, grad_conv2_bias, learning_rate)
        _sgd_update(self.fc1_weights, grad_fc1_weights, learning_rate)
        _sgd_update(self.fc1_bias, grad_fc1_bias, learning_rate)
        _sgd_update(self.fc2_weights, grad_fc2_weights, learning_rate)
        _sgd_update(self.fc2_bias, grad_fc2_bias, learning_rate)
        _sgd_update(self.fc3_weights, grad_fc3_weights, learning_rate)
        _sgd_update(self.fc3_bias, grad_fc3_bias, learning_rate)

    def parameters(self) raises -> List[AnyTensor]:
        """Return all trainable parameters.

        Returns:
            List of parameter tensors (10 total: conv1/2 kernel/bias, fc1/2/3 weights/bias)

        Note:
            This method copies the parameter tensors. For in-place updates,
            use update_parameters() instead.
        """
        var params: List[AnyTensor] = []
        params.append(self.conv1_kernel)
        params.append(self.conv1_bias)
        params.append(self.conv2_kernel)
        params.append(self.conv2_bias)
        params.append(self.fc1_weights)
        params.append(self.fc1_bias)
        params.append(self.fc2_weights)
        params.append(self.fc2_bias)
        params.append(self.fc3_weights)
        params.append(self.fc3_bias)
        return params^

    def zero_grad(mut self) raises:
        """Reset all parameter gradients to zero.

        Note:
            LeNet5 uses manual gradient computation in compute_gradients().
            Gradients are computed fresh each forward/backward pass, so this
            is a no-op. Included for Model trait conformance.
        """
        # Gradients are computed fresh each batch in compute_gradients()
        # No persistent gradient accumulators to reset
        pass


def _sgd_update(mut param: AnyTensor, grad: AnyTensor, lr: Float32) raises:
    """SGD parameter update: param = param - lr * grad.

    Deprecated: prefer AdamW via `update_parameters_adamw` for the grokking
    experiment example. Retained for backward compatibility.
    """
    var numel = param.numel()
    var param_data = param._data.bitcast[Float32]()
    var grad_data = grad._data.bitcast[Float32]()

    for i in range(numel):
        param_data[i] -= lr * grad_data[i]


# ============================================================================
# AdamW Optimizer State
# ============================================================================


struct AdamWState(Movable):
    """First and second moment buffers for AdamW, one (m, v) pair per parameter.

    The buffers are zero-initialized to match each model parameter shape and
    dtype. The caller (training loop) owns the timestep `t`.
    """

    var m_conv1_kernel: AnyTensor
    var v_conv1_kernel: AnyTensor
    var m_conv1_bias: AnyTensor
    var v_conv1_bias: AnyTensor
    var m_conv2_kernel: AnyTensor
    var v_conv2_kernel: AnyTensor
    var m_conv2_bias: AnyTensor
    var v_conv2_bias: AnyTensor
    var m_fc1_weights: AnyTensor
    var v_fc1_weights: AnyTensor
    var m_fc1_bias: AnyTensor
    var v_fc1_bias: AnyTensor
    var m_fc2_weights: AnyTensor
    var v_fc2_weights: AnyTensor
    var m_fc2_bias: AnyTensor
    var v_fc2_bias: AnyTensor
    var m_fc3_weights: AnyTensor
    var v_fc3_weights: AnyTensor
    var m_fc3_bias: AnyTensor
    var v_fc3_bias: AnyTensor

    def __init__(out self, model: LeNet5) raises:
        self.m_conv1_kernel = zeros_like(model.conv1_kernel)
        self.v_conv1_kernel = zeros_like(model.conv1_kernel)
        self.m_conv1_bias = zeros_like(model.conv1_bias)
        self.v_conv1_bias = zeros_like(model.conv1_bias)
        self.m_conv2_kernel = zeros_like(model.conv2_kernel)
        self.v_conv2_kernel = zeros_like(model.conv2_kernel)
        self.m_conv2_bias = zeros_like(model.conv2_bias)
        self.v_conv2_bias = zeros_like(model.conv2_bias)
        self.m_fc1_weights = zeros_like(model.fc1_weights)
        self.v_fc1_weights = zeros_like(model.fc1_weights)
        self.m_fc1_bias = zeros_like(model.fc1_bias)
        self.v_fc1_bias = zeros_like(model.fc1_bias)
        self.m_fc2_weights = zeros_like(model.fc2_weights)
        self.v_fc2_weights = zeros_like(model.fc2_weights)
        self.m_fc2_bias = zeros_like(model.fc2_bias)
        self.v_fc2_bias = zeros_like(model.fc2_bias)
        self.m_fc3_weights = zeros_like(model.fc3_weights)
        self.v_fc3_weights = zeros_like(model.fc3_weights)
        self.m_fc3_bias = zeros_like(model.fc3_bias)
        self.v_fc3_bias = zeros_like(model.fc3_bias)


def update_parameters_adamw(
    mut model: LeNet5,
    mut state: AdamWState,
    lr: Float64,
    t: Int,
    weight_decay: Float64,
    grad_conv1_kernel: AnyTensor,
    grad_conv1_bias: AnyTensor,
    grad_conv2_kernel: AnyTensor,
    grad_conv2_bias: AnyTensor,
    grad_fc1_weights: AnyTensor,
    grad_fc1_bias: AnyTensor,
    grad_fc2_weights: AnyTensor,
    grad_fc2_bias: AnyTensor,
    grad_fc3_weights: AnyTensor,
    grad_fc3_bias: AnyTensor,
) raises:
    """Apply AdamW update to all 10 parameters, mutating model and state."""
    (
        model.conv1_kernel,
        state.m_conv1_kernel,
        state.v_conv1_kernel,
    ) = adamw_step(
        model.conv1_kernel,
        grad_conv1_kernel,
        state.m_conv1_kernel,
        state.v_conv1_kernel,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.conv1_bias, state.m_conv1_bias, state.v_conv1_bias) = adamw_step(
        model.conv1_bias,
        grad_conv1_bias,
        state.m_conv1_bias,
        state.v_conv1_bias,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (
        model.conv2_kernel,
        state.m_conv2_kernel,
        state.v_conv2_kernel,
    ) = adamw_step(
        model.conv2_kernel,
        grad_conv2_kernel,
        state.m_conv2_kernel,
        state.v_conv2_kernel,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.conv2_bias, state.m_conv2_bias, state.v_conv2_bias) = adamw_step(
        model.conv2_bias,
        grad_conv2_bias,
        state.m_conv2_bias,
        state.v_conv2_bias,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.fc1_weights, state.m_fc1_weights, state.v_fc1_weights) = adamw_step(
        model.fc1_weights,
        grad_fc1_weights,
        state.m_fc1_weights,
        state.v_fc1_weights,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.fc1_bias, state.m_fc1_bias, state.v_fc1_bias) = adamw_step(
        model.fc1_bias,
        grad_fc1_bias,
        state.m_fc1_bias,
        state.v_fc1_bias,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.fc2_weights, state.m_fc2_weights, state.v_fc2_weights) = adamw_step(
        model.fc2_weights,
        grad_fc2_weights,
        state.m_fc2_weights,
        state.v_fc2_weights,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.fc2_bias, state.m_fc2_bias, state.v_fc2_bias) = adamw_step(
        model.fc2_bias,
        grad_fc2_bias,
        state.m_fc2_bias,
        state.v_fc2_bias,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.fc3_weights, state.m_fc3_weights, state.v_fc3_weights) = adamw_step(
        model.fc3_weights,
        grad_fc3_weights,
        state.m_fc3_weights,
        state.v_fc3_weights,
        t,
        lr,
        weight_decay=weight_decay,
    )
    (model.fc3_bias, state.m_fc3_bias, state.v_fc3_bias) = adamw_step(
        model.fc3_bias,
        grad_fc3_bias,
        state.m_fc3_bias,
        state.v_fc3_bias,
        t,
        lr,
        weight_decay=weight_decay,
    )
