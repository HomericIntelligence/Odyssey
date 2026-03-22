"""AlexNet Model for CIFAR-10 Classification

Classic AlexNet architecture adapted for CIFAR-10 dataset (32x32 RGB images).

Architecture:
    Input (32x32x3) ->
    Conv2D(96 filters, 11x11, stride=4, padding=2) -> ReLU -> MaxPool(3x3, stride=2) ->
    Conv2D(256 filters, 5x5, padding=2) -> ReLU -> MaxPool(3x3, stride=2) ->
    Conv2D(384 filters, 3x3, padding=1) -> ReLU ->
    Conv2D(384 filters, 3x3, padding=1) -> ReLU ->
    Conv2D(256 filters, 3x3, padding=1) -> ReLU -> MaxPool(3x3, stride=2) ->
    Flatten ->
    Linear(4096) -> ReLU -> Dropout(0.5) ->
    Linear(4096) -> ReLU -> Dropout(0.5) ->
    Linear(num_classes)

References:
    - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
      ImageNet classification with deep convolutional neural networks.
      Advances in Neural Information Processing Systems, 25, 1097-1105.
    - CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from shared.core import AnyTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.dropout import dropout, dropout_backward
from shared.core.initializers import he_uniform, xavier_uniform
from shared.core.shape import conv2d_output_shape, pool_output_shape
from shared.training.optimizers import sgd_momentum_update_inplace
from shared.training.model_utils import (
    save_model_weights,
    load_model_weights,
    get_model_parameter_names,
)
from collections import List


# ============================================================================
# Architecture Hyperparameters
# ============================================================================
# All architecture dimensions are defined here for easy modification.
# Change these values to experiment with different model sizes.

# Input dimensions (CIFAR-10)
comptime INPUT_HEIGHT = 32
comptime INPUT_WIDTH = 32
comptime INPUT_CHANNELS = 3

# Conv layer 1 hyperparameters
comptime CONV1_OUT_CHANNELS = 96
comptime CONV1_KERNEL_SIZE = 11
comptime CONV1_STRIDE = 4
comptime CONV1_PADDING = 2

# Pool layer 1 hyperparameters
comptime POOL1_KERNEL_SIZE = 3
comptime POOL1_STRIDE = 2
comptime POOL1_PADDING = 0

# Conv layer 2 hyperparameters
comptime CONV2_OUT_CHANNELS = 256
comptime CONV2_KERNEL_SIZE = 5
comptime CONV2_STRIDE = 1
comptime CONV2_PADDING = 2

# Pool layer 2 hyperparameters
comptime POOL2_KERNEL_SIZE = 3
comptime POOL2_STRIDE = 2
comptime POOL2_PADDING = 0

# Conv layer 3 hyperparameters
comptime CONV3_OUT_CHANNELS = 384
comptime CONV3_KERNEL_SIZE = 3
comptime CONV3_STRIDE = 1
comptime CONV3_PADDING = 1

# Conv layer 4 hyperparameters
comptime CONV4_OUT_CHANNELS = 384
comptime CONV4_KERNEL_SIZE = 3
comptime CONV4_STRIDE = 1
comptime CONV4_PADDING = 1

# Conv layer 5 hyperparameters
comptime CONV5_OUT_CHANNELS = 256
comptime CONV5_KERNEL_SIZE = 3
comptime CONV5_STRIDE = 1
comptime CONV5_PADDING = 1

# Pool layer 3 hyperparameters
comptime POOL3_KERNEL_SIZE = 3
comptime POOL3_STRIDE = 2
comptime POOL3_PADDING = 0

# Fully connected layer sizes
comptime FC1_OUT_FEATURES = 4096
comptime FC2_OUT_FEATURES = 4096


fn compute_flattened_size() -> Int:
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

    # After conv3
    var h5, w5 = conv2d_output_shape(
        h4,
        w4,
        CONV3_KERNEL_SIZE,
        CONV3_KERNEL_SIZE,
        CONV3_STRIDE,
        CONV3_PADDING,
    )

    # After conv4
    var h6, w6 = conv2d_output_shape(
        h5,
        w5,
        CONV4_KERNEL_SIZE,
        CONV4_KERNEL_SIZE,
        CONV4_STRIDE,
        CONV4_PADDING,
    )

    # After conv5
    var h7, w7 = conv2d_output_shape(
        h6,
        w6,
        CONV5_KERNEL_SIZE,
        CONV5_KERNEL_SIZE,
        CONV5_STRIDE,
        CONV5_PADDING,
    )

    # After pool3
    var h8, w8 = pool_output_shape(
        h7, w7, POOL3_KERNEL_SIZE, POOL3_STRIDE, POOL3_PADDING
    )

    return CONV5_OUT_CHANNELS * h8 * w8


struct AlexNet:
    """AlexNet model for CIFAR-10 classification.

    Attributes:
        num_classes: Number of output classes (10 for CIFAR-10)
        dropout_rate: Dropout probability for FC layers (default 0.5)
        conv1_kernel: First conv layer weights (96, 3, 11, 11)
        conv1_bias: First conv layer bias (96,)
        conv2_kernel: Second conv layer weights (256, 96, 5, 5)
        conv2_bias: Second conv layer bias (256,)
        conv3_kernel: Third conv layer weights (384, 256, 3, 3)
        conv3_bias: Third conv layer bias (384,)
        conv4_kernel: Fourth conv layer weights (384, 384, 3, 3)
        conv4_bias: Fourth conv layer bias (384,)
        conv5_kernel: Fifth conv layer weights (256, 384, 3, 3)
        conv5_bias: Fifth conv layer bias (256,)
        fc1_weights: First FC layer weights (4096, 256)
        fc1_bias: First FC layer bias (4096,)
        fc2_weights: Second FC layer weights (4096, 4096)
        fc2_bias: Second FC layer bias (4096,)
        fc3_weights: Third FC layer weights (num_classes, 4096)
        fc3_bias: Third FC layer bias (num_classes,).
    """

    var num_classes: Int
    var dropout_rate: Float32

    # Convolutional layer parameters
    var conv1_kernel: AnyTensor
    var conv1_bias: AnyTensor
    var conv2_kernel: AnyTensor
    var conv2_bias: AnyTensor
    var conv3_kernel: AnyTensor
    var conv3_bias: AnyTensor
    var conv4_kernel: AnyTensor
    var conv4_bias: AnyTensor
    var conv5_kernel: AnyTensor
    var conv5_bias: AnyTensor

    # Fully connected layer parameters
    var fc1_weights: AnyTensor
    var fc1_bias: AnyTensor
    var fc2_weights: AnyTensor
    var fc2_bias: AnyTensor
    var fc3_weights: AnyTensor
    var fc3_bias: AnyTensor

    fn __init__(
        out self, num_classes: Int = 10, dropout_rate: Float32 = 0.5
    ) raises:
        """Initialize AlexNet model with random weights.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10).
            dropout_rate: Dropout probability for FC layers (default: 0.5).
        """
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Compute derived dimensions using shared shape utilities
        var flattened_size = compute_flattened_size()

        # Conv1: INPUT_CHANNELS -> CONV1_OUT_CHANNELS
        var conv1_shape: List[Int] = [
            CONV1_OUT_CHANNELS,
            INPUT_CHANNELS,
            CONV1_KERNEL_SIZE,
            CONV1_KERNEL_SIZE,
        ]
        self.conv1_kernel = he_uniform(conv1_shape, DType.float32)
        var conv1_bias_shape: List[Int] = [CONV1_OUT_CHANNELS]
        self.conv1_bias = zeros(conv1_bias_shape, DType.float32)

        # Conv2: CONV1_OUT_CHANNELS -> CONV2_OUT_CHANNELS
        var conv2_shape: List[Int] = [
            CONV2_OUT_CHANNELS,
            CONV1_OUT_CHANNELS,
            CONV2_KERNEL_SIZE,
            CONV2_KERNEL_SIZE,
        ]
        self.conv2_kernel = he_uniform(conv2_shape, DType.float32)
        var conv2_bias_shape: List[Int] = [CONV2_OUT_CHANNELS]
        self.conv2_bias = zeros(conv2_bias_shape, DType.float32)

        # Conv3: CONV2_OUT_CHANNELS -> CONV3_OUT_CHANNELS
        var conv3_shape: List[Int] = [
            CONV3_OUT_CHANNELS,
            CONV2_OUT_CHANNELS,
            CONV3_KERNEL_SIZE,
            CONV3_KERNEL_SIZE,
        ]
        self.conv3_kernel = he_uniform(conv3_shape, DType.float32)
        var conv3_bias_shape: List[Int] = [CONV3_OUT_CHANNELS]
        self.conv3_bias = zeros(conv3_bias_shape, DType.float32)

        # Conv4: CONV3_OUT_CHANNELS -> CONV4_OUT_CHANNELS
        var conv4_shape: List[Int] = [
            CONV4_OUT_CHANNELS,
            CONV3_OUT_CHANNELS,
            CONV4_KERNEL_SIZE,
            CONV4_KERNEL_SIZE,
        ]
        self.conv4_kernel = he_uniform(conv4_shape, DType.float32)
        var conv4_bias_shape: List[Int] = [CONV4_OUT_CHANNELS]
        self.conv4_bias = zeros(conv4_bias_shape, DType.float32)

        # Conv5: CONV4_OUT_CHANNELS -> CONV5_OUT_CHANNELS
        var conv5_shape: List[Int] = [
            CONV5_OUT_CHANNELS,
            CONV4_OUT_CHANNELS,
            CONV5_KERNEL_SIZE,
            CONV5_KERNEL_SIZE,
        ]
        self.conv5_kernel = he_uniform(conv5_shape, DType.float32)
        var conv5_bias_shape: List[Int] = [CONV5_OUT_CHANNELS]
        self.conv5_bias = zeros(conv5_bias_shape, DType.float32)

        # FC1: flattened_size -> FC1_OUT_FEATURES (derived from conv/pool layers)
        var fc1_shape: List[Int] = [FC1_OUT_FEATURES, flattened_size]
        self.fc1_weights = xavier_uniform(fc1_shape, DType.float32)
        var fc1_bias_shape: List[Int] = [FC1_OUT_FEATURES]
        self.fc1_bias = zeros(fc1_bias_shape, DType.float32)

        # FC2: FC1_OUT_FEATURES -> FC2_OUT_FEATURES
        var fc2_shape: List[Int] = [FC2_OUT_FEATURES, FC1_OUT_FEATURES]
        self.fc2_weights = xavier_uniform(fc2_shape, DType.float32)
        var fc2_bias_shape: List[Int] = [FC2_OUT_FEATURES]
        self.fc2_bias = zeros(fc2_bias_shape, DType.float32)

        # FC3: FC2_OUT_FEATURES -> num_classes
        var fc3_shape: List[Int] = [num_classes, FC2_OUT_FEATURES]
        self.fc3_weights = xavier_uniform(fc3_shape, DType.float32)
        var fc3_bias_shape: List[Int] = [num_classes]
        self.fc3_bias = zeros(fc3_bias_shape, DType.float32)

    fn forward(
        mut self, input: AnyTensor, training: Bool = True
    ) raises -> AnyTensor:
        """Forward pass through AlexNet.

        Args:
            input: Input tensor of shape (batch, 3, 32, 32).
            training: Whether in training mode (applies dropout if True).

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

        # Conv3 + ReLU
        var conv3_out = conv2d(
            pool2_out,
            self.conv3_kernel,
            self.conv3_bias,
            stride=CONV3_STRIDE,
            padding=CONV3_PADDING,
        )
        var relu3_out = relu(conv3_out)

        # Conv4 + ReLU
        var conv4_out = conv2d(
            relu3_out,
            self.conv4_kernel,
            self.conv4_bias,
            stride=CONV4_STRIDE,
            padding=CONV4_PADDING,
        )
        var relu4_out = relu(conv4_out)

        # Conv5 + ReLU + MaxPool
        var conv5_out = conv2d(
            relu4_out,
            self.conv5_kernel,
            self.conv5_bias,
            stride=CONV5_STRIDE,
            padding=CONV5_PADDING,
        )
        var relu5_out = relu(conv5_out)
        var pool3_out = maxpool2d(
            relu5_out,
            kernel_size=POOL3_KERNEL_SIZE,
            stride=POOL3_STRIDE,
            padding=POOL3_PADDING,
        )

        # Flatten: (batch, 256, 1, 1) -> (batch, 256)
        var pool3_shape = pool3_out.shape()
        var batch_size = pool3_shape[0]
        var flattened_size = pool3_shape[1] * pool3_shape[2] * pool3_shape[3]

        var flatten_shape = List[Int]()
        flatten_shape.append(batch_size)
        flatten_shape.append(flattened_size)
        var flattened = pool3_out.reshape(flatten_shape)

        # FC1 + ReLU + Dropout
        var fc1_out = linear(flattened, self.fc1_weights, self.fc1_bias)
        var relu6_out = relu(fc1_out)
        var drop1_result = dropout(
            relu6_out, Float64(self.dropout_rate), training
        )
        var drop1_out = drop1_result[0]

        # FC2 + ReLU + Dropout
        var fc2_out = linear(drop1_out, self.fc2_weights, self.fc2_bias)
        var relu7_out = relu(fc2_out)
        var drop2_result = dropout(
            relu7_out, Float64(self.dropout_rate), training
        )
        var drop2_out = drop2_result[0]

        # FC3 (output logits)
        var output = linear(drop2_out, self.fc3_weights, self.fc3_bias)

        return output

    fn predict(mut self, input: AnyTensor) raises -> Int:
        """Predict class for a single input.

        Args:
            input: Input tensor of shape (1, 3, 32, 32).

        Returns:
            Predicted class index (0 to num_classes-1).
        """
        var logits = self.forward(input, training=False)

        # Find argmax
        var logits_shape = logits.shape()
        var max_idx = 0
        var max_val = logits[0]

        for i in range(1, logits_shape[1]):
            if logits[i] > max_val:
                max_val = logits[i]
                max_idx = i

        return max_idx

    fn save_weights(self, weights_dir: String) raises:
        """Save model weights to directory.

        Args:
            weights_dir: Directory to save weight files (one file per parameter).

        Note:
            Creates directory if it doesn't exist. Each parameter saved as:
            - conv1_kernel.weights, conv1_bias.weights, etc.
        """
        # Collect all parameters
        var parameters: List[AnyTensor] = []
        parameters.append(self.conv1_kernel)
        parameters.append(self.conv1_bias)
        parameters.append(self.conv2_kernel)
        parameters.append(self.conv2_bias)
        parameters.append(self.conv3_kernel)
        parameters.append(self.conv3_bias)
        parameters.append(self.conv4_kernel)
        parameters.append(self.conv4_bias)
        parameters.append(self.conv5_kernel)
        parameters.append(self.conv5_bias)
        parameters.append(self.fc1_weights)
        parameters.append(self.fc1_bias)
        parameters.append(self.fc2_weights)
        parameters.append(self.fc2_bias)
        parameters.append(self.fc3_weights)
        parameters.append(self.fc3_bias)

        # Get standard parameter names
        var param_names = get_model_parameter_names("alexnet")

        # Save using shared utility
        save_model_weights(parameters, weights_dir, param_names)

    fn load_weights(mut self, weights_dir: String) raises:
        """Load model weights from directory.

        Args:
            weights_dir: Directory containing weight files.

        Raises:
            Error: If weight files are missing or have incompatible shapes.
        """
        # Get standard parameter names
        var param_names = get_model_parameter_names("alexnet")

        # Create empty list for loaded parameters
        var loaded_params: List[AnyTensor] = []

        # Load using shared utility
        load_model_weights(loaded_params, weights_dir, param_names)

        # Assign loaded parameters to model fields
        self.conv1_kernel = loaded_params[0]
        self.conv1_bias = loaded_params[1]
        self.conv2_kernel = loaded_params[2]
        self.conv2_bias = loaded_params[3]
        self.conv3_kernel = loaded_params[4]
        self.conv3_bias = loaded_params[5]
        self.conv4_kernel = loaded_params[6]
        self.conv4_bias = loaded_params[7]
        self.conv5_kernel = loaded_params[8]
        self.conv5_bias = loaded_params[9]
        self.fc1_weights = loaded_params[10]
        self.fc1_bias = loaded_params[11]
        self.fc2_weights = loaded_params[12]
        self.fc2_bias = loaded_params[13]
        self.fc3_weights = loaded_params[14]
        self.fc3_bias = loaded_params[15]

    fn update_parameters(
        mut self,
        learning_rate: Float32,
        momentum: Float32,
        grad_conv1_kernel: AnyTensor,
        grad_conv1_bias: AnyTensor,
        grad_conv2_kernel: AnyTensor,
        grad_conv2_bias: AnyTensor,
        grad_conv3_kernel: AnyTensor,
        grad_conv3_bias: AnyTensor,
        grad_conv4_kernel: AnyTensor,
        grad_conv4_bias: AnyTensor,
        grad_conv5_kernel: AnyTensor,
        grad_conv5_bias: AnyTensor,
        grad_fc1_weights: AnyTensor,
        grad_fc1_bias: AnyTensor,
        grad_fc2_weights: AnyTensor,
        grad_fc2_bias: AnyTensor,
        grad_fc3_weights: AnyTensor,
        grad_fc3_bias: AnyTensor,
        mut velocity_conv1_kernel: AnyTensor,
        mut velocity_conv1_bias: AnyTensor,
        mut velocity_conv2_kernel: AnyTensor,
        mut velocity_conv2_bias: AnyTensor,
        mut velocity_conv3_kernel: AnyTensor,
        mut velocity_conv3_bias: AnyTensor,
        mut velocity_conv4_kernel: AnyTensor,
        mut velocity_conv4_bias: AnyTensor,
        mut velocity_conv5_kernel: AnyTensor,
        mut velocity_conv5_bias: AnyTensor,
        mut velocity_fc1_weights: AnyTensor,
        mut velocity_fc1_bias: AnyTensor,
        mut velocity_fc2_weights: AnyTensor,
        mut velocity_fc2_bias: AnyTensor,
        mut velocity_fc3_weights: AnyTensor,
        mut velocity_fc3_bias: AnyTensor,
    ) raises:
        """Update parameters using SGD with momentum.

        Args:
            learning_rate: Learning rate for gradient descent.
            momentum: Momentum factor (typically 0.9).
            grad_conv1_kernel: Gradient for conv1 kernel weights.
            grad_conv1_bias: Gradient for conv1 bias.
            grad_conv2_kernel: Gradient for conv2 kernel weights.
            grad_conv2_bias: Gradient for conv2 bias.
            grad_conv3_kernel: Gradient for conv3 kernel weights.
            grad_conv3_bias: Gradient for conv3 bias.
            grad_conv4_kernel: Gradient for conv4 kernel weights.
            grad_conv4_bias: Gradient for conv4 bias.
            grad_conv5_kernel: Gradient for conv5 kernel weights.
            grad_conv5_bias: Gradient for conv5 bias.
            grad_fc1_weights: Gradient for fc1 weights.
            grad_fc1_bias: Gradient for fc1 bias.
            grad_fc2_weights: Gradient for fc2 weights.
            grad_fc2_bias: Gradient for fc2 bias.
            grad_fc3_weights: Gradient for fc3 weights.
            grad_fc3_bias: Gradient for fc3 bias.
            velocity_conv1_kernel: Velocity for conv1 kernel weights.
            velocity_conv1_bias: Velocity for conv1 bias.
            velocity_conv2_kernel: Velocity for conv2 kernel weights.
            velocity_conv2_bias: Velocity for conv2 bias.
            velocity_conv3_kernel: Velocity for conv3 kernel weights.
            velocity_conv3_bias: Velocity for conv3 bias.
            velocity_conv4_kernel: Velocity for conv4 kernel weights.
            velocity_conv4_bias: Velocity for conv4 bias.
            velocity_conv5_kernel: Velocity for conv5 kernel weights.
            velocity_conv5_bias: Velocity for conv5 bias.
            velocity_fc1_weights: Velocity for fc1 weights.
            velocity_fc1_bias: Velocity for fc1 bias.
            velocity_fc2_weights: Velocity for fc2 weights.
            velocity_fc2_bias: Velocity for fc2 bias.
            velocity_fc3_weights: Velocity for fc3 weights.
            velocity_fc3_bias: Velocity for fc3 bias.
        """
        # SGD with momentum update: v = momentum * v - lr * grad; param = param + v
        # Convert Float32 parameters to Float64 for shared library
        var lr = Float64(learning_rate)
        var mom = Float64(momentum)
        sgd_momentum_update_inplace(
            self.conv1_kernel, grad_conv1_kernel, velocity_conv1_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv1_bias, grad_conv1_bias, velocity_conv1_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv2_kernel, grad_conv2_kernel, velocity_conv2_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv2_bias, grad_conv2_bias, velocity_conv2_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv3_kernel, grad_conv3_kernel, velocity_conv3_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv3_bias, grad_conv3_bias, velocity_conv3_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv4_kernel, grad_conv4_kernel, velocity_conv4_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv4_bias, grad_conv4_bias, velocity_conv4_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv5_kernel, grad_conv5_kernel, velocity_conv5_kernel, lr, mom
        )
        sgd_momentum_update_inplace(
            self.conv5_bias, grad_conv5_bias, velocity_conv5_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc1_weights, grad_fc1_weights, velocity_fc1_weights, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc1_bias, grad_fc1_bias, velocity_fc1_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc2_weights, grad_fc2_weights, velocity_fc2_weights, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc2_bias, grad_fc2_bias, velocity_fc2_bias, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc3_weights, grad_fc3_weights, velocity_fc3_weights, lr, mom
        )
        sgd_momentum_update_inplace(
            self.fc3_bias, grad_fc3_bias, velocity_fc3_bias, lr, mom
        )


fn main() raises:
    """Entry point for build validation.

    This function exists solely to allow `mojo build` to compile the module.
    The actual model is used as a library by train.mojo and inference.mojo.
    """
    print("AlexNet model module compiled successfully")
