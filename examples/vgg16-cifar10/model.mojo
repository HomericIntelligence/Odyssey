"""VGG-16 Model for CIFAR-10 Classification

Classic VGG-16 architecture adapted for CIFAR-10 dataset (32x32 RGB images).

Architecture:
    VGG-16 consists of 13 convolutional layers and 3 fully connected layers,
    organized into 5 blocks. Each block uses 3x3 convolutions followed by max pooling.

    Input (32x32x3) ->

    Block 1:
        Conv2D(64, 3x3, pad=1) -> ReLU ->
        Conv2D(64, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (16x16x64)

    Block 2:
        Conv2D(128, 3x3, pad=1) -> ReLU ->
        Conv2D(128, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (8x8x128)

    Block 3:
        Conv2D(256, 3x3, pad=1) -> ReLU ->
        Conv2D(256, 3x3, pad=1) -> ReLU ->
        Conv2D(256, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (4x4x256)

    Block 4:
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (2x2x512)

    Block 5:
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        Conv2D(512, 3x3, pad=1) -> ReLU ->
        MaxPool(2x2, stride=2) -> (1x1x512)

    Flatten (512x1x1 = 512) ->
    Linear(512 -> 512) -> ReLU -> Dropout(0.5) ->
    Linear(512 -> 512) -> ReLU -> Dropout(0.5) ->
    Linear(512 -> 10)

Key Innovation:
    - Depth: 16 layers (very deep for 2014)
    - Uniform 3x3 convolutions throughout
    - Simplicity: repetitive architecture, easy to implement
    - Small receptive fields with more non-linearities

References:
    - Simonyan, K., & Zisserman, A. (2014).
      Very deep convolutional networks for large-scale image recognition.
      arXiv preprint arXiv:1409.1556.
    - CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from shared.core import AnyTensor, zeros
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.dropout import dropout, dropout_backward
from shared.core.initializers import he_uniform
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
# VGG-16 uses uniform 3x3 convolutions with padding=1 (preserving size),
# followed by 2x2 max pooling with stride=2 (halving size).

# Input dimensions (CIFAR-10)
comptime INPUT_HEIGHT = 32
comptime INPUT_WIDTH = 32
comptime INPUT_CHANNELS = 3

# VGG-16 uses uniform conv parameters: 3x3 kernel, stride=1, padding=1
comptime CONV_KERNEL_SIZE = 3
comptime CONV_STRIDE = 1
comptime CONV_PADDING = 1

# VGG-16 uses uniform pool parameters: 2x2 kernel, stride=2, padding=0
comptime POOL_KERNEL_SIZE = 2
comptime POOL_STRIDE = 2
comptime POOL_PADDING = 0

# Block 1 channel sizes
comptime BLOCK1_CHANNELS = 64

# Block 2 channel sizes
comptime BLOCK2_CHANNELS = 128

# Block 3 channel sizes
comptime BLOCK3_CHANNELS = 256

# Block 4 channel sizes
comptime BLOCK4_CHANNELS = 512

# Block 5 channel sizes
comptime BLOCK5_CHANNELS = 512

# Fully connected layer sizes
comptime FC1_OUT_FEATURES = 512
comptime FC2_OUT_FEATURES = 512


fn compute_flattened_size() -> Int:
    """Compute the flattened feature size after all conv/pool layers.

    VGG-16 uses 5 pooling layers, each halving spatial dimensions:
    32 -> 16 -> 8 -> 4 -> 2 -> 1

    Returns:
        Number of features after flattening (channels * height * width).
    """
    # VGG-16: Each block has 3x3 convs with padding=1 (size preserved)
    # followed by 2x2 pool with stride=2 (size halved)
    var h, w = INPUT_HEIGHT, INPUT_WIDTH

    # After Block 1 pool: 32 -> 16
    h, w = pool_output_shape(h, w, POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING)

    # After Block 2 pool: 16 -> 8
    h, w = pool_output_shape(h, w, POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING)

    # After Block 3 pool: 8 -> 4
    h, w = pool_output_shape(h, w, POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING)

    # After Block 4 pool: 4 -> 2
    h, w = pool_output_shape(h, w, POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING)

    # After Block 5 pool: 2 -> 1
    h, w = pool_output_shape(h, w, POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING)

    return BLOCK5_CHANNELS * h * w


struct VGG16:
    """VGG-16 model for CIFAR-10 classification.

    Attributes:
        num_classes: Number of output classes (10 for CIFAR-10)
        dropout_rate: Dropout probability for FC layers (default 0.5)

        # Block 1 (2 conv layers)
        conv1_1_kernel, conv1_1_bias: First conv layer (64, 3, 3, 3)
        conv1_2_kernel, conv1_2_bias: Second conv layer (64, 64, 3, 3)

        # Block 2 (2 conv layers)
        conv2_1_kernel, conv2_1_bias: Third conv layer (128, 64, 3, 3)
        conv2_2_kernel, conv2_2_bias: Fourth conv layer (128, 128, 3, 3)

        # Block 3 (3 conv layers)
        conv3_1_kernel, conv3_1_bias: Fifth conv layer (256, 128, 3, 3)
        conv3_2_kernel, conv3_2_bias: Sixth conv layer (256, 256, 3, 3)
        conv3_3_kernel, conv3_3_bias: Seventh conv layer (256, 256, 3, 3)

        # Block 4 (3 conv layers)
        conv4_1_kernel, conv4_1_bias: Eighth conv layer (512, 256, 3, 3)
        conv4_2_kernel, conv4_2_bias: Ninth conv layer (512, 512, 3, 3)
        conv4_3_kernel, conv4_3_bias: Tenth conv layer (512, 512, 3, 3)

        # Block 5 (3 conv layers)
        conv5_1_kernel, conv5_1_bias: Eleventh conv layer (512, 512, 3, 3)
        conv5_2_kernel, conv5_2_bias: Twelfth conv layer (512, 512, 3, 3)
        conv5_3_kernel, conv5_3_bias: Thirteenth conv layer (512, 512, 3, 3)

        # Fully connected layers
        fc1_weights, fc1_bias: First FC layer (512, 512)
        fc2_weights, fc2_bias: Second FC layer (512, 512)
        fc3_weights, fc3_bias: Third FC layer (num_classes, 512).
    """

    var num_classes: Int
    var dropout_rate: Float32

    # Block 1 parameters
    var conv1_1_kernel: AnyTensor
    var conv1_1_bias: AnyTensor
    var conv1_2_kernel: AnyTensor
    var conv1_2_bias: AnyTensor

    # Block 2 parameters
    var conv2_1_kernel: AnyTensor
    var conv2_1_bias: AnyTensor
    var conv2_2_kernel: AnyTensor
    var conv2_2_bias: AnyTensor

    # Block 3 parameters
    var conv3_1_kernel: AnyTensor
    var conv3_1_bias: AnyTensor
    var conv3_2_kernel: AnyTensor
    var conv3_2_bias: AnyTensor
    var conv3_3_kernel: AnyTensor
    var conv3_3_bias: AnyTensor

    # Block 4 parameters
    var conv4_1_kernel: AnyTensor
    var conv4_1_bias: AnyTensor
    var conv4_2_kernel: AnyTensor
    var conv4_2_bias: AnyTensor
    var conv4_3_kernel: AnyTensor
    var conv4_3_bias: AnyTensor

    # Block 5 parameters
    var conv5_1_kernel: AnyTensor
    var conv5_1_bias: AnyTensor
    var conv5_2_kernel: AnyTensor
    var conv5_2_bias: AnyTensor
    var conv5_3_kernel: AnyTensor
    var conv5_3_bias: AnyTensor

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
        """Initialize VGG-16 model with random weights.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10).
            dropout_rate: Dropout probability for FC layers (default: 0.5).
        """
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Compute derived dimensions using shared shape utilities
        var flattened_size = compute_flattened_size()

        # Block 1: INPUT_CHANNELS -> BLOCK1_CHANNELS
        self.conv1_1_kernel = he_uniform(
            [
                BLOCK1_CHANNELS,
                INPUT_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv1_1_bias = zeros([BLOCK1_CHANNELS], DType.float32)
        self.conv1_2_kernel = he_uniform(
            [
                BLOCK1_CHANNELS,
                BLOCK1_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv1_2_bias = zeros([BLOCK1_CHANNELS], DType.float32)

        # Block 2: BLOCK1_CHANNELS -> BLOCK2_CHANNELS
        self.conv2_1_kernel = he_uniform(
            [
                BLOCK2_CHANNELS,
                BLOCK1_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv2_1_bias = zeros([BLOCK2_CHANNELS], DType.float32)
        self.conv2_2_kernel = he_uniform(
            [
                BLOCK2_CHANNELS,
                BLOCK2_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv2_2_bias = zeros([BLOCK2_CHANNELS], DType.float32)

        # Block 3: BLOCK2_CHANNELS -> BLOCK3_CHANNELS
        self.conv3_1_kernel = he_uniform(
            [
                BLOCK3_CHANNELS,
                BLOCK2_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv3_1_bias = zeros([BLOCK3_CHANNELS], DType.float32)
        self.conv3_2_kernel = he_uniform(
            [
                BLOCK3_CHANNELS,
                BLOCK3_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv3_2_bias = zeros([BLOCK3_CHANNELS], DType.float32)
        self.conv3_3_kernel = he_uniform(
            [
                BLOCK3_CHANNELS,
                BLOCK3_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv3_3_bias = zeros([BLOCK3_CHANNELS], DType.float32)

        # Block 4: BLOCK3_CHANNELS -> BLOCK4_CHANNELS
        self.conv4_1_kernel = he_uniform(
            [
                BLOCK4_CHANNELS,
                BLOCK3_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv4_1_bias = zeros([BLOCK4_CHANNELS], DType.float32)
        self.conv4_2_kernel = he_uniform(
            [
                BLOCK4_CHANNELS,
                BLOCK4_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv4_2_bias = zeros([BLOCK4_CHANNELS], DType.float32)
        self.conv4_3_kernel = he_uniform(
            [
                BLOCK4_CHANNELS,
                BLOCK4_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv4_3_bias = zeros([BLOCK4_CHANNELS], DType.float32)

        # Block 5: BLOCK4_CHANNELS -> BLOCK5_CHANNELS
        self.conv5_1_kernel = he_uniform(
            [
                BLOCK5_CHANNELS,
                BLOCK4_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv5_1_bias = zeros([BLOCK5_CHANNELS], DType.float32)
        self.conv5_2_kernel = he_uniform(
            [
                BLOCK5_CHANNELS,
                BLOCK5_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv5_2_bias = zeros([BLOCK5_CHANNELS], DType.float32)
        self.conv5_3_kernel = he_uniform(
            [
                BLOCK5_CHANNELS,
                BLOCK5_CHANNELS,
                CONV_KERNEL_SIZE,
                CONV_KERNEL_SIZE,
            ],
            DType.float32,
        )
        self.conv5_3_bias = zeros([BLOCK5_CHANNELS], DType.float32)

        # FC1: flattened_size -> FC1_OUT_FEATURES (derived from conv/pool layers)
        self.fc1_weights = he_uniform(
            [FC1_OUT_FEATURES, flattened_size], DType.float32
        )
        self.fc1_bias = zeros([FC1_OUT_FEATURES], DType.float32)

        # FC2: FC1_OUT_FEATURES -> FC2_OUT_FEATURES
        self.fc2_weights = he_uniform(
            [FC2_OUT_FEATURES, FC1_OUT_FEATURES], DType.float32
        )
        self.fc2_bias = zeros([FC2_OUT_FEATURES], DType.float32)

        # FC3: FC2_OUT_FEATURES -> num_classes
        self.fc3_weights = he_uniform(
            [num_classes, FC2_OUT_FEATURES], DType.float32
        )
        self.fc3_bias = zeros([num_classes], DType.float32)

    fn forward(
        mut self, input: AnyTensor, training: Bool = True
    ) raises -> AnyTensor:
        """Forward pass through VGG-16.

        Args:
            input: Input tensor of shape (batch, 3, 32, 32).
            training: Whether in training mode (applies dropout if True).

        Returns:
            Output logits of shape (batch, num_classes).
        """
        # Block 1: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv1_1 = conv2d(
            input,
            self.conv1_1_kernel,
            self.conv1_1_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu1_1 = relu(conv1_1)
        var conv1_2 = conv2d(
            relu1_1,
            self.conv1_2_kernel,
            self.conv1_2_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu1_2 = relu(conv1_2)
        var pool1 = maxpool2d(
            relu1_2,
            kernel_size=POOL_KERNEL_SIZE,
            stride=POOL_STRIDE,
            padding=POOL_PADDING,
        )  # 32x32 -> 16x16

        # Block 2: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv2_1 = conv2d(
            pool1,
            self.conv2_1_kernel,
            self.conv2_1_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu2_1 = relu(conv2_1)
        var conv2_2 = conv2d(
            relu2_1,
            self.conv2_2_kernel,
            self.conv2_2_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu2_2 = relu(conv2_2)
        var pool2 = maxpool2d(
            relu2_2,
            kernel_size=POOL_KERNEL_SIZE,
            stride=POOL_STRIDE,
            padding=POOL_PADDING,
        )  # 16x16 -> 8x8

        # Block 3: Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv3_1 = conv2d(
            pool2,
            self.conv3_1_kernel,
            self.conv3_1_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu3_1 = relu(conv3_1)
        var conv3_2 = conv2d(
            relu3_1,
            self.conv3_2_kernel,
            self.conv3_2_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu3_2 = relu(conv3_2)
        var conv3_3 = conv2d(
            relu3_2,
            self.conv3_3_kernel,
            self.conv3_3_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu3_3 = relu(conv3_3)
        var pool3 = maxpool2d(
            relu3_3,
            kernel_size=POOL_KERNEL_SIZE,
            stride=POOL_STRIDE,
            padding=POOL_PADDING,
        )  # 8x8 -> 4x4

        # Block 4: Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv4_1 = conv2d(
            pool3,
            self.conv4_1_kernel,
            self.conv4_1_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu4_1 = relu(conv4_1)
        var conv4_2 = conv2d(
            relu4_1,
            self.conv4_2_kernel,
            self.conv4_2_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu4_2 = relu(conv4_2)
        var conv4_3 = conv2d(
            relu4_2,
            self.conv4_3_kernel,
            self.conv4_3_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu4_3 = relu(conv4_3)
        var pool4 = maxpool2d(
            relu4_3,
            kernel_size=POOL_KERNEL_SIZE,
            stride=POOL_STRIDE,
            padding=POOL_PADDING,
        )  # 4x4 -> 2x2

        # Block 5: Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU -> MaxPool
        var conv5_1 = conv2d(
            pool4,
            self.conv5_1_kernel,
            self.conv5_1_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu5_1 = relu(conv5_1)
        var conv5_2 = conv2d(
            relu5_1,
            self.conv5_2_kernel,
            self.conv5_2_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu5_2 = relu(conv5_2)
        var conv5_3 = conv2d(
            relu5_2,
            self.conv5_3_kernel,
            self.conv5_3_bias,
            stride=CONV_STRIDE,
            padding=CONV_PADDING,
        )
        var relu5_3 = relu(conv5_3)
        var pool5 = maxpool2d(
            relu5_3,
            kernel_size=POOL_KERNEL_SIZE,
            stride=POOL_STRIDE,
            padding=POOL_PADDING,
        )  # 2x2 -> 1x1

        # Flatten: (batch, 512, 1, 1) -> (batch, 512)
        var pool5_shape = pool5.shape()
        var batch_size = pool5_shape[0]
        var flattened_size = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]

        var flatten_shape = List[Int]()
        flatten_shape.append(batch_size)
        flatten_shape.append(flattened_size)
        var flattened = pool5.reshape(flatten_shape)

        # FC1 + ReLU + Dropout
        var fc1 = linear(flattened, self.fc1_weights, self.fc1_bias)
        var relu_fc1 = relu(fc1)
        var drop1_result = dropout(
            relu_fc1, Float64(self.dropout_rate), training
        )
        var drop1 = drop1_result[0]

        # FC2 + ReLU + Dropout
        var fc2 = linear(drop1, self.fc2_weights, self.fc2_bias)
        var relu_fc2 = relu(fc2)
        var drop2_result = dropout(
            relu_fc2, Float64(self.dropout_rate), training
        )
        var drop2 = drop2_result[0]

        # FC3 (output logits)
        var output = linear(drop2, self.fc3_weights, self.fc3_bias)

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
            - conv1_1_kernel.weights, conv1_1_bias.weights, etc.
        """
        # Collect all parameters in order
        var parameters: List[AnyTensor] = []
        parameters.append(self.conv1_1_kernel)
        parameters.append(self.conv1_1_bias)
        parameters.append(self.conv1_2_kernel)
        parameters.append(self.conv1_2_bias)
        parameters.append(self.conv2_1_kernel)
        parameters.append(self.conv2_1_bias)
        parameters.append(self.conv2_2_kernel)
        parameters.append(self.conv2_2_bias)
        parameters.append(self.conv3_1_kernel)
        parameters.append(self.conv3_1_bias)
        parameters.append(self.conv3_2_kernel)
        parameters.append(self.conv3_2_bias)
        parameters.append(self.conv3_3_kernel)
        parameters.append(self.conv3_3_bias)
        parameters.append(self.conv4_1_kernel)
        parameters.append(self.conv4_1_bias)
        parameters.append(self.conv4_2_kernel)
        parameters.append(self.conv4_2_bias)
        parameters.append(self.conv4_3_kernel)
        parameters.append(self.conv4_3_bias)
        parameters.append(self.conv5_1_kernel)
        parameters.append(self.conv5_1_bias)
        parameters.append(self.conv5_2_kernel)
        parameters.append(self.conv5_2_bias)
        parameters.append(self.conv5_3_kernel)
        parameters.append(self.conv5_3_bias)
        parameters.append(self.fc1_weights)
        parameters.append(self.fc1_bias)
        parameters.append(self.fc2_weights)
        parameters.append(self.fc2_bias)
        parameters.append(self.fc3_weights)
        parameters.append(self.fc3_bias)

        # Get standard parameter names
        var param_names = get_model_parameter_names("vgg16")

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
        var param_names = get_model_parameter_names("vgg16")

        # Create empty list for loaded parameters
        var loaded_params: List[AnyTensor] = []

        # Load using shared utility
        load_model_weights(loaded_params, weights_dir, param_names)

        # Assign loaded parameters to model fields
        self.conv1_1_kernel = loaded_params[0]
        self.conv1_1_bias = loaded_params[1]
        self.conv1_2_kernel = loaded_params[2]
        self.conv1_2_bias = loaded_params[3]
        self.conv2_1_kernel = loaded_params[4]
        self.conv2_1_bias = loaded_params[5]
        self.conv2_2_kernel = loaded_params[6]
        self.conv2_2_bias = loaded_params[7]
        self.conv3_1_kernel = loaded_params[8]
        self.conv3_1_bias = loaded_params[9]
        self.conv3_2_kernel = loaded_params[10]
        self.conv3_2_bias = loaded_params[11]
        self.conv3_3_kernel = loaded_params[12]
        self.conv3_3_bias = loaded_params[13]
        self.conv4_1_kernel = loaded_params[14]
        self.conv4_1_bias = loaded_params[15]
        self.conv4_2_kernel = loaded_params[16]
        self.conv4_2_bias = loaded_params[17]
        self.conv4_3_kernel = loaded_params[18]
        self.conv4_3_bias = loaded_params[19]
        self.conv5_1_kernel = loaded_params[20]
        self.conv5_1_bias = loaded_params[21]
        self.conv5_2_kernel = loaded_params[22]
        self.conv5_2_bias = loaded_params[23]
        self.conv5_3_kernel = loaded_params[24]
        self.conv5_3_bias = loaded_params[25]
        self.fc1_weights = loaded_params[26]
        self.fc1_bias = loaded_params[27]
        self.fc2_weights = loaded_params[28]
        self.fc2_bias = loaded_params[29]
        self.fc3_weights = loaded_params[30]
        self.fc3_bias = loaded_params[31]


fn main() raises:
    """Entry point for build validation.

    This function exists solely to allow `mojo build` to compile the module.
    The actual model is used as a library by train.mojo and inference.mojo.
    """
    print("VGG-16 model module compiled successfully")
