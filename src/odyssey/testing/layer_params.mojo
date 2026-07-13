"""Test fixtures for model layer testing.

Provides reusable parameter creation patterns to reduce code duplication
in model tests. Each fixture encapsulates the parameter creation logic
for a specific layer type.

Usage:
    from odyssey.testing.test_fixtures import ConvFixture, LinearFixture

    # Create conv layer parameters
    var fixture = ConvFixture(in_channels=3, out_channels=64, kernel_size=5)
    var kernel = fixture.kernel
    var bias = fixture.bias

    # Create linear layer parameters
    var fc_fixture = LinearFixture(in_features=120, out_features=84)
    var weights = fc_fixture.weights
    var fc_bias = fc_fixture.bias
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.initializers import kaiming_uniform


struct ConvFixture:
    """Fixture for convolutional layer parameters.

    Creates kernel and bias tensors with proper Kaiming initialization.
    Reduces duplication in model tests that repeatedly create conv parameters.

    Example:
        ```mojo
        # Before (duplicated across 10+ tests):
        var kernel_shape = List[Int]()
        kernel_shape.append(6)
        kernel_shape.append(1)
        kernel_shape.append(5)
        kernel_shape.append(5)
        var fan_in = 1 * 5 * 5
        var fan_out = 6 * 5 * 5
        var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)
        var bias = zeros([6], dtype)

        # After (single call):
        var fixture = ConvFixture(1, 6, 5, dtype)
        var kernel = fixture.kernel
        var bias = fixture.bias
        ```
    """

    var kernel: AnyTensor
    var bias: AnyTensor
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int

    def __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        dtype: DType = DType.float32,
    ) raises:
        """Initialize conv layer parameters with Kaiming initialization.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size (assumes square kernel).
            dtype: Data type for parameters (default: float32).
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Create kernel with Kaiming initialization
        var kernel_shape = List[Int]()
        kernel_shape.append(out_channels)
        kernel_shape.append(in_channels)
        kernel_shape.append(kernel_size)
        kernel_shape.append(kernel_size)

        var fan_in = in_channels * kernel_size * kernel_size
        var fan_out = out_channels * kernel_size * kernel_size

        self.kernel = kaiming_uniform(
            fan_in, fan_out, kernel_shape, dtype=dtype
        )

        # Create bias (zeros)
        var bias_shape = List[Int]()
        bias_shape.append(out_channels)
        self.bias = zeros(bias_shape, dtype)


struct LinearFixture:
    """Fixture for linear (fully connected) layer parameters.

    Creates weights and bias tensors with proper Kaiming initialization.
    Reduces duplication in model tests that repeatedly create FC parameters.

    Example:
        ```mojo
        # Before (duplicated across 6+ tests):
        var weights_shape = List[Int]()
        weights_shape.append(120)
        weights_shape.append(400)
        var weights = kaiming_uniform(400, 120, weights_shape, dtype=dtype)
        var bias = zeros([120], dtype)

        # After (single call):
        var fixture = LinearFixture(400, 120, dtype)
        var weights = fixture.weights
        var bias = fixture.bias
        ```
    """

    var weights: AnyTensor
    var bias: AnyTensor
    var in_features: Int
    var out_features: Int

    def __init__(
        out self,
        in_features: Int,
        out_features: Int,
        dtype: DType = DType.float32,
    ) raises:
        """Initialize linear layer parameters with Kaiming initialization.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            dtype: Data type for parameters (default: float32).
        """
        self.in_features = in_features
        self.out_features = out_features

        # Create weights with Kaiming initialization
        var weights_shape = List[Int]()
        weights_shape.append(out_features)
        weights_shape.append(in_features)

        self.weights = kaiming_uniform(
            in_features, out_features, weights_shape, dtype=dtype
        )

        # Create bias (zeros)
        var bias_shape = List[Int]()
        bias_shape.append(out_features)
        self.bias = zeros(bias_shape, dtype)
