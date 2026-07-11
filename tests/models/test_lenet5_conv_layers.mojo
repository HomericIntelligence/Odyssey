"""LeNet-5 Convolutional Layer Tests.

Tests Conv1 and Conv2 layers independently with special FP-representable values.

Workaround for Issue #2942: This file contains <10 tests to avoid heap corruption
bug that occurs after running 15+ cumulative tests.

Tests:
- Conv1 (1→6 channels, 5x5 kernel): forward float32, float16, backward float32
- Conv2 (6→16 channels, 5x5 kernel): forward float32, float16, backward float32
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.testing.layer_params import ConvFixture
from odyssey.testing.layer_testers import LayerTester


# ============================================================================
# Test Fixtures
# ============================================================================


def create_conv1_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv1 layer parameters (1→6, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=1, out_channels=6, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


def create_conv2_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv2 layer parameters (6→16, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=6, out_channels=16, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


# ============================================================================
# Conv1 Tests
# ============================================================================


def test_conv1_forward_float32() raises:
    """Test Conv1 forward pass (1→6 channels, 5x5 kernel) with float32."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=28,
        input_w=28,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=0,
    )


def test_conv1_forward_float16() raises:
    """Test Conv1 forward pass with float16."""
    var dtype = DType.float16
    var _result = create_conv1_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=28,
        input_w=28,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=0,
    )


def test_conv1_backward_float32() raises:
    """Test Conv1 backward pass with gradient checking (small tensor: 8x8)."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer_backward(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=0,
    )


# ============================================================================
# Conv2 Tests
# ============================================================================


def test_conv2_forward_float32() raises:
    """Test Conv2 forward pass (6→16 channels, 5x5 kernel) with float32."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    # Input after pool1 is smaller: 14x14
    LayerTester.test_conv_layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=14,
        input_w=14,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=0,
    )


def test_conv2_forward_float16() raises:
    """Test Conv2 forward pass with float16."""
    var dtype = DType.float16
    var _result = create_conv2_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=14,
        input_w=14,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=0,
    )


def test_conv2_backward_float32() raises:
    """Test Conv2 backward pass with gradient checking (small tensor: 8x8)."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer_backward(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=0,
    )


def main() raises:
    """Run all convolutional layer tests."""
    print("LeNet-5 Convolutional Layer Tests")
    print("=" * 50)

    # Conv1 tests
    print("  test_conv1_forward_float32...", end="")
    test_conv1_forward_float32()
    print(" OK")

    print("  test_conv1_forward_float16...", end="")
    test_conv1_forward_float16()
    print(" OK")

    print("  test_conv1_backward_float32...", end="")
    test_conv1_backward_float32()
    print(" OK")

    # Conv2 tests
    print("  test_conv2_forward_float32...", end="")
    test_conv2_forward_float32()
    print(" OK")

    print("  test_conv2_forward_float16...", end="")
    test_conv2_forward_float16()
    print(" OK")

    print("  test_conv2_backward_float32...", end="")
    test_conv2_backward_float32()
    print(" OK")

    print("\n✅ All convolutional layer tests passed (6/6)")
