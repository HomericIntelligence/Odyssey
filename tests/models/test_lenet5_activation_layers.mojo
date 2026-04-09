"""LeNet-5 Activation Layer Tests

Tests ReLU activation layers independently with special FP-representable values.

Workaround for Issue #2942: This file contains <10 tests to avoid heap corruption
bug that occurs after running 15+ cumulative tests.

Tests:
- ReLU1: forward float32, float16, backward float32
"""

from shared.testing.layer_testers import LayerTester


# ============================================================================
# ReLU Tests
# ============================================================================


def test_relu_forward_float32() raises:
    """Test ReLU activation forward pass with float32."""
    var dtype = DType.float32
    var shape: List[Int] = [2, 16, 8, 8]

    LayerTester.test_activation_layer(shape, dtype, activation="relu")


def test_relu_forward_float16() raises:
    """Test ReLU activation forward pass with float16."""
    var dtype = DType.float16
    var shape: List[Int] = [2, 16, 8, 8]

    LayerTester.test_activation_layer(shape, dtype, activation="relu")


def test_relu_backward_float32() raises:
    """Test ReLU backward pass with gradient checking."""
    var dtype = DType.float32
    var shape: List[Int] = [2, 8, 4, 4]

    LayerTester.test_activation_layer_backward(shape, dtype, activation="relu")


def main() raises:
    """Run all activation layer tests."""
    print("LeNet-5 Activation Layer Tests")
    print("=" * 50)

    print("  test_relu_forward_float32...", end="")
    test_relu_forward_float32()
    print(" OK")

    print("  test_relu_forward_float16...", end="")
    test_relu_forward_float16()
    print(" OK")

    print("  test_relu_backward_float32...", end="")
    test_relu_backward_float32()
    print(" OK")

    print("\n✅ All activation layer tests passed (3/3)")
