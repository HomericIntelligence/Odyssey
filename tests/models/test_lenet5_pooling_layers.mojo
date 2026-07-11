"""LeNet-5 Pooling Layer Tests.

Tests MaxPool layers independently with special FP-representable values.

Workaround for Issue #2942: This file contains <10 tests to avoid heap corruption
bug that occurs after running 15+ cumulative tests.

Tests:
- MaxPool1 (2x2, stride 2): forward float32, float16
- MaxPool2 (2x2, stride 2): forward float32, float16
"""

from odyssey.testing.layer_testers import LayerTester


# ============================================================================
# MaxPool Tests
# ============================================================================


def test_maxpool1_forward_float32() raises:
    """Test MaxPool1 (2x2, stride 2) forward pass with float32."""
    var dtype = DType.float32

    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


def test_maxpool1_forward_float16() raises:
    """Test MaxPool1 forward pass with float16."""
    var dtype = DType.float16

    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


def test_maxpool2_forward_float32() raises:
    """Test MaxPool2 (2x2, stride 2) forward pass with float32."""
    var dtype = DType.float32

    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


def test_maxpool2_forward_float16() raises:
    """Test MaxPool2 forward pass with float16."""
    var dtype = DType.float16

    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


def main() raises:
    """Run all pooling layer tests."""
    print("LeNet-5 Pooling Layer Tests")
    print("=" * 50)

    print("  test_maxpool1_forward_float32...", end="")
    test_maxpool1_forward_float32()
    print(" OK")

    print("  test_maxpool1_forward_float16...", end="")
    test_maxpool1_forward_float16()
    print(" OK")

    print("  test_maxpool2_forward_float32...", end="")
    test_maxpool2_forward_float32()
    print(" OK")

    print("  test_maxpool2_forward_float16...", end="")
    test_maxpool2_forward_float16()
    print(" OK")

    print("\n✅ All pooling layer tests passed (4/4)")
