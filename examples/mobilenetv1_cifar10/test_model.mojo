"""Quick integration test for MobileNetV1 model."""

from model import MobileNetV1
from shared.tensor.any_tensor import AnyTensor, zeros


def main() raises:
    print("Testing MobileNetV1 Model")
    print("=" * 60)

    print("Initializing MobileNetV1...")
    var model = MobileNetV1(num_classes=10)
    print("✓ Model initialized")

    print(
        "\nCreating dummy input (batch=2, channels=3, height=32, width=32)..."
    )
    var input = zeros([2, 3, 32, 32], DType.float32)
    var input_data = input._data.bitcast[Float32]()
    for i in range(2 * 3 * 32 * 32):
        input_data[i] = Float32(0.1)
    print("✓ Input created")

    print("\nTesting inference mode...")
    var logits_inf = model.forward(input, training=False)
    print(
        "  Output shape: ("
        + String(logits_inf.shape()[0])
        + ", "
        + String(logits_inf.shape()[1])
        + ")"
    )
    if logits_inf.shape()[0] == 2 and logits_inf.shape()[1] == 10:
        print("✓ Inference mode PASSED")
    else:
        print("✗ Inference mode FAILED")
        return

    print("\nTesting training mode...")
    var logits_train = model.forward(input, training=True)
    print(
        "  Output shape: ("
        + String(logits_train.shape()[0])
        + ", "
        + String(logits_train.shape()[1])
        + ")"
    )
    if logits_train.shape()[0] == 2 and logits_train.shape()[1] == 10:
        print("✓ Training mode PASSED")
    else:
        print("✗ Training mode FAILED")
        return

    print("\n" + "=" * 60)
    print("✓ MobileNetV1 ALL TESTS PASSED!")
    print("=" * 60)
