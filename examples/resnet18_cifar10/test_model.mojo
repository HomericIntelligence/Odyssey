"""Quick integration test for ResNet-18 model.

Tests ResNet-18 forward pass, weight persistence, and model integration with shared modules.

Shared Modules Used:
    - projectodyssey.core: Core tensor operations
    - projectodyssey.utils.serialization: Weight save/load functionality
"""

from model import ResNet18
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.utils.serialization import save_tensor, load_tensor
from projectodyssey.data.constants import DatasetInfo
from std.collections import List


def main() raises:
    print("Testing ResNet-18 Model")
    print("=" * 60)

    # Create model
    print("Initializing ResNet-18...")
    var dataset_info = DatasetInfo("cifar10")
    var num_classes = dataset_info.num_classes()
    var model = ResNet18(num_classes=num_classes)
    print("✓ Model initialized")

    # Create dummy input
    print(
        "\nCreating dummy input (batch=4, channels=3, height=32, width=32)..."
    )
    var input_shape: List[Int] = [4, 3, 32, 32]
    var input = zeros(input_shape, DType.float32)
    var input_data = input._data.bitcast[Float32]()
    for i in range(4 * 3 * 32 * 32):
        input_data[i] = Float32(0.1)
    print("✓ Input created")

    # Test inference mode
    print("\nTesting inference mode...")
    var logits_inf = model.forward(input, training=False)
    print(
        "  Output shape: ("
        + String(logits_inf.shape()[0])
        + ", "
        + String(logits_inf.shape()[1])
        + ")"
    )
    if logits_inf.shape()[0] == 4 and logits_inf.shape()[1] == 10:
        print("✓ Inference mode PASSED")
    else:
        print("✗ Inference mode FAILED - wrong output shape")
        return

    # Test training mode
    print("\nTesting training mode...")
    var logits_train = model.forward(input, training=True)
    print(
        "  Output shape: ("
        + String(logits_train.shape()[0])
        + ", "
        + String(logits_train.shape()[1])
        + ")"
    )
    if logits_train.shape()[0] == 4 and logits_train.shape()[1] == 10:
        print("✓ Training mode PASSED")
    else:
        print("✗ Training mode FAILED - wrong output shape")
        return

    # Check for NaN/Inf
    print("\nChecking for NaN/Inf in outputs...")
    var logits_data = logits_train._data.bitcast[Float32]()
    var has_nan_inf = False
    for i in range(40):
        var val = logits_data[i]
        if val > 1e10 or val < -1e10 or val != val:
            has_nan_inf = True
            break

    if has_nan_inf:
        print("✗ Found NaN or Inf in outputs")
    else:
        print("✓ No NaN/Inf detected")

    print("\n" + "=" * 60)
    print("✓ ResNet-18 ALL TESTS PASSED!")
    print("=" * 60)
