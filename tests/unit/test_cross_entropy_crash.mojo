"""Test to reproduce the exact cross-entropy crash from training.

This test reproduces the segmentation fault that occurs during LeNet-5 training
when cross_entropy is called with logits shape (2, 47) and one-hot targets.

Stack trace shows crash at:
#11 shared::core::extensor::AnyTensor::__init__ at line 107
#12 shared::core::loss::cross_entropy at line 293
#13 train::compute_gradients at line 127
"""

from shared.core.extensor import AnyTensor, zeros
from shared.core.loss import cross_entropy
from testing import assert_raises


fn test_cross_entropy_small_batch() raises:
    """Test cross-entropy with small batch (2 samples, 47 classes)."""
    print("\n=== Test 1: Cross-Entropy with Small Batch ===")
    print("Creating logits shape (2, 47) and one-hot targets...")

    # Create logits: (2, 47) - 2 samples, 47 classes
    var batch_size = 2
    var num_classes = 47
    var logits_shape = List[Int]()
    logits_shape.append(batch_size)
    logits_shape.append(num_classes)

    var logits = zeros(logits_shape, DType.float32)
    print("Logits created:", logits.shape()[0], "x", logits.shape()[1])

    # Fill with some test values
    for i in range(logits.numel()):
        logits._set_float64(i, 0.1 * Float64(i % 10))

    # Create one-hot targets: (2, 47)
    var targets = zeros(logits_shape, DType.float32)
    print("Targets created:", targets.shape()[0], "x", targets.shape()[1])

    # Set one-hot: first sample class 5, second sample class 10
    targets._set_float64(5, 1.0)  # First sample, class 5
    targets._set_float64(47 + 10, 1.0)  # Second sample, class 10

    print("Calling cross_entropy...")
    print("This should crash with segmentation fault at AnyTensor.__init__:107")

    try:
        var loss = cross_entropy(logits, targets)
        var shape = loss.shape()
        var shape_str = "["
        for i in range(len(shape)):
            shape_str += String(shape[i])
            if i < len(shape) - 1:
                shape_str += ", "
        shape_str += "]"
        print("Loss computed successfully! Shape:", shape_str)
        print("Loss value:", loss[0])
    except e:
        print("ERROR:", e)
        raise e^


fn test_cross_entropy_varying_sizes() raises:
    """Test cross-entropy with various batch and class sizes."""
    print("\n=== Test 2: Cross-Entropy with Varying Sizes ===")

    var test_configs = List[Tuple[Int, Int]]()
    test_configs.append((1, 10))  # Tiny
    test_configs.append((2, 47))  # Actual crash case
    test_configs.append((4, 47))  # Double batch
    test_configs.append((8, 100))  # Larger

    for i in range(len(test_configs)):
        var batch_size = test_configs[i][0]
        var num_classes = test_configs[i][1]

        print("\nTesting batch_size=", batch_size, "num_classes=", num_classes)

        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(num_classes)

        var logits = zeros(shape, DType.float32)
        var targets = zeros(shape, DType.float32)

        # Set one-hot for each sample
        for b in range(batch_size):
            var class_idx = b % num_classes
            var flat_idx = b * num_classes + class_idx
            targets._set_float64(flat_idx, 1.0)

        print("  Calling cross_entropy...")
        try:
            var loss = cross_entropy(logits, targets)
            print("  SUCCESS: Loss computed")
        except e:
            print("  CRASH:", e)
            raise e^


fn test_cross_entropy_edge_cases() raises:
    """Test cross-entropy with edge cases that might trigger memory issues."""
    print("\n=== Test 3: Cross-Entropy Edge Cases ===")

    # Test 1: Single sample, many classes
    print("\nTest 3.1: Single sample, 1000 classes")
    var shape1 = List[Int]()
    shape1.append(1)
    shape1.append(1000)
    var logits1 = zeros(shape1, DType.float32)
    var targets1 = zeros(shape1, DType.float32)
    targets1._set_float64(42, 1.0)

    try:
        var loss1 = cross_entropy(logits1, targets1)
        print("  SUCCESS")
    except e:
        print("  CRASH:", e)
        raise e^

    # Test 2: Many samples, few classes
    print("\nTest 3.2: 100 samples, 10 classes")
    var shape2 = List[Int]()
    shape2.append(100)
    shape2.append(10)
    var logits2 = zeros(shape2, DType.float32)
    var targets2 = zeros(shape2, DType.float32)

    for b in range(100):
        targets2._set_float64(b * 10 + (b % 10), 1.0)

    try:
        var loss2 = cross_entropy(logits2, targets2)
        print("  SUCCESS")
    except e:
        print("  CRASH:", e)
        raise e^


fn main() raises:
    """Run all crash reproduction tests."""
    print("=" * 60)
    print("CRASH REPRODUCTION TESTS - Cross-Entropy")
    print("=" * 60)

    # Test 1: Exact crash case
    test_cross_entropy_small_batch()

    # Test 2: Varying sizes
    test_cross_entropy_varying_sizes()

    # Test 3: Edge cases
    test_cross_entropy_edge_cases()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - No crash detected!")
    print("=" * 60)
