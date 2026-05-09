"""Integration Tests for All CNN Architectures

This script tests all 4 classic CNN architectures on CIFAR-10:
    1. ResNet-18
    2. GoogLeNet (Inception-v1)
    3. MobileNetV1
    4. VGG-16 (if available)

Tests:
    - Model initialization
    - Forward pass with dummy data
    - Output shape verification
    - Inference mode vs training mode
    - Basic sanity checks

Usage:
    mojo run tests/integration/test_all_architectures.mojo
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones
import std.sys as sys


def test_model_forward[
    ForwardFn: def(AnyTensor, Bool) raises -> AnyTensor
](
    model_name: String,
    forward_fn: ForwardFn,
    batch_size: Int = 4,
) raises -> Bool:
    """Test a model's forward pass with dummy data.

    Args:
        model_name: Name of the model being tested.
        forward_fn: Function that takes (input, training) and returns logits.
        batch_size: Batch size for test input.

    Returns:
        True if test passes, False otherwise.
    """
    print("\n" + String("=" * 60))
    print("Testing " + String(model_name))
    print(String("=" * 60))

    try:
        # Create dummy input (batch_size, 3, 32, 32)
        print("Creating dummy input: (" + String(batch_size) + ", 3, 32, 32)")
        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(3)
        shape.append(32)
        shape.append(32)
        var input = zeros(shape, DType.float32)

        # Fill with some non-zero values to avoid all-zero gradients
        var input_data = input._data.bitcast[Float32]()
        for i in range(batch_size * 3 * 32 * 32):
            input_data[i] = Float32(0.1)  # Small non-zero value

        print("✓ Input created successfully")

        # Test inference mode
        print("\nTesting inference mode...")
        var logits_inference = forward_fn(input, False)

        var batch_out = logits_inference.shape()[0]
        var classes_out = logits_inference.shape()[1]

        print(
            "  Output shape: ("
            + String(batch_out)
            + ", "
            + String(classes_out)
            + ")"
        )

        if batch_out != batch_size:
            print(
                "✗ FAIL: Expected batch size "
                + String(batch_size)
                + ", got "
                + String(batch_out)
            )
            return False

        if classes_out != 10:
            print("✗ FAIL: Expected 10 classes, got " + String(classes_out))
            return False

        print("✓ Inference mode passed")

        # Test training mode
        print("\nTesting training mode...")
        var logits_training = forward_fn(input, True)

        var batch_train = logits_training.shape()[0]
        var classes_train = logits_training.shape()[1]

        print(
            "  Output shape: ("
            + String(batch_train)
            + ", "
            + String(classes_train)
            + ")"
        )

        if batch_train != batch_size:
            print(
                "✗ FAIL: Expected batch size "
                + String(batch_size)
                + ", got "
                + String(batch_train)
            )
            return False

        if classes_train != 10:
            print("✗ FAIL: Expected 10 classes, got " + String(classes_train))
            return False

        print("✓ Training mode passed")

        # Check that outputs are different between training and inference
        # (due to batch norm, dropout, etc.)
        print("\nChecking mode differences...")
        var same_count = 0
        var total_count = batch_size * 10
        var logits_inf_data = logits_inference._data.bitcast[Float32]()
        var logits_train_data = logits_training._data.bitcast[Float32]()

        for i in range(total_count):
            if logits_inf_data[i] == logits_train_data[i]:
                same_count += 1

        # Note: For some models without dropout, outputs might be same
        # Just report, don't fail
        var same_pct = Float32(same_count) / Float32(total_count) * 100.0
        print("  " + String(same_pct) + "% of outputs are identical")

        if same_pct == 100.0:
            print(
                "  ⚠ Warning: Outputs identical (no batch norm running stats"
                " difference)"
            )
        else:
            print("✓ Outputs differ between modes (expected)")

        # Verify outputs are finite (not NaN or Inf)
        print("\nChecking for NaN/Inf...")
        var has_nan_inf = False
        for i in range(total_count):
            var val = logits_train_data[i]
            # Simple check: if value is too large, might be inf
            if val > 1e10 or val < -1e10 or val != val:  # NaN check: val != val
                has_nan_inf = True
                break

        if has_nan_inf:
            print("✗ FAIL: Found NaN or Inf in outputs")
            return False
        else:
            print("✓ No NaN/Inf detected")

        print("\n" + String("=" * 60))
        print("✓ " + String(model_name) + " PASSED ALL TESTS")
        print(String("=" * 60) + "\n")

        return True

    except e:
        print("\n" + String("=" * 60))
        print("✗ " + String(model_name) + " FAILED")
        print("Error: " + String(e))
        print(String("=" * 60) + "\n")
        return False


def main() raises:
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS FOR ALL CNN ARCHITECTURES")
    print("=" * 60)
    print("\nTesting 5 classic CNN architectures:")
    print("  1. ResNet-18")
    print("  2. GoogLeNet (Inception-v1)")
    print("  3. MobileNetV1")
    print("  4. VGG-16 (if available)")
    print()

    var results: List[Bool] = []
    var model_names = List[String]()

    # Test 1: ResNet-18
    print("\n" + "=" * 60)
    print("TEST 1/4: ResNet-18")
    print("=" * 60)
    print("NOTE: Import paths use underscores in directory names")
    print("Attempting: examples/resnet18_cifar10/model.mojo")
    print()

    # Note: Mojo imports use the actual file system path
    # We'll need to run this from the repo root and check if models work
    var resnet_passed = False
    print("⚠ Manual test required - cannot dynamically import")
    results.append(resnet_passed)
    model_names.append("ResNet-18")

    # Test 2: GoogLeNet
    print("\n" + "=" * 60)
    print("TEST 2/4: GoogLeNet (Inception-v1)")
    print("=" * 60)
    var googlenet_passed = False
    print("⚠ Manual test required - cannot dynamically import")
    results.append(googlenet_passed)
    model_names.append("GoogLeNet")

    # Test 3: MobileNetV1
    print("\n" + "=" * 60)
    print("TEST 3/4: MobileNetV1")
    print("=" * 60)
    var mobilenet_passed = False
    print("⚠ Manual test required - cannot dynamically import")
    results.append(mobilenet_passed)
    model_names.append("MobileNetV1")

    # Test 4: VGG-16
    print("\n" + "=" * 60)
    print("TEST 4/4: VGG-16")
    print("=" * 60)
    var vgg_passed = False
    print("⚠ Manual test required - cannot dynamically import")
    results.append(vgg_passed)
    model_names.append("VGG-16")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print()

    var total_tests = len(results)
    var passed_tests = 0

    for i in range(total_tests):
        var status = "✓ PASS" if results[i] else "✗ FAIL"
        print(String(status) + ": " + String(model_names[i]))
        if results[i]:
            passed_tests += 1

    print()
    print(
        "Total: "
        + String(passed_tests)
        + "/"
        + String(total_tests)
        + " tests passed"
    )

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠ " + String(total_tests - passed_tests) + " test(s) failed")

    print("=" * 60)
