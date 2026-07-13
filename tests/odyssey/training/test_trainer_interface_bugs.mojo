"""Tests for List[Int] constructor bugs in trainer_interface.mojo.

This test file demonstrates the bug in trainer_interface.mojo that uses the
unsafe pattern= List[Int]() followed by list[i] = value.

These tests SHOULD FAIL before the fix is applied, demonstrating the bug.
After fixing, they should PASS.

Bug tested:
- Line 270: DataLoader.next() - var batch_labels_shape = List[Int]()
"""

# Import AnyTensor and trainer interface
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import ones
from odyssey.training.trainer_interface import DataLoader


# ============================================================================
# Test DataLoader.next() bug (Line 270)
# ============================================================================


def test_dataloader_next_normal_batch() raises:
    """Test DataLoader.next() with normal batch (triggers bug at line 270).

    Bug: Line 270 uses List[Int]() then is passed to AnyTensor.
    This crashes because the list has undefined size.
    """
    # Create dummy dataset
    var data_shape = List[Int]()
    data_shape.append(100)  # num samples
    data_shape.append(10)  # feature dim
    var data = ones(data_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ones(labels_shape, DType.int32)

    # Create dataloader with batch_size=32
    var loader = DataLoader(data, labels, batch_size=32)

    # Get first batch - this will crash due to bug at line 270
    var batch = loader.next()

    # If we get here, the bug is fixed
    print("    ✓ Got first batch successfully")


def test_dataloader_next_small_batch() raises:
    """Test DataLoader.next() with small batch (triggers bug at line 270).

    Bug: Even with batch_size=1, the bug still crashes.
    """
    # Create small dataset
    var data_shape = List[Int]()
    data_shape.append(10)  # num samples
    data_shape.append(5)  # feature dim
    var data = ones(data_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = ones(labels_shape, DType.int32)

    # Create dataloader with batch_size=1
    var loader = DataLoader(data, labels, batch_size=1)

    # Get first batch - this will crash due to bug at line 270
    var batch = loader.next()

    # If we get here, the bug is fixed
    print("    ✓ Got small batch successfully")


def test_dataloader_next_large_batch() raises:
    """Test DataLoader.next() with large batch (triggers bug at line 270).

    Bug: Larger batch size makes the bug more likely to manifest.
    """
    # Create large dataset
    var data_shape = List[Int]()
    data_shape.append(1000)  # num samples
    data_shape.append(50)  # feature dim
    var data = ones(data_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(1000)
    var labels = ones(labels_shape, DType.int32)

    # Create dataloader with large batch_size=256
    var loader = DataLoader(data, labels, batch_size=256)

    # Get first batch - this will crash due to bug at line 270
    var batch = loader.next()

    # If we get here, the bug is fixed
    print("    ✓ Got large batch successfully")


def test_dataloader_next_partial_last_batch() raises:
    """Test DataLoader.next() with partial last batch (triggers bug).

    Bug: The last batch has actual_batch_size < batch_size, which still
    triggers the List[Int] constructor bug at line 270.
    """
    # Create dataset that doesn't divide evenly by batch size
    var data_shape = List[Int]()
    data_shape.append(50)  # num samples
    data_shape.append(10)  # feature dim
    var data = ones(data_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(50)
    var labels = ones(labels_shape, DType.int32)

    # Create dataloader with batch_size=32
    # This gives: batch 1 (32), batch 2 (18 - partial)
    var loader = DataLoader(data, labels, batch_size=32)

    # Get first full batch
    var batch1 = loader.next()
    print("    ✓ Got first full batch")

    # Get second partial batch - this will crash due to bug at line 270
    var batch2 = loader.next()
    print("    ✓ Got partial last batch")


def test_dataloader_multiple_iterations() raises:
    """Test DataLoader with multiple next() calls (triggers bug repeatedly).

    Bug: Each call to next() triggers the List[Int] constructor bug.
    """
    # Create dataset
    var data_shape = List[Int]()
    data_shape.append(100)
    data_shape.append(20)
    var data = ones(data_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ones(labels_shape, DType.int32)

    # Create dataloader
    var loader = DataLoader(data, labels, batch_size=25)

    # Iterate through all batches
    var batch_count = 0
    while loader.has_next():
        var batch = loader.next()  # Each call crashes at line 270
        batch_count += 1

    # If we get here, the bug is fixed
    print("    ✓ Processed", batch_count, "batches successfully")


# ============================================================================
# Main test runner
# ============================================================================


def main() raises:
    """Run all trainer_interface.mojo bug tests.

    These tests demonstrate the List[Int] constructor bug in trainer_interface.mojo.
    They SHOULD FAIL before fix is applied, and PASS after.
    """
    print("Running trainer_interface.mojo List[Int] constructor bug tests...")
    print("WARNING: These tests may crash before fix is applied!")
    print("")

    print("  Testing DataLoader.next() bug...")
    try:
        test_dataloader_next_normal_batch()
    except e:
        print("    ✗ normal batch CRASHED:", String(e))

    try:
        test_dataloader_next_small_batch()
    except e:
        print("    ✗ small batch CRASHED:", String(e))

    try:
        test_dataloader_next_large_batch()
    except e:
        print("    ✗ large batch CRASHED:", String(e))

    try:
        test_dataloader_next_partial_last_batch()
    except e:
        print("    ✗ partial last batch CRASHED:", String(e))

    try:
        test_dataloader_multiple_iterations()
    except e:
        print("    ✗ multiple iterations CRASHED:", String(e))

    print("")
    print("trainer_interface.mojo bug tests completed!")
    print("If any tests crashed, the bug is still present.")
    print("After fixing, all tests should pass.")
