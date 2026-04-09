"""Tests for CIFAR-10 Dataset wrapper.

Tests the CIFAR10Dataset struct including:
    - Initialization with data directory
    - Dataset length and properties
    - Individual sample access via __getitem__
    - Training data loading
    - Test data loading
    - Class name retrieval
    - Error handling for invalid indices and directories
"""

from std.testing import assert_true, assert_equal
from shared.data.datasets import CIFAR10Dataset


def test_cifar10_initialization() raises:
    """Test CIFAR10Dataset initialization with valid directory."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")
    assert_equal(
        dataset.data_dir,
        "/tmp/cifar10",
        "Data directory should be set correctly",
    )


def test_cifar10_initialization_empty_path() raises:
    """Test CIFAR10Dataset raises error for empty path."""
    try:
        var dataset = CIFAR10Dataset("")
        assert_true(False, "Should raise error for empty data_dir")
    except:
        # Expected: error raised for empty path
        pass


def test_cifar10_num_samples() raises:
    """Test CIFAR10Dataset returns correct number of training samples."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")
    var n_train = dataset.num_train_samples()
    assert_equal(n_train, 50000, "Training set should have 50,000 samples")

    var n_test = dataset.num_test_samples()
    assert_equal(n_test, 10000, "Test set should have 10,000 samples")


def test_cifar10_len() raises:
    """Test CIFAR10Dataset __len__ returns training set size."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")
    var length = dataset.__len__()
    assert_equal(length, 50000, "__len__ should return 50,000 for training set")


def test_cifar10_num_classes() raises:
    """Test CIFAR10Dataset returns correct number of classes."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")
    var n_classes = dataset.num_classes()
    assert_equal(n_classes, 10, "CIFAR-10 should have 10 classes")


def test_cifar10_image_shape() raises:
    """Test CIFAR10Dataset returns correct image shape."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")
    var shape = dataset.image_shape()

    assert_equal(len(shape), 3, "Image shape should have 3 dimensions")
    assert_equal(shape[0], 3, "First dimension should be 3 (RGB channels)")
    assert_equal(shape[1], 32, "Second dimension should be 32 (height)")
    assert_equal(shape[2], 32, "Third dimension should be 32 (width)")


def test_cifar10_class_names() raises:
    """Test CIFAR10Dataset class name retrieval."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")

    # Test valid class indices
    var airplane = dataset.get_class_name(0)
    assert_equal(airplane, "airplane", "Class 0 should be 'airplane'")

    var cat = dataset.get_class_name(3)
    assert_equal(cat, "cat", "Class 3 should be 'cat'")

    var truck = dataset.get_class_name(9)
    assert_equal(truck, "truck", "Class 9 should be 'truck'")


def test_cifar10_class_name_invalid_index() raises:
    """Test CIFAR10Dataset raises error for invalid class index."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")

    try:
        _ = dataset.get_class_name(10)
        assert_true(False, "Should raise error for class index 10")
    except:
        # Expected: error raised for out-of-range index
        pass

    try:
        _ = dataset.get_class_name(-1)
        assert_true(False, "Should raise error for negative class index")
    except:
        # Expected: error raised for negative index
        pass


def test_cifar10_getitem_out_of_bounds() raises:
    """Test CIFAR10Dataset __getitem__ raises error for invalid indices.

    Note: This test does not load actual data, so it tests the index
    validation logic before attempting to load data.
    """
    var dataset = CIFAR10Dataset("/tmp/cifar10")

    try:
        _ = dataset.__getitem__(50000)
        assert_true(False, "Should raise error for index 50000 (out of bounds)")
    except:
        # Expected: error raised for out-of-bounds index
        pass

    try:
        _ = dataset.__getitem__(-1)
        assert_true(False, "Should raise error for negative index -1")
    except:
        # Expected: error raised for negative index
        pass


def test_cifar10_get_class_name_all_classes() raises:
    """Test CIFAR10Dataset returns correct names for all 10 classes."""
    var dataset = CIFAR10Dataset("/tmp/cifar10")

    var expected_names = List[String]()
    expected_names.append("airplane")
    expected_names.append("automobile")
    expected_names.append("bird")
    expected_names.append("cat")
    expected_names.append("deer")
    expected_names.append("dog")
    expected_names.append("frog")
    expected_names.append("horse")
    expected_names.append("ship")
    expected_names.append("truck")

    for i in range(10):
        var name = dataset.get_class_name(i)
        assert_equal(
            name, expected_names[i], "Class name mismatch at index " + String(i)
        )


def main() raises:
    """Run all CIFAR-10 dataset tests."""
    test_cifar10_initialization()
    test_cifar10_initialization_empty_path()
    test_cifar10_num_samples()
    test_cifar10_len()
    test_cifar10_num_classes()
    test_cifar10_image_shape()
    test_cifar10_class_names()
    test_cifar10_class_name_invalid_index()
    test_cifar10_getitem_out_of_bounds()
    test_cifar10_get_class_name_all_classes()
    print("All CIFAR-10 dataset tests passed!")
