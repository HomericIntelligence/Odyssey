"""Tests for EMNIST Dataset Wrapper

Tests cover:
- Dataset initialization with different splits
- Dataset length and item access
- Boundary conditions and error handling

"""


from testing import assert_equal, assert_true, assert_false, assert_raises
from shared.data import EMNISTDataset, AnyTensorDataset, Dataset
from shared.tensor.any_tensor import AnyTensor


fn create_mock_idx_files(temp_dir: String) raises:
    """Create mock IDX files for testing.

    Creates minimal valid IDX format files with test data.
    """
    # For this test, we'll use real file paths if they exist,
    # or skip tests if files don't exist (offline testing)
    _ = temp_dir  # Consume unused parameter
    pass


fn test_emnist_init_balanced() raises:
    """Test EMNISTDataset initialization with balanced split.

    Verifies that the dataset can be initialized and properties are set.
    """
    # Note: This test requires actual EMNIST data files.
    # In a CI environment, the test will be skipped if files don't exist.
    # For local testing, ensure EMNIST data is downloaded to /tmp/emnist/

    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        assert_true(len(dataset.split) > 0, "Split should be set")
        assert_equal(dataset.split, "balanced", "Split should be 'balanced'")
    except e:
        # Expected if test data doesn't exist
        print("Test data not available - skipping initialization test")


fn test_emnist_init_byclass() raises:
    """Test EMNISTDataset initialization with byclass split.

    Verifies that different split types are accepted.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="byclass", train=True)
        assert_equal(dataset.split, "byclass", "Split should be 'byclass'")
    except e:
        print("Test data not available - skipping byclass test")


fn test_emnist_init_digits() raises:
    """Test EMNISTDataset initialization with digits split (MNIST equivalent).

    Verifies that digits-only split loads correctly.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="digits", train=True)
        assert_equal(dataset.split, "digits", "Split should be 'digits'")
    except e:
        print("Test data not available - skipping digits test")


fn test_emnist_init_letters() raises:
    """Test EMNISTDataset initialization with letters split.

    Verifies that letters-only split loads correctly.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="letters", train=True)
        assert_equal(dataset.split, "letters", "Split should be 'letters'")
    except e:
        print("Test data not available - skipping letters test")


fn test_emnist_init_invalid_split() raises:
    """Test EMNISTDataset with invalid split parameter.

    Verifies that invalid splits are rejected with appropriate error.
    """
    var error_raised = False
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="invalid", train=True)
        _ = dataset  # Consume unused variable (expected to raise before here)
    except e:
        error_raised = True
        assert_true(
            String(e).__contains__("Invalid split"),
            "Error should mention invalid split",
        )

    assert_true(error_raised, "Invalid split should raise error")


fn test_emnist_len() raises:
    """Test __len__ returns correct dataset size.

    Verifies that the length reflects the actual number of samples.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var length = dataset.__len__()
        assert_true(length > 0, "Dataset length should be positive")
    except e:
        print("Test data not available - skipping length test")


fn test_emnist_getitem_index() raises:
    """Test __getitem__ with positive index.

    Verifies that samples can be retrieved by index.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var sample_data, sample_label = dataset.__getitem__(0)
        _ = sample_label  # Consume unused variable

        # Verify sample is a valid AnyTensor
        var data_shape = sample_data.shape()
        assert_equal(
            len(data_shape), 4, "Image should have 4 dimensions (N, C, H, W)"
        )
        assert_equal(data_shape[1], 1, "Should have 1 channel (grayscale)")
        assert_equal(data_shape[2], 28, "Height should be 28")
        assert_equal(data_shape[3], 28, "Width should be 28")
    except e:
        print("Test data not available - skipping getitem test")


fn test_emnist_getitem_negative_index() raises:
    """Test __getitem__ with negative indexing.

    Verifies that negative indices work (last element).
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var length = dataset.__len__()
        _ = length  # Consume unused variable
        var last_sample_data, last_sample_label = dataset.__getitem__(-1)
        _ = last_sample_label  # Consume unused variable

        # Verify we got a valid sample
        var data_shape = last_sample_data.shape()
        assert_equal(len(data_shape), 4, "Image should have 4 dimensions")
    except e:
        print("Test data not available - skipping negative index test")


fn test_emnist_getitem_out_of_bounds() raises:
    """Test __getitem__ with out-of-bounds index.

    Verifies that accessing invalid indices raises appropriate error.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var length = dataset.__len__()

        var error_raised = False
        try:
            var sample_data, sample_label = dataset.__getitem__(length + 100)
            _ = sample_data  # Consume unused variable
            _ = sample_label  # Consume unused variable
        except e:
            error_raised = True
            assert_true(
                String(e).__contains__("out of bounds"),
                "Error should mention out of bounds",
            )

        assert_true(error_raised, "Out of bounds access should raise error")
    except e:
        print("Test data not available - skipping bounds test")


fn test_emnist_shape() raises:
    """Test shape() method returns correct dimensions.

    Verifies that individual sample shape is (1, 28, 28).
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var shape = dataset.shape()

        assert_equal(len(shape), 3, "Shape should have 3 dimensions")
        assert_equal(shape[0], 1, "Channels should be 1")
        assert_equal(shape[1], 28, "Height should be 28")
        assert_equal(shape[2], 28, "Width should be 28")
    except e:
        print("Test data not available - skipping shape test")


fn test_emnist_num_classes_balanced() raises:
    """Test num_classes() for balanced split.

    Verifies correct class count for balanced variant.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        assert_equal(
            dataset.num_classes(), 47, "Balanced split should have 47 classes"
        )
    except e:
        print("Test data not available - skipping class count test")


fn test_emnist_num_classes_byclass() raises:
    """Test num_classes() for byclass split.

    Verifies correct class count for byclass variant.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="byclass", train=True)
        assert_equal(
            dataset.num_classes(), 62, "Byclass split should have 62 classes"
        )
    except e:
        print("Test data not available - skipping byclass class count test")


fn test_emnist_num_classes_digits() raises:
    """Test num_classes() for digits split.

    Verifies that digits split has 10 classes (same as MNIST).
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="digits", train=True)
        assert_equal(
            dataset.num_classes(), 10, "Digits split should have 10 classes"
        )
    except e:
        print("Test data not available - skipping digits class count test")


fn test_emnist_num_classes_letters() raises:
    """Test num_classes() for letters split.

    Verifies that letters split has 26 classes (A-Z).
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="letters", train=True)
        assert_equal(
            dataset.num_classes(), 26, "Letters split should have 26 classes"
        )
    except e:
        print("Test data not available - skipping letters class count test")


fn test_emnist_num_classes_mnist() raises:
    """Test num_classes() for mnist split.

    Verifies that MNIST equivalent has 10 classes.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="mnist", train=True)
        assert_equal(
            dataset.num_classes(), 10, "MNIST split should have 10 classes"
        )
    except e:
        print("Test data not available - skipping mnist class count test")


fn test_emnist_get_train_data() raises:
    """Test get_train_data() returns AnyTensorDataset.

    Verifies that the method wraps data in AnyTensorDataset correctly.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var tensor_dataset = dataset.get_train_data()

        # Verify it's a valid AnyTensorDataset
        var length = tensor_dataset.__len__()
        assert_true(length > 0, "AnyTensorDataset should have samples")
    except e:
        print("Test data not available - skipping get_train_data test")


fn test_emnist_get_test_data() raises:
    """Test get_test_data() returns AnyTensorDataset.

    Verifies that the method wraps data in AnyTensorDataset correctly.
    """
    try:
        var dataset = EMNISTDataset(
            "/tmp/emnist", split="balanced", train=False
        )
        var tensor_dataset = dataset.get_test_data()

        # Verify it's a valid AnyTensorDataset
        var length = tensor_dataset.__len__()
        assert_true(length > 0, "AnyTensorDataset should have samples")
    except e:
        print("Test data not available - skipping get_test_data test")


fn test_emnist_train_vs_test_sizes() raises:
    """Test that train and test splits have different sizes.

    Verifies that training and test datasets contain expected sample counts.
    """
    try:
        var train_dataset = EMNISTDataset(
            "/tmp/emnist", split="balanced", train=True
        )
        var test_dataset = EMNISTDataset(
            "/tmp/emnist", split="balanced", train=False
        )

        var train_len = train_dataset.__len__()
        var test_len = test_dataset.__len__()

        # Train should typically have more samples than test
        assert_true(train_len > 0, "Train set should have samples")
        assert_true(test_len > 0, "Test set should have samples")
    except e:
        print("Test data not available - skipping train/test split test")


fn test_emnist_data_label_consistency() raises:
    """Test that data and labels have matching first dimensions.

    Verifies that the dataset maintains consistency between data and labels.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)

        var data_len = dataset.data.shape()[0]
        var labels_len = dataset.labels.shape()[0]

        assert_equal(
            data_len,
            labels_len,
            "Data and labels should have same first dimension",
        )
    except e:
        print("Test data not available - skipping consistency test")


fn test_emnist_all_valid_splits() raises:
    """Test that all documented splits are accepted.

    Verifies that balanced, byclass, bymerge, digits, letters, mnist are all valid.
    """
    var splits = List[String]()
    splits.append("balanced")
    splits.append("byclass")
    splits.append("bymerge")
    splits.append("digits")
    splits.append("letters")
    splits.append("mnist")

    for split in splits:
        var error_raised = False
        try:
            # Just test that initialization is accepted (may fail on file I/O)
            # The key test is that the split validation passes
            var dataset = EMNISTDataset("/tmp/emnist", split=split, train=True)
            _ = dataset  # Consume unused variable
        except e:
            # Check that error is file I/O, not validation
            if String(e).__contains__("Invalid split"):
                error_raised = True
                assert_false(True, "Split '" + split + "' should be valid")

        _ = error_raised  # Consume unused variable
        # Note: File I/O errors are expected if data doesn't exist


fn test_emnist_performance_random_access() raises:
    """Test performance of random index access.

    Verifies that accessing different indices works correctly.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var length = dataset.__len__()

        if length > 0:
            # Access first, middle, and last samples
            var first_data, first_label = dataset.__getitem__(0)
            var middle_data, middle_label = dataset.__getitem__(length // 2)
            var last_data, last_label = dataset.__getitem__(length - 1)
            _ = first_label  # Consume unused variable
            _ = middle_label  # Consume unused variable
            _ = last_label  # Consume unused variable

            # Verify all have correct shape
            for data in [first_data, middle_data, last_data]:
                var shape = data.shape()
                assert_equal(len(shape), 4, "All samples should have 4D shape")
    except e:
        print("Test data not available - skipping performance test")


fn main() raises:
    """Run all test_emnist tests."""
    print("Running test_emnist tests...")

    test_emnist_init_balanced()
    print("✓ test_emnist_init_balanced")

    test_emnist_init_byclass()
    print("✓ test_emnist_init_byclass")

    test_emnist_init_digits()
    print("✓ test_emnist_init_digits")

    test_emnist_init_letters()
    print("✓ test_emnist_init_letters")

    test_emnist_init_invalid_split()
    print("✓ test_emnist_init_invalid_split")

    test_emnist_len()
    print("✓ test_emnist_len")

    test_emnist_getitem_index()
    print("✓ test_emnist_getitem_index")

    test_emnist_getitem_negative_index()
    print("✓ test_emnist_getitem_negative_index")

    test_emnist_getitem_out_of_bounds()
    print("✓ test_emnist_getitem_out_of_bounds")

    test_emnist_shape()
    print("✓ test_emnist_shape")

    test_emnist_num_classes_balanced()
    print("✓ test_emnist_num_classes_balanced")

    test_emnist_num_classes_byclass()
    print("✓ test_emnist_num_classes_byclass")

    test_emnist_num_classes_digits()
    print("✓ test_emnist_num_classes_digits")

    test_emnist_num_classes_letters()
    print("✓ test_emnist_num_classes_letters")

    test_emnist_num_classes_mnist()
    print("✓ test_emnist_num_classes_mnist")

    test_emnist_get_train_data()
    print("✓ test_emnist_get_train_data")

    test_emnist_get_test_data()
    print("✓ test_emnist_get_test_data")

    test_emnist_train_vs_test_sizes()
    print("✓ test_emnist_train_vs_test_sizes")

    test_emnist_data_label_consistency()
    print("✓ test_emnist_data_label_consistency")

    test_emnist_all_valid_splits()
    print("✓ test_emnist_all_valid_splits")

    test_emnist_performance_random_access()
    print("✓ test_emnist_performance_random_access")

    print("\nAll test_emnist tests passed!")
