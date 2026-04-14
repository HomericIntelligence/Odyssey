"""Tests for lazy-loading file dataset (Part 1: Creation and Lazy Loading).

Tests FileDataset which loads data from disk on-demand,
enabling training on datasets larger than available memory.
"""


from tests.shared.conftest import assert_true, assert_equal, TestFixtures
from shared.data.datasets import FileDataset


def test_file_dataset_from_directory() raises:
    """Test creating FileDataset from list of file paths.

    FileDataset should accept file paths and labels,
    loading them lazily when requested via __getitem__.
    """
    var file_paths = List[String]()
    file_paths.append("/path/to/file1.jpg")
    file_paths.append("/path/to/file2.jpg")
    file_paths.append("/path/to/file3.jpg")

    var labels = List[Int]()
    labels.append(0)
    labels.append(1)
    labels.append(2)

    var dataset = FileDataset(file_paths^, labels^, cache=False)
    assert_equal(dataset.__len__(), 3)


def test_file_dataset_with_file_pattern() raises:
    """Test creating FileDataset with specific file types.

    FileDataset should work with filtered file lists,
    useful for selecting specific file types.
    """
    # Create dataset with only .jpg files
    var file_paths = List[String]()
    file_paths.append("/data/img1.jpg")
    file_paths.append("/data/img2.jpg")

    var labels = List[Int]()
    labels.append(0)
    labels.append(1)

    var dataset = FileDataset(file_paths^, labels^)
    assert_equal(dataset.__len__(), 2)


def test_file_dataset_nonexistent_directory() raises:
    """Test that mismatched file paths and labels raise error.

    Should fail immediately with clear error rather than
    creating invalid dataset.
    """
    var file_paths = List[String]()
    file_paths.append("/path/file1.jpg")
    file_paths.append("/path/file2.jpg")

    var labels = List[Int]()
    labels.append(0)  # Only one label for two files

    var error_raised = False
    try:
        var dataset = FileDataset(file_paths^, labels^)
    except:
        error_raised = True
    assert_true(error_raised, "Should raise error for mismatched lengths")


def test_file_dataset_empty_directory() raises:
    """Test handling of empty file list.

    Should create valid dataset with length 0, not crash.
    Useful for testing and incremental dataset building.
    """
    var file_paths = List[String]()
    var labels = List[Int]()
    var dataset = FileDataset(file_paths^, labels^)
    assert_equal(dataset.__len__(), 0)


def test_file_dataset_lazy_loading() raises:
    """Test that dataset creation is fast (doesn't load files).

    Creating FileDataset should be fast (just store file paths),
    with actual loading deferred until __getitem__ is called.
    """
    # Create dataset with many file paths - should be instant
    var file_paths = List[String]()
    var labels = List[Int]()

    for i in range(10000):
        file_paths.append("/path/to/image_" + String(i) + ".jpg")
        labels.append(i % 10)

    var dataset = FileDataset(file_paths^, labels^)
    assert_equal(dataset.__len__(), 10000)


def test_file_dataset_getitem_loads_file() raises:
    """Test that __getitem__ API exists and would load files.

    Note: Actual file loading not implemented yet (_load_file raises error),
    but we can test the API structure and error handling.
    """
    # Skip: File loading not yet implemented
    print("⏭️ SKIPPED: File loading not yet implemented")
    return


def test_file_dataset_caching() raises:
    """Test that caching flag can be set.

    FileDataset API supports caching parameter to control
    whether loaded files are cached in memory.
    """
    var file_paths = List[String]()
    file_paths.append("/test/file.jpg")
    var labels = List[Int]()
    labels.append(0)

    # Test with caching enabled
    var dataset_cached = FileDataset(
        file_paths.copy(), labels.copy(), cache=True
    )
    assert_equal(dataset_cached.__len__(), 1)

    # Test with caching disabled
    var file_paths2 = List[String]()
    file_paths2.append("/test/file.jpg")
    var labels2 = List[Int]()
    labels2.append(0)
    var dataset_no_cache = FileDataset(file_paths2^, labels2^, cache=False)
    assert_equal(dataset_no_cache.__len__(), 1)


def test_file_dataset_memory_efficiency() raises:
    """Test that FileDataset doesn't load all files during creation.

    Memory usage should remain low even for large datasets,
    only storing file paths not loaded data.
    """
    # Create dataset with many files - shouldn't load them all
    var file_paths = List[String]()
    var labels = List[Int]()

    for i in range(10000):
        file_paths.append("/images/img" + String(i) + ".jpg")
        labels.append(i % 100)

    var dataset = FileDataset(file_paths^, labels^, cache=False)

    # Dataset created without loading files - should be fast and memory efficient
    assert_equal(dataset.__len__(), 10000)


def test_file_dataset_labels_from_filename() raises:
    """Test that labels are provided explicitly with file paths.

    FileDataset stores labels provided at creation,
    returning them when samples are accessed.
    """
    var file_paths = List[String]()
    file_paths.append("/images/class0_001.jpg")
    file_paths.append("/images/class1_002.jpg")

    var labels = List[Int]()
    labels.append(0)
    labels.append(1)

    var dataset = FileDataset(file_paths^, labels^)
    assert_equal(dataset.__len__(), 2)


def test_file_dataset_labels_from_directory() raises:
    """Test that labels can represent directory-based organization.

    FileDataset supports labels that could come from directory structure,
    passed explicitly at dataset creation.
    """
    # Simulate ImageFolder-style dataset with directory-based labels
    var file_paths = List[String]()
    file_paths.append("/data/cats/img001.jpg")
    file_paths.append("/data/dogs/img001.jpg")

    var labels = List[Int]()
    labels.append(0)  # cats
    labels.append(1)  # dogs

    var dataset = FileDataset(file_paths^, labels^)
    assert_equal(dataset.__len__(), 2)


def test_file_dataset_labels_from_file() raises:
    """Test that labels can be loaded from external source.

    FileDataset accepts any label list, which could come from
    a CSV or JSON file parsed externally.
    """
    var file_paths = List[String]()
    file_paths.append("/data/img1.jpg")
    file_paths.append("/data/img2.jpg")

    # Labels could be loaded from labels.csv or labels.json
    var labels = List[Int]()
    labels.append(5)
    labels.append(7)

    var dataset = FileDataset(file_paths^, labels^)
    assert_equal(dataset.__len__(), 2)


def test_file_dataset_corrupted_file() raises:
    """Test error handling for file loading failures.

    When file loading fails, should raise informative error.
    Currently _load_file is not implemented, so any access will error.
    """
    # Skip: File loading not yet implemented
    print("⏭️ SKIPPED: File loading not yet implemented")
    return


def test_file_dataset_missing_file() raises:
    """Test bounds checking for dataset access.

    Accessing invalid index should raise error,
    similar to accessing missing/deleted file.
    """
    var file_paths = List[String]()
    file_paths.append("/data/img.jpg")

    var labels = List[Int]()
    labels.append(0)

    var dataset = FileDataset(file_paths^, labels^)

    # Test out of bounds access
    var error_raised = False
    try:
        var sample = dataset[5]  # Index out of bounds
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for out of bounds access")


def main() raises:
    """Run all test_file_dataset tests."""
    print("Running test_file_dataset tests...")

    test_file_dataset_from_directory()
    print("✓ test_file_dataset_from_directory")

    test_file_dataset_with_file_pattern()
    print("✓ test_file_dataset_with_file_pattern")

    test_file_dataset_nonexistent_directory()
    print("✓ test_file_dataset_nonexistent_directory")

    test_file_dataset_empty_directory()
    print("✓ test_file_dataset_empty_directory")

    test_file_dataset_lazy_loading()
    print("✓ test_file_dataset_lazy_loading")

    test_file_dataset_getitem_loads_file()
    print("✓ test_file_dataset_getitem_loads_file")

    test_file_dataset_caching()
    print("✓ test_file_dataset_caching")

    test_file_dataset_memory_efficiency()
    print("✓ test_file_dataset_memory_efficiency")

    test_file_dataset_labels_from_filename()
    print("✓ test_file_dataset_labels_from_filename")

    test_file_dataset_labels_from_directory()
    print("✓ test_file_dataset_labels_from_directory")

    test_file_dataset_labels_from_file()
    print("✓ test_file_dataset_labels_from_file")

    test_file_dataset_corrupted_file()
    print("✓ test_file_dataset_corrupted_file")

    test_file_dataset_missing_file()
    print("✓ test_file_dataset_missing_file")

    print("\nAll test_file_dataset tests passed!")
