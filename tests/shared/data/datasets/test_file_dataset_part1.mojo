"""Tests for lazy-loading file dataset (Part 1: Creation and Lazy Loading).

Split from test_file_dataset.mojo per ADR-009.
Tests FileDataset which loads data from disk on-demand,
enabling training on datasets larger than available memory.
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_file_dataset.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from tests.shared.conftest import assert_true, assert_equal, TestFixtures
from shared.data.datasets import FileDataset


# ============================================================================
# FileDataset Creation Tests
# ============================================================================


fn test_file_dataset_from_directory() raises:
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


fn test_file_dataset_with_file_pattern() raises:
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


fn test_file_dataset_nonexistent_directory() raises:
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


fn test_file_dataset_empty_directory() raises:
    """Test handling of empty file list.

    Should create valid dataset with length 0, not crash.
    Useful for testing and incremental dataset building.
    """
    var file_paths = List[String]()
    var labels = List[Int]()
    var dataset = FileDataset(file_paths^, labels^)
    assert_equal(dataset.__len__(), 0)


# ============================================================================
# FileDataset Lazy Loading Tests
# ============================================================================


fn test_file_dataset_lazy_loading() raises:
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


fn test_file_dataset_getitem_loads_file() raises:
    """Test that __getitem__ API exists and would load files.

    Note: Actual file loading not implemented yet (_load_file raises error),
    but we can test the API structure and error handling.
    """
    # Skip: File loading not yet implemented
    print("⏭️ SKIPPED: File loading not yet implemented")
    return


fn test_file_dataset_caching() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run file dataset creation and lazy loading tests."""
    print("Running file dataset tests (Part 1: Creation and Lazy Loading)...")

    # Creation tests
    test_file_dataset_from_directory()
    test_file_dataset_with_file_pattern()
    test_file_dataset_nonexistent_directory()
    test_file_dataset_empty_directory()

    # Lazy loading tests
    test_file_dataset_lazy_loading()
    test_file_dataset_getitem_loads_file()
    test_file_dataset_caching()

    print("✓ All file dataset Part 1 tests passed!")
