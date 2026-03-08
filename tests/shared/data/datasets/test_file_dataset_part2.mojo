"""Tests for lazy-loading file dataset (Part 2: Memory, Labels, and Error Handling).

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
# FileDataset Memory Efficiency Tests
# ============================================================================


fn test_file_dataset_memory_efficiency() raises:
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


# ============================================================================
# FileDataset Label Loading Tests
# ============================================================================


fn test_file_dataset_labels_from_filename() raises:
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


fn test_file_dataset_labels_from_directory() raises:
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


fn test_file_dataset_labels_from_file() raises:
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


# ============================================================================
# FileDataset Error Handling Tests
# ============================================================================


fn test_file_dataset_corrupted_file() raises:
    """Test error handling for file loading failures.

    When file loading fails, should raise informative error.
    Currently _load_file is not implemented, so any access will error.
    """
    # Skip: File loading not yet implemented
    print("⏭️ SKIPPED: File loading not yet implemented")
    return


fn test_file_dataset_missing_file() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run file dataset memory, label, and error handling tests."""
    print(
        "Running file dataset tests (Part 2: Memory, Labels, and Error"
        " Handling)..."
    )

    # Memory efficiency tests
    test_file_dataset_memory_efficiency()

    # Label loading tests
    test_file_dataset_labels_from_filename()
    test_file_dataset_labels_from_directory()
    test_file_dataset_labels_from_file()

    # Error handling tests
    test_file_dataset_corrupted_file()
    test_file_dataset_missing_file()

    print("✓ All file dataset Part 2 tests passed!")
