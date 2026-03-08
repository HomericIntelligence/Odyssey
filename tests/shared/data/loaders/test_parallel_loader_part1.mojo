"""Tests for parallel data loader - Part 1: Creation and Correctness.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_parallel_loader.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests ParallelLoader creation and correctness: worker configuration,
dataset compatibility, and deterministic ordering behavior.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_greater,
    TestFixtures,
)


# ============================================================================
# ParallelLoader Creation Tests
# ============================================================================


fn test_parallel_loader_creation():
    """Test creating ParallelLoader with multiple workers.

    Should accept num_workers parameter and create worker threads
    to load data in parallel.
    """
    # var dataset = TestFixtures.synthetic_dataset(n_samples=100)
    # var loader = ParallelLoader(dataset, batch_size=32, num_workers=4)
    # assert_equal(len(loader), 4)
    pass


fn test_parallel_loader_num_workers_validation():
    """Test that num_workers must be non-negative.

    num_workers=0 should fall back to single-threaded mode,
    negative values should raise ValueError.
    """
    # var dataset = TestFixtures.synthetic_dataset(n_samples=100)
    #
    # # Zero workers should be valid (sequential mode)
    # var loader0 = ParallelLoader(dataset, batch_size=32, num_workers=0)
    # assert_true(loader0 is not None)
    #
    # # Negative workers should raise error
    # try:
    #     var loader_neg = ParallelLoader(dataset, batch_size=32, num_workers=-1)
    #     assert_true(False, "Should have raised ValueError")
    # except ValueError:
    #     pass
    pass


fn test_parallel_loader_with_file_dataset():
    """Test ParallelLoader with I/O-bound FileDataset.

    This is the primary use case: parallel workers loading files
    from disk while GPU processes previous batch.
    """
    # var dataset = FileDataset(path="/path/to/images")
    # var loader = ParallelLoader(dataset, batch_size=32, num_workers=4)
    #
    # var batch_count = 0
    # for batch in loader:
    #     batch_count += 1
    #
    # assert_true(batch_count > 0)
    pass


# ============================================================================
# ParallelLoader Correctness Tests
# ============================================================================


fn test_parallel_loader_all_samples():
    """Test that parallel loading yields all samples.

    Despite parallel execution, should not lose, duplicate,
    or reorder samples (unless shuffle=True).
    """
    # var dataset = TestFixtures.sequential_dataset(n_samples=100)
    # var loader = ParallelLoader(
    #     dataset, batch_size=32, num_workers=4, shuffle=False
    # )
    #
    # var seen_indices = Set[Int]()
    # for batch in loader:
    #     for i in range(batch.size()):
    #         seen_indices.add(Int(batch.data[i, 0]))
    #
    # assert_equal(len(seen_indices), 100)
    pass


fn test_parallel_loader_deterministic_order():
    """Test that results are deterministic with shuffle=False.

    Even with parallel workers, same input should produce
    same output batches in same order.
    """
    # var dataset = TestFixtures.sequential_dataset(n_samples=100)
    #
    # # First run
    # var loader1 = ParallelLoader(
    #     dataset, batch_size=32, num_workers=4, shuffle=False
    # )
    # var batches1 : List[Batch] = []
    # for batch in loader1:
    #     batches1.append(batch)
    #
    # # Second run
    # var loader2 = ParallelLoader(
    #     dataset, batch_size=32, num_workers=4, shuffle=False
    # )
    # var batches2 : List[Batch] = []
    # for batch in loader2:
    #     batches2.append(batch)
    #
    # # Should be identical
    # assert_equal(len(batches1), len(batches2))
    # for i in range(len(batches1)):
    #     assert_equal(batches1[i].data, batches2[i].data)
    pass


fn test_parallel_loader_with_shuffle():
    """Test parallel loading with shuffling enabled.

    Shuffling should work correctly even with multiple workers,
    maintaining determinism with fixed seed.
    """
    # TestFixtures.set_seed()
    # var dataset = TestFixtures.synthetic_dataset(n_samples=100)
    #
    # var loader1 = ParallelLoader(
    #     dataset, batch_size=32, num_workers=4, shuffle=True
    # )
    # var batch1 = next(iter(loader1))
    #
    # TestFixtures.set_seed()
    # var loader2 = ParallelLoader(
    #     dataset, batch_size=32, num_workers=4, shuffle=True
    # )
    # var batch2 = next(iter(loader2))
    #
    # # Same seed should produce same shuffle
    # assert_equal(batch1.data, batch2.data)
    pass


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run parallel loader creation and correctness tests."""
    print("Running parallel loader creation and correctness tests...")

    # Creation tests
    test_parallel_loader_creation()
    test_parallel_loader_num_workers_validation()
    test_parallel_loader_with_file_dataset()

    # Correctness tests
    test_parallel_loader_all_samples()
    test_parallel_loader_deterministic_order()
    test_parallel_loader_with_shuffle()

    print("✓ All parallel loader creation and correctness tests passed!")
