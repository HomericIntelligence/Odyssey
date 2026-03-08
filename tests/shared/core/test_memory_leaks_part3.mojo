# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_memory_leaks.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Memory leak detection tests for ExTensor - Part 3: Edge Cases & Destructor.

Tests verify:
1. Edge cases (empty tensor, 1D tensor, different dtypes)
2. Destructor edge cases
"""

from shared.core import ExTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_equal_int


# ============================================================================
# Edge Cases
# ============================================================================


fn test_empty_tensor_lifecycle() raises:
    """Test empty tensor (0 elements) creation and destruction."""
    for _ in range(1000):
        var empty = zeros(List[Int](), DType.float32)
        assert_equal_int(
            empty._refcount[], 1, "Empty tensor should have refcount 1"
        )

    assert_true(True, "Empty tensor lifecycle test completed")


fn test_1d_tensor_lifecycle() raises:
    """Test 1D tensor lifecycle."""
    for _ in range(1000):
        var shape = List[Int]()
        shape.append(100)
        var tensor = zeros(shape, DType.float32)
        assert_equal_int(tensor._refcount[], 1, "Should have refcount 1")

    assert_true(True, "1D tensor lifecycle test completed")


fn test_different_dtypes_lifecycle() raises:
    """Test tensor lifecycle with different dtypes."""
    var dtypes = List[DType]()
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)
    dtypes.append(DType.uint8)
    for i in range(len(dtypes)):
        for _ in range(100):
            var shape = List[Int]()
            shape.append(50)
            shape.append(50)
            var tensor = zeros(shape, dtypes[i])
            assert_equal_int(tensor._refcount[], 1, "Should have refcount 1")

    assert_true(True, "Different dtypes lifecycle test completed")


# ============================================================================
# Destructor Edge Cases
# ============================================================================


fn test_destructor_with_valid_refcount() raises:
    """Test destructor handles normal case correctly."""
    var tensor = zeros([10], DType.float32)
    # Verify refcount pointer is valid (non-null)
    var refcount_value = tensor._refcount[]
    assert_equal_int(refcount_value, 1, "Should have valid refcount")
    assert_true(True, "Destructor edge case test completed")


fn test_view_destructor_does_not_decrement_refcount() raises:
    """Test view destructor doesn't decrement refcount incorrectly."""
    var original = zeros([12], DType.float32)
    var initial_refcount = original._refcount[]

    if True:
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var view = original.reshape(shape)
        assert_true(view._is_view, "Should be view")
        # reshape calls copy() which increments refcount
        var inner_refcount = original._refcount[]
        assert_equal_int(
            inner_refcount,
            initial_refcount + 1,
            "View creation should increment refcount",
        )

    # After view destruction, refcount should return to initial
    var final_refcount = original._refcount[]
    assert_equal_int(
        final_refcount,
        initial_refcount,
        "View destruction should decrement refcount",
    )


# ============================================================================
# Test Runner
# ============================================================================


fn main() raises:
    """Run edge case and destructor tests."""
    print("Running memory leak tests - Part 3...")

    print("  Edge case tests...")
    test_empty_tensor_lifecycle()
    test_1d_tensor_lifecycle()
    test_different_dtypes_lifecycle()
    print("    Passed")

    print("  Destructor edge case tests...")
    test_destructor_with_valid_refcount()
    test_view_destructor_does_not_decrement_refcount()
    print("    Passed")

    print("\nAll Part 3 memory leak tests passed!")
