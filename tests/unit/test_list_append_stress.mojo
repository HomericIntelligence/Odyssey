"""Stress test for List[Int] append operations.

The crash occurs during List[Int].append(0) in AnyTensor.__init__ at line 107.
This test stresses List operations to identify if there's a memory issue with
List itself or with the specific pattern used in AnyTensor.
"""

from std.collections import List
from std.testing import assert_equal


def test_list_basic_append() raises:
    """Test basic List append operations."""
    print("\n=== Test 1: Basic List Append ===")

    var lst = List[Int]()
    print("Appending 10 elements...")

    for i in range(10):
        lst.append(i)
        print("  Appended", i, "len =", len(lst))

    assert_equal(len(lst), 10)
    print("SUCCESS: All elements appended")


def test_list_large_append() raises:
    """Test List with many append operations."""
    print("\n=== Test 2: Large List Append ===")

    var sizes = List[Int]()
    sizes.append(100)
    sizes.append(1000)
    sizes.append(10000)
    sizes.append(100000)

    for i in range(len(sizes)):
        var size = sizes[i]
        print("\nAppending", size, "elements...")

        var lst = List[Int]()
        for j in range(size):
            lst.append(j)

        print("  SUCCESS: len =", len(lst))
        assert_equal(len(lst), size)


def test_list_rapid_allocations() raises:
    """Test rapid List allocations and deallocations."""
    print("\n=== Test 3: Rapid List Allocations ===")

    print("Creating and destroying 1000 lists...")
    for _ in range(1000):
        var lst = List[Int]()
        for j in range(10):
            lst.append(j)
        # List goes out of scope and is destroyed

    print("SUCCESS: 1000 lists created and destroyed")


def test_list_append_pattern_from_anytensor() raises:
    """Test the exact append pattern used in AnyTensor.__init__.

    This reproduces the pattern from lines 104-110:
        self._strides = List[Int]()
        var stride = 1
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides.append(0)  # Preallocate <- LINE 107
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides[i] = stride
            stride *= self._shape[i]
    """
    print("\n=== Test 4: AnyTensor Stride Pattern ===")

    # Test case 1: 1D shape
    print("\nTest case 1: 1D shape [2]")
    var shape1 = List[Int]()
    shape1.append(2)
    test_stride_pattern(shape1)

    # Test case 2: 2D shape
    print("\nTest case 2: 2D shape [2, 3]")
    var shape2 = List[Int]()
    shape2.append(2)
    shape2.append(3)
    test_stride_pattern(shape2)

    # Test case 3: Exact crash case
    print("\nTest case 3: 2D shape [2, 47] (CRASH CASE)")
    var shape3 = List[Int]()
    shape3.append(2)
    shape3.append(47)
    test_stride_pattern(shape3)

    # Test case 4: 3D shape
    print("\nTest case 4: 3D shape [10, 20, 30]")
    var shape4 = List[Int]()
    shape4.append(10)
    shape4.append(20)
    shape4.append(30)
    test_stride_pattern(shape4)


def test_stride_pattern(shape: List[Int]) raises:
    """Helper function to test stride pattern for a given shape."""
    print("  Testing pattern with shape of", len(shape), "dimensions...")

    # Step 1: Preallocate with zeros (backward iteration)
    print("  Step 1: Preallocate with append(0)...")
    var strides = List[Int]()
    for i in range(len(shape) - 1, -1, -1):
        print("    i =", i, "appending 0...")
        strides.append(0)  # THIS IS LINE 107
        print("    len =", len(strides))

    print("  Step 1 complete: len =", len(strides))

    # Step 2: Fill in actual stride values
    print("  Step 2: Calculate strides...")
    var stride = 1
    for i in range(len(shape) - 1, -1, -1):
        print("    i =", i, "stride =", stride)
        strides[i] = stride
        stride *= shape[i]

    print("  Step 2 complete: strides =", end=" ")
    for i in range(len(strides)):
        print(strides[i], end=" ")
    print()

    assert_equal(len(strides), len(shape))
    print("  SUCCESS")


def test_list_reverse_iteration_append() raises:
    """Test List append during reverse iteration (specific to crash pattern)."""
    print("\n=== Test 5: Reverse Iteration Append ===")

    print("Testing reverse iteration append with various sizes...")

    var sizes = List[Int]()
    sizes.append(1)
    sizes.append(2)
    sizes.append(3)
    sizes.append(10)
    sizes.append(47)  # Exact crash case
    sizes.append(100)

    for size_idx in range(len(sizes)):
        var size = sizes[size_idx]
        print("\n  Size:", size)

        var lst = List[Int]()

        # Reverse iteration append (like in AnyTensor)
        print("    Reverse iteration append...")
        for _ in range(size):
            lst.append(0)

        print("    len =", len(lst))
        assert_equal(len(lst), size)

        # Now write values in reverse order
        print("    Writing values...")
        for i in range(size - 1, -1, -1):
            lst[i] = i

        print("    SUCCESS")


def test_list_memory_stress() raises:
    """Stress test List memory allocation."""
    print("\n=== Test 6: List Memory Stress ===")

    print("Creating many lists sequentially...")
    var total_lists = 0

    for _ in range(100):
        var lst = List[Int]()
        for j in range(100):
            lst.append(j)
        total_lists += 1
        # List goes out of scope

    print("  Created", total_lists, "lists")
    print("  Each with 100 elements")
    print("SUCCESS")


def test_list_growth_pattern() raises:
    """Test List growth patterns to identify reallocation issues."""
    print("\n=== Test 7: List Growth Pattern ===")

    print("Testing List growth with incremental appends...")
    var lst = List[Int]()

    for i in range(1000):
        lst.append(i)
        if i % 100 == 0:
            print("  Appended", i + 1, "elements, len =", len(lst))

    assert_equal(len(lst), 1000)
    print("SUCCESS")


def test_list_zero_append_stress() raises:
    """Stress test specifically appending zeros (exact crash pattern)."""
    print("\n=== Test 8: Zero Append Stress ===")

    print("Appending zeros in various patterns...")

    # Pattern 1: Sequential forward
    print("\n  Pattern 1: Forward append of zeros...")
    var lst1 = List[Int]()
    for _ in range(100):
        lst1.append(0)
    print("    len =", len(lst1))

    # Pattern 2: Sequential backward (like AnyTensor)
    print("\n  Pattern 2: Backward loop, forward append of zeros...")
    var lst2 = List[Int]()
    for _ in range(100):
        lst2.append(0)
    print("    len =", len(lst2))

    # Pattern 3: Large size
    print("\n  Pattern 3: Large backward loop...")
    var lst3 = List[Int]()
    for _ in range(10000):
        lst3.append(0)
    print("    len =", len(lst3))

    print("SUCCESS: All patterns completed")


def main() raises:
    """Run all List stress tests."""
    print("=" * 60)
    print("LIST APPEND STRESS TESTS")
    print("=" * 60)

    test_list_basic_append()
    test_list_large_append()
    test_list_rapid_allocations()
    test_list_append_pattern_from_anytensor()
    test_list_reverse_iteration_append()
    test_list_memory_stress()
    test_list_growth_pattern()
    test_list_zero_append_stress()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
