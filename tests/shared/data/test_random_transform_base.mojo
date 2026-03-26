"""Tests for RandomTransformBase probability handling.

Verifies that RandomTransformBase correctly handles probability-based
decisions for random transforms.
"""

from shared.data.random_transform_base import RandomTransformBase
from shared.testing import assert_true, assert_false, assert_equal


fn test_random_transform_base_creation() raises:
    """Test RandomTransformBase initialization."""
    var base = RandomTransformBase(0.5)
    assert_equal(base.p, 0.5, "Probability should be 0.5")

    var base_high = RandomTransformBase(0.9)
    assert_equal(base_high.p, 0.9, "Probability should be 0.9")

    var base_low = RandomTransformBase(0.1)
    assert_equal(base_low.p, 0.1, "Probability should be 0.1")


fn test_random_transform_base_should_apply():
    """Test should_apply method returns Bool."""
    var base = RandomTransformBase(0.5)

    # Call multiple times to ensure it returns a Bool
    for _ in range(10):
        var result = base.should_apply()
        # Just verify it returns a Bool (true or false)
        _ = result


fn test_random_transform_base_extreme_probabilities() raises:
    """Test RandomTransformBase with extreme probability values."""
    # Test p=0.0 (should rarely apply)
    var base_zero = RandomTransformBase(0.0)
    _ = base_zero.should_apply()

    # Test p=1.0 (should always apply)
    var base_one = RandomTransformBase(1.0)
    _ = base_one.should_apply()

    # Just verify they can be created and called without errors
    assert_equal(base_zero.p, 0.0, "Zero probability should be 0.0")
    assert_equal(base_one.p, 1.0, "Unit probability should be 1.0")


fn test_random_transform_base_default() raises:
    """Test RandomTransformBase default probability."""
    var base = RandomTransformBase()
    assert_equal(base.p, 0.5, "Default probability should be 0.5")


fn main() raises:
    """Run all tests."""
    print("Running RandomTransformBase tests...")

    test_random_transform_base_creation()
    print("  PASS: test_random_transform_base_creation")

    test_random_transform_base_should_apply()
    print("  PASS: test_random_transform_base_should_apply")

    test_random_transform_base_extreme_probabilities()
    print("  PASS: test_random_transform_base_extreme_probabilities")

    test_random_transform_base_default()
    print("  PASS: test_random_transform_base_default")

    print("All tests passed!")
