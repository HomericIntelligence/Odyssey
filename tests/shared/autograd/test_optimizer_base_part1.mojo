"""Tests for optimizer base learning rate functionality.

Tests the OptimizerBase learning rate get/set methods and validation:
- Learning rate get/set for SGD, Adam, AdaGrad, RMSprop
- Learning rate validation

Split from test_optimizer_base.mojo to comply with ADR-009 (≤10 fn test_ per file).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_optimizer_base.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true
from tests.shared.conftest import assert_almost_equal
from shared.tensor.any_tensor import AnyTensor
from shared.autograd import Variable, GradientTape, SGD, Adam, AdaGrad, RMSprop
from shared.autograd.optimizer_base import validate_learning_rate


# ============================================================================
# Learning Rate Get/Set Tests
# ============================================================================


fn test_sgd_get_set_lr() raises:
    """Test SGD learning rate get/set methods."""
    var optimizer = SGD(learning_rate=0.01)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.01, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.001)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.001, tolerance=1e-10)


fn test_adam_get_set_lr() raises:
    """Test Adam learning rate get/set methods."""
    var optimizer = Adam(learning_rate=0.001)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.001, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.0001)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.0001, tolerance=1e-10)


fn test_adagrad_get_set_lr() raises:
    """Test AdaGrad learning rate get/set methods."""
    var optimizer = AdaGrad(learning_rate=0.01)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.01, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.005)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.005, tolerance=1e-10)


fn test_rmsprop_get_set_lr() raises:
    """Test RMSprop learning rate get/set methods."""
    var optimizer = RMSprop(learning_rate=0.01)

    # Test get_lr
    var lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.01, tolerance=1e-10)

    # Test set_lr
    optimizer.set_lr(0.02)
    lr = optimizer.get_lr()
    assert_almost_equal(lr, 0.02, tolerance=1e-10)


fn test_set_lr_validation() raises:
    """Test that set_lr validates learning rate is positive."""
    var optimizer = SGD(learning_rate=0.01)

    # Should raise error for non-positive learning rate
    try:
        optimizer.set_lr(0.0)
        assert_true(False, "Should have raised error for lr=0.0")
    except e:
        assert_true(True, "Correctly raised error for lr=0.0")

    try:
        optimizer.set_lr(-0.01)
        assert_true(False, "Should have raised error for negative lr")
    except e:
        assert_true(True, "Correctly raised error for negative lr")


fn test_validate_learning_rate_function() raises:
    """Test the validate_learning_rate utility function."""
    # Should not raise for positive values
    validate_learning_rate(0.01)
    validate_learning_rate(0.001)
    validate_learning_rate(1.0)

    # Should raise for non-positive values
    try:
        validate_learning_rate(0.0)
        assert_true(False, "Should have raised error for lr=0.0")
    except e:
        assert_true(True, "Correctly raised error for lr=0.0")

    try:
        validate_learning_rate(-0.01)
        assert_true(False, "Should have raised error for negative lr")
    except e:
        assert_true(True, "Correctly raised error for negative lr")


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run optimizer base learning rate tests."""
    print("Running learning rate get/set tests...")
    test_sgd_get_set_lr()
    test_adam_get_set_lr()
    test_adagrad_get_set_lr()
    test_rmsprop_get_set_lr()
    test_set_lr_validation()
    test_validate_learning_rate_function()

    print("\nAll optimizer base part1 tests passed! ✓")
