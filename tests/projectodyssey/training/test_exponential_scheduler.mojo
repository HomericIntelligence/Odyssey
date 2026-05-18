"""Tests for ExponentialLR scheduler.

Tests cover:
- ExponentialLR: Exponential decay scheduler
- Decay calculation and progression

All tests use the real scheduler implementation.
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less_or_equal,
    TestFixtures,
)
from projectodyssey.training.schedulers import ExponentialLR


# ============================================================================
# ExponentialLR Tests
# ============================================================================


def test_exponential_lr_initialization() raises:
    """Test ExponentialLR scheduler initialization."""
    var scheduler = ExponentialLR(base_lr=0.1, gamma=0.95)

    assert_almost_equal(scheduler.base_lr, 0.1)
    assert_almost_equal(scheduler.gamma, 0.95)


def test_exponential_lr_epoch_zero() raises:
    """Test ExponentialLR at epoch 0 (initial learning rate).

    At epoch 0, LR should equal base_lr.
    Formula: lr = base_lr * gamma^0 = base_lr.
    """
    var scheduler = ExponentialLR(base_lr=0.1, gamma=0.95)

    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 0.1)


def test_exponential_lr_epoch_one() raises:
    """Test ExponentialLR at epoch 1.

    At epoch 1, LR should equal base_lr * gamma.
    Formula: lr = base_lr * gamma^1.
    """
    var scheduler = ExponentialLR(base_lr=0.1, gamma=0.95)

    var lr1 = scheduler.get_lr(epoch=1)
    assert_almost_equal(lr1, 0.095, tolerance=1e-6)


def test_exponential_lr_exponential_decay() raises:
    """Test ExponentialLR decays exponentially over epochs."""
    var scheduler = ExponentialLR(base_lr=0.1, gamma=0.95)

    var previous_lr = scheduler.get_lr(0)
    for epoch in range(1, 51):
        var current_lr = scheduler.get_lr(epoch)

        # LR should decrease or stay the same
        assert_less_or_equal(current_lr, previous_lr)
        previous_lr = current_lr


def test_exponential_lr_different_gamma() raises:
    """Test ExponentialLR with different gamma values."""
    var scheduler_aggressive = ExponentialLR(base_lr=0.1, gamma=0.9)
    var scheduler_gradual = ExponentialLR(base_lr=0.1, gamma=0.99)

    var lr_agg_10 = scheduler_aggressive.get_lr(epoch=10)
    var lr_grad_10 = scheduler_gradual.get_lr(epoch=10)

    # Aggressive decay (gamma=0.9) should result in lower LR than gradual (gamma=0.99)
    assert_less_or_equal(lr_agg_10, lr_grad_10)


def main() raises:
    """Run all ExponentialLR tests."""
    print("Running ExponentialLR tests...")
    test_exponential_lr_initialization()
    test_exponential_lr_epoch_zero()
    test_exponential_lr_epoch_one()
    test_exponential_lr_exponential_decay()
    test_exponential_lr_different_gamma()

    print("All ExponentialLR tests passed! ✓")
