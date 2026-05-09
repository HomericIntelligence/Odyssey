"""Unit tests for Step Learning Rate Scheduler.

Tests cover:
- Step decay at fixed intervals
- Learning rate reduction by gamma factor
- Edge cases (step size, gamma values)

All tests use the real StepLR implementation.
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less_or_equal,
    TestFixtures,
)
from shared.training.schedulers import StepLR


def test_step_scheduler_initialization() raises:
    """Test StepLR scheduler initialization with hyperparameters."""
    var scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.1)

    # Verify initial parameters
    assert_equal(scheduler.step_size, 10)
    assert_almost_equal(scheduler.gamma, 0.1)
    assert_almost_equal(scheduler.base_lr, 0.1)


def test_step_scheduler_reduces_lr_at_step() raises:
    """Test StepLR reduces learning rate at specified step.

    At epoch = step_size, LR *= gamma
    LR remains constant between steps.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.1)

    # Initial LR (epoch 0)
    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 1.0)

    # Steps 1-4: LR unchanged
    for epoch in range(1, 5):
        var lr = scheduler.get_lr(epoch=epoch)
        assert_almost_equal(lr, 1.0)

    # Step 5: LR reduced
    var lr5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr5, 0.1)


def test_step_scheduler_multiple_steps() raises:
    """Test StepLR continues reducing LR at each step interval.

    LR reduction continues every step_size epochs:
    - Epoch 0-4: LR = initial_lr
    - Epoch 5-9: LR = initial_lr * gamma
    - Epoch 10-14: LR = initial_lr * gamma^2
    """
    var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.1)

    # Epoch 0: LR = 1.0
    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 1.0)

    # Epoch 5: LR = 0.1
    var lr5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr5, 0.1)

    # Epoch 10: LR = 0.01
    var lr10 = scheduler.get_lr(epoch=10)
    assert_almost_equal(lr10, 0.01)


def test_step_scheduler_different_gamma_values() raises:
    """Test StepLR with different gamma (reduction factor) values.

    gamma determines how much LR is reduced:
    - gamma=0.1: LR reduced to 10% of previous value
    - gamma=0.5: LR reduced to 50% of previous value
    - gamma=0.9: LR reduced to 90% of previous value.
    """
    # Test gamma = 0.5
    var scheduler1 = StepLR(base_lr=1.0, step_size=1, gamma=0.5)
    assert_almost_equal(scheduler1.get_lr(1), 0.5)
    assert_almost_equal(scheduler1.get_lr(2), 0.25)

    # Test gamma = 0.9 (smaller reduction)
    var scheduler2 = StepLR(base_lr=1.0, step_size=1, gamma=0.9)
    assert_almost_equal(scheduler2.get_lr(1), 0.9)
    assert_almost_equal(scheduler2.get_lr(2), 0.81)


def test_step_scheduler_gamma_one() raises:
    """Test StepLR with gamma=1.0 (no reduction).

    gamma=1.0 should result in no LR change.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=1.0)

    # LR should remain constant
    for epoch in range(0, 11):
        assert_almost_equal(scheduler.get_lr(epoch), 1.0)


def test_step_scheduler_different_step_sizes() raises:
    """Test StepLR with different step_size values.

    step_size determines frequency of LR reduction:
    - step_size=1: Reduce every epoch
    - step_size=10: Reduce every 10 epochs.
    """
    # Test step_size = 1 (reduce every epoch)
    var scheduler1 = StepLR(base_lr=1.0, step_size=1, gamma=0.5)
    assert_almost_equal(scheduler1.get_lr(0), 1.0)
    assert_almost_equal(scheduler1.get_lr(1), 0.5)
    assert_almost_equal(scheduler1.get_lr(2), 0.25)

    # Test step_size = 10 (reduce every 10 epochs)
    var scheduler2 = StepLR(base_lr=1.0, step_size=10, gamma=0.5)

    # Epochs 0-9: No change
    for epoch in range(0, 10):
        assert_almost_equal(scheduler2.get_lr(epoch), 1.0)

    # Epoch 10: Reduce
    assert_almost_equal(scheduler2.get_lr(10), 0.5)


def test_step_scheduler_zero_step_size() raises:
    """Test StepLR with step_size=0 returns base_lr.

    The implementation handles this gracefully by returning base_lr.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=0, gamma=0.1)

    # Should return base_lr (defensive behavior)
    assert_almost_equal(scheduler.get_lr(0), 1.0)
    assert_almost_equal(scheduler.get_lr(10), 1.0)


def test_step_scheduler_negative_gamma() raises:
    """Test StepLR with negative gamma.

    While mathematically valid, negative gamma would make LR oscillate.
    Current implementation allows it (no validation).
    """
    var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=-0.5)

    # Mathematically: 1.0 * (-0.5)^1 = -0.5
    var lr1 = scheduler.get_lr(1)
    assert_almost_equal(lr1, -0.5)


def test_step_scheduler_very_small_lr() raises:
    """Test StepLR continues to reduce LR even when very small.

    LR can become arbitrarily small (no minimum threshold).
    """
    var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=0.1)

    # After 10 steps: LR = 1.0 * 0.1^10 = 1e-10
    var lr10 = scheduler.get_lr(10)
    assert_almost_equal(lr10, 1e-10, tolerance=1e-15)


def test_step_scheduler_property_monotonic_decrease() raises:
    """Property: Learning rate should never increase (for 0 < gamma < 1).

    StepLR should only decrease or maintain LR, never increase it.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.5)

    var previous_lr = scheduler.get_lr(0)
    for epoch in range(1, 51):
        var current_lr = scheduler.get_lr(epoch)

        # LR should not increase
        assert_less_or_equal(current_lr, previous_lr)
        previous_lr = current_lr


def test_step_scheduler_formula_accuracy() raises:
    """Test StepLR matches the mathematical formula exactly.

    Formula: lr = base_lr * gamma^(epoch // step_size).
    """
    var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

    # Epoch 0: gamma^0 = 1.0
    assert_almost_equal(scheduler.get_lr(0), 0.1)

    # Epoch 29: gamma^0 = 1.0
    assert_almost_equal(scheduler.get_lr(29), 0.1)

    # Epoch 30: gamma^1 = 0.1
    assert_almost_equal(scheduler.get_lr(30), 0.01)

    # Epoch 60: gamma^2 = 0.01
    assert_almost_equal(scheduler.get_lr(60), 0.001)


def main() raises:
    """Run all test_step_scheduler tests."""
    print("Running test_step_scheduler tests...")

    test_step_scheduler_initialization()
    print("✓ test_step_scheduler_initialization")

    test_step_scheduler_reduces_lr_at_step()
    print("✓ test_step_scheduler_reduces_lr_at_step")

    test_step_scheduler_multiple_steps()
    print("✓ test_step_scheduler_multiple_steps")

    test_step_scheduler_different_gamma_values()
    print("✓ test_step_scheduler_different_gamma_values")

    test_step_scheduler_gamma_one()
    print("✓ test_step_scheduler_gamma_one")

    test_step_scheduler_different_step_sizes()
    print("✓ test_step_scheduler_different_step_sizes")

    test_step_scheduler_zero_step_size()
    print("✓ test_step_scheduler_zero_step_size")

    test_step_scheduler_negative_gamma()
    print("✓ test_step_scheduler_negative_gamma")

    test_step_scheduler_very_small_lr()
    print("✓ test_step_scheduler_very_small_lr")

    test_step_scheduler_property_monotonic_decrease()
    print("✓ test_step_scheduler_property_monotonic_decrease")

    test_step_scheduler_formula_accuracy()
    print("✓ test_step_scheduler_formula_accuracy")

    print("\nAll test_step_scheduler tests passed!")
