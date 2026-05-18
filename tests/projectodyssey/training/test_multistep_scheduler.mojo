"""Tests for MultiStepLR scheduler.

Tests cover:
- MultiStepLR: Multi-step decay scheduler
- Milestone-based learning rate reduction

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
from projectodyssey.training.schedulers import MultiStepLR


# ============================================================================
# MultiStepLR Tests
# ============================================================================


def test_multistep_lr_initialization() raises:
    """Test MultiStepLR scheduler initialization."""
    var milestones = List[Int]()
    milestones.append(30)
    milestones.append(60)

    var scheduler = MultiStepLR(base_lr=0.1, milestones=milestones, gamma=0.1)

    assert_almost_equal(scheduler.base_lr, 0.1)
    assert_equal(len(scheduler.milestones), 2)
    assert_almost_equal(scheduler.gamma, 0.1)


def test_multistep_lr_before_first_milestone() raises:
    """Test MultiStepLR before first milestone.

    Before any milestone, LR should equal base_lr.
    """
    var milestones = List[Int]()
    milestones.append(30)
    milestones.append(60)

    var scheduler = MultiStepLR(base_lr=0.1, milestones=milestones, gamma=0.1)

    for epoch in range(30):
        var lr = scheduler.get_lr(epoch=epoch)
        assert_almost_equal(lr, 0.1, tolerance=1e-6)


def test_multistep_lr_at_first_milestone() raises:
    """Test MultiStepLR at first milestone.

    At first milestone, LR should be multiplied by gamma once.
    Formula: lr = base_lr * gamma^1.
    """
    var milestones = List[Int]()
    milestones.append(30)
    milestones.append(60)

    var scheduler = MultiStepLR(base_lr=0.1, milestones=milestones, gamma=0.1)

    var lr_at_milestone = scheduler.get_lr(epoch=30)
    assert_almost_equal(lr_at_milestone, 0.01, tolerance=1e-6)


def test_multistep_lr_between_milestones() raises:
    """Test MultiStepLR between milestones.

    Between milestones, LR should remain constant.
    """
    var milestones = List[Int]()
    milestones.append(30)
    milestones.append(60)

    var scheduler = MultiStepLR(base_lr=0.1, milestones=milestones, gamma=0.1)

    var lr_30 = scheduler.get_lr(epoch=30)
    var lr_45 = scheduler.get_lr(epoch=45)
    var lr_59 = scheduler.get_lr(epoch=59)

    assert_almost_equal(lr_30, 0.01, tolerance=1e-6)
    assert_almost_equal(lr_45, 0.01, tolerance=1e-6)
    assert_almost_equal(lr_59, 0.01, tolerance=1e-6)


def test_multistep_lr_at_second_milestone() raises:
    """Test MultiStepLR at second milestone.

    At second milestone, LR should be multiplied by gamma twice.
    Formula: lr = base_lr * gamma^2.
    """
    var milestones = List[Int]()
    milestones.append(30)
    milestones.append(60)

    var scheduler = MultiStepLR(base_lr=0.1, milestones=milestones, gamma=0.1)

    var lr_at_milestone = scheduler.get_lr(epoch=60)
    assert_almost_equal(lr_at_milestone, 0.001, tolerance=1e-6)


def test_multistep_lr_after_last_milestone() raises:
    """Test MultiStepLR after last milestone.

    After all milestones, LR should remain constant at reduced value.
    """
    var milestones = List[Int]()
    milestones.append(30)
    milestones.append(60)

    var scheduler = MultiStepLR(base_lr=0.1, milestones=milestones, gamma=0.1)

    var lr_70 = scheduler.get_lr(epoch=70)
    var lr_100 = scheduler.get_lr(epoch=100)

    assert_almost_equal(lr_70, 0.001, tolerance=1e-6)
    assert_almost_equal(lr_100, 0.001, tolerance=1e-6)


def main() raises:
    """Run all MultiStepLR tests."""
    print("Running MultiStepLR tests...")
    test_multistep_lr_initialization()
    test_multistep_lr_before_first_milestone()
    test_multistep_lr_at_first_milestone()
    test_multistep_lr_between_milestones()
    test_multistep_lr_at_second_milestone()
    test_multistep_lr_after_last_milestone()

    print("All MultiStepLR tests passed! ✓")
