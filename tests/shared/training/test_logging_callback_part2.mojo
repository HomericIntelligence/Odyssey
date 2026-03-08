# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_logging_callback.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for Logging Callback (Part 2).

Tests cover:
- Callback interface (train end)
- Edge cases

All tests use the real LoggingCallback implementation.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_greater,
    TestFixtures,
)
from shared.training.callbacks import LoggingCallback
from shared.training.base import TrainingState


# ============================================================================
# Callback Interface Tests
# ============================================================================


fn test_logging_callback_on_train_end() raises:
    """Test on_train_end does not affect log count."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=10, learning_rate=0.1)

    # Set log count to some value
    for epoch in range(5):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    var count_before = logger.log_count

    # Call on_train_end
    _ = logger.on_train_end(state)

    # Log count should be unchanged
    assert_equal(logger.log_count, count_before)


# ============================================================================
# Edge Cases
# ============================================================================


fn test_logging_callback_zero_interval() raises:
    """Test LoggingCallback with log_interval=0 (undefined behavior)."""
    # Note: Division by zero may occur in modulo operation
    # This test just verifies the callback can be created
    var logger = LoggingCallback(log_interval=0)
    assert_equal(logger.log_interval, 0)


fn test_logging_callback_large_interval() raises:
    """Test LoggingCallback with very large log_interval."""
    var logger = LoggingCallback(log_interval=1000)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Run for 10 epochs
    for epoch in range(10):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    # Should have logged only once (epoch 0)
    assert_equal(logger.log_count, 1)

    # Run until next log point
    state.epoch = 1000
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 2)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run logging callback tests part 2."""
    print("Running callback interface tests...")
    test_logging_callback_on_train_end()

    print("Running edge cases...")
    test_logging_callback_zero_interval()
    test_logging_callback_large_interval()

    print("\nAll logging callback part 2 tests passed! ✓")
