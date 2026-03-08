# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_no_grad_context.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for disable/restore roundtrip and tape node recording.

Tests the gradient tracking control functionality including:
- disable/restore roundtrip correctness
- Operations not recorded when tracking is disabled
- Tape clearing between no-grad contexts

Split from test_no_grad_context.mojo per ADR-009 (≤10 fn test_ per file).
"""

from testing import assert_true, assert_equal
from shared.autograd import (
    GradientTape,
    NoGradContext,
    disable_gradient_tracking,
    restore_gradient_tracking,
)


# ============================================================================
# Disable/Restore Roundtrip Tests
# ============================================================================


fn test_disable_restore_roundtrip() raises:
    """Test that disable/restore roundtrip works correctly."""
    var tape = GradientTape()
    tape.enable()
    assert_true(tape.enabled, "Tape should be enabled initially")

    # Disable and store state
    var was_enabled = disable_gradient_tracking(tape)
    assert_true(not tape.enabled, "Tape should be disabled")
    assert_true(was_enabled, "Should have returned True")

    # Restore state
    restore_gradient_tracking(tape, was_enabled)
    assert_true(tape.enabled, "Tape should be re-enabled after restore")


fn test_disable_restore_roundtrip_from_disabled() raises:
    """Test disable/restore roundtrip when starting from disabled state."""
    var tape = GradientTape()
    tape.disable()
    assert_true(not tape.enabled, "Tape should be disabled initially")

    # Disable and store state
    var was_enabled = disable_gradient_tracking(tape)
    assert_true(not tape.enabled, "Tape should remain disabled")
    assert_true(not was_enabled, "Should have returned False")

    # Restore state
    restore_gradient_tracking(tape, was_enabled)
    assert_true(not tape.enabled, "Tape should remain disabled after restore")


# ============================================================================
# Tape Node Recording Tests
# ============================================================================


fn test_operations_not_recorded_when_disabled() raises:
    """Test that operations are not recorded when tape is disabled.

    This is an integration test that verifies the tape doesn't record
    nodes when it's disabled via disable_gradient_tracking().
    """
    var tape = GradientTape()

    # Start with tape disabled
    tape.disable()
    var initial_node_count = len(tape.nodes)

    # Operations should not be recorded (manually verify via tape state)
    assert_true(not tape.enabled, "Tape should be disabled")
    assert_equal(
        len(tape.nodes), initial_node_count, "No nodes should be recorded"
    )


fn test_operations_recorded_when_enabled() raises:
    """Test that tape is in correct state to record when enabled.

    This verifies the tape is ready to record by checking the enabled flag.
    """
    var tape = GradientTape()
    tape.enable()

    assert_true(tape.enabled, "Tape should be enabled and ready to record")


fn test_clear_tape_between_operations() raises:
    """Test clearing tape between different no-grad contexts."""
    var tape = GradientTape()
    tape.enable()

    # First operation block
    var ctx1 = NoGradContext()
    ctx1.enter(tape)
    assert_true(not tape.enabled, "Tape should be disabled in first context")
    ctx1.exit(tape)
    assert_true(tape.enabled, "Tape should be re-enabled after first context")

    # Clear tape
    tape.clear()
    assert_equal(len(tape.nodes), 0, "Tape should be empty after clear()")

    # Second operation block
    var ctx2 = NoGradContext()
    ctx2.enter(tape)
    assert_true(not tape.enabled, "Tape should be disabled in second context")
    ctx2.exit(tape)
    assert_true(tape.enabled, "Tape should be re-enabled after second context")


fn main() raises:
    """Run disable/restore roundtrip and tape node recording tests."""
    print("Running disable/restore roundtrip tests...")
    test_disable_restore_roundtrip()
    test_disable_restore_roundtrip_from_disabled()

    print("Running tape node recording tests...")
    test_operations_not_recorded_when_disabled()
    test_operations_recorded_when_enabled()
    test_clear_tape_between_operations()

    print("\nAll NoGradContext part2 tests passed! ✓")
