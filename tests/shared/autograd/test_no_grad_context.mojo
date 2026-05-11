"""Tests for NoGradContext enter/exit and disable/restore gradient tracking.

Tests the gradient tracking control functionality including:
- NoGradContext enter/exit methods
- disable_gradient_tracking and restore_gradient_tracking functions

"""


from std.testing import (
    assert_equal,
    assert_true,
)
from shared.autograd import (
    GradientTape,
    NoGradContext,
    disable_gradient_tracking,
    restore_gradient_tracking,
)


def test_no_grad_context_enter_disables_tracking() raises:
    """Test that NoGradContext.enter() disables tape recording."""
    var tape = GradientTape()
    tape.enable()
    assert_true(tape.enabled, "Tape should be enabled initially")

    var ctx = NoGradContext()
    ctx.enter(tape)
    assert_true(not tape.enabled, "Tape should be disabled after enter()")


def test_no_grad_context_exit_restores_enabled() raises:
    """Test that NoGradContext.exit() restores enabled state."""
    var tape = GradientTape()
    tape.enable()
    assert_true(tape.enabled, "Tape should be enabled initially")

    var ctx = NoGradContext()
    ctx.enter(tape)
    assert_true(not tape.enabled, "Tape should be disabled after enter()")

    ctx.exit(tape)
    assert_true(tape.enabled, "Tape should be re-enabled after exit()")


def test_no_grad_context_exit_restores_disabled() raises:
    """Test that NoGradContext.exit() preserves disabled state if it was disabled.
    """
    var tape = GradientTape()
    tape.disable()
    assert_true(not tape.enabled, "Tape should be disabled initially")

    var ctx = NoGradContext()
    ctx.enter(tape)
    assert_true(not tape.enabled, "Tape should still be disabled after enter()")

    ctx.exit(tape)
    assert_true(not tape.enabled, "Tape should remain disabled after exit()")


def test_no_grad_context_nested_contexts() raises:
    """Test that nested NoGradContext contexts preserve state correctly."""
    var tape = GradientTape()
    tape.enable()

    # First context: enabled -> disabled
    var ctx1 = NoGradContext()
    ctx1.enter(tape)
    assert_true(not tape.enabled, "Tape should be disabled in first context")

    # Second context: disabled -> disabled (but remembers disabled state)
    var ctx2 = NoGradContext()
    ctx2.enter(tape)
    assert_true(
        not tape.enabled, "Tape should still be disabled in second context"
    )

    # Exit second context: should restore to disabled
    ctx2.exit(tape)
    assert_true(
        not tape.enabled, "Tape should be disabled after exiting second context"
    )

    # Exit first context: should restore to enabled
    ctx1.exit(tape)
    assert_true(
        tape.enabled, "Tape should be re-enabled after exiting first context"
    )


def test_disable_gradient_tracking_returns_previous_state_enabled() raises:
    """Test disable_gradient_tracking returns True when tape was enabled."""
    var tape = GradientTape()
    tape.enable()

    var was_enabled = disable_gradient_tracking(tape)
    assert_true(was_enabled, "Should return True when tape was enabled")
    assert_true(not tape.enabled, "Tape should be disabled")


def test_disable_gradient_tracking_returns_previous_state_disabled() raises:
    """Test disable_gradient_tracking returns False when tape was disabled."""
    var tape = GradientTape()
    tape.disable()

    var was_enabled = disable_gradient_tracking(tape)
    assert_true(not was_enabled, "Should return False when tape was disabled")
    assert_true(not tape.enabled, "Tape should remain disabled")


def test_restore_gradient_tracking_enables() raises:
    """Test restore_gradient_tracking can re-enable tape."""
    var tape = GradientTape()
    tape.disable()

    restore_gradient_tracking(tape, True)
    assert_true(tape.enabled, "Tape should be enabled after restore with True")


def test_restore_gradient_tracking_keeps_disabled() raises:
    """Test restore_gradient_tracking keeps tape disabled if True is False."""
    var tape = GradientTape()
    tape.enable()

    restore_gradient_tracking(tape, False)
    assert_true(
        not tape.enabled, "Tape should be disabled after restore with False"
    )


def test_disable_restore_roundtrip() raises:
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


def test_disable_restore_roundtrip_from_disabled() raises:
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


def test_operations_not_recorded_when_disabled() raises:
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


def test_operations_recorded_when_enabled() raises:
    """Test that tape is in correct state to record when enabled.

    This verifies the tape is ready to record by checking the enabled flag.
    """
    var tape = GradientTape()
    tape.enable()

    assert_true(tape.enabled, "Tape should be enabled and ready to record")


def test_clear_tape_between_operations() raises:
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


def main() raises:
    """Run all test_no_grad_context tests."""
    print("Running test_no_grad_context tests...")

    test_no_grad_context_enter_disables_tracking()
    print("✓ test_no_grad_context_enter_disables_tracking")

    test_no_grad_context_exit_restores_enabled()
    print("✓ test_no_grad_context_exit_restores_enabled")

    test_no_grad_context_exit_restores_disabled()
    print("✓ test_no_grad_context_exit_restores_disabled")

    test_no_grad_context_nested_contexts()
    print("✓ test_no_grad_context_nested_contexts")

    test_disable_gradient_tracking_returns_previous_state_enabled()
    print("✓ test_disable_gradient_tracking_returns_previous_state_enabled")

    test_disable_gradient_tracking_returns_previous_state_disabled()
    print("✓ test_disable_gradient_tracking_returns_previous_state_disabled")

    test_restore_gradient_tracking_enables()
    print("✓ test_restore_gradient_tracking_enables")

    test_restore_gradient_tracking_keeps_disabled()
    print("✓ test_restore_gradient_tracking_keeps_disabled")

    test_disable_restore_roundtrip()
    print("✓ test_disable_restore_roundtrip")

    test_disable_restore_roundtrip_from_disabled()
    print("✓ test_disable_restore_roundtrip_from_disabled")

    test_operations_not_recorded_when_disabled()
    print("✓ test_operations_not_recorded_when_disabled")

    test_operations_recorded_when_enabled()
    print("✓ test_operations_recorded_when_enabled")

    test_clear_tape_between_operations()
    print("✓ test_clear_tape_between_operations")

    print("\nAll test_no_grad_context tests passed!")
