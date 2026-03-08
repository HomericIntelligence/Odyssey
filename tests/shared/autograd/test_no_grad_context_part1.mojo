# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_no_grad_context.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for NoGradContext enter/exit and disable/restore gradient tracking.

Tests the gradient tracking control functionality including:
- NoGradContext enter/exit methods
- disable_gradient_tracking and restore_gradient_tracking functions

Split from test_no_grad_context.mojo per ADR-009 (≤10 fn test_ per file).
"""

from testing import assert_true
from shared.autograd import (
    GradientTape,
    NoGradContext,
    disable_gradient_tracking,
    restore_gradient_tracking,
)


# ============================================================================
# NoGradContext Enter/Exit Tests
# ============================================================================


fn test_no_grad_context_enter_disables_tracking() raises:
    """Test that NoGradContext.enter() disables tape recording."""
    var tape = GradientTape()
    tape.enable()
    assert_true(tape.enabled, "Tape should be enabled initially")

    var ctx = NoGradContext()
    ctx.enter(tape)
    assert_true(not tape.enabled, "Tape should be disabled after enter()")


fn test_no_grad_context_exit_restores_enabled() raises:
    """Test that NoGradContext.exit() restores enabled state."""
    var tape = GradientTape()
    tape.enable()
    assert_true(tape.enabled, "Tape should be enabled initially")

    var ctx = NoGradContext()
    ctx.enter(tape)
    assert_true(not tape.enabled, "Tape should be disabled after enter()")

    ctx.exit(tape)
    assert_true(tape.enabled, "Tape should be re-enabled after exit()")


fn test_no_grad_context_exit_restores_disabled() raises:
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


fn test_no_grad_context_nested_contexts() raises:
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


# ============================================================================
# Disable/Restore Gradient Tracking Tests
# ============================================================================


fn test_disable_gradient_tracking_returns_previous_state_enabled() raises:
    """Test disable_gradient_tracking returns True when tape was enabled."""
    var tape = GradientTape()
    tape.enable()

    var was_enabled = disable_gradient_tracking(tape)
    assert_true(was_enabled, "Should return True when tape was enabled")
    assert_true(not tape.enabled, "Tape should be disabled")


fn test_disable_gradient_tracking_returns_previous_state_disabled() raises:
    """Test disable_gradient_tracking returns False when tape was disabled."""
    var tape = GradientTape()
    tape.disable()

    var was_enabled = disable_gradient_tracking(tape)
    assert_true(not was_enabled, "Should return False when tape was disabled")
    assert_true(not tape.enabled, "Tape should remain disabled")


fn test_restore_gradient_tracking_enables() raises:
    """Test restore_gradient_tracking can re-enable tape."""
    var tape = GradientTape()
    tape.disable()

    restore_gradient_tracking(tape, True)
    assert_true(tape.enabled, "Tape should be enabled after restore with True")


fn test_restore_gradient_tracking_keeps_disabled() raises:
    """Test restore_gradient_tracking keeps tape disabled if True is False."""
    var tape = GradientTape()
    tape.enable()

    restore_gradient_tracking(tape, False)
    assert_true(
        not tape.enabled, "Tape should be disabled after restore with False"
    )


fn main() raises:
    """Run NoGradContext enter/exit and disable/restore tests."""
    print("Running NoGradContext enter/exit tests...")
    test_no_grad_context_enter_disables_tracking()
    test_no_grad_context_exit_restores_enabled()
    test_no_grad_context_exit_restores_disabled()
    test_no_grad_context_nested_contexts()

    print("Running disable/restore gradient tracking tests...")
    test_disable_gradient_tracking_returns_previous_state_enabled()
    test_disable_gradient_tracking_returns_previous_state_disabled()
    test_restore_gradient_tracking_enables()
    test_restore_gradient_tracking_keeps_disabled()

    print("\nAll NoGradContext part1 tests passed! ✓")
