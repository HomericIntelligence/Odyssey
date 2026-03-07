"""Tests for the Apple Silicon guard in PrecisionConfig.bf16().

These tests verify that:
- The guard helper raises an error when Apple Silicon is simulated.
- The guard helper does not raise on non-Apple Silicon (e.g., Linux CI).
- PrecisionConfig.bf16() succeeds on non-Apple Silicon.
- The error message matches the expected string.
"""

from shared.training.precision_config import PrecisionConfig, _check_bf16_platform_support


fn test_check_bf16_platform_support_raises_on_apple() raises:
    """Test that _check_bf16_platform_support raises when is_apple=True."""
    print("Testing _check_bf16_platform_support raises on Apple Silicon...")

    var caught = False
    try:
        _check_bf16_platform_support(True)
    except e:
        caught = True
        var msg = String(e)
        if "Apple Silicon" not in msg:
            raise Error(
                "Error message should mention 'Apple Silicon', got: " + msg
            )
        if "fp16" not in msg:
            raise Error(
                "Error message should suggest fp16 alternative, got: " + msg
            )

    if not caught:
        raise Error(
            "_check_bf16_platform_support(True) should raise an error"
        )

    print("✓ _check_bf16_platform_support raises on simulated Apple Silicon")


fn test_check_bf16_platform_support_no_raise_on_non_apple() raises:
    """Test that _check_bf16_platform_support does not raise when is_apple=False."""
    print("Testing _check_bf16_platform_support passes on non-Apple Silicon...")

    _check_bf16_platform_support(False)

    print("✓ _check_bf16_platform_support does not raise on non-Apple Silicon")


fn test_bf16_succeeds_on_non_apple_silicon() raises:
    """Test that PrecisionConfig.bf16() succeeds on Linux CI (non-Apple Silicon)."""
    print("Testing PrecisionConfig.bf16() succeeds on non-Apple Silicon...")

    # On Linux CI, is_apple_silicon() returns False, so bf16() should not raise.
    var config = PrecisionConfig.bf16()

    if config.compute_dtype != DType.bfloat16:
        raise Error("BF16 config should use bfloat16 compute dtype")
    if config.storage_dtype != DType.bfloat16:
        raise Error("BF16 config should use bfloat16 storage dtype")

    print("✓ PrecisionConfig.bf16() succeeds on non-Apple Silicon")


fn test_bf16_error_message_content() raises:
    """Test the exact content of the Apple Silicon error message."""
    print("Testing Apple Silicon error message content...")

    var caught = False
    var error_msg = String("")
    try:
        _check_bf16_platform_support(True)
    except e:
        caught = True
        error_msg = String(e)

    if not caught:
        raise Error("Expected error was not raised")

    if "BF16" not in error_msg and "bfloat16" not in error_msg:
        raise Error(
            "Error message should mention BF16/bfloat16, got: " + error_msg
        )
    if "Apple Silicon" not in error_msg:
        raise Error(
            "Error message should mention Apple Silicon, got: " + error_msg
        )
    if "PrecisionConfig.fp16()" not in error_msg:
        raise Error(
            "Error message should suggest PrecisionConfig.fp16(), got: "
            + error_msg
        )

    print("✓ Error message has correct content")


fn main() raises:
    """Run all Apple Silicon guard tests."""
    print("=" * 60)
    print("BF16 APPLE SILICON GUARD TESTS")
    print("=" * 60)
    print()

    test_check_bf16_platform_support_raises_on_apple()
    test_check_bf16_platform_support_no_raise_on_non_apple()
    test_bf16_succeeds_on_non_apple_silicon()
    test_bf16_error_message_content()

    print()
    print("=" * 60)
    print("ALL BF16 APPLE SILICON GUARD TESTS PASSED! ✓")
    print("=" * 60)
