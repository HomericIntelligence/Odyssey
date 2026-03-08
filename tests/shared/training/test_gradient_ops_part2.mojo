# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_ops.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for gradient zeroing operations and workflow (Part 2 of 2).

Verifies correctness and performance of in-place gradient zeroing
and the complete gradient accumulation workflow.

Test Coverage:
- zero_gradient_inplace: Complete zeroing across dtypes
- gradient_ops_workflow: End-to-end accumulation, scaling, and zeroing

All tests use small tensors for fast runtime.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from shared.training.gradient_ops import (
    accumulate_gradient_inplace,
    scale_gradient_inplace,
    zero_gradient_inplace,
)
from shared.testing.assertions import (
    assert_true,
    assert_close_float,
    assert_equal_float,
)


fn test_zero_gradient_float32() raises:
    """Test gradient zeroing with float32."""
    # Create tensor with non-zero values
    var grad = full([100], 42.0, DType.float32)

    # Zero the gradient
    zero_gradient_inplace(grad)

    # Verify all zeros
    for i in range(100):
        var val = grad._get_float64(i)
        assert_equal_float(
            Float32(val), Float32(0.0), "Zeroed gradient should be exactly 0.0"
        )


fn test_zero_gradient_float16() raises:
    """Test gradient zeroing with float16."""
    # Create tensor with non-zero values
    var grad = full([50], 3.14, DType.float16)

    # Zero the gradient
    zero_gradient_inplace(grad)

    # Verify all zeros
    for i in range(50):
        var val = grad._get_float64(i)
        assert_equal_float(
            Float32(val), Float32(0.0), "Zeroed gradient should be exactly 0.0"
        )


fn test_zero_gradient_large_tensor() raises:
    """Test gradient zeroing with larger tensor (vectorization test)."""
    # Create large tensor with non-zero values
    var grad = full([1000], 123.456, DType.float32)

    # Zero the gradient
    zero_gradient_inplace(grad)

    # Verify all zeros
    for i in range(1000):
        var val = grad._get_float64(i)
        assert_equal_float(
            Float32(val), Float32(0.0), "Zeroed gradient should be exactly 0.0"
        )


fn test_gradient_ops_workflow() raises:
    """Test complete gradient accumulation workflow."""
    # Simulate gradient accumulation over 4 mini-batches
    var batch_size = 4
    var accumulated = zeros([200], DType.float32)

    # Accumulate gradients from 4 mini-batches
    for batch in range(batch_size):
        var batch_grad = full([200], Float64(batch + 1), DType.float32)
        accumulate_gradient_inplace(accumulated, batch_grad)

    # accumulated now contains: 1 + 2 + 3 + 4 = 10.0

    # Average over batch size
    scale_gradient_inplace(accumulated, 1.0 / Float32(batch_size))

    # Verify averaged gradient
    for i in range(200):
        var val = accumulated._get_float64(i)
        assert_close_float(
            val,
            2.5,
            atol=1e-5,
            message="Averaged gradient should equal (1+2+3+4)/4 = 2.5",
        )

    # Zero for next iteration
    zero_gradient_inplace(accumulated)

    # Verify zeroed
    for i in range(200):
        var val = accumulated._get_float64(i)
        assert_equal_float(
            Float32(val), Float32(0.0), "Should be zero after zeroing"
        )


fn main() raises:
    """Run gradient zeroing and workflow tests (Part 2 of 2)."""
    print("Testing Gradient Operations - Part 2 (Zero & Workflow)...")
    print("=" * 70)

    print("\n[1/4] Testing gradient zeroing (float32)...")
    test_zero_gradient_float32()
    print("✓ PASSED")

    print("[2/4] Testing gradient zeroing (float16)...")
    test_zero_gradient_float16()
    print("✓ PASSED")

    print("[3/4] Testing gradient zeroing (large tensor)...")
    test_zero_gradient_large_tensor()
    print("✓ PASSED")

    print("[4/4] Testing complete gradient workflow...")
    test_gradient_ops_workflow()
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("All 4 gradient zero/workflow tests PASSED! ✓")
