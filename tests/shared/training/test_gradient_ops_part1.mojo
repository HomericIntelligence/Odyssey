# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_ops.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for gradient accumulation and scaling operations (Part 1 of 2).

Verifies correctness and performance of in-place gradient accumulation
and scaling operations.

Test Coverage:
- accumulate_gradient_inplace: Correctness across dtypes, shapes
- scale_gradient_inplace: Correctness for scaling factors

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


fn test_accumulate_gradient_float32() raises:
    """Test gradient accumulation with float32."""
    # Create tensors
    var accumulated = zeros([100], DType.float32)
    var grad1 = ones([100], DType.float32)
    var grad2 = full([100], 2.0, DType.float32)

    # Accumulate gradients
    accumulate_gradient_inplace(accumulated, grad1)  # accumulated = 1.0
    accumulate_gradient_inplace(accumulated, grad2)  # accumulated = 3.0

    # Verify results
    for i in range(100):
        var val = accumulated._get_float64(i)
        assert_close_float(
            val, 3.0, atol=1e-6, message="Accumulation should equal 1.0 + 2.0"
        )


fn test_accumulate_gradient_float16() raises:
    """Test gradient accumulation with float16."""
    # Create tensors
    var accumulated = zeros([50], DType.float16)
    var grad1 = ones([50], DType.float16)
    var grad2 = ones([50], DType.float16)

    # Accumulate gradients
    accumulate_gradient_inplace(accumulated, grad1)  # accumulated = 1.0
    accumulate_gradient_inplace(accumulated, grad2)  # accumulated = 2.0

    # Verify results
    for i in range(50):
        var val = accumulated._get_float64(i)
        assert_close_float(
            val, 2.0, atol=1e-3, message="FP16 accumulation should equal 2.0"
        )


fn test_accumulate_gradient_large_tensor() raises:
    """Test gradient accumulation with larger tensor (vectorization test)."""
    # Create large tensors to exercise vectorized path
    var accumulated = zeros([1000], DType.float32)
    var grad = ones([1000], DType.float32)

    # Accumulate multiple times
    for _ in range(10):
        accumulate_gradient_inplace(accumulated, grad)

    # Verify results
    for i in range(1000):
        var val = accumulated._get_float64(i)
        assert_close_float(
            val,
            10.0,
            atol=1e-5,
            message="Should accumulate to 10.0 after 10 iterations",
        )


fn test_accumulate_gradient_mismatched_shapes_fails() raises:
    """Test that mismatched shapes raise an error."""
    var accumulated = zeros([100], DType.float32)
    var grad = ones([50], DType.float32)

    var caught_error = False
    try:
        accumulate_gradient_inplace(accumulated, grad)
    except:
        caught_error = True

    assert_true(
        caught_error, "Accumulation with mismatched shapes should raise error"
    )


fn test_accumulate_gradient_mismatched_dtypes_fails() raises:
    """Test that mismatched dtypes raise an error."""
    var accumulated = zeros([100], DType.float32)
    var grad = ones([100], DType.float16)

    var caught_error = False
    try:
        accumulate_gradient_inplace(accumulated, grad)
    except:
        caught_error = True

    assert_true(
        caught_error, "Accumulation with mismatched dtypes should raise error"
    )


fn test_scale_gradient_float32() raises:
    """Test gradient scaling with float32."""
    # Create tensor with value 10.0
    var grad = full([100], 10.0, DType.float32)

    # Scale by 0.5
    scale_gradient_inplace(grad, 0.5)

    # Verify results
    for i in range(100):
        var val = grad._get_float64(i)
        assert_close_float(
            val, 5.0, atol=1e-6, message="Scaling 10.0 by 0.5 should give 5.0"
        )


fn test_scale_gradient_averaging() raises:
    """Test gradient scaling for mini-batch averaging."""
    # Simulate accumulated gradient from 4 mini-batches
    var grad = full([50], 4.0, DType.float32)

    # Average over 4 mini-batches
    scale_gradient_inplace(grad, 0.25)

    # Verify results
    for i in range(50):
        var val = grad._get_float64(i)
        assert_close_float(
            val,
            1.0,
            atol=1e-6,
            message="Averaging 4.0 over 4 batches should give 1.0",
        )


fn test_scale_gradient_large_tensor() raises:
    """Test gradient scaling with larger tensor (vectorization test)."""
    # Create large tensor
    var grad = full([1000], 100.0, DType.float32)

    # Scale by 0.01
    scale_gradient_inplace(grad, 0.01)

    # Verify results
    for i in range(1000):
        var val = grad._get_float64(i)
        assert_close_float(
            val, 1.0, atol=1e-5, message="Scaling 100.0 by 0.01 should give 1.0"
        )


fn main() raises:
    """Run gradient accumulation and scaling tests (Part 1 of 2)."""
    print("Testing Gradient Operations - Part 1 (Accumulate & Scale)...")
    print("=" * 70)

    print("\n[1/8] Testing gradient accumulation (float32)...")
    test_accumulate_gradient_float32()
    print("✓ PASSED")

    print("[2/8] Testing gradient accumulation (float16)...")
    test_accumulate_gradient_float16()
    print("✓ PASSED")

    print("[3/8] Testing gradient accumulation (large tensor)...")
    test_accumulate_gradient_large_tensor()
    print("✓ PASSED")

    print("[4/8] Testing accumulation with mismatched shapes (error case)...")
    test_accumulate_gradient_mismatched_shapes_fails()
    print("✓ PASSED")

    print("[5/8] Testing accumulation with mismatched dtypes (error case)...")
    test_accumulate_gradient_mismatched_dtypes_fails()
    print("✓ PASSED")

    print("[6/8] Testing gradient scaling (float32)...")
    test_scale_gradient_float32()
    print("✓ PASSED")

    print("[7/8] Testing gradient scaling (averaging)...")
    test_scale_gradient_averaging()
    print("✓ PASSED")

    print("[8/8] Testing gradient scaling (large tensor)...")
    test_scale_gradient_large_tensor()
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("All 8 gradient accumulate/scale tests PASSED! ✓")
