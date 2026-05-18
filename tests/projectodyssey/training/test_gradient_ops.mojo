"""Tests for gradient accumulation and scaling operations.

Verifies correctness and performance of in-place gradient accumulation
and scaling operations.

Test Coverage:
- accumulate_gradient_inplace: Correctness across dtypes, shapes
- scale_gradient_inplace: Correctness for scaling factors

All tests use small tensors for fast runtime.
"""


from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, full
from projectodyssey.training.gradient_ops import (
    accumulate_gradient_inplace,
    scale_gradient_inplace,
    zero_gradient_inplace,
)
from projectodyssey.testing.assertions import (
    assert_true,
    assert_close_float,
    assert_equal_float,
)


def test_accumulate_gradient_float32() raises:
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


def test_accumulate_gradient_float16() raises:
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


def test_accumulate_gradient_large_tensor() raises:
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


def test_accumulate_gradient_mismatched_shapes_fails() raises:
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


def test_accumulate_gradient_mismatched_dtypes_fails() raises:
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


def test_scale_gradient_float32() raises:
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


def test_scale_gradient_averaging() raises:
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


def test_scale_gradient_large_tensor() raises:
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


def test_zero_gradient_float32() raises:
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


def test_zero_gradient_float16() raises:
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


def test_zero_gradient_large_tensor() raises:
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


def test_gradient_ops_workflow() raises:
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


def main() raises:
    """Run all test_gradient_ops tests."""
    print("Running test_gradient_ops tests...")

    test_accumulate_gradient_float32()
    print("✓ test_accumulate_gradient_float32")

    test_accumulate_gradient_float16()
    print("✓ test_accumulate_gradient_float16")

    test_accumulate_gradient_large_tensor()
    print("✓ test_accumulate_gradient_large_tensor")

    test_accumulate_gradient_mismatched_shapes_fails()
    print("✓ test_accumulate_gradient_mismatched_shapes_fails")

    test_accumulate_gradient_mismatched_dtypes_fails()
    print("✓ test_accumulate_gradient_mismatched_dtypes_fails")

    test_scale_gradient_float32()
    print("✓ test_scale_gradient_float32")

    test_scale_gradient_averaging()
    print("✓ test_scale_gradient_averaging")

    test_scale_gradient_large_tensor()
    print("✓ test_scale_gradient_large_tensor")

    test_zero_gradient_float32()
    print("✓ test_zero_gradient_float32")

    test_zero_gradient_float16()
    print("✓ test_zero_gradient_float16")

    test_zero_gradient_large_tensor()
    print("✓ test_zero_gradient_large_tensor")

    test_gradient_ops_workflow()
    print("✓ test_gradient_ops_workflow")

    print("\nAll test_gradient_ops tests passed!")
