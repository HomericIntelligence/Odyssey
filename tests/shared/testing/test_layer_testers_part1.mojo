# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_layer_testers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for layer_testers module (Part 1 of 2)

Tests LayerTester utility functions:
- test_layer_dtype_consistency
- test_layer_no_invalid_values
- test_activation_layer_backward (relu, sigmoid)

Split from test_layer_testers.mojo per ADR-009 (≤10 fn test_ per file).
See test_layer_testers_part2.mojo for remaining tests.
"""

from math import isnan, isinf
from shared.testing.layer_testers import LayerTester
from shared.testing.special_values import (
    create_ones_tensor,
    create_special_value_tensor,
    create_seeded_random_tensor,
)
from shared.testing.assertions import assert_true
from shared.testing.tensor_factory import zeros_tensor
from shared.core.extensor import ExTensor


fn test_layer_dtype_consistency_passes() raises:
    """Test that dtype consistency check passes when dtypes match."""
    var input = create_ones_tensor([1, 3, 32, 32], DType.float32)
    var output = create_ones_tensor([1, 64, 30, 30], DType.float32)

    # Should not raise - dtypes match
    LayerTester.test_layer_dtype_consistency(input, output, "TestLayer")


fn test_layer_dtype_consistency_different_shapes() raises:
    """Test that dtype consistency works with different shapes."""
    var input = create_ones_tensor([1, 3], DType.float16)
    var output = create_ones_tensor([1, 10], DType.float16)

    # Should not raise - dtypes match even though shapes differ
    LayerTester.test_layer_dtype_consistency(input, output, "TestFC")


fn test_layer_no_invalid_values_passes() raises:
    """Test that invalid value check passes when no NaN/Inf."""
    var output = create_ones_tensor([2, 3, 4, 5], DType.float32)

    # Should not raise - all values are 1.0 (finite)
    LayerTester.test_layer_no_invalid_values(output, "TestLayer")


fn test_layer_no_invalid_values_with_zeros() raises:
    """Test that invalid value check passes with zero values."""
    var output = create_special_value_tensor([3, 3], DType.float32, 0.0)

    # Should not raise - zeros are valid
    LayerTester.test_layer_no_invalid_values(output, "TestLayer")


fn test_layer_no_invalid_values_with_halves() raises:
    """Test that invalid value check passes with 0.5 values."""
    var output = create_special_value_tensor([2, 2], DType.float32, 0.5)

    # Should not raise - 0.5 is valid
    LayerTester.test_layer_no_invalid_values(output, "TestLayer")


fn test_layer_tester_utility_functions() raises:
    """Test that LayerTester utility functions are accessible."""
    # This test verifies that we can call the static methods without errors

    var input = create_ones_tensor([1, 3, 32, 32], DType.float32)
    var output = create_ones_tensor([1, 64, 30, 30], DType.float32)

    # Test dtype consistency
    LayerTester.test_layer_dtype_consistency(input, output, "Conv1")

    # Test no invalid values
    LayerTester.test_layer_no_invalid_values(output, "Conv1")


fn test_activation_layer_backward_relu() raises:
    """Test ReLU activation backward pass with gradient checking."""
    # Use small shape to avoid timeout
    LayerTester.test_activation_layer_backward(
        shape=[2, 3, 4, 4], dtype=DType.float32, activation="relu"
    )


fn test_activation_layer_backward_sigmoid() raises:
    """Test Sigmoid activation backward pass with gradient checking."""
    # Use small shape to avoid timeout
    LayerTester.test_activation_layer_backward(
        shape=[2, 3, 4, 4], dtype=DType.float32, activation="sigmoid"
    )


fn main() raises:
    print("Testing layer_testers module (Part 1)...")

    # Test dtype consistency
    test_layer_dtype_consistency_passes()
    print("✓ test_layer_dtype_consistency_passes")

    test_layer_dtype_consistency_different_shapes()
    print("✓ test_layer_dtype_consistency_different_shapes")

    # Test invalid value checking
    test_layer_no_invalid_values_passes()
    print("✓ test_layer_no_invalid_values_passes")

    test_layer_no_invalid_values_with_zeros()
    print("✓ test_layer_no_invalid_values_with_zeros")

    test_layer_no_invalid_values_with_halves()
    print("✓ test_layer_no_invalid_values_with_halves")

    # Test utility functions
    test_layer_tester_utility_functions()
    print("✓ test_layer_tester_utility_functions")

    # Test backward pass methods
    test_activation_layer_backward_relu()
    print("✓ test_activation_layer_backward_relu")

    test_activation_layer_backward_sigmoid()
    print("✓ test_activation_layer_backward_sigmoid")

    print("\n✅ All layer_testers Part 1 tests passed!")
