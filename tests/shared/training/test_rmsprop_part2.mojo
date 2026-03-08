# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_rmsprop.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for RMSprop optimizer (part 2 of 2).

Tests cover:
- Alpha parameter behavior
- Epsilon preventing division by zero
- Batch parameter updates

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones
from shared.training.optimizers.rmsprop import rmsprop_step, rmsprop_step_simple


# ============================================================================
# RMSprop Advanced Tests
# ============================================================================


fn test_rmsprop_alpha_parameter() raises:
    """Test that alpha parameter controls averaging."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    gradients._data.bitcast[Float32]()[0] = 0.1

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    # High alpha (0.99) - slow adaptation
    var result_high = rmsprop_step(
        params,
        gradients,
        square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf,
    )
    var square_avg_high = result_high[1]

    # Low alpha (0.5) - fast adaptation
    var result_low = rmsprop_step(
        params,
        gradients,
        square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.5,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf,
    )
    var square_avg_low = result_low[1]

    # Low alpha should result in larger square_avg update
    # alpha=0.99: 0.99 * 0.0 + 0.01 * 0.01 = 0.0001
    # alpha=0.5: 0.5 * 0.0 + 0.5 * 0.01 = 0.005
    assert_true(
        square_avg_low._data.bitcast[Float32]()[0]
        > square_avg_high._data.bitcast[Float32]()[0]
    )


fn test_rmsprop_epsilon_prevents_division_by_zero() raises:
    """Test that epsilon prevents division by zero."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)  # Zero square_avg
    var buf = zeros(shape, DType.float32)

    # This should not crash despite zero square_avg
    var result_eps = rmsprop_step(
        params,
        gradients,
        square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf,
    )
    var new_params = result_eps[0]

    # Result should be finite
    var val = new_params._data.bitcast[Float32]()[0]
    assert_true(val == val)  # Not NaN
    assert_true(val > -1e10 and val < 1e10)  # Not infinite


fn test_rmsprop_batch_update() raises:
    """Test rmsprop with batch of parameters."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(5)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)

    # Set different gradient values (non-zero to ensure parameter updates)
    for i in range(50):
        gradients._data.bitcast[Float32]()[i] = Float32(i + 1) * 0.01

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var result_batch = rmsprop_step(
        params,
        gradients,
        square_avg,
        t=1,
        learning_rate=0.01,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf,
    )
    var new_params = result_batch[0]
    var new_square_avg = result_batch[1]

    # All parameters should have been updated
    var all_different = True
    for i in range(50):
        if (
            new_params._data.bitcast[Float32]()[i]
            == params._data.bitcast[Float32]()[i]
        ):
            all_different = False
            break

    assert_true(all_different)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run RMSprop optimizer tests (part 2)."""
    print("Running RMSprop optimizer tests (part 2)...")

    test_rmsprop_alpha_parameter()
    print("✓ test_rmsprop_alpha_parameter")

    test_rmsprop_epsilon_prevents_division_by_zero()
    print("✓ test_rmsprop_epsilon_prevents_division_by_zero")

    test_rmsprop_batch_update()
    print("✓ test_rmsprop_batch_update")

    print("\nAll RMSprop optimizer tests (part 2) passed!")
