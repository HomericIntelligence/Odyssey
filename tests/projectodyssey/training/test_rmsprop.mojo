"""Tests for RMSprop optimizer (part 1 of 2).

Tests cover:
- Basic parameter updates
- Running average of squared gradients
- Momentum support
- Weight decay support
- Simplified version without momentum
- Numerical correctness

All tests use pure functional API.
"""


from tests.projectodyssey.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_shape_equal,
    TestFixtures,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.training.optimizers.rmsprop import (
    rmsprop_step,
    rmsprop_step_simple,
)


def test_rmsprop_step_shapes() raises:
    """Test that rmsprop_step returns correct shapes."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var result = rmsprop_step(
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
    var new_params = result[0]
    var new_square_avg = result[1]
    _ = result[2]

    # Check shapes
    assert_equal(new_params.shape()[0], 4)
    assert_equal(new_params.shape()[1], 10)
    assert_equal(new_square_avg.shape()[0], 4)
    assert_equal(new_square_avg.shape()[1], 10)


def test_rmsprop_simple_shapes() raises:
    """Test that rmsprop_step_simple returns correct shapes."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)

    var result2 = rmsprop_step_simple(
        params,
        gradients,
        square_avg,
        learning_rate=0.01,
        alpha=0.99,
        epsilon=1e-8,
    )
    var new_params = result2[0]
    var new_square_avg = result2[1]

    # Check shapes
    assert_equal(new_params.shape()[0], 4)
    assert_equal(new_params.shape()[1], 10)
    assert_equal(new_square_avg.shape()[0], 4)
    assert_equal(new_square_avg.shape()[1], 10)


def test_rmsprop_step_parameter_update() raises:
    """Test that rmsprop_step updates parameters correctly."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var gradients = ones(shape, DType.float32)
    gradients.set(0, Float32(0.1))

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var result3 = rmsprop_step(
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
    var new_params = result3[0]
    var new_square_avg = result3[1]

    # First step:
    # square_avg = 0.9 * 0.0 + 0.1 * (0.1)^2 = 0.001
    # normalized_grad = 0.1 / (sqrt(0.001) + 1e-8) ≈ 0.1 / 0.0316 ≈ 3.16
    # new_params = 1.0 - 0.1 * 3.16 = 1.0 - 0.316 = 0.684

    assert_true(
        new_params._data.bitcast[Float32]()[0] < 1.0
    )  # Parameter should decrease
    assert_almost_equal(
        new_params._data.bitcast[Float32]()[0], Float32(0.684), tolerance=0.01
    )

    # Check that square_avg was updated
    assert_almost_equal(
        new_square_avg._data.bitcast[Float32]()[0],
        Float32(0.001),
        tolerance=1e-5,
    )


def test_rmsprop_simple_parameter_update() raises:
    """Test that rmsprop_step_simple updates parameters correctly."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var gradients = ones(shape, DType.float32)
    gradients.set(0, Float32(0.1))

    var square_avg = zeros(shape, DType.float32)

    var result2 = rmsprop_step_simple(
        params,
        gradients,
        square_avg,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
    )
    var new_params = result2[0]
    var new_square_avg = result2[1]

    # Should produce same result as rmsprop_step with momentum=0.0
    assert_true(new_params._data.bitcast[Float32]()[0] < 1.0)
    assert_almost_equal(
        new_params._data.bitcast[Float32]()[0], Float32(0.684), tolerance=0.01
    )


def test_rmsprop_square_avg_accumulation() raises:
    """Test that square_avg accumulates correctly over multiple steps."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    gradients.set(0, Float32(0.1))

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    # Step 1
    var result1 = rmsprop_step(
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
    var params1 = result1[0]
    var square_avg1 = result1[1]
    var buf1 = result1[2]

    # square_avg after step 1: 0.9 * 0.0 + 0.1 * 0.01 = 0.001

    # Step 2
    var result2 = rmsprop_step(
        params1,
        gradients,
        square_avg1,
        t=2,
        learning_rate=0.01,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        buf=buf1,
    )
    var params2 = result2[0]
    var square_avg2 = result2[1]
    var buf2 = result2[2]

    # square_avg after step 2: 0.9 * 0.001 + 0.1 * 0.01 = 0.0009 + 0.001 = 0.0019

    assert_almost_equal(
        square_avg2._data.bitcast[Float32]()[0], Float32(0.0019), tolerance=1e-5
    )

    # Square avg should be increasing
    assert_true(
        square_avg2._data.bitcast[Float32]()[0]
        > square_avg1._data.bitcast[Float32]()[0]
    )


def test_rmsprop_with_momentum() raises:
    """Test rmsprop with momentum."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var gradients = ones(shape, DType.float32)
    gradients.set(0, Float32(0.1))

    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    # Step 1 with momentum
    var result_step1 = rmsprop_step(
        params,
        gradients,
        square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.9,
        buf=buf,
    )
    var params1 = result_step1[0]
    var square_avg1 = result_step1[1]
    var buf1 = result_step1[2]

    # buf should now contain momentum-weighted gradient
    assert_true(buf1._data.bitcast[Float32]()[0] != 0.0)

    # Step 2 with momentum
    var result_step2 = rmsprop_step(
        params1,
        gradients,
        square_avg1,
        t=2,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.0,
        momentum=0.9,
        buf=buf1,
    )
    var params2 = result_step2[0]
    var square_avg2 = result_step2[1]
    var buf2 = result_step2[2]

    # With momentum, buf accumulates and parameter updates should be larger
    assert_true(
        buf2._data.bitcast[Float32]()[0] > buf1._data.bitcast[Float32]()[0]
    )


def test_rmsprop_with_weight_decay() raises:
    """Test rmsprop with weight decay."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    params.set(0, Float32(1.0))

    var gradients = zeros(shape, DType.float32)  # Zero gradient
    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var result_decay = rmsprop_step(
        params,
        gradients,
        square_avg,
        t=1,
        learning_rate=0.1,
        alpha=0.9,
        epsilon=1e-8,
        weight_decay=0.01,
        momentum=0.0,
        buf=buf,
    )
    var new_params = result_decay[0]

    # With weight decay, parameters should decrease even with zero gradient
    # grad_with_decay = grad + weight_decay * params = 0.0 + 0.01 * 1.0 = 0.01
    assert_true(new_params._data.bitcast[Float32]()[0] < 1.0)


def test_rmsprop_zero_gradient() raises:
    """Test that rmsprop handles zero gradients correctly."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    var gradients = zeros(shape, DType.float32)
    var square_avg = zeros(shape, DType.float32)
    var buf = zeros(shape, DType.float32)

    var result_zero_grad = rmsprop_step(
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
    var new_params = result_zero_grad[0]
    var new_square_avg = result_zero_grad[1]

    # With zero gradient and no weight decay, parameters should not change
    assert_almost_equal(
        new_params._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )


def test_rmsprop_alpha_parameter() raises:
    """Test that alpha parameter controls averaging."""
    var shape = List[Int]()
    shape.append(1)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)
    gradients.set(0, Float32(0.1))

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


def test_rmsprop_epsilon_prevents_division_by_zero() raises:
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


def test_rmsprop_batch_update() raises:
    """Test rmsprop with batch of parameters."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(5)

    var params = ones(shape, DType.float32)
    var gradients = ones(shape, DType.float32)

    # Set different gradient values (non-zero to ensure parameter updates)
    for i in range(50):
        gradients.set(i, Float32(Float32(i + 1) * 0.01))

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


def main() raises:
    """Run all test_rmsprop tests."""
    print("Running test_rmsprop tests...")

    test_rmsprop_step_shapes()
    print("✓ test_rmsprop_step_shapes")

    test_rmsprop_simple_shapes()
    print("✓ test_rmsprop_simple_shapes")

    test_rmsprop_step_parameter_update()
    print("✓ test_rmsprop_step_parameter_update")

    test_rmsprop_simple_parameter_update()
    print("✓ test_rmsprop_simple_parameter_update")

    test_rmsprop_square_avg_accumulation()
    print("✓ test_rmsprop_square_avg_accumulation")

    test_rmsprop_with_momentum()
    print("✓ test_rmsprop_with_momentum")

    test_rmsprop_with_weight_decay()
    print("✓ test_rmsprop_with_weight_decay")

    test_rmsprop_zero_gradient()
    print("✓ test_rmsprop_zero_gradient")

    test_rmsprop_alpha_parameter()
    print("✓ test_rmsprop_alpha_parameter")

    test_rmsprop_epsilon_prevents_division_by_zero()
    print("✓ test_rmsprop_epsilon_prevents_division_by_zero")

    test_rmsprop_batch_update()
    print("✓ test_rmsprop_batch_update")

    print("\nAll test_rmsprop tests passed!")
