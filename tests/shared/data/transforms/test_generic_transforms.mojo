"""Tests for generic data transformation utilities.

Tests identity transforms and lambda transforms.
"""


from shared.tensor.any_tensor import AnyTensor
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_false,
    assert_almost_equal,
    assert_greater,
    assert_less,
    TestFixtures,
)
from shared.data.transforms import Transform
from shared.data.generic_transforms import (
    IdentityTransform,
    LambdaTransform,
    ConditionalTransform,
    ClampTransform,
    DebugTransform,
    BatchTransform,
    ToFloat32,
    ToInt32,
    SequentialTransform,
    AnyTransform,
)


def test_identity_basic() raises:
    """Test identity transform returns input unchanged."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)
    var identity = IdentityTransform()

    var result = identity(data)

    assert_equal(result.num_elements(), 3)
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


def test_identity_preserves_values() raises:
    """Test identity preserves all values exactly."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(-1.5)
    values.append(42.0)
    values.append(100.5)
    var data = AnyTensor(values^)
    var identity = IdentityTransform()

    var result = identity(data)

    for i in range(data.num_elements()):
        assert_almost_equal(result[i], data[i])


def test_identity_empty_tensor() raises:
    """Test identity handles empty tensor."""
    var values = List[Float32]()
    var data = AnyTensor(values^)
    var identity = IdentityTransform()

    var result = identity(data)

    assert_equal(result.num_elements(), 0)


def test_lambda_double_values() raises:
    """Test lambda transform doubles all values."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var transform = LambdaTransform[double_fn]()
    var result = transform(data)

    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 4.0)
    assert_almost_equal(result[2], 6.0)


def test_lambda_add_constant() raises:
    """Test lambda transform adds constant."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    def add_ten(value: Float32) -> Float32:
        return value + 10.0

    var transform = LambdaTransform[add_ten]()
    var result = transform(data)

    assert_almost_equal(result[0], 11.0)
    assert_almost_equal(result[1], 12.0)
    assert_almost_equal(result[2], 13.0)


def test_lambda_square_values() raises:
    """Test lambda transform squares values."""
    var values = List[Float32]()
    values.append(2.0)
    values.append(3.0)
    values.append(4.0)
    var data = AnyTensor(values^)

    def square(value: Float32) -> Float32:
        return value * value

    var transform = LambdaTransform[square]()
    var result = transform(data)

    assert_almost_equal(result[0], 4.0)
    assert_almost_equal(result[1], 9.0)
    assert_almost_equal(result[2], 16.0)


def test_lambda_negative_values() raises:
    """Test lambda transform with negative values."""
    var values = List[Float32]()
    values.append(-1.0)
    values.append(-2.0)
    values.append(-3.0)
    var data = AnyTensor(values^)

    def abs_value(value: Float32) -> Float32:
        return abs(value)

    var transform = LambdaTransform[abs_value]()
    var result = transform(data)

    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


def test_conditional_always_apply() raises:
    """Test conditional transform with always-true predicate."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    def always_true(tensor: AnyTensor) raises -> Bool:
        return True

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform[double_fn]()
    var conditional = ConditionalTransform[
        always_true, LambdaTransform[double_fn]
    ](base_transform^)

    var result = conditional(data)

    # Should apply transform
    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 4.0)
    assert_almost_equal(result[2], 6.0)


def test_conditional_never_apply() raises:
    """Test conditional transform with always-false predicate."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    def always_false(tensor: AnyTensor) raises -> Bool:
        return False

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform[double_fn]()
    var conditional = ConditionalTransform[
        always_false, LambdaTransform[double_fn]
    ](base_transform^)

    var result = conditional(data)

    # Should NOT apply transform (return original)
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


def test_conditional_based_on_size() raises:
    """Test conditional transform based on tensor size."""
    var small_values = List[Float32]()
    small_values.append(1.0)
    small_values.append(2.0)
    var small_data = AnyTensor(small_values^)
    var large_values = List[Float32]()
    large_values.append(1.0)
    large_values.append(2.0)
    large_values.append(3.0)
    large_values.append(4.0)
    var large_data = AnyTensor(large_values^)

    def is_large(tensor: AnyTensor) raises -> Bool:
        return tensor.num_elements() > 3

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform[double_fn]()
    var conditional = ConditionalTransform[
        is_large, LambdaTransform[double_fn]
    ](base_transform^)

    var result_small = conditional(small_data)
    var result_large = conditional(large_data)

    # Small should not be transformed
    assert_almost_equal(result_small[0], 1.0)
    assert_almost_equal(result_small[1], 2.0)

    # Large should be transformed
    assert_almost_equal(result_large[0], 2.0)
    assert_almost_equal(result_large[1], 4.0)


def test_conditional_based_on_values() raises:
    """Test conditional transform based on tensor values."""
    var positive_values = List[Float32]()
    positive_values.append(1.0)
    positive_values.append(2.0)
    positive_values.append(3.0)
    var positive_data = AnyTensor(positive_values^)
    var mixed_values = List[Float32]()
    mixed_values.append(-1.0)
    mixed_values.append(2.0)
    mixed_values.append(3.0)
    var mixed_data = AnyTensor(mixed_values^)

    def all_positive(tensor: AnyTensor) raises -> Bool:
        for i in range(tensor.num_elements()):
            if tensor[i] < 0.0:
                return False
        return True

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform[double_fn]()
    var conditional = ConditionalTransform[
        all_positive, LambdaTransform[double_fn]
    ](base_transform^)

    var result_positive = conditional(positive_data)
    var result_mixed = conditional(mixed_data)

    # Positive should be transformed
    assert_almost_equal(result_positive[0], 2.0)

    # Mixed should not be transformed
    assert_almost_equal(result_mixed[0], -1.0)


def test_clamp_basic() raises:
    """Test clamp limits values to range."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(0.5)
    values.append(1.0)
    values.append(1.5)
    values.append(2.0)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(0.3, 1.2)

    var result = clamp(data)

    assert_almost_equal(result[0], 0.3)  # Clamped to min
    assert_almost_equal(result[1], 0.5)  # Unchanged
    assert_almost_equal(result[2], 1.0)  # Unchanged
    assert_almost_equal(result[3], 1.2)  # Clamped to max
    assert_almost_equal(result[4], 1.2)  # Clamped to max


def test_clamp_all_below_min() raises:
    """Test clamp when all values below minimum."""
    var values = List[Float32]()
    values.append(-5.0)
    values.append(-2.0)
    values.append(0.0)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(1.0, 10.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 1.0)
    assert_almost_equal(result[2], 1.0)


def test_clamp_all_above_max() raises:
    """Test clamp when all values above maximum."""
    var values = List[Float32]()
    values.append(15.0)
    values.append(20.0)
    values.append(25.0)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(1.0, 10.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 10.0)
    assert_almost_equal(result[1], 10.0)
    assert_almost_equal(result[2], 10.0)


def test_clamp_all_in_range() raises:
    """Test clamp when all values already in range."""
    var values = List[Float32]()
    values.append(2.0)
    values.append(5.0)
    values.append(8.0)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(1.0, 10.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 5.0)
    assert_almost_equal(result[2], 8.0)


def test_clamp_negative_range() raises:
    """Test clamp with negative range."""
    var values = List[Float32]()
    values.append(-10.0)
    values.append(-5.0)
    values.append(0.0)
    values.append(5.0)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(-8.0, -2.0)

    var result = clamp(data)

    assert_almost_equal(result[0], -8.0)  # Clamped to min
    assert_almost_equal(result[1], -5.0)  # Unchanged
    assert_almost_equal(result[2], -2.0)  # Clamped to max
    assert_almost_equal(result[3], -2.0)  # Clamped to max


def test_clamp_zero_crossing() raises:
    """Test clamp range crossing zero."""
    var values = List[Float32]()
    values.append(-5.0)
    values.append(-1.0)
    values.append(0.0)
    values.append(1.0)
    values.append(5.0)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(-2.0, 2.0)

    var result = clamp(data)

    assert_almost_equal(result[0], -2.0)
    assert_almost_equal(result[1], -1.0)
    assert_almost_equal(result[2], 0.0)
    assert_almost_equal(result[3], 1.0)
    assert_almost_equal(result[4], 2.0)


def test_debug_passthrough() raises:
    """Test debug transform passes data through unchanged."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)
    var debug = DebugTransform("test")

    var result = debug(data)

    # Should pass through unchanged
    assert_equal(result.num_elements(), 3)
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


def test_debug_with_empty_tensor() raises:
    """Test debug transform with empty tensor."""
    var values = List[Float32]()
    var data = AnyTensor(values^)
    var debug = DebugTransform("empty_test")

    var result = debug(data)

    assert_equal(result.num_elements(), 0)


def test_debug_with_large_tensor() raises:
    """Test debug transform with large tensor."""
    var values = List[Float32]()
    for i in range(100):
        values.append(Float32(i))

    var data = AnyTensor(values^)
    var debug = DebugTransform("large_test")

    var result = debug(data)

    # Should pass through unchanged
    assert_equal(result.num_elements(), 100)
    for i in range(100):
        var expected = Float32(i)
        assert_almost_equal(Float32(result[i]), expected)


def test_to_float32_preserves_values() raises:
    """Test ToFloat32 preserves float values."""
    var values = List[Float32]()
    values.append(1.5)
    values.append(2.5)
    values.append(3.5)
    var data = AnyTensor(values^)
    var converter = ToFloat32()

    var result = converter(data)

    assert_almost_equal(result[0], 1.5)
    assert_almost_equal(result[1], 2.5)
    assert_almost_equal(result[2], 3.5)


def test_to_int32_truncates() raises:
    """Test ToInt32 truncates float values."""
    var values = List[Float32]()
    values.append(1.9)
    values.append(2.5)
    values.append(3.1)
    var data = AnyTensor(values^)
    var converter = ToInt32()

    var result = converter(data)

    assert_equal(Int(result[0]), 1)
    assert_equal(Int(result[1]), 2)
    assert_equal(Int(result[2]), 3)


def test_to_int32_negative() raises:
    """Test ToInt32 handles negative values."""
    var values = List[Float32]()
    values.append(-1.9)
    values.append(-2.5)
    values.append(-3.1)
    var data = AnyTensor(values^)
    var converter = ToInt32()

    var result = converter(data)

    assert_equal(Int(result[0]), -1)
    assert_equal(Int(result[1]), -2)
    assert_equal(Int(result[2]), -3)


def test_to_int32_zero() raises:
    """Test ToInt32 handles zero."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(0.1)
    values.append(-0.1)
    var data = AnyTensor(values^)
    var converter = ToInt32()

    var result = converter(data)

    assert_equal(Int(result[0]), 0)
    assert_equal(Int(result[1]), 0)
    assert_equal(Int(result[2]), 0)


def test_sequential_basic() raises:
    """Test sequential application of transforms."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    def add_one(value: Float32) -> Float32:
        return value + 1.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform.from_lambda[double_fn]())
    transforms.append(AnyTransform.from_lambda[add_one]())

    var sequential = SequentialTransform(transforms^)
    var result = sequential(data)

    # Should apply double then add_one: (1*2)+1=3, (2*2)+1=5, (3*2)+1=7
    assert_almost_equal(result[0], 3.0)
    assert_almost_equal(result[1], 5.0)
    assert_almost_equal(result[2], 7.0)


def test_sequential_single_transform() raises:
    """Test sequential with single transform."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform.from_lambda[double_fn]())

    var sequential = SequentialTransform(transforms^)
    var result = sequential(data)

    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 4.0)
    assert_almost_equal(result[2], 6.0)


def test_sequential_empty() raises:
    """Test sequential with no transforms."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    var transforms: List[AnyTransform] = []
    var sequential = SequentialTransform(transforms^)

    var result = sequential(data)

    # Should act like identity
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


def test_sequential_with_clamp() raises:
    """Test sequential including clamp transform."""
    var values = List[Float32]()
    values.append(0.5)
    values.append(1.5)
    values.append(2.5)
    var data = AnyTensor(values^)

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var transforms: List[AnyTransform] = []
    transforms.append(
        AnyTransform.from_lambda[double_fn]()
    )  # Double: 1.0, 3.0, 5.0
    transforms.append(AnyTransform(ClampTransform(0.0, 4.0)))  # Clamp to [0, 4]

    var sequential = SequentialTransform(transforms^)
    var result = sequential(data)

    assert_almost_equal(result[0], 1.0)  # 0.5*2 = 1.0
    assert_almost_equal(result[1], 3.0)  # 1.5*2 = 3.0
    assert_almost_equal(result[2], 4.0)  # 2.5*2 = 5.0, clamped to 4.0


def test_sequential_deterministic() raises:
    """Test sequential produces same result on repeated calls."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = AnyTensor(values^)

    def triple(value: Float32) -> Float32:
        return value * 3.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform.from_lambda[triple]())
    transforms.append(AnyTransform(ClampTransform(0.0, 5.0)))

    var sequential = SequentialTransform(transforms^)

    var result1 = sequential(data)
    var result2 = sequential(data)

    # Should be identical
    for i in range(result1.num_elements()):
        assert_almost_equal(result1[i], result2[i])


def test_batch_transform_basic() raises:
    """Test batch transform applies to multiple tensors."""
    var values1 = List[Float32]()
    values1.append(1.0)
    values1.append(2.0)
    var values2 = List[Float32]()
    values2.append(3.0)
    values2.append(4.0)
    var values3 = List[Float32]()
    values3.append(5.0)
    values3.append(6.0)

    var tensors: List[AnyTensor] = []
    tensors.append(AnyTensor(values1^))
    tensors.append(AnyTensor(values2^))
    tensors.append(AnyTensor(values3^))

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var batch = BatchTransform(AnyTransform.from_lambda[double_fn]())

    var results = batch(tensors)

    # Check first tensor
    assert_almost_equal(results[0][0], 2.0)
    assert_almost_equal(results[0][1], 4.0)

    # Check second tensor
    assert_almost_equal(results[1][0], 6.0)
    assert_almost_equal(results[1][1], 8.0)

    # Check third tensor
    assert_almost_equal(results[2][0], 10.0)
    assert_almost_equal(results[2][1], 12.0)


def test_batch_transform_empty_list() raises:
    """Test batch transform with empty list."""
    var tensors: List[AnyTensor] = []

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var batch = BatchTransform(AnyTransform.from_lambda[double_fn]())

    var results = batch(tensors)

    assert_equal(len(results), 0)


def test_batch_transform_single_tensor() raises:
    """Test batch transform with single tensor."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)

    var tensors: List[AnyTensor] = []
    tensors.append(AnyTensor(values^))

    def add_ten(value: Float32) -> Float32:
        return value + 10.0

    var batch = BatchTransform(AnyTransform.from_lambda[add_ten]())

    var results = batch(tensors)

    assert_equal(len(results), 1)
    assert_almost_equal(results[0][0], 11.0)
    assert_almost_equal(results[0][1], 12.0)
    assert_almost_equal(results[0][2], 13.0)


def test_batch_transform_different_sizes() raises:
    """Test batch transform with different sized tensors."""
    var values1 = List[Float32]()
    values1.append(1.0)
    var values2 = List[Float32]()
    values2.append(2.0)
    values2.append(3.0)
    var values3 = List[Float32]()
    values3.append(4.0)
    values3.append(5.0)
    values3.append(6.0)

    var tensors: List[AnyTensor] = []
    tensors.append(AnyTensor(values1^))
    tensors.append(AnyTensor(values2^))
    tensors.append(AnyTensor(values3^))

    def double_fn(value: Float32) -> Float32:
        return value * 2.0

    var batch = BatchTransform(AnyTransform.from_lambda[double_fn]())

    var results = batch(tensors)

    # First tensor (size 1)
    assert_equal(results[0].num_elements(), 1)
    assert_almost_equal(results[0][0], 2.0)

    # Second tensor (size 2)
    assert_equal(results[1].num_elements(), 2)
    assert_almost_equal(results[1][0], 4.0)
    assert_almost_equal(results[1][1], 6.0)

    # Third tensor (size 3)
    assert_equal(results[2].num_elements(), 3)
    assert_almost_equal(results[2][0], 8.0)
    assert_almost_equal(results[2][1], 10.0)
    assert_almost_equal(results[2][2], 12.0)


def test_batch_transform_with_clamp() raises:
    """Test batch transform with clamp."""
    var values1 = List[Float32]()
    values1.append(0.0)
    values1.append(5.0)
    values1.append(10.0)
    var values2 = List[Float32]()
    values2.append(15.0)
    values2.append(20.0)
    values2.append(25.0)

    var tensors: List[AnyTensor] = []
    tensors.append(AnyTensor(values1^))
    tensors.append(AnyTensor(values2^))

    var base_transform = ClampTransform(2.0, 18.0)
    var batch = BatchTransform(AnyTransform(base_transform^))

    var results = batch(tensors)

    # First tensor
    assert_almost_equal(results[0][0], 2.0)  # Clamped
    assert_almost_equal(results[0][1], 5.0)  # Unchanged
    assert_almost_equal(results[0][2], 10.0)  # Unchanged

    # Second tensor
    assert_almost_equal(results[1][0], 15.0)  # Unchanged
    assert_almost_equal(results[1][1], 18.0)  # Clamped
    assert_almost_equal(results[1][2], 18.0)  # Clamped


def test_integration_preprocessing_pipeline() raises:
    """Test typical preprocessing pipeline."""
    var values = List[Float32]()
    values.append(-5.0)
    values.append(0.0)
    values.append(5.0)
    values.append(10.0)
    values.append(15.0)
    var data = AnyTensor(values^)

    def normalize(value: Float32) -> Float32:
        # Scale to [0, 1]
        return (value + 5.0) / 20.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform.from_lambda[normalize]())  # Normalize
    transforms.append(AnyTransform(ClampTransform(0.0, 1.0)))  # Ensure bounds
    transforms.append(AnyTransform(DebugTransform("pipeline")))  # Debug

    var pipeline = SequentialTransform(transforms^)
    var result = pipeline(data)

    # Check values are properly normalized and clamped
    assert_almost_equal(result[0], 0.0)
    assert_almost_equal(result[1], 0.25)
    assert_almost_equal(result[2], 0.5)
    assert_almost_equal(result[3], 0.75)
    assert_almost_equal(result[4], 1.0)


def test_integration_conditional_augmentation() raises:
    """Test conditional augmentation pipeline."""
    var large_values = List[Float32]()
    large_values.append(1.0)
    large_values.append(2.0)
    large_values.append(3.0)
    large_values.append(4.0)
    var large_data = AnyTensor(large_values^)
    var small_values = List[Float32]()
    small_values.append(1.0)
    small_values.append(2.0)
    var small_data = AnyTensor(small_values^)

    def is_large_enough(tensor: AnyTensor) raises -> Bool:
        return tensor.num_elements() >= 3

    def augment(value: Float32) -> Float32:
        return value * 1.5

    var base_transform = LambdaTransform[augment]()
    var conditional = ConditionalTransform[
        is_large_enough, LambdaTransform[augment]
    ](base_transform^)

    var result_large = conditional(large_data)
    var result_small = conditional(small_data)

    # Large should be augmented
    assert_almost_equal(result_large[0], 1.5)
    assert_almost_equal(result_large[1], 3.0)

    # Small should NOT be augmented
    assert_almost_equal(result_small[0], 1.0)
    assert_almost_equal(result_small[1], 2.0)


def test_integration_batch_preprocessing() raises:
    """Test batch preprocessing pipeline."""
    var values1 = List[Float32]()
    values1.append(100.0)
    values1.append(200.0)
    var values2 = List[Float32]()
    values2.append(300.0)
    values2.append(400.0)

    var batch: List[AnyTensor] = []
    batch.append(AnyTensor(values1^))
    batch.append(AnyTensor(values2^))

    def scale_down(value: Float32) -> Float32:
        return value / 100.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform.from_lambda[scale_down]())
    transforms.append(AnyTransform(ClampTransform(0.0, 5.0)))

    var pipeline = SequentialTransform(transforms^)
    var batch_transform = BatchTransform(AnyTransform(pipeline^))

    var results = batch_transform(batch)

    # First batch item
    assert_almost_equal(results[0][0], 1.0)
    assert_almost_equal(results[0][1], 2.0)

    # Second batch item
    assert_almost_equal(results[1][0], 3.0)
    assert_almost_equal(results[1][1], 4.0)


def test_integration_type_conversion_pipeline() raises:
    """Test type conversion in pipeline."""
    var values = List[Float32]()
    values.append(1.9)
    values.append(2.5)
    values.append(3.1)
    var data = AnyTensor(values^)

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(ToInt32()))  # Convert to int (truncate)
    transforms.append(AnyTransform(ToFloat32()))  # Convert back to float
    transforms.append(AnyTransform(ClampTransform(0.0, 3.0)))  # Clamp result

    var pipeline = SequentialTransform(transforms^)
    var result = pipeline(data)

    # Should truncate then clamp
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


def test_edge_case_very_large_values() raises:
    """Test transforms with very large values."""
    var values = List[Float32]()
    values.append(1e6)
    values.append(1e7)
    values.append(1e8)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(0.0, 1e9)

    var result = clamp(data)

    assert_almost_equal(result[0], 1e6)
    assert_almost_equal(result[1], 1e7)
    assert_almost_equal(result[2], 1e8)


def test_edge_case_very_small_values() raises:
    """Test transforms with very small values."""
    var values = List[Float32]()
    values.append(1e-6)
    values.append(1e-7)
    values.append(1e-8)
    var data = AnyTensor(values^)
    var clamp = ClampTransform(1e-9, 1.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 1e-6, tolerance=1e-9)
    assert_almost_equal(result[1], 1e-7, tolerance=1e-9)
    assert_almost_equal(result[2], 1e-8, tolerance=1e-9)


def test_edge_case_all_zeros() raises:
    """Test transforms with all zeros."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(0.0)
    values.append(0.0)
    var data = AnyTensor(values^)

    def add_one(value: Float32) -> Float32:
        return value + 1.0

    var transform = LambdaTransform[add_one]()
    var result = transform(data)

    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 1.0)
    assert_almost_equal(result[2], 1.0)


def test_edge_case_single_element() raises:
    """Test transforms with single element."""
    var values = List[Float32]()
    values.append(42.0)
    var data = AnyTensor(values^)

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(ClampTransform(0.0, 100.0)))
    transforms.append(AnyTransform(DebugTransform("single")))

    var pipeline = SequentialTransform(transforms^)
    var result = pipeline(data)

    assert_equal(result.num_elements(), 1)
    assert_almost_equal(result[0], 42.0)


def main() raises:
    """Run all test_generic_transforms tests."""
    print("Running test_generic_transforms tests...")

    test_identity_basic()
    print("✓ test_identity_basic")

    test_identity_preserves_values()
    print("✓ test_identity_preserves_values")

    test_identity_empty_tensor()
    print("✓ test_identity_empty_tensor")

    test_lambda_double_values()
    print("✓ test_lambda_double_values")

    test_lambda_add_constant()
    print("✓ test_lambda_add_constant")

    test_lambda_square_values()
    print("✓ test_lambda_square_values")

    test_lambda_negative_values()
    print("✓ test_lambda_negative_values")

    test_conditional_always_apply()
    print("✓ test_conditional_always_apply")

    test_conditional_never_apply()
    print("✓ test_conditional_never_apply")

    test_conditional_based_on_size()
    print("✓ test_conditional_based_on_size")

    test_conditional_based_on_values()
    print("✓ test_conditional_based_on_values")

    test_clamp_basic()
    print("✓ test_clamp_basic")

    test_clamp_all_below_min()
    print("✓ test_clamp_all_below_min")

    test_clamp_all_above_max()
    print("✓ test_clamp_all_above_max")

    test_clamp_all_in_range()
    print("✓ test_clamp_all_in_range")

    test_clamp_negative_range()
    print("✓ test_clamp_negative_range")

    test_clamp_zero_crossing()
    print("✓ test_clamp_zero_crossing")

    test_debug_passthrough()
    print("✓ test_debug_passthrough")

    test_debug_with_empty_tensor()
    print("✓ test_debug_with_empty_tensor")

    test_debug_with_large_tensor()
    print("✓ test_debug_with_large_tensor")

    test_to_float32_preserves_values()
    print("✓ test_to_float32_preserves_values")

    test_to_int32_truncates()
    print("✓ test_to_int32_truncates")

    test_to_int32_negative()
    print("✓ test_to_int32_negative")

    test_to_int32_zero()
    print("✓ test_to_int32_zero")

    test_sequential_basic()
    print("✓ test_sequential_basic")

    test_sequential_single_transform()
    print("✓ test_sequential_single_transform")

    test_sequential_empty()
    print("✓ test_sequential_empty")

    test_sequential_with_clamp()
    print("✓ test_sequential_with_clamp")

    test_sequential_deterministic()
    print("✓ test_sequential_deterministic")

    test_batch_transform_basic()
    print("✓ test_batch_transform_basic")

    test_batch_transform_empty_list()
    print("✓ test_batch_transform_empty_list")

    test_batch_transform_single_tensor()
    print("✓ test_batch_transform_single_tensor")

    test_batch_transform_different_sizes()
    print("✓ test_batch_transform_different_sizes")

    test_batch_transform_with_clamp()
    print("✓ test_batch_transform_with_clamp")

    test_integration_preprocessing_pipeline()
    print("✓ test_integration_preprocessing_pipeline")

    test_integration_conditional_augmentation()
    print("✓ test_integration_conditional_augmentation")

    test_integration_batch_preprocessing()
    print("✓ test_integration_batch_preprocessing")

    test_integration_type_conversion_pipeline()
    print("✓ test_integration_type_conversion_pipeline")

    test_edge_case_very_large_values()
    print("✓ test_edge_case_very_large_values")

    test_edge_case_very_small_values()
    print("✓ test_edge_case_very_small_values")

    test_edge_case_all_zeros()
    print("✓ test_edge_case_all_zeros")

    test_edge_case_single_element()
    print("✓ test_edge_case_single_element")

    print("\nAll test_generic_transforms tests passed!")
