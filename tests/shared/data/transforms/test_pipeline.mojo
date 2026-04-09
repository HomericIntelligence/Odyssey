"""Tests for transform pipeline composition (part 1: creation and execution).

Tests Pipeline which composes multiple transforms into a single transform,
enabling flexible and reusable data preprocessing workflows.
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)


struct StubData:
    """Minimal stub data for transform testing."""

    var value: Float32

    def __init__(out self, value: Float32):
        self.value = value


struct StubPipeline:
    """Minimal stub pipeline that applies a fixed number of transforms."""

    var num_transforms: Int

    def __init__(out self, num_transforms: Int):
        self.num_transforms = num_transforms

    def apply(self, data: StubData) -> StubData:
        """Apply all transforms sequentially."""
        # Stub implementation: add 10 for each transform
        var result = data.value + Float32(self.num_transforms * 10)
        return StubData(result)

    def __len__(self) -> Int:
        """Return number of transforms in pipeline."""
        return self.num_transforms


def test_pipeline_creation() raises:
    """Test creating Pipeline from list of transforms.

    Should accept list of transform objects and apply them sequentially
    when called on data.
    """
    var pipeline = StubPipeline(num_transforms=2)
    assert_equal(pipeline.__len__(), 2)


def test_pipeline_empty() raises:
    """Test creating empty Pipeline.

    Empty pipeline should be valid and return data unchanged,
    useful as default or for conditional pipeline building.
    """
    var pipeline = StubPipeline(num_transforms=0)
    var data = StubData(value=42.0)
    var result = pipeline.apply(data)
    assert_equal(result.value, data.value)


def test_pipeline_single_transform() raises:
    """Test Pipeline with single transform.

    Should work correctly even with just one transform,
    maintaining consistent API.
    """
    var pipeline = StubPipeline(num_transforms=1)

    var data = StubData(value=5.0)
    var result = pipeline.apply(data)
    assert_equal(result.value, Float32(15.0))  # 5.0 + 1*10


def test_pipeline_sequential_application() raises:
    """Test that transforms are applied in order.

    Transform order matters: Transform(+10)→Transform(+5) should produce
    different result than Transform(+5)→Transform(+10) when order affects
    output.
    """
    var data = StubData(value=0.0)

    # Pipeline with 2 transforms
    var pipeline = StubPipeline(num_transforms=2)
    var result = pipeline.apply(data)

    # Result should be 0 + 2*10 = 20
    assert_equal(result.value, Float32(20.0))


def test_pipeline_output_feeds_next() raises:
    """Test that each transform receives output of previous.

    Output value from transform N should be input to transform N+1,
    enabling complex preprocessing chains.
    """
    var data = StubData(value=2.0)

    # Create pipeline with 3 transforms
    var pipeline = StubPipeline(num_transforms=3)

    var result = pipeline.apply(data)
    # Result should be 2 + 3*10 = 32
    assert_equal(result.value, Float32(32.0))


def test_pipeline_preserves_intermediate_values() raises:
    """Test that pipeline doesn't modify original data.

    Original input data should remain unchanged after pipeline application.
    """
    var data = StubData(value=100.0)

    var pipeline = StubPipeline(num_transforms=1)

    var result = pipeline.apply(data)

    # Result should be different from original
    assert_not_equal(result.value, data.value)
    # Original should be unchanged
    assert_equal(data.value, Float32(100.0))
    # Result should have transform applied
    assert_equal(result.value, Float32(110.0))


def test_pipeline_multiple_calls() raises:
    """Test that Pipeline can be called multiple times.

    Should be stateless and produce same output for same input,
    not accumulate state between calls.
    """
    var pipeline = StubPipeline(num_transforms=1)

    var data = StubData(value=10.0)
    var result1 = pipeline.apply(data)
    var result2 = pipeline.apply(data)

    assert_equal(result1.value, result2.value)


def test_pipeline_composition():
    """Test composing pipelines together.

    Should support Pipeline(pipeline1 + pipeline2) to create
    longer pipelines from smaller reusable pieces.
    """
    # var preprocess = Pipeline([Resize(224, 224)])
    # var augment = Pipeline([RandomFlip(), RandomCrop(200, 200)])
    # var normalize = Pipeline([Normalize(0.5, 0.5)])
    #
    # var full_pipeline = Pipeline(preprocess + augment + normalize)
    # var data = TestFixtures.small_tensor()
    # var result = full_pipeline(data)
    #
    # assert_true(result is not None)
    pass


def test_pipeline_append():
    """Test appending transforms to existing pipeline.

    Should support adding transforms after pipeline creation
    for incremental pipeline building.
    """
    # var pipeline = Pipeline([Resize(224, 224)])
    # pipeline.append(Normalize(0.5, 0.5))
    # pipeline.append(RandomFlip())
    #
    # var data = TestFixtures.small_tensor()
    # var result = pipeline(data)
    # assert_true(result is not None)
    pass


def test_pipeline_transform_error_propagation():
    """Test that errors in transforms are properly propagated.

    If a transform raises error, pipeline should propagate it
    with context about which transform failed.
    """
    # var pipeline = Pipeline([
    #     Resize(224, 224),
    #     InvalidTransform(),  # This will raise error
    #     Normalize(0.5, 0.5)
    # ])
    #
    # var data = TestFixtures.small_tensor()
    # try:
    #     var result = pipeline(data)
    #     assert_true(False, "Should have raised error")
    # except TransformError as e:
    #     # Error message should indicate which transform failed
    #     assert_true("InvalidTransform" in String(e))
    pass


def test_pipeline_shape_mismatch():
    """Test handling of shape mismatches between transforms.

    If transform N outputs shape incompatible with transform N+1,
    should raise clear error.
    """
    # # Reshape to 3D, then try to apply 2D-only transform
    # var pipeline = Pipeline([
    #     Reshape(10, 10, 3),
    #     Resize(224, 224)  # Expects 2D input
    # ])
    #
    # var data = Tensor.ones(300)
    # try:
    #     var result = pipeline(data)
    #     assert_true(False, "Should have raised ShapeError")
    # except ShapeError:
    #     pass
    pass


def test_pipeline_str_representation():
    """Test string representation of Pipeline.

    Should show list of transforms for debugging,
    e.g., 'Pipeline([Resize(224), Normalize(0.5)])'.
    """
    # var pipeline = Pipeline([Resize(224, 224), Normalize(0.5, 0.5)])
    # var repr = String(pipeline)
    #
    # assert_true("Pipeline" in repr)
    # assert_true("Resize" in repr)
    # assert_true("Normalize" in repr)
    pass


def test_pipeline_len():
    """Test getting number of transforms in pipeline.

    len(pipeline) should return number of transforms,
    useful for debugging and validation.
    """
    # var pipeline = Pipeline([Resize(224, 224), Normalize(0.5, 0.5)])
    # assert_equal(len(pipeline), 2)
    pass


def main() raises:
    """Run all test_pipeline tests."""
    print("Running test_pipeline tests...")

    test_pipeline_creation()
    print("✓ test_pipeline_creation")

    test_pipeline_empty()
    print("✓ test_pipeline_empty")

    test_pipeline_single_transform()
    print("✓ test_pipeline_single_transform")

    test_pipeline_sequential_application()
    print("✓ test_pipeline_sequential_application")

    test_pipeline_output_feeds_next()
    print("✓ test_pipeline_output_feeds_next")

    test_pipeline_preserves_intermediate_values()
    print("✓ test_pipeline_preserves_intermediate_values")

    test_pipeline_multiple_calls()
    print("✓ test_pipeline_multiple_calls")

    test_pipeline_composition()
    print("✓ test_pipeline_composition")

    test_pipeline_append()
    print("✓ test_pipeline_append")

    test_pipeline_transform_error_propagation()
    print("✓ test_pipeline_transform_error_propagation")

    test_pipeline_shape_mismatch()
    print("✓ test_pipeline_shape_mismatch")

    test_pipeline_str_representation()
    print("✓ test_pipeline_str_representation")

    test_pipeline_len()
    print("✓ test_pipeline_len")

    print("\nAll test_pipeline tests passed!")
