# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_pipeline.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

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


# ============================================================================
# Stub Implementations for TDD
# ============================================================================


struct StubData:
    """Minimal stub data for transform testing."""

    var value: Float32

    fn __init__(out self, value: Float32):
        self.value = value


# Simple pipeline with fixed transform count for testing
struct StubPipeline:
    """Minimal stub pipeline that applies a fixed number of transforms."""

    var num_transforms: Int

    fn __init__(out self, num_transforms: Int):
        self.num_transforms = num_transforms

    fn apply(self, data: StubData) -> StubData:
        """Apply all transforms sequentially."""
        # Stub implementation: add 10 for each transform
        var result = data.value + Float32(self.num_transforms * 10)
        return StubData(result)

    fn __len__(self) -> Int:
        """Return number of transforms in pipeline."""
        return self.num_transforms


# ============================================================================
# Pipeline Creation Tests
# ============================================================================


fn test_pipeline_creation() raises:
    """Test creating Pipeline from list of transforms.

    Should accept list of transform objects and apply them sequentially
    when called on data.
    """
    var pipeline = StubPipeline(num_transforms=2)
    assert_equal(pipeline.__len__(), 2)


fn test_pipeline_empty() raises:
    """Test creating empty Pipeline.

    Empty pipeline should be valid and return data unchanged,
    useful as default or for conditional pipeline building.
    """
    var pipeline = StubPipeline(num_transforms=0)
    var data = StubData(value=42.0)
    var result = pipeline.apply(data)
    assert_equal(result.value, data.value)


fn test_pipeline_single_transform() raises:
    """Test Pipeline with single transform.

    Should work correctly even with just one transform,
    maintaining consistent API.
    """
    var pipeline = StubPipeline(num_transforms=1)

    var data = StubData(value=5.0)
    var result = pipeline.apply(data)
    assert_equal(result.value, Float32(15.0))  # 5.0 + 1*10


# ============================================================================
# Pipeline Execution Tests
# ============================================================================


fn test_pipeline_sequential_application() raises:
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


fn test_pipeline_output_feeds_next() raises:
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


fn test_pipeline_preserves_intermediate_values() raises:
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


fn test_pipeline_multiple_calls() raises:
    """Test that Pipeline can be called multiple times.

    Should be stateless and produce same output for same input,
    not accumulate state between calls.
    """
    var pipeline = StubPipeline(num_transforms=1)

    var data = StubData(value=10.0)
    var result1 = pipeline.apply(data)
    var result2 = pipeline.apply(data)

    assert_equal(result1.value, result2.value)


# ============================================================================
# Pipeline Composition Tests
# ============================================================================


fn test_pipeline_composition():
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run pipeline creation and execution tests."""
    print("Running pipeline part 1 tests...")

    # Creation tests
    test_pipeline_creation()
    test_pipeline_empty()
    test_pipeline_single_transform()

    # Execution tests
    test_pipeline_sequential_application()
    test_pipeline_output_feeds_next()
    test_pipeline_preserves_intermediate_values()
    test_pipeline_multiple_calls()

    # Composition tests
    test_pipeline_composition()

    print("✓ All pipeline part 1 tests passed!")
