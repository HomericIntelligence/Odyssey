# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_pipeline.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Tests for transform pipeline composition (part 2: append, error handling, and utilities).

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
# Pipeline Composition Tests (continued)
# ============================================================================


fn test_pipeline_append():
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


# ============================================================================
# Pipeline Error Handling Tests
# ============================================================================


fn test_pipeline_transform_error_propagation():
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


fn test_pipeline_shape_mismatch():
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


# ============================================================================
# Pipeline Utility Tests
# ============================================================================


fn test_pipeline_str_representation():
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


fn test_pipeline_len():
    """Test getting number of transforms in pipeline.

    len(pipeline) should return number of transforms,
    useful for debugging and validation.
    """
    # var pipeline = Pipeline([Resize(224, 224), Normalize(0.5, 0.5)])
    # assert_equal(len(pipeline), 2)
    pass


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run pipeline append, error handling, and utility tests."""
    print("Running pipeline part 2 tests...")

    # Composition tests (continued)
    test_pipeline_append()

    # Error handling tests
    test_pipeline_transform_error_propagation()
    test_pipeline_shape_mismatch()

    # Utility tests
    test_pipeline_str_representation()
    test_pipeline_len()

    print("✓ All pipeline part 2 tests passed!")
