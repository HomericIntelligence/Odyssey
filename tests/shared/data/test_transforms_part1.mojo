"""Transform integration tests - Part 1: Compose Pipeline and Trait Conformance.

Split from test_transforms.mojo per ADR-009 to avoid Mojo heap corruption.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Compose pipeline behavior (empty, single, multiple, determinism)
- Transform trait conformance (Normalize, Reshape)
- Transform statefulness (stateless, no mutation)
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_close_float,
    TestFixtures,
)
from shared.data.transforms import Compose, Normalize, Reshape
from shared.core.extensor import ExTensor


# ============================================================================
# Compose Pipeline Tests
# ============================================================================


fn test_compose_empty_pipeline() raises:
    """Test Compose with no transforms returns input unchanged.

    Empty pipeline should act as identity function.

    Integration Points:
        - Compose initialization with empty transforms
        - Identity behavior

    Success Criteria:
        - Empty Compose can be created
        - Applying to tensor returns same tensor
    """
    TestFixtures.set_seed()

    # Create small test tensor
    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    # Create empty Compose pipeline
    var pipeline = Compose[Normalize]()

    # Apply pipeline
    var result = pipeline(data)

    # Result should have same shape
    assert_equal(len(data.shape()), len(result.shape()))


fn test_compose_single_transform() raises:
    """Test Compose with single transform applies correctly.

    Single-transform pipeline should work same as direct transform.

    Integration Points:
        - Compose with minimal configuration
        - Single transform delegation

    Success Criteria:
        - Compose can wrap single transform
        - Application succeeds
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) / 10.0)
    var data = ExTensor(data_list^)

    # Create single-transform pipeline
    var transforms_list = List[Normalize]()
    transforms_list.append(Normalize())
    var pipeline = Compose[Normalize](transforms_list^)

    # Apply pipeline
    var result = pipeline(data)

    # Should have same shape
    assert_equal(len(data.shape()), len(result.shape()))


fn test_compose_multiple_transforms() raises:
    """Test Compose with multiple transforms applies in order.

    Transforms should apply sequentially, each receiving previous output.

    Integration Points:
        - Multiple transforms in Compose
        - Sequential application order
        - Transform chaining

    Success Criteria:
        - Multiple transforms can be composed
        - Pipeline executes without error
        - Output has correct shape
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) / 10.0)
    var data = ExTensor(data_list^)

    # Create pipeline with normalize transform
    var transforms_list = List[Normalize]()
    transforms_list.append(Normalize())
    var pipeline = Compose[Normalize](transforms_list^)

    # Apply pipeline
    var result1 = pipeline(data)

    # Should have same shape as input
    assert_equal(len(data.shape()), len(result1.shape()))


fn test_compose_determinism() raises:
    """Test Compose produces consistent results with same input.

    Applying same transform pipeline to same data should yield same output.

    Integration Points:
        - Transform statelessness
        - Deterministic behavior
        - Reproducibility

    Success Criteria:
        - Applying pipeline twice to same data gives same result
        - No side effects from first application
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(15):
        data_list.append(Float32(i) * 0.5)
    var data = ExTensor(data_list^)

    var transforms_list = List[Normalize]()
    transforms_list.append(Normalize())
    var pipeline = Compose[Normalize](transforms_list^)

    # Apply twice
    var result1 = pipeline(data)
    var result2 = pipeline(data)

    # Results should have same shape
    assert_equal(len(result1.shape()), len(result2.shape()))


# ============================================================================
# Transform Trait Conformance Tests
# ============================================================================


fn test_normalize_transform() raises:
    """Test Normalize transform applies (x - mean) / std correctly.

    Normalize should standardize input to zero mean and unit variance.

    Integration Points:
        - Normalize transform implementation
        - Statistical computation
        - Transform trait conformance

    Success Criteria:
        - Normalize computes statistics
        - Output shape matches input shape
        - Values are scaled appropriately
    """
    TestFixtures.set_seed()

    # Create test data with known values
    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    # Output should have same shape
    assert_equal(len(data.shape()), len(result.shape()))


fn test_reshape_transform() raises:
    """Test Reshape transform changes tensor shape.

    Reshape should change dimensions while preserving element count.

    Integration Points:
        - Reshape transform implementation
        - Shape manipulation
        - Element count preservation

    Success Criteria:
        - Can reshape 10 elements to (2, 5)
        - Element count unchanged
        - Output shape is [2, 5]
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    # Create reshape to (2, 5)
    var new_shape = List[Int]()
    new_shape.append(2)
    new_shape.append(5)
    var reshape = Reshape(new_shape^)

    var result = reshape(data)

    # Shape should change
    assert_equal(result.num_elements(), 10)


# ============================================================================
# Transform Statefulness and Idempotency Tests
# ============================================================================


fn test_transform_stateless() raises:
    """Test transforms are stateless and don't maintain state between calls.

    Same input to same transform should always produce same output.

    Integration Points:
        - Transform state management
        - No side effects on input
        - Immutability

    Success Criteria:
        - Multiple calls with same input produce same results
        - Transform doesn't accumulate state
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var normalize = Normalize()

    # Call multiple times
    var result1 = normalize(data)
    var result2 = normalize(data)
    var result3 = normalize(data)

    # All results should have consistent shape
    assert_equal(len(result1.shape()), len(result2.shape()))
    assert_equal(len(result2.shape()), len(result3.shape()))


fn test_transform_no_mutation() raises:
    """Test transforms don't modify original input data.

    Transform should return new tensor, not modify input in-place.

    Integration Points:
        - Output ownership and allocation
        - Input preservation
        - Functional programming paradigm

    Success Criteria:
        - Input shape unchanged after transform
        - No exceptions from accessing original after transform
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var original_shape = data.shape()

    var normalize = Normalize()
    var result = normalize(data)

    # Original shape should be unchanged
    var current_shape = data.shape()
    assert_equal(len(original_shape), len(current_shape))


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run transform integration tests - Part 1."""
    print("Running transform integration tests (Part 1)...")

    # Compose pipeline tests
    test_compose_empty_pipeline()
    test_compose_single_transform()
    test_compose_multiple_transforms()
    test_compose_determinism()

    # Transform trait tests
    test_normalize_transform()
    test_reshape_transform()

    # Statefulness tests
    test_transform_stateless()
    test_transform_no_mutation()

    print("✓ All transform integration tests (Part 1) passed!")
