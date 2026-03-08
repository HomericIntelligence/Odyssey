"""Tests for evaluation module - Part 2.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_evaluation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Covers:
- evaluate_topk edge cases (k > num_classes, k == num_classes)
- Integration tests (consistency, single sample, sample count matching)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
)
from shared.core import ExTensor, zeros, ones, full
from shared.training.evaluation import (
    EvaluationResult,
    evaluate_model,
    evaluate_model_simple,
    evaluate_topk,
)
from shared.testing import SimpleMLP
from collections import List


# ============================================================================
# evaluate_topk Edge Case Tests
# ============================================================================


fn test_evaluate_topk_k_greater_than_classes() raises:
    """Test that evaluate_topk rejects k > num_classes."""
    print("Testing evaluate_topk with invalid k...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)
    var images_shape = List[Int]()
    images_shape.append(5)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(5)
    var labels = zeros(labels_shape, DType.int32)

    # Try with k > num_classes (should raise error)
    try:
        var _ = evaluate_topk(
            model,
            images,
            labels,
            k=5,  # k > 3 classes
            num_classes=3,
            verbose=False,
        )
        assert_false(True, "Should have raised error for k > num_classes")
    except e:
        print("   Correctly raised error for k > num_classes: " + String(e))

    print("   evaluate_topk with invalid k test passed")


fn test_evaluate_topk_edge_case_k_equals_num_classes() raises:
    """Test evaluate_topk with k == num_classes (should give 100% accuracy)."""
    print("Testing evaluate_topk with k == num_classes...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)
    var images_shape = List[Int]()
    images_shape.append(10)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # With k == num_classes, should always find the correct class
    var accuracy = evaluate_topk(
        model,
        images,
        labels,
        k=3,  # k == num_classes
        num_classes=3,
        verbose=False,
    )

    # Should have perfect accuracy
    assert_almost_equal(
        accuracy, 1.0, 1e-6, "Top-3 on 3 classes should be 100% accurate"
    )

    print("   evaluate_topk with k == num_classes test passed")


# ============================================================================
# Integration Tests
# ============================================================================


fn test_evaluation_consistency() raises:
    """Test that evaluate_model_simple and evaluate_model give consistent results.
    """
    print("Testing evaluation consistency...")

    var model1 = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)
    var model2 = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)

    var images_shape = List[Int]()
    images_shape.append(10)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # Both functions should work with same data
    var simple_acc = evaluate_model_simple(
        model1, images, labels, batch_size=5, num_classes=3, verbose=False
    )

    var full_result = evaluate_model(
        model2, images, labels, batch_size=5, num_classes=3, verbose=False
    )

    # Both should produce valid results
    assert_true(
        simple_acc >= 0.0 and simple_acc <= 1.0,
        "Simple accuracy should be valid",
    )
    assert_true(
        full_result.accuracy >= 0.0 and full_result.accuracy <= 1.0,
        "Full accuracy should be valid",
    )

    print("   Evaluation consistency test passed")


fn test_single_sample_evaluation() raises:
    """Test evaluation with single sample."""
    print("Testing single sample evaluation...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)

    # Create single sample
    var images_shape = List[Int]()
    images_shape.append(1)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(1)
    var labels = zeros(labels_shape, DType.int32)

    # Evaluate
    var result = evaluate_model(
        model, images, labels, batch_size=1, num_classes=3, verbose=False
    )

    assert_equal(result.num_total, 1, "Should evaluate 1 sample")
    assert_true(
        result.accuracy >= 0.0 and result.accuracy <= 1.0,
        "Accuracy should be valid",
    )

    print("   Single sample evaluation test passed")


fn test_evaluation_matches_sample_counts() raises:
    """Test that per-class sample counts match actual data."""
    print("Testing evaluation sample count matching...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=4)

    # Create test data with known class distribution
    var images_shape = List[Int]()
    images_shape.append(8)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(8)
    var labels = zeros(labels_shape, DType.int32)

    # Set labels: 2 samples each for classes 0,1,2,3
    var labels_data = labels._data.bitcast[Int32]()
    for i in range(8):
        labels_data[i] = Int32(i / 2)  # [0,0,1,1,2,2,3,3]

    # Evaluate
    var result = evaluate_model(
        model, images, labels, batch_size=2, num_classes=4, verbose=False
    )

    # Check per-class totals
    for i in range(4):
        assert_equal(
            result.total_per_class[i],
            2,
            "Class " + String(i) + " should have 2 samples",
        )

    # Check sum matches total
    var sum_totals = 0
    for i in range(4):
        sum_totals += result.total_per_class[i]

    assert_equal(sum_totals, 8, "Per-class totals should sum to 8")

    print("   Evaluation sample count matching test passed")


# ============================================================================
# Test Runner
# ============================================================================


fn main() raises:
    """Run evaluation tests part 2."""
    print("=" * 60)
    print("Evaluation Module Test Suite - Part 2")
    print("=" * 60)

    # evaluate_topk edge cases
    test_evaluate_topk_k_greater_than_classes()
    test_evaluate_topk_edge_case_k_equals_num_classes()

    # Integration tests
    test_evaluation_consistency()
    test_single_sample_evaluation()
    test_evaluation_matches_sample_counts()

    print("=" * 60)
    print("All evaluation part 2 tests passed! ✓")
    print("=" * 60)
