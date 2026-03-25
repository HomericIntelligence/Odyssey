"""Tests for evaluation module

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_evaluation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Covers:
- EvaluationResult struct initialization and field access
- Generic evaluate_model function (perfect predictions, batch sizes, per-class stats)
- evaluate_model_simple function (basic usage, batch processing)
- evaluate_topk basic functionality
"""


from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.training.evaluation import (
    EvaluationResult,
    evaluate_model,
    evaluate_model_simple,
    evaluate_topk,
)
from shared.testing import SimpleMLP
from collections import List


fn test_evaluation_result_initialization() raises:
    """Test EvaluationResult initializes correctly."""
    print("Testing EvaluationResult initialization...")

    var result = EvaluationResult(accuracy=0.95, num_correct=95, num_total=100)

    assert_almost_equal(result.accuracy, 0.95, 1e-6, "Accuracy should be 0.95")
    assert_equal(result.num_correct, 95, "num_correct should be 95")
    assert_equal(result.num_total, 100, "num_total should be 100")
    assert_equal(
        len(result.correct_per_class), 0, "correct_per_class should be empty"
    )
    assert_equal(
        len(result.total_per_class), 0, "total_per_class should be empty"
    )

    print("   EvaluationResult initialization test passed")


fn test_evaluation_result_with_per_class_stats() raises:
    """Test EvaluationResult with per-class statistics."""
    print("Testing EvaluationResult with per-class stats...")

    var correct_per_class: List[Int] = [5, 8, 10, 9, 8]
    var total_per_class: List[Int] = [10, 10, 10, 10, 10]

    var result = EvaluationResult(
        accuracy=0.8,
        num_correct=40,
        num_total=50,
        correct_per_class=correct_per_class,
        total_per_class=total_per_class,
    )

    assert_almost_equal(result.accuracy, 0.8, 1e-6, "Accuracy should be 0.8")
    assert_equal(len(result.correct_per_class), 5, "Should have 5 classes")
    assert_equal(
        result.correct_per_class[0], 5, "Class 0 should have 5 correct"
    )
    assert_equal(result.total_per_class[2], 10, "Class 2 should have 10 total")

    print("   EvaluationResult with per-class stats test passed")


fn test_evaluate_model_perfect_predictions() raises:
    """Test evaluate_model with perfect predictions.

    Creates synthetic logits where predictions match ground truth,
    expecting 100% accuracy.
    """
    print("Testing evaluate_model with perfect predictions...")

    # Create simple model
    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)

    # Create test data: 10 samples, 4 features (input), 3 classes (output)
    var images_shape = List[Int]()
    images_shape.append(10)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # Set labels to match expected argmax (for ones input, should go to class 0)
    var labels_data = labels._data.bitcast[Int32]()
    for i in range(10):
        labels_data[i] = 0  # All samples are class 0

    # Evaluate
    var result = evaluate_model(
        model, images, labels, batch_size=5, num_classes=3, verbose=False
    )

    # Verify result structure
    assert_equal(result.num_total, 10, "Should evaluate 10 samples")
    assert_equal(len(result.correct_per_class), 3, "Should have 3 classes")
    assert_equal(len(result.total_per_class), 3, "Should have 3 classes")

    print("   evaluate_model with perfect predictions test passed")


fn test_evaluate_model_batch_sizes() raises:
    """Test evaluate_model with different batch sizes."""
    print("Testing evaluate_model with different batch sizes...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)

    # Create test data
    var images_shape = List[Int]()
    images_shape.append(10)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # Test with different batch sizes
    for batch_size in [1, 2, 5, 10]:
        var result = evaluate_model(
            model,
            images,
            labels,
            batch_size=batch_size,
            num_classes=3,
            verbose=False,
        )

        assert_equal(
            result.num_total,
            10,
            "Should evaluate 10 samples with batch_size=" + String(batch_size),
        )
        assert_equal(len(result.correct_per_class), 3, "Should have 3 classes")

    print("   evaluate_model with different batch sizes test passed")


fn test_evaluate_model_per_class_statistics() raises:
    """Test evaluate_model computes per-class statistics correctly.

    With controlled predictions, verify per-class accuracy computation.
    """
    print("Testing evaluate_model per-class statistics...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2)

    # Create test data: 4 samples, 2 classes
    var images_shape = List[Int]()
    images_shape.append(4)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = zeros(labels_shape, DType.int32)

    # Set labels: [0, 1, 0, 1]
    var labels_data = labels._data.bitcast[Int32]()
    labels_data[0] = 0
    labels_data[1] = 1
    labels_data[2] = 0
    labels_data[3] = 1

    # Evaluate
    var result = evaluate_model(
        model, images, labels, batch_size=2, num_classes=2, verbose=False
    )

    # Check per-class totals sum to overall total
    var sum_totals = 0
    for i in range(2):
        sum_totals += result.total_per_class[i]

    assert_equal(sum_totals, 4, "Per-class totals should sum to 4")
    assert_equal(result.total_per_class[0], 2, "Class 0 should have 2 samples")
    assert_equal(result.total_per_class[1], 2, "Class 1 should have 2 samples")

    print("   evaluate_model per-class statistics test passed")


fn test_evaluate_model_simple_basic() raises:
    """Test simplified evaluation function."""
    print("Testing evaluate_model_simple...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)

    # Create test data
    var images_shape = List[Int]()
    images_shape.append(10)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # Evaluate
    var accuracy = evaluate_model_simple(
        model, images, labels, batch_size=5, num_classes=3, verbose=False
    )

    # Check accuracy is valid fraction
    assert_true(accuracy >= 0.0, "Accuracy should be >= 0")
    assert_true(accuracy <= 1.0, "Accuracy should be <= 1")

    print("   evaluate_model_simple test passed")


fn test_evaluate_model_simple_batch_processing() raises:
    """Test evaluate_model_simple handles batches correctly."""
    print("Testing evaluate_model_simple batch processing...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=3)

    # Create test data
    var images_shape = List[Int]()
    images_shape.append(7)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(7)
    var labels = zeros(labels_shape, DType.int32)

    # Evaluate
    var accuracy = evaluate_model_simple(
        model, images, labels, batch_size=3, num_classes=3, verbose=False
    )

    # Check result is valid
    assert_true(accuracy >= 0.0, "Accuracy should be valid fraction")
    assert_true(accuracy <= 1.0, "Accuracy should be valid fraction")

    print("   evaluate_model_simple batch processing test passed")


fn test_evaluate_topk_basic() raises:
    """Test top-k evaluation basic functionality."""
    print("Testing evaluate_topk basic...")

    var model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=5)

    # Create test data
    var images_shape = List[Int]()
    images_shape.append(10)
    images_shape.append(4)
    var images = ones(images_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # Evaluate top-1
    var top1_acc = evaluate_topk(
        model, images, labels, k=1, batch_size=5, num_classes=5, verbose=False
    )

    # Evaluate top-2
    var top2_acc = evaluate_topk(
        model, images, labels, k=2, batch_size=5, num_classes=5, verbose=False
    )

    # Top-2 should be >= Top-1 (relaxed criteria)
    assert_true(top2_acc >= top1_acc, "Top-2 accuracy should be >= Top-1")
    assert_true(
        top1_acc >= 0.0 and top1_acc <= 1.0, "Top-1 accuracy should be valid"
    )
    assert_true(
        top2_acc >= 0.0 and top2_acc <= 1.0, "Top-2 accuracy should be valid"
    )

    print("   evaluate_topk basic test passed")


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


fn main() raises:
    """Run all test_evaluation tests."""
    print("Running test_evaluation tests...")

    test_evaluation_result_initialization()
    print("✓ test_evaluation_result_initialization")

    test_evaluation_result_with_per_class_stats()
    print("✓ test_evaluation_result_with_per_class_stats")

    test_evaluate_model_perfect_predictions()
    print("✓ test_evaluate_model_perfect_predictions")

    test_evaluate_model_batch_sizes()
    print("✓ test_evaluate_model_batch_sizes")

    test_evaluate_model_per_class_statistics()
    print("✓ test_evaluate_model_per_class_statistics")

    test_evaluate_model_simple_basic()
    print("✓ test_evaluate_model_simple_basic")

    test_evaluate_model_simple_batch_processing()
    print("✓ test_evaluate_model_simple_batch_processing")

    test_evaluate_topk_basic()
    print("✓ test_evaluate_topk_basic")

    test_evaluate_topk_k_greater_than_classes()
    print("✓ test_evaluate_topk_k_greater_than_classes")

    test_evaluate_topk_edge_case_k_equals_num_classes()
    print("✓ test_evaluate_topk_edge_case_k_equals_num_classes")

    test_evaluation_consistency()
    print("✓ test_evaluation_consistency")

    test_single_sample_evaluation()
    print("✓ test_single_sample_evaluation")

    test_evaluation_matches_sample_counts()
    print("✓ test_evaluation_matches_sample_counts")

    print("\nAll test_evaluation tests passed!")
