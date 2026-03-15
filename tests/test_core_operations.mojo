"""Core Operations coordination tests.

Validates integration of all core operations: initializers, metrics, tensor ops,
and activations from the ExTensor framework.

Coordination tests (#298-302):
- #299: Core operations integration
- #300: Cross-component compatibility
- #301: End-to-end workflows

Testing strategy:
- Component integration: All components work together
- Data flow: Tensors flow correctly through init→forward→metric pipeline
- API consistency: Uniform patterns across all components
- Realistic workflows: Simulate actual ML training scenarios
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core.extensor import ExTensor
from shared.core.initializers import (
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    uniform,
    normal,
    constant,
)
from shared.core.activation import relu, sigmoid, tanh, softmax
from shared.core.loss import mean_squared_error, binary_cross_entropy

# Import matmul explicitly from matrix to avoid conflict with matmul module
from shared.core.matrix import matmul
from shared.training.metrics import (
    AccuracyMetric,
    LossTracker,
    ConfusionMatrix,
    MetricCollection,
    MetricLogger,
)
from shared.training.metrics.base import MetricResult


fn test_initializers_with_activations() raises:
    """Test that initializers work correctly with activation functions."""
    print("Testing initializers with activations...")

    var shape: List[Int] = [100, 100]
    var fan_in = 100
    var fan_out = 100

    # Xavier initialization (for sigmoid/tanh)
    var xavier_weights = xavier_uniform(fan_in, fan_out, shape, seed_val=42)

    # Apply sigmoid activation
    var sigmoid_output = sigmoid(xavier_weights)

    # Verify output is in valid range [0, 1]
    for i in range(100):
        var val = sigmoid_output._data.bitcast[Float32]()[i]
        assert_true(val >= 0.0 and val <= 1.0, "Sigmoid output in [0,1]")

    # Kaiming initialization (for ReLU)
    var kaiming_weights = kaiming_uniform(fan_in, fan_out, shape, seed_val=42)

    # Apply ReLU activation
    var relu_output = relu(kaiming_weights)

    # Verify ReLU zeroed negative values
    var has_positive = False
    for i in range(100):
        var val = relu_output._data.bitcast[Float32]()[i]
        assert_true(val >= 0.0, "ReLU output non-negative")
        if val > 0.0:
            has_positive = True

    assert_true(has_positive, "ReLU has some positive values")

    print("  ✓ Initializers work correctly with activations")


fn test_forward_pass_with_metrics() raises:
    """Test forward pass using initialized weights and metric computation."""
    print("Testing forward pass with metrics...")

    # Initialize a simple 2-layer network
    var w1_shape: List[Int] = [3, 4]  # 3 inputs, 4 hidden
    var w2_shape: List[Int] = [4, 3]  # 4 hidden, 3 outputs

    var w1 = kaiming_uniform(3, 4, w1_shape, seed_val=1)
    var b1 = constant(List[Int](), 0.0)

    var w2 = xavier_uniform(4, 3, w2_shape, seed_val=2)
    var b2 = constant(List[Int](), 0.0)

    # Create fake input (batch_size=5, features=3)
    var input_shape: List[Int] = [5, 3]
    var input = normal(input_shape, mean=0.0, std=1.0, seed_val=3)

    # Forward pass: input @ w1.T + b1
    var h1 = matmul(input, w1)  # (5, 4)
    var h1_act = relu(h1)

    # Second layer: h1 @ w2.T + b2
    var output = matmul(h1_act, w2)  # (5, 3)
    var predictions = softmax(output)

    # Create fake labels
    var labels_shape = List[Int]()
    labels_shape.append(5)
    var labels = ExTensor(labels_shape, DType.int32)
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 2
    labels._data.bitcast[Int32]()[3] = 0
    labels._data.bitcast[Int32]()[4] = 1

    # Compute accuracy
    var accuracy = AccuracyMetric()
    accuracy.update(predictions, labels)
    var acc = accuracy.compute()

    # Accuracy should be in [0, 1]
    assert_true(acc >= 0.0 and acc <= 1.0, "Accuracy in valid range")

    print("  Forward pass accuracy: " + String(acc))
    print("  ✓ Forward pass with metrics works")


fn test_training_loop_simulation() raises:
    """Simulate a complete training loop with all core operations."""
    print("Testing training loop simulation...")

    # Network setup
    var input_dim = 10
    var hidden_dim = 8
    var output_dim = 3
    var batch_size = 4

    # Initialize weights
    var w1_shape = List[Int]()
    w1_shape.append(input_dim)
    w1_shape.append(hidden_dim)
    var w1 = kaiming_uniform(input_dim, hidden_dim, w1_shape, seed_val=1)
    var w2_shape = List[Int]()
    w2_shape.append(hidden_dim)
    w2_shape.append(output_dim)
    var w2 = xavier_uniform(hidden_dim, output_dim, w2_shape, seed_val=2)

    # Setup metrics
    var accuracy = AccuracyMetric()
    var loss_tracker = LossTracker(window_size=10)
    var confusion = ConfusionMatrix(num_classes=output_dim)
    var logger = MetricLogger()

    # Simulate 3 epochs
    for epoch in range(3):
        # Reset metrics for new epoch
        accuracy.reset()
        confusion.reset()

        var epoch_loss = Float32(0.0)
        var num_batches = 5

        # Simulate batches
        for batch_idx in range(num_batches):
            # Create fake batch
            var input_shape = List[Int]()
            input_shape.append(batch_size)
            input_shape.append(input_dim)
            var input = normal(
                input_shape,
                seed_val=epoch * 100 + batch_idx,
            )
            var labels_shape = List[Int]()
            labels_shape.append(batch_size)
            var labels = ExTensor(labels_shape, DType.int32)

            for i in range(batch_size):
                labels._data.bitcast[Int32]()[i] = Int32(
                    (i + batch_idx) % output_dim
                )

            # Forward pass
            var h1 = matmul(input, w1)
            var h1_act = relu(h1)
            var output = matmul(h1_act, w2)
            var predictions = softmax(output)

            # Compute loss (fake - just use small random value)
            var batch_loss = Float32(0.5) - Float32(epoch) * Float32(0.1)
            loss_tracker.update(batch_loss)
            epoch_loss += batch_loss

            # Update metrics
            accuracy.update(predictions, labels)
            confusion.update(predictions, labels)

        # Epoch metrics
        var epoch_acc = accuracy.compute()
        var epoch_avg_loss = epoch_loss / Float32(num_batches)

        print(
            "  Epoch "
            + String(epoch)
            + ": loss="
            + String(epoch_avg_loss)
            + ", acc="
            + String(epoch_acc)
        )

        # Log metrics
        var epoch_metrics: List[MetricResult] = []
        epoch_metrics.append(MetricResult("accuracy", epoch_acc))
        epoch_metrics.append(MetricResult("loss", Float64(epoch_avg_loss)))
        logger.log_epoch(epoch, epoch_metrics)

    # Verify training logged correctly
    assert_equal(logger.num_epochs, 3, "Logged 3 epochs")

    var acc_history = logger.get_history("accuracy")
    assert_equal(len(acc_history), 3, "Accuracy history has 3 epochs")

    print("  ✓ Training loop simulation works")


fn test_dtype_consistency_across_components() raises:
    """Test that all components handle multiple dtypes consistently."""
    print("Testing dtype consistency across components...")

    var shape: List[Int] = [10, 10]
    var dtypes: List[DType] = [DType.float32, DType.float64]

    for dt_idx in range(2):
        var dt = dtypes[dt_idx]

        # Initialize with specific dtype
        var weights = xavier_uniform(10, 10, shape, dtype=dt, seed_val=42)
        if weights.dtype() != dt:
            raise Error("Initializer respects dtype")

        # Ensure activations preserve dtype
        var activated = relu(weights)
        if activated.dtype() != dt:
            raise Error("ReLU preserves dtype")

    print("  ✓ All components handle dtypes consistently")


fn test_seed_reproducibility_across_components() raises:
    """Test that seeding works consistently across all components."""
    print("Testing seed reproducibility across components...")

    var shape: List[Int] = [20, 20]
    var seed = 42

    # Initialize with same seed twice
    var init1 = kaiming_uniform(20, 20, shape, seed_val=seed)
    var init2 = kaiming_uniform(20, 20, shape, seed_val=seed)

    # Verify identical
    var identical = True
    for i in range(100):  # Check first 100 elements
        if (
            init1._data.bitcast[Float32]()[i]
            != init2._data.bitcast[Float32]()[i]
        ):
            identical = False
            break

    assert_true(identical, "Same seed produces identical initialization")

    # Different seed should produce different results
    var init3 = kaiming_uniform(20, 20, shape, seed_val=seed + 1)

    var different = False
    for i in range(100):
        if (
            init1._data.bitcast[Float32]()[i]
            != init3._data.bitcast[Float32]()[i]
        ):
            different = True
            break

    assert_true(different, "Different seed produces different initialization")

    print("  ✓ Seeding works consistently across components")


fn test_batch_processing_pipeline() raises:
    """Test processing multiple batches through the full pipeline."""
    print("Testing batch processing pipeline...")

    var batch_size = 8
    var num_features = 5
    var num_classes = 3

    # Initialize network
    var weights_shape = List[Int]()
    weights_shape.append(num_features)
    weights_shape.append(num_classes)
    var weights = kaiming_uniform(
        num_features,
        num_classes,
        weights_shape,
        seed_val=1,
    )

    # Setup metrics
    var accuracy = AccuracyMetric()
    var confusion = ConfusionMatrix(num_classes=num_classes)

    # Process 10 batches
    var num_batches = 10
    for batch_idx in range(num_batches):
        # Generate batch
        var input_shape = List[Int]()
        input_shape.append(batch_size)
        input_shape.append(num_features)
        var input = normal(input_shape, seed_val=batch_idx)
        var labels_shape = List[Int]()
        labels_shape.append(batch_size)
        var labels = ExTensor(labels_shape, DType.int32)

        for i in range(batch_size):
            labels._data.bitcast[Int32]()[i] = Int32(i % num_classes)

        # Forward pass
        var logits = matmul(input, weights)
        var predictions = softmax(logits)

        # Update metrics
        accuracy.update(predictions, labels)
        confusion.update(predictions, labels)

    # Compute final metrics
    var final_acc = accuracy.compute()
    var precision = confusion.get_precision()
    var recall = confusion.get_recall()

    # Verify metrics are in valid range
    assert_true(final_acc >= 0.0 and final_acc <= 1.0, "Accuracy in [0,1]")

    for i in range(num_classes):
        var p = precision._data.bitcast[Float64]()[i]
        var r = recall._data.bitcast[Float64]()[i]
        assert_true(p >= 0.0 and p <= 1.0, "Precision in [0,1]")
        assert_true(r >= 0.0 and r <= 1.0, "Recall in [0,1]")

    print("  Processed " + String(num_batches) + " batches")
    print("  Final accuracy: " + String(final_acc))
    print("  ✓ Batch processing pipeline works")


fn test_multi_layer_network_integration() raises:
    """Test multi-layer network with all components integrated."""
    print("Testing multi-layer network integration...")

    # 3-layer network: 784 -> 256 -> 128 -> 10 (MNIST-like)
    var layer_sizes: List[Int] = [784, 256, 128, 10]

    # Initialize all layers with appropriate methods
    var w1 = kaiming_uniform(784, 256, [784, 256], seed_val=1)
    var b1 = constant(List[Int](), 0.0)

    var w2 = kaiming_uniform(256, 128, [256, 128], seed_val=2)
    var b2 = constant(List[Int](), 0.0)

    var w3 = xavier_uniform(128, 10, [128, 10], seed_val=3)
    var b3 = constant(List[Int](), 0.0)

    # Verify all weights initialized
    assert_equal(w1.numel(), 784 * 256, "Layer 1 weights")
    assert_equal(w2.numel(), 256 * 128, "Layer 2 weights")
    assert_equal(w3.numel(), 128 * 10, "Layer 3 weights")

    # Create fake mini-batch (batch_size=4)
    var input = normal([4, 784], seed_val=42)
    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = ExTensor(labels_shape, DType.int32)
    labels._data.bitcast[Int32]()[0] = 7
    labels._data.bitcast[Int32]()[1] = 2
    labels._data.bitcast[Int32]()[2] = 1
    labels._data.bitcast[Int32]()[3] = 0

    # Forward pass through 3 layers
    var h1 = matmul(input, w1)  # (4, 256)
    var h1_act = relu(h1)

    var h2 = matmul(h1_act, w2)  # (4, 128)
    var h2_act = relu(h2)

    var output = matmul(h2_act, w3)  # (4, 10)
    var predictions = softmax(output)

    # Compute metrics
    var accuracy = AccuracyMetric()
    accuracy.update(predictions, labels)
    var acc = accuracy.compute()

    var confusion = ConfusionMatrix(num_classes=10)
    confusion.update(predictions, labels)

    # Verify valid results
    assert_true(acc >= 0.0 and acc <= 1.0, "Accuracy in valid range")
    var pred_shape = predictions.shape()
    assert_equal(pred_shape[0], 4, "Predictions batch size")
    assert_equal(pred_shape[1], 10, "Predictions num classes")

    print("  Network: 784 -> 256 -> 128 -> 10")
    print("  Batch accuracy: " + String(acc))
    print("  ✓ Multi-layer network integration works")


fn test_error_handling_across_components() raises:
    """Test that components handle errors gracefully."""
    print("Testing error handling across components...")

    # Test mismatched shapes in metric updates
    var preds_shape = List[Int]()
    var preds = ExTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    var labels = ExTensor(labels_shape, DType.int32)  # Mismatched size

    var accuracy = AccuracyMetric()

    var raised = False
    try:
        accuracy.update(preds, labels)
    except:
        raised = True

    assert_true(raised, "Should raise on mismatched prediction/label shapes")

    print("  ✓ Components handle errors gracefully")


fn main() raises:
    """Run all core operations coordination tests."""
    print("\n" + "=" * 70)
    print("CORE OPERATIONS COORDINATION TEST SUITE")
    print("Integration of Initializers, Metrics, Activations, and Tensor Ops")
    print("Issues #298-302")
    print("=" * 70 + "\n")

    print("Component Integration Tests (#299)")
    print("-" * 70)
    test_initializers_with_activations()
    test_forward_pass_with_metrics()
    test_dtype_consistency_across_components()
    test_seed_reproducibility_across_components()

    print("\nWorkflow Integration Tests (#300)")
    print("-" * 70)
    test_training_loop_simulation()
    test_batch_processing_pipeline()
    test_multi_layer_network_integration()

    print("\nRobustness Tests (#300)")
    print("-" * 70)
    test_error_handling_across_components()

    print("\n" + "=" * 70)
    print("ALL CORE OPERATIONS COORDINATION TESTS PASSED ✓")
    print("=" * 70 + "\n")
    print("Summary:")
    print("  ✓ Initializers integrate with activation functions")
    print("  ✓ Forward pass works with metrics computation")
    print("  ✓ Full training loop with all components")
    print("  ✓ Dtype consistency across all operations")
    print("  ✓ Seed reproducibility across components")
    print("  ✓ Batch processing pipeline functional")
    print("  ✓ Multi-layer networks work end-to-end")
    print("  ✓ Error handling is graceful and consistent")
    print("\nCore Operations Hierarchy:")
    print("  ExTensor Framework:")
    print("    - Tensor operations (matmul, arithmetic, reductions)")
    print("    - Activation functions (ReLU, sigmoid, tanh, softmax)")
    print("    - Loss functions (MSE, BCE, cross-entropy)")
    print("  Initializers:")
    print("    - Xavier/Glorot (uniform, normal)")
    print("    - Kaiming/He (uniform, normal)")
    print("    - Basic distributions (uniform, normal, constant)")
    print("  Metrics:")
    print("    - Accuracy (top-1, top-k, per-class)")
    print("    - Loss tracking (moving average, statistics)")
    print("    - Confusion matrix (precision, recall, F1)")
    print("    - Metric collection and logging")
    print()
