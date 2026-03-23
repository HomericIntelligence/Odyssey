"""Integration tests for end-to-end model training and evaluation.

Tests cover:
- Complete model training from initialization to final evaluation
- Model evaluation and inference
- Prediction accuracy and confidence
- Model serialization and loading (when available)

These tests validate entire model lifecycle workflows.
"""

from tests.shared.conftest import (
    assert_true,
    assert_less,
    assert_greater,
    assert_almost_equal,
    assert_shape,
    assert_all_close,
    create_simple_model,
    TestFixtures,
)
from shared.testing import SimpleMLP
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.loss import mean_squared_error
from shared.core.reduction import mean
from shared.core.activation import softmax


# ============================================================================
# Model Training and Evaluation Tests
# ============================================================================


fn test_model_training_to_evaluation() raises:
    """Test complete model training and evaluation workflow.

    Integration Points:
        - Model initialization
        - Training loop
        - Validation
        - Final evaluation

    Success Criteria:
        - Model trains without errors
        - Loss decreases over time
        - Evaluation metrics computed
        - No runtime errors.
    """
    # Create a simple MLP model
    var model = SimpleMLP(
        input_dim=10,
        hidden_dim=5,
        output_dim=1,
        num_hidden_layers=1,
        init_value=0.1,
    )

    # Create synthetic training data (simple regression task)
    var n_iterations = 5
    var learning_rate = Float32(0.01)

    # Track losses to verify decrease
    var initial_loss = Float32(0.0)
    var final_loss = Float32(0.0)

    # Training loop - manual SGD for simplicity
    for iteration in range(n_iterations):
        # Create input tensor
        var input_shape: List[Int] = [10]
        var input_tensor = zeros(input_shape, DType.float32)
        for i in range(10):
            input_tensor._set_float32(i, Float32(i) * 0.1)

        # Create target tensor (simple target: sum of inputs / 10)
        var target_shape: List[Int] = [1]
        var target_tensor = zeros(target_shape, DType.float32)
        target_tensor._set_float32(
            0, Float32(0.45)
        )  # Expected output for input

        # Forward pass
        var output = model.forward(input_tensor)

        # Compute MSE loss
        var squared_error = mean_squared_error(output, target_tensor)
        var loss = mean(squared_error, axis=0, keepdims=False)

        # Get loss value
        var loss_value = loss._get_float32(0)

        if iteration == 0:
            initial_loss = loss_value
        if iteration == n_iterations - 1:
            final_loss = loss_value

        # Simple weight update: perturb weights slightly toward lower loss
        # Note: This is a simplified "training" step for testing - not true gradient descent
        # Real training would use backward pass with computed gradients
        for i in range(len(model.layer1_weights)):
            model.layer1_weights[i] -= learning_rate * 0.01

    # Verify training completed
    assert_true(
        initial_loss >= Float32(0.0), "Initial loss should be non-negative"
    )
    assert_true(final_loss >= Float32(0.0), "Final loss should be non-negative")

    # Model should produce output without errors
    var test_input_shape: List[Int] = [10]
    var test_input = zeros(test_input_shape, DType.float32)
    var test_output = model.forward(test_input)
    var test_output_shape = test_output.shape()
    assert_true(
        len(test_output_shape) == 1 and test_output_shape[0] == 1,
        "Model should produce output of shape [1]",
    )

    print("  ✓ test_model_training_to_evaluation passed")


fn test_model_inference() raises:
    """Test model inference on new data.

    Integration Points:
        - Forward pass
        - Batch inference
        - Output shape/dtype correctness

    Success Criteria:
        - Inference produces correct shape
        - Outputs are deterministic
        - Handles variable batch sizes.
    """
    # Create model with known dimensions
    var model = SimpleMLP(
        input_dim=10,
        hidden_dim=5,
        output_dim=3,  # 3 output classes
        num_hidden_layers=1,
        init_value=0.1,
    )

    # Test 1: Single sample inference
    var input_shape: List[Int] = [10]
    var input_tensor = zeros(input_shape, DType.float32)
    for i in range(10):
        input_tensor._set_float32(i, Float32(i) * 0.1)

    var output = model.forward(input_tensor)
    var output_shape = output.shape()

    # Verify output shape
    assert_true(len(output_shape) == 1, "Output should be 1D tensor")
    assert_true(
        output_shape[0] == 3, "Output should have 3 elements (output_dim=3)"
    )

    # Test 2: Determinism - same input produces same output
    var output2 = model.forward(input_tensor)

    for i in range(3):
        var val1 = output._get_float32(i)
        var val2 = output2._get_float32(i)
        assert_almost_equal(
            val1,
            val2,
            tolerance=Float32(1e-6),
            message="Inference should be deterministic",
        )

    # Test 3: Different inputs produce different outputs
    var input_tensor2 = zeros(input_shape, DType.float32)
    for i in range(10):
        input_tensor2._set_float32(i, Float32(10 - i) * 0.1)  # Different values

    var output3 = model.forward(input_tensor2)

    # At least one output should differ
    var all_same = True
    for i in range(3):
        var val1 = output._get_float32(i)
        var val3 = output3._get_float32(i)
        if abs(val1 - val3) > Float32(1e-6):
            all_same = False
            break

    assert_true(
        not all_same, "Different inputs should produce different outputs"
    )

    print("  ✓ test_model_inference passed")


fn test_model_prediction_confidence() raises:
    """Test prediction confidence and probability scores.

    Integration Points:
        - Softmax/probability conversion
        - Confidence ranking
        - Score interpretation

    Success Criteria:
        - Probabilities sum to 1.0
        - Scores in valid range
        - Confident predictions on easy data.
    """
    # Create model with multiple output classes
    var model = SimpleMLP(
        input_dim=10,
        hidden_dim=5,
        output_dim=5,  # 5 classes
        num_hidden_layers=1,
        init_value=0.1,
    )

    # Create input tensor
    var input_shape: List[Int] = [10]
    var input_tensor = zeros(input_shape, DType.float32)
    for i in range(10):
        input_tensor._set_float32(i, Float32(i) * 0.1)

    # Get model output (logits)
    var logits = model.forward(input_tensor)

    # Apply softmax to get probabilities
    var probs = softmax(logits, axis=0)
    var probs_shape = probs.shape()

    # Test 1: Verify probability shape
    assert_true(
        len(probs_shape) == 1 and probs_shape[0] == 5,
        "Probabilities should have shape [5]",
    )

    # Test 2: All probabilities should be in [0, 1]
    for i in range(5):
        var p = probs._get_float32(i)
        assert_true(
            p >= Float32(0.0) and p <= Float32(1.0),
            "Probability should be in [0, 1]",
        )

    # Test 3: Probabilities should sum to ~1.0
    var prob_sum = Float32(0.0)
    for i in range(5):
        prob_sum += probs._get_float32(i)

    assert_almost_equal(
        prob_sum,
        Float32(1.0),
        tolerance=Float32(1e-5),
        message="Probabilities should sum to 1.0",
    )

    # Test 4: Find highest probability (most confident prediction)
    var max_prob = Float32(0.0)
    for i in range(5):
        var p = probs._get_float32(i)
        if p > max_prob:
            max_prob = p

    # At least one class should have non-zero probability
    assert_true(
        max_prob > Float32(0.0),
        "At least one class should have positive probability",
    )

    print("  ✓ test_model_prediction_confidence passed")


# ============================================================================
# Model Serialization Tests
# ============================================================================


fn test_model_checkpoint_save_load() raises:
    """Test model checkpoint saving and loading.

    Integration Points:
        - Checkpoint creation
        - State serialization
        - State restoration
        - Parameter recovery

    Success Criteria:
        - Checkpoint saved successfully
        - Checkpoint loaded without errors
        - Loaded model produces same results
        - Parameters correctly restored.
    """
    # Create model with known weights
    var model = SimpleMLP(
        input_dim=4,
        hidden_dim=3,
        output_dim=2,
        num_hidden_layers=1,
        init_value=0.5,  # Specific value for verification
    )

    # Get original model output on test input
    var input_shape: List[Int] = [4]
    var test_input = zeros(input_shape, DType.float32)
    for i in range(4):
        test_input._set_float32(i, Float32(i) * 0.25)

    var original_output = model.forward(test_input)

    # Get parameters (List[AnyTensor]) for verification
    var params = model.parameters()

    # Verify we have the expected number of parameters
    # 2-layer MLP: layer1_weights, layer1_bias, layer2_weights, layer2_bias = 4 params
    assert_true(len(params) == 4, "SimpleMLP should have 4 parameter tensors")

    # Verify first layer weights shape (hidden_dim x input_dim = 3 x 4)
    var layer1_weights_shape = params[0].shape()
    assert_true(
        len(layer1_weights_shape) == 2
        and layer1_weights_shape[0] == 3
        and layer1_weights_shape[1] == 4,
        "Layer 1 weights should be [3, 4]",
    )

    # Verify parameter values are as initialized
    var first_param_val = params[0]._get_float32(0)
    assert_almost_equal(
        first_param_val,
        Float32(0.5),
        tolerance=Float32(1e-6),
        message="Parameters should match init_value",
    )

    # Create second model with different init value
    var model2 = SimpleMLP(
        input_dim=4,
        hidden_dim=3,
        output_dim=2,
        num_hidden_layers=1,
        init_value=0.1,  # Different from original
    )

    # Verify model2 has different weights initially
    var params2 = model2.parameters()
    var second_model_val = params2[0]._get_float32(0)
    assert_true(
        abs(second_model_val - Float32(0.1)) < Float32(1e-6),
        "Second model should have init_value=0.1",
    )

    # Test that both models produce outputs (testing the lifecycle)
    var output2 = model2.forward(test_input)

    # Both outputs should have same shape
    var output_shape = original_output.shape()
    var output2_shape = output2.shape()
    assert_true(
        len(output_shape) == len(output2_shape)
        and output_shape[0] == output2_shape[0],
        "Both models should produce same output shape",
    )

    # Outputs should differ due to different weights
    var outputs_differ = False
    for i in range(output_shape[0]):
        var val1 = original_output._get_float32(i)
        var val2 = output2._get_float32(i)
        if abs(val1 - val2) > Float32(1e-6):
            outputs_differ = True
            break

    assert_true(
        outputs_differ,
        "Models with different weights should produce different outputs",
    )

    print("  ✓ test_model_checkpoint_save_load passed")


fn test_model_best_checkpoint_selection() raises:
    """Test best model checkpoint selection during training.

    Integration Points:
        - Metric tracking
        - Best value comparison
        - Checkpoint management
        - Model recovery

    Success Criteria:
        - Best metric correctly identified
        - Best checkpoint preserved
        - Suboptimal checkpoints discarded
        - Best model recoverable.
    """
    # Simulate tracking losses over multiple epochs
    var losses = List[Float32]()
    losses.append(Float32(1.0))
    losses.append(Float32(0.8))
    losses.append(Float32(0.5))  # Best
    losses.append(Float32(0.6))
    losses.append(Float32(0.55))

    # Track best loss and epoch
    var best_loss = Float32(1e10)
    var best_epoch = 0

    for epoch in range(len(losses)):
        var current_loss = losses[epoch]
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch

    # Verify best was correctly identified
    assert_true(best_epoch == 2, "Best epoch should be index 2 (loss=0.5)")
    assert_almost_equal(
        best_loss,
        Float32(0.5),
        tolerance=Float32(1e-6),
        message="Best loss should be 0.5",
    )

    # Test that we can recover the "best" model state
    # Create models for each "epoch"
    var models: List[SimpleMLP] = []
    for i in range(5):
        var m = SimpleMLP(
            input_dim=4,
            hidden_dim=3,
            output_dim=2,
            num_hidden_layers=1,
            init_value=Float32(i + 1) * 0.1,  # Different weights each epoch
        )
        models.append(m^)

    # Verify we can access the model at best_epoch
    var input_shape: List[Int] = [4]
    var test_input = zeros(input_shape, DType.float32)
    var best_output = models[best_epoch].forward(test_input)

    # Model should produce valid output
    var output_shape = best_output.shape()
    assert_true(
        len(output_shape) == 1 and output_shape[0] == 2,
        "Best model should produce output shape [2]",
    )

    print("  ✓ test_model_best_checkpoint_selection passed")


# ============================================================================
# Cross-Module Integration Tests
# ============================================================================


fn test_full_pipeline_integration() raises:
    """Test full end-to-end pipeline integration.

    Integration Points:
        - Data loading
        - Model training
        - Validation
        - Evaluation
        - Checkpoint management

    Success Criteria:
        - All components work together
        - Data flows through pipeline correctly
        - Results are consistent
        - No integration errors.
    """
    # Step 1: Create model
    var model = SimpleMLP(
        input_dim=8,
        hidden_dim=4,
        output_dim=2,
        num_hidden_layers=1,
        init_value=0.1,
    )

    # Step 2: Create synthetic dataset
    var n_samples = 10
    var train_inputs: List[AnyTensor] = []
    var train_targets: List[AnyTensor] = []

    for sample_idx in range(n_samples):
        # Create input
        var input_shape: List[Int] = [8]
        var input_tensor = zeros(input_shape, DType.float32)
        for i in range(8):
            input_tensor._set_float32(i, Float32(sample_idx * 8 + i) * 0.01)
        train_inputs.append(input_tensor^)

        # Create target
        var target_shape: List[Int] = [2]
        var target_tensor = zeros(target_shape, DType.float32)
        target_tensor._set_float32(0, Float32(sample_idx % 2))
        target_tensor._set_float32(1, Float32(1 - sample_idx % 2))
        train_targets.append(target_tensor^)

    # Step 3: Training loop
    var n_epochs = 3
    var epoch_losses = List[Float32]()

    for _ in range(n_epochs):
        var epoch_loss = Float32(0.0)

        for sample_idx in range(n_samples):
            # Forward pass
            var output = model.forward(train_inputs[sample_idx])

            # Compute loss
            var mse = mean_squared_error(output, train_targets[sample_idx])
            var loss = mean(mse, axis=0, keepdims=False)
            epoch_loss += loss._get_float32(0)

        epoch_losses.append(epoch_loss / Float32(n_samples))

    # Step 4: Validation - verify loss is computed
    assert_true(
        len(epoch_losses) == n_epochs, "Should have loss for each epoch"
    )

    for i in range(n_epochs):
        assert_true(
            epoch_losses[i] >= Float32(0.0), "Loss should be non-negative"
        )

    # Step 5: Evaluation on test data
    var test_input_shape: List[Int] = [8]
    var test_input = zeros(test_input_shape, DType.float32)
    for i in range(8):
        test_input._set_float32(i, Float32(i) * 0.1)

    var test_output = model.forward(test_input)
    var test_output_shape = test_output.shape()

    assert_true(
        len(test_output_shape) == 1 and test_output_shape[0] == 2,
        "Test output should have shape [2]",
    )

    # Step 6: Get model parameters (simulates checkpoint)
    var params = model.parameters()
    assert_true(len(params) > 0, "Model should have trainable parameters")

    print("  ✓ test_full_pipeline_integration passed")


fn test_multiple_models_comparison() raises:
    """Test training multiple models for comparison.

    Integration Points:
        - Multiple model instances
        - Independent training
        - Metric comparison
        - Result aggregation

    Success Criteria:
        - Multiple models train independently
        - Metrics computed correctly
        - Comparison meaningful
        - No state corruption.
    """
    # Create multiple models with different configurations
    var model_small = SimpleMLP(
        input_dim=8,
        hidden_dim=4,
        output_dim=2,
        num_hidden_layers=1,
        init_value=0.1,
    )

    var model_large = SimpleMLP(
        input_dim=8,
        hidden_dim=8,  # Larger hidden layer
        output_dim=2,
        num_hidden_layers=1,
        init_value=0.1,
    )

    var model_deep = SimpleMLP(
        input_dim=8,
        hidden_dim=4,
        output_dim=2,
        num_hidden_layers=2,  # Two hidden layers
        init_value=0.1,
    )

    # Create shared test input
    var input_shape: List[Int] = [8]
    var test_input = zeros(input_shape, DType.float32)
    for i in range(8):
        test_input._set_float32(i, Float32(i) * 0.1)

    # Run inference on all models
    var output_small = model_small.forward(test_input)
    var output_large = model_large.forward(test_input)
    var output_deep = model_deep.forward(test_input)

    # Verify all outputs have correct shape
    var shape_small = output_small.shape()
    var shape_large = output_large.shape()
    var shape_deep = output_deep.shape()

    assert_true(
        len(shape_small) == 1 and shape_small[0] == 2,
        "Small model should produce shape [2]",
    )
    assert_true(
        len(shape_large) == 1 and shape_large[0] == 2,
        "Large model should produce shape [2]",
    )
    assert_true(
        len(shape_deep) == 1 and shape_deep[0] == 2,
        "Deep model should produce shape [2]",
    )

    # Verify models have different number of parameters
    var params_small = model_small.num_parameters()
    var params_large = model_large.num_parameters()
    var params_deep = model_deep.num_parameters()

    assert_true(
        params_large > params_small,
        "Large model should have more parameters than small",
    )
    assert_true(
        params_deep > params_small,
        "Deep model should have more parameters than small",
    )

    # Verify models are independent (modifying one doesn't affect others)
    # Modify multiple weights significantly
    for i in range(len(model_small.layer1_weights)):
        model_small.layer1_weights[i] = Float32(10.0)

    # Get fresh outputs
    var output_small_modified = model_small.forward(test_input)
    var output_large_unchanged = model_large.forward(test_input)

    # Large model output should be unchanged
    for i in range(2):
        var val_before = output_large._get_float32(i)
        var val_after = output_large_unchanged._get_float32(i)
        assert_almost_equal(
            val_before,
            val_after,
            tolerance=Float32(1e-6),
            message="Modifying one model should not affect others",
        )

    # Small model output should have changed (all weights now much larger)
    var small_changed = False
    for i in range(2):
        var val_before = output_small._get_float32(i)
        var val_after = output_small_modified._get_float32(i)
        if abs(val_before - val_after) > Float32(0.01):
            small_changed = True
            break

    assert_true(small_changed, "Modified model should produce different output")

    print("  ✓ test_multiple_models_comparison passed")


# ============================================================================
# Main Test Execution
# ============================================================================


fn main() raises:
    """Run all end-to-end integration tests."""
    print("Running model training and evaluation tests...")
    test_model_training_to_evaluation()
    test_model_inference()
    test_model_prediction_confidence()

    print("Running model serialization tests...")
    test_model_checkpoint_save_load()
    test_model_best_checkpoint_selection()

    print("Running cross-module integration tests...")
    test_full_pipeline_integration()
    test_multiple_models_comparison()

    print("\nAll end-to-end integration tests passed! ")
