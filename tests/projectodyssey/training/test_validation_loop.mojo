"""Unit tests for Validation Loop (evaluation without weight updates).

Tests cover:
- ValidationLoop constructor initialization
- validation_step() standalone function
- validate() standalone function
- ValidationLoop.run() full validation
- ValidationLoop.run_subset() subset validation
- Metrics tracking (loss updated in TrainingMetrics)
- No weight updates during validation (forward-only)

Issue #3082: Re-enable validation loop tests after ValidationLoop implementation.
Blockers resolved: ValidationLoop (Issue #34), DataLoader, TrainingMetrics all implemented.
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_equal,
    assert_equal_int,
    assert_almost_equal,
    assert_less,
    assert_greater,
    assert_not_equal_tensor,
)
from projectodyssey.training.loops.validation_loop import (
    ValidationLoop,
    validation_step,
    validate,
)
from projectodyssey.training.trainer_interface import (
    DataLoader,
    DataBatch,
    TrainingMetrics,
)
from projectodyssey.training.metrics import ConfusionMatrix
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import ones, randn, zeros

# ============================================================================
# Helper functions
# ============================================================================


def simple_forward(data: AnyTensor) raises -> AnyTensor:
    """Simple forward: returns ones matching data shape."""
    return ones(data.shape(), data.dtype())


def simple_loss(pred: AnyTensor, labels: AnyTensor) raises -> AnyTensor:
    """Simple loss: returns scalar ones tensor."""
    return ones([1], DType.float32)


def create_val_loader(n_batches: Int = 3) raises -> DataLoader:
    """Create a DataLoader with n_batches * 4 samples, batch_size=4, feature_dim=10.
    """
    var n_samples = n_batches * 4
    var data = ones([n_samples, 10], DType.float32)
    var labels = zeros([n_samples, 1], DType.float32)
    return DataLoader(data^, labels^, batch_size=4)


# ============================================================================
# ValidationLoop Initialization Tests
# ============================================================================


def test_validation_loop_init_defaults() raises:
    """Test ValidationLoop constructor defaults."""
    var vloop = ValidationLoop()
    assert_true(vloop.compute_accuracy)
    assert_true(not vloop.compute_confusion)
    assert_equal(vloop.num_classes, 10)
    print("  test_validation_loop_init_defaults: PASSED")


def test_validation_loop_init_custom() raises:
    """Test ValidationLoop constructor stores custom values."""
    var vloop = ValidationLoop(
        compute_accuracy=False, compute_confusion=True, num_classes=5
    )
    assert_true(not vloop.compute_accuracy)
    assert_true(vloop.compute_confusion)
    assert_equal(vloop.num_classes, 5)
    print("  test_validation_loop_init_custom: PASSED")


# ============================================================================
# validation_step() Tests
# ============================================================================


def test_validation_step_returns_float() raises:
    """Test validation_step returns a Float64 loss value."""
    var data = ones([4, 10], DType.float32)
    var labels = zeros([4, 1], DType.float32)
    var loss = validation_step(simple_forward, simple_loss, data, labels)
    # Loss from ones tensor = 1.0
    assert_almost_equal(loss, Float64(1.0), Float64(1e-5))
    print("  test_validation_step_returns_float: PASSED")


def test_validation_step_no_grad() raises:
    """Test validation_step completes without error (forward-only, no backward).
    """
    var data = randn([4, 10], DType.float32, seed=42)
    var labels = zeros([4, 1], DType.float32)
    var loss = validation_step(simple_forward, simple_loss, data, labels)
    assert_greater(loss, Float64(-1e10))
    print("  test_validation_step_no_grad: PASSED")


# ============================================================================
# validate() Function Tests
# ============================================================================


def test_validate_runs_full_loader() raises:
    """Test validate() iterates all batches and returns average loss."""
    var loader = create_val_loader(n_batches=3)
    var avg_loss = validate(simple_forward, simple_loss, loader)
    # Each batch returns loss=1.0, average over 3 batches = 1.0
    assert_almost_equal(avg_loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_runs_full_loader: PASSED")


def test_validate_returns_positive_loss() raises:
    """Test validate() returns non-negative loss."""
    var loader = create_val_loader(n_batches=2)
    var avg_loss = validate(simple_forward, simple_loss, loader)
    assert_greater(avg_loss, Float64(-1e-10))
    print("  test_validate_returns_positive_loss: PASSED")


# ============================================================================
# ValidationLoop.run() Tests
# ============================================================================


def test_validation_loop_run_basic() raises:
    """Test ValidationLoop.run() returns valid loss."""
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run(simple_forward, simple_loss, loader, metrics)
    assert_greater(val_loss, Float64(-1e-10))
    print("  test_validation_loop_run_basic: PASSED")


def test_validation_loop_run_updates_metrics() raises:
    """Test ValidationLoop.run() updates TrainingMetrics.val_loss."""
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run(simple_forward, simple_loss, loader, metrics)
    assert_almost_equal(metrics.val_loss, val_loss, Float64(1e-10))
    print("  test_validation_loop_run_updates_metrics: PASSED")


def test_validation_loop_run_resets_loader() raises:
    """Test run() resets a partially-consumed DataLoader before iterating.

    Strategy: Create a loader with exactly 2 batches, then exhaust it by
    setting current_batch = num_batches. Without reset(), has_next() returns
    False immediately -> 0 batches processed -> division by zero. With reset(),
    the loader restarts and processes exactly 2 batches -> valid loss.

    This proves run() calls val_loader.reset() internally through validate()
    (line 94 of validation_loop.mojo).
    """
    var vloop = ValidationLoop()
    # 2 batches total (8 samples, batch_size=4)
    var loader = create_val_loader(n_batches=2)
    # Pre-exhaust: advance to end so has_next() returns False
    loader.current_batch = loader.num_batches
    assert_true(not loader.has_next())
    var metrics = TrainingMetrics()
    # run() calls reset() internally via validate(), so it should process 2 batches
    var val_loss = vloop.run(simple_forward, simple_loss, loader, metrics)
    # Valid loss proves 2 batches were processed after reset (not 0)
    assert_greater(val_loss, Float64(-1e-10))
    assert_less(val_loss, Float64(1e10))
    print("  test_validation_loop_run_resets_loader: PASSED")


def test_validation_loop_run_compute_accuracy_false() raises:
    """Test ValidationLoop.run() with compute_accuracy=False skips accuracy.

    When compute_accuracy=False, run() should not compute accuracy and
    metrics.val_accuracy should remain at its default value of 0.0.
    """
    var vloop = ValidationLoop(compute_accuracy=False)
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run(simple_forward, simple_loss, loader, metrics)
    # Loss should still be valid
    assert_greater(val_loss, Float64(-1e-10))
    # Accuracy should remain 0.0 since compute_accuracy=False
    assert_almost_equal(metrics.val_accuracy, Float64(0.0), Float64(1e-10))
    print("  test_validation_loop_run_compute_accuracy_false: PASSED")


def test_validation_loop_run_accuracy_tracked() raises:
    """Test ValidationLoop.run() stores computed accuracy in TrainingMetrics.val_accuracy.

    When compute_accuracy=True (default), run() must pass the actual computed
    accuracy to update_val_metrics(), not a hardcoded 0.0.

    simple_forward returns ones([batch, 10]) -> argmax of each row is index 0
    (all values equal, first index wins). Labels are zeros([n, 1]) -> all label=0.
    argmax=0 == label=0 -> accuracy = 1.0.
    """
    var vloop = ValidationLoop()  # compute_accuracy=True by default
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    _ = vloop.run(simple_forward, simple_loss, loader, metrics)
    # simple_forward: ones([batch,10]) -> argmax=0; labels=zeros -> label=0
    # All predictions correct -> accuracy = 1.0
    assert_almost_equal(metrics.val_accuracy, Float64(1.0), Float64(1e-5))
    print("  test_validation_loop_run_accuracy_tracked: PASSED")


# ============================================================================
# ValidationLoop.run_subset() Tests
# ============================================================================


def test_validation_loop_run_subset_limited() raises:
    """Test run_subset(max_batches=2) with 5-batch loader processes only 2 batches.
    """
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=5)
    var metrics = TrainingMetrics()
    # With max_batches=2, only 2 batches processed; loss from ones = 1.0
    var val_loss = vloop.run_subset(
        simple_forward, simple_loss, loader, 2, metrics
    )
    assert_almost_equal(val_loss, Float64(1.0), Float64(1e-5))
    print("  test_validation_loop_run_subset_limited: PASSED")


def test_validation_loop_run_subset_loss_valid() raises:
    """Test run_subset returns valid Float64 loss."""
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run_subset(
        simple_forward, simple_loss, loader, 1, metrics
    )
    assert_greater(val_loss, Float64(-1e-10))
    assert_less(val_loss, Float64(1e10))
    print("  test_validation_loop_run_subset_loss_valid: PASSED")


def test_validation_loop_run_subset_resets_loader() raises:
    """Test run_subset() resets a partially-consumed DataLoader before iterating.

    Strategy: Create a loader with exactly 2 batches, then exhaust it by
    setting current_batch = num_batches. Without reset(), has_next() returns
    False immediately -> 0 batches processed -> division by zero. With reset(),
    the loader restarts and processes exactly 2 batches -> valid loss.

    This proves run_subset() calls val_loader.reset() internally (line 255 of
    validation_loop.mojo).
    """
    var vloop = ValidationLoop()
    # 2 batches total (8 samples, batch_size=4)
    var loader = create_val_loader(n_batches=2)
    # Pre-exhaust: advance to end so has_next() returns False
    loader.current_batch = loader.num_batches
    assert_true(not loader.has_next())
    var metrics = TrainingMetrics()
    # run_subset calls reset() internally, so it should process 2 batches
    var val_loss = vloop.run_subset(
        simple_forward, simple_loss, loader, 2, metrics
    )
    # Valid loss proves 2 batches were processed after reset (not 0)
    assert_greater(val_loss, Float64(-1e-10))
    assert_less(val_loss, Float64(1e10))
    print("  test_validation_loop_run_subset_resets_loader: PASSED")


def test_validation_loop_run_subset_updates_val_accuracy() raises:
    """Test run_subset() computes and updates val_accuracy (not hardcoded 0.0).

    Creates a ValidationLoop with compute_accuracy=True, runs run_subset() on
    a loader with ones predictions and zero labels, and asserts that
    metrics.val_accuracy > 0.0 (not hardcoded 0.0).

    This verifies the fix for issue #3680 where run_subset() was calling
    metrics.update_val_metrics(avg_loss, 0.0) with a hardcoded accuracy.
    """
    var vloop = ValidationLoop(compute_accuracy=True)
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run_subset(
        simple_forward, simple_loss, loader, 3, metrics
    )
    # simple_forward returns ones, labels are zeros, so accuracy should be > 0.0
    assert_greater(metrics.val_accuracy, Float64(0.0))
    print("  test_validation_loop_run_subset_updates_val_accuracy: PASSED")


# ============================================================================
# No-Weight-Update Property Tests
# ============================================================================


def test_validation_loop_no_weight_updates() raises:
    """Validate that validation runs forward-only without optimizer step.

    Since ValidationLoop has no optimizer, calling run() multiple times
    on the same loader with the same forward function produces the same loss.
    """
    var vloop = ValidationLoop()
    var metrics1 = TrainingMetrics()
    var metrics2 = TrainingMetrics()

    var loader1 = create_val_loader(n_batches=3)
    var loader2 = create_val_loader(n_batches=3)

    var loss1 = vloop.run(simple_forward, simple_loss, loader1, metrics1)
    var loss2 = vloop.run(simple_forward, simple_loss, loader2, metrics2)

    # Same inputs and forward fn -> same loss every time (no weight mutation)
    assert_almost_equal(loss1, loss2, Float64(1e-10))
    print("  test_validation_loop_no_weight_updates: PASSED")


# ============================================================================
# Confusion Matrix Integration Tests
# ============================================================================


def test_validation_loop_confusion_matrix_basic() raises:
    """Exact TP/TN/FP/FN across MULTIPLE batches with an asymmetric matrix.

    Upgraded per #3684: DataLoader.next() now slices the real dataset
    (self.data.slice(...) / self.labels.slice(...)) instead of returning
    zero-initialized placeholder batches, so per-sample predictions are
    meaningful end-to-end and confusion-matrix cell counts can be asserted
    exactly (not just smoke-tested for a non-negative loss).

    This complements test_validation_loop_confusion_matrix_integration
    (single balanced batch, #3185) by exercising the parts that only real
    slicing makes testable:
      - the multi-batch offset path (start_idx = current_batch * batch_size),
      - data/label slice ALIGNMENT across a batch boundary — an off-by-one
        between data.slice and labels.slice would corrupt an asymmetric matrix,
      - an asymmetric count fixture (2,1,1,2), so a degenerate all-class-0
        forward or a slice misalignment cannot accidentally satisfy it.

    Fixture (6 samples, 2 logit columns, identity forward, batch_size=3 => 2 batches):
        idx  logits    argmax(pred)  label   cell
         0   [1,0]        0            0      [0,0] TN   -- batch 0
         1   [1,0]        0            1      [1,0] FN
         2   [0,1]        1            1      [1,1] TP
         3   [0,1]        1            0      [0,1] FP   -- batch 1
         4   [0,1]        1            1      [1,1] TP
         5   [1,0]        0            0      [0,0] TN
    Confusion matrix (row=true, col=pred), layout idx = true*2 + pred:
                pred=0   pred=1
        true=0    2        1      ([0,0]=TN=2, [0,1]=FP=1)
        true=1    1        2      ([1,0]=FN=1, [1,1]=TP=2)
    """
    var vloop = ValidationLoop(compute_confusion=True, num_classes=2)

    var n_samples = 6
    var data_shape = List[Int]()
    data_shape.append(n_samples)
    data_shape.append(2)
    var data = AnyTensor(data_shape, DType.float32)
    # Per-row logits [c0, c1]; argmax picks the larger column.
    var rows: List[Int] = [0, 0, 1, 1, 1, 0]  # desired argmax per sample
    for i in range(n_samples):
        var hot = rows[i]
        data.set(i * 2 + 0, Float32(1.0) if hot == 0 else Float32(0.0))
        data.set(i * 2 + 1, Float32(1.0) if hot == 1 else Float32(0.0))

    var labels_shape = List[Int]()
    labels_shape.append(n_samples)
    var labels = AnyTensor(labels_shape, DType.int32)
    var label_vals: List[Int] = [0, 1, 1, 0, 1, 0]
    for i in range(n_samples):
        labels.set(i, Int32(label_vals[i]))

    # batch_size=3 with 6 samples => 2 batches, exercising the start_idx advance.
    var loader = DataLoader(data^, labels^, batch_size=3)
    var metrics = TrainingMetrics()
    # identity_forward preserves logits so argmax matches the crafted rows.
    var val_loss = vloop.run(identity_forward, simple_loss, loader, metrics)
    assert_greater(val_loss, Float64(-1e-10))

    # ValidationLoop.run() populates vloop.confusion_matrix when
    # compute_confusion=True. Assert exact cell counts (matrix is int32,
    # layout idx = true*num_classes + pred). Positive class = 1.
    var cm = vloop.confusion_matrix.matrix
    assert_equal_int(Int(cm.load[DType.int32](0)), 2)  # [0,0] TN
    assert_equal_int(Int(cm.load[DType.int32](1)), 1)  # [0,1] FP
    assert_equal_int(Int(cm.load[DType.int32](2)), 1)  # [1,0] FN
    assert_equal_int(Int(cm.load[DType.int32](3)), 2)  # [1,1] TP
    print("  test_validation_loop_confusion_matrix_basic: PASSED")


def test_confusion_matrix_binary_counts() raises:
    """Test ConfusionMatrix cell counts with known binary predictions.

    Fixture: y_true=[0,1,0,1], y_pred=[0,1,1,0]
    Expected confusion matrix (row=true, col=pred):
        pred=0  pred=1
    true=0   1       1    (TN=1, FP=1)
    true=1   1       1    (FN=1, TP=1)
    """
    var cm = ConfusionMatrix(num_classes=2)

    var preds_shape = List[Int]()
    preds_shape.append(4)
    var preds = AnyTensor(preds_shape, DType.int32)
    preds.set(0, Int32(0))
    preds.set(1, Int32(1))
    preds.set(2, Int32(1))
    preds.set(3, Int32(0))

    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = AnyTensor(labels_shape, DType.int32)
    labels.set(0, Int32(0))
    labels.set(1, Int32(1))
    labels.set(2, Int32(0))
    labels.set(3, Int32(1))

    cm.update(preds, labels)

    var raw = cm.normalize(mode="none")
    # Matrix layout: raw[row*2 + col] where row=true, col=pred
    # [0,0]=TN=1, [0,1]=FP=1, [1,0]=FN=1, [1,1]=TP=1
    assert_equal_int(Int(raw._data.bitcast[Float64]()[0]), 1)  # TN
    assert_equal_int(Int(raw._data.bitcast[Float64]()[1]), 1)  # FP
    assert_equal_int(Int(raw._data.bitcast[Float64]()[2]), 1)  # FN
    assert_equal_int(Int(raw._data.bitcast[Float64]()[3]), 1)  # TP
    print("  test_confusion_matrix_binary_counts: PASSED")


def test_confusion_matrix_all_correct() raises:
    """Test ConfusionMatrix with all-correct predictions yields pure diagonal.

    Fixture: y_true=[0,0,1,1], y_pred=[0,0,1,1]
    Expected: TN=2, FP=0, FN=0, TP=2
    """
    var cm = ConfusionMatrix(num_classes=2)

    var preds_shape = List[Int]()
    preds_shape.append(4)
    var preds = AnyTensor(preds_shape, DType.int32)
    preds.set(0, Int32(0))
    preds.set(1, Int32(0))
    preds.set(2, Int32(1))
    preds.set(3, Int32(1))

    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = AnyTensor(labels_shape, DType.int32)
    labels.set(0, Int32(0))
    labels.set(1, Int32(0))
    labels.set(2, Int32(1))
    labels.set(3, Int32(1))

    cm.update(preds, labels)

    var raw = cm.normalize(mode="none")
    assert_equal_int(Int(raw._data.bitcast[Float64]()[0]), 2)  # TN=2
    assert_equal_int(Int(raw._data.bitcast[Float64]()[1]), 0)  # FP=0
    assert_equal_int(Int(raw._data.bitcast[Float64]()[2]), 0)  # FN=0
    assert_equal_int(Int(raw._data.bitcast[Float64]()[3]), 2)  # TP=2
    print("  test_confusion_matrix_all_correct: PASSED")


def test_confusion_matrix_all_wrong() raises:
    """Test ConfusionMatrix with all-wrong predictions yields zero diagonal.

    Fixture: y_true=[0,0,1,1], y_pred=[1,1,0,0]
    Expected: TN=0, FP=2, FN=2, TP=0
    """
    var cm = ConfusionMatrix(num_classes=2)

    var preds_shape = List[Int]()
    preds_shape.append(4)
    var preds = AnyTensor(preds_shape, DType.int32)
    preds.set(0, Int32(1))
    preds.set(1, Int32(1))
    preds.set(2, Int32(0))
    preds.set(3, Int32(0))

    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = AnyTensor(labels_shape, DType.int32)
    labels.set(0, Int32(0))
    labels.set(1, Int32(0))
    labels.set(2, Int32(1))
    labels.set(3, Int32(1))

    cm.update(preds, labels)

    var raw = cm.normalize(mode="none")
    assert_equal_int(Int(raw._data.bitcast[Float64]()[0]), 0)  # TN=0
    assert_equal_int(Int(raw._data.bitcast[Float64]()[1]), 2)  # FP=2
    assert_equal_int(Int(raw._data.bitcast[Float64]()[2]), 2)  # FN=2
    assert_equal_int(Int(raw._data.bitcast[Float64]()[3]), 0)  # TP=0
    print("  test_confusion_matrix_all_wrong: PASSED")


# ============================================================================
# Helper: identity forward (returns input unchanged for controlled logits)
# ============================================================================


def identity_forward(data: AnyTensor) raises -> AnyTensor:
    """Identity forward: returns the input data unchanged.

    Used to control predictions via crafted input logits.
    """
    return data


# ============================================================================
# ValidationLoop Confusion Matrix Integration Test (Issue #3185)
# ============================================================================


def test_validation_loop_confusion_matrix_integration() raises:
    """Integration test: ValidationLoop populates confusion matrix with correct counts.

    Constructs ValidationLoop(compute_confusion=True, num_classes=2), runs
    validation with crafted 2-column logit data and known int32 labels,
    then inspects vloop.confusion_matrix to verify exact cell counts.

    Fixture (single batch of 4 samples):
        Data (logits):  [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        -> argmax:      [0, 1, 1, 0]
        Labels:         [0, 1, 0, 1]

    Expected confusion matrix (row=true, col=pred):
            pred=0  pred=1
    true=0    1       1    (TN=1, FP=1)
    true=1    1       1    (FN=1, TP=1)

    Closes #3185.
    """
    var vloop = ValidationLoop(
        compute_confusion=True, compute_accuracy=False, num_classes=2
    )

    # Construct data: 4 samples, 2 logit columns
    var n_samples = 4
    var data_shape = List[Int]()
    data_shape.append(n_samples)
    data_shape.append(2)
    var data = AnyTensor(data_shape, DType.float32)
    # Row 0: [1.0, 0.0] -> argmax=0
    data.set(0, Float32(1.0))
    data.set(1, Float32(0.0))
    # Row 1: [0.0, 1.0] -> argmax=1
    data.set(2, Float32(0.0))
    data.set(3, Float32(1.0))
    # Row 2: [0.0, 1.0] -> argmax=1
    data.set(4, Float32(0.0))
    data.set(5, Float32(1.0))
    # Row 3: [1.0, 0.0] -> argmax=0
    data.set(6, Float32(1.0))
    data.set(7, Float32(0.0))

    var labels_shape = List[Int]()
    labels_shape.append(n_samples)
    var labels = AnyTensor(labels_shape, DType.int32)
    labels.set(0, Int32(0))
    labels.set(1, Int32(1))
    labels.set(2, Int32(0))
    labels.set(3, Int32(1))

    var loader = DataLoader(data^, labels^, batch_size=4)
    var metrics = TrainingMetrics()

    # Run validation through ValidationLoop (the full integration path)
    _ = vloop.run(identity_forward, simple_loss, loader, metrics)

    # Inspect the confusion matrix stored on ValidationLoop
    var raw = vloop.confusion_matrix.normalize(mode="none")
    # Matrix layout: raw[row*2 + col] where row=true, col=pred
    assert_equal_int(Int(raw._data.bitcast[Float64]()[0]), 1)  # [0,0] TN=1
    assert_equal_int(Int(raw._data.bitcast[Float64]()[1]), 1)  # [0,1] FP=1
    assert_equal_int(Int(raw._data.bitcast[Float64]()[2]), 1)  # [1,0] FN=1
    assert_equal_int(Int(raw._data.bitcast[Float64]()[3]), 1)  # [1,1] TP=1
    print("  test_validation_loop_confusion_matrix_integration: PASSED")


# ============================================================================
# Test Main
# ============================================================================


def main() raises:
    """Run all validation loop tests."""
    print("Running ValidationLoop initialization tests...")
    test_validation_loop_init_defaults()
    test_validation_loop_init_custom()

    print("Running validation_step() tests...")
    test_validation_step_returns_float()
    test_validation_step_no_grad()

    print("Running validate() function tests...")
    test_validate_runs_full_loader()
    test_validate_returns_positive_loss()

    print("Running ValidationLoop.run() tests...")
    test_validation_loop_run_basic()
    test_validation_loop_run_updates_metrics()
    test_validation_loop_run_compute_accuracy_false()
    test_validation_loop_run_resets_loader()
    test_validation_loop_run_accuracy_tracked()

    print("Running ValidationLoop.run_subset() tests...")
    test_validation_loop_run_subset_limited()
    test_validation_loop_run_subset_loss_valid()
    test_validation_loop_run_subset_resets_loader()
    test_validation_loop_run_subset_updates_val_accuracy()

    print("Running no-weight-update property tests...")
    test_validation_loop_no_weight_updates()

    print("Running confusion matrix integration tests...")
    test_validation_loop_confusion_matrix_basic()
    test_confusion_matrix_binary_counts()
    test_confusion_matrix_all_correct()
    test_confusion_matrix_all_wrong()
    test_validation_loop_confusion_matrix_integration()

    print("\nAll validation loop tests passed!")
