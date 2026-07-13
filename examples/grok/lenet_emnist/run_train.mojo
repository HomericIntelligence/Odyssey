"""
CLI Wrapper for LeNet-5 Training (Grokking Experiment Variant).

Fork of examples/lenet_emnist/run_train.mojo with:
  * AdamW optimizer (Adam + decoupled weight decay) instead of SGD.
  * --subset-size flag to train on a small fixed subset of EMNIST (memorization regime).
  * --max-batches flag to cap batches per epoch (fast smoke tests).
  * Per-epoch structured log line with train_loss / train_acc / test_loss /
    test_acc / weight_l2_norm (regex-parseable by analyze_phases.py).
  * MultiMetricCheckpointer: tracks best/min/min_after_max/max_after_min
    per requested metric and writes weight + JSON sidecar.

Autograd note: ProjectOdyssey's autograd substrate (Variable conv/pool/linear
ops, automatic backward dispatch) is in-progress (see #5452). This example
keeps the existing manual backward path; AdamW is wired in via the bare
functional `adamw_step()` from `projectodyssey.training.optimizers.adamw`.

Usage:
    mojo run examples/grok/lenet_emnist/run_train.mojo \
        --epochs 1000 --batch-size 64 --lr 0.001 --weight-decay 1.0 \
        --subset-size 1000 \
        --track-metric test_acc:max,test_loss:both,train_loss:min
"""

from model import LeNet5, AdamWState, update_parameters_adamw
from projectodyssey.data.constants import DatasetInfo
from projectodyssey.data.formats import (
    load_idx_images,
    load_idx_labels,
    normalize_images,
    one_hot_encode,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.core.pooling import maxpool2d, maxpool2d_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.loss import cross_entropy, cross_entropy_backward
from projectodyssey.core.numerical_safety import compute_tensor_l2_norm
from projectodyssey.training.evaluation import evaluate_model_simple
from projectodyssey.utils.arg_parser import create_training_parser
from std.collections import List, Dict


struct TrainConfig(Movable):
    """Training configuration for the grokking-experiment variant."""

    var epochs: Int
    var batch_size: Int
    var learning_rate: Float64
    var weight_decay: Float64
    var subset_size: Int
    var max_batches: Int
    var log_every: Int
    var checkpoint_every: Int
    var track_metric: String
    var data_dir: String
    var weights_dir: String

    def __init__(out self):
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        self.subset_size = 0
        self.max_batches = 0
        self.log_every = 1
        self.checkpoint_every = 1
        self.track_metric = "test_acc:max"
        self.data_dir = "datasets/emnist"
        self.weights_dir = "lenet5_weights"


def parse_args() raises -> TrainConfig:
    """Parse command line arguments using enhanced argument parser."""
    var parser = create_training_parser()
    parser.add_argument("weights-dir", "string", "lenet5_weights")
    parser.add_argument("data-dir", "string", "datasets/emnist")
    parser.add_argument("weight-decay", "float", "0.01")
    parser.add_argument("subset-size", "int", "0")
    parser.add_argument("max-batches", "int", "0")
    parser.add_argument("log-every", "int", "1")
    parser.add_argument("checkpoint-every", "int", "1")
    # NOTE: arg_parser does not support repeated flags; pass comma-separated
    # list e.g. --track-metric test_acc:max,test_loss:both
    parser.add_argument("track-metric", "string", "test_acc:max")

    var args = parser.parse()

    var config = TrainConfig()
    config.epochs = args.resolve_int("epochs", 10)
    config.batch_size = args.resolve_int("batch-size", 32)
    config.learning_rate = args.resolve_float("lr", 0.001)
    config.weight_decay = args.resolve_float("weight-decay", 0.01)
    config.subset_size = args.resolve_int("subset-size", 0)
    config.max_batches = args.resolve_int("max-batches", 0)
    config.log_every = args.resolve_int("log-every", 1)
    config.checkpoint_every = args.resolve_int("checkpoint-every", 1)
    config.track_metric = args.resolve_string("track-metric", "test_acc:max")
    config.data_dir = args.resolve_string("data-dir", "datasets/emnist")
    config.weights_dir = args.resolve_string("weights-dir", "lenet5_weights")

    return config^


# ============================================================================
# Forward + backward + gradient computation + AdamW update (fused)
# ============================================================================


def forward_backward_step(
    mut model: LeNet5,
    mut optim_state: AdamWState,
    t: Int,
    input: AnyTensor,
    labels: AnyTensor,
    learning_rate: Float64,
    weight_decay: Float64,
) raises -> Float32:
    """Run forward + backward, return loss and gradients (no parameter update).

    Manual gradient computation (autograd path is in-progress; see #5452).
    """
    # Conv1 + ReLU + MaxPool
    var conv1_out = conv2d(
        input, model.conv1_kernel, model.conv1_bias, stride=1, padding=0
    )
    var relu1_out = relu(conv1_out)
    var pool1_out = maxpool2d(relu1_out, kernel_size=2, stride=2, padding=0)

    # Conv2 + ReLU + MaxPool
    var conv2_out = conv2d(
        pool1_out, model.conv2_kernel, model.conv2_bias, stride=1, padding=0
    )
    var relu2_out = relu(conv2_out)
    var pool2_out = maxpool2d(relu2_out, kernel_size=2, stride=2, padding=0)

    # Flatten
    var pool2_shape = pool2_out.shape()
    var batch_size = pool2_shape[0]
    var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    var flatten_shape: List[Int] = [batch_size, flattened_size]
    var flattened = pool2_out.reshape(flatten_shape)

    # FC1 + ReLU
    var fc1_out = linear(flattened, model.fc1_weights, model.fc1_bias)
    var relu3_out = relu(fc1_out)

    # FC2 + ReLU
    var fc2_out = linear(relu3_out, model.fc2_weights, model.fc2_bias)
    var relu4_out = relu(fc2_out)

    # FC3 (logits)
    var logits = linear(relu4_out, model.fc3_weights, model.fc3_bias)

    # Loss
    var loss_tensor = cross_entropy(logits, labels)
    var loss = loss_tensor._data.bitcast[Float32]()[0]

    # Backward
    var grad_output_shape: List[Int] = [1]
    var grad_output = zeros(grad_output_shape, logits.dtype())
    grad_output.set(0, Float32(1.0))
    var grad_logits = cross_entropy_backward(grad_output, logits, labels)

    var fc3_grads = linear_backward(grad_logits, relu4_out, model.fc3_weights)
    var grad_fc2_out = relu_backward(fc3_grads.grad_input, fc2_out)
    var fc2_grads = linear_backward(grad_fc2_out, relu3_out, model.fc2_weights)
    var grad_fc1_out = relu_backward(fc2_grads.grad_input, fc1_out)
    var fc1_grads = linear_backward(grad_fc1_out, flattened, model.fc1_weights)
    var grad_pool2_out = fc1_grads.grad_input.reshape(pool2_shape)
    var grad_relu2_out = maxpool2d_backward(
        grad_pool2_out, relu2_out, kernel_size=2, stride=2, padding=0
    )
    var grad_conv2_out = relu_backward(grad_relu2_out, conv2_out)
    var conv2_grads = conv2d_backward(
        grad_conv2_out, pool1_out, model.conv2_kernel, stride=1, padding=0
    )
    var grad_relu1_out = maxpool2d_backward(
        conv2_grads.grad_input, relu1_out, kernel_size=2, stride=2, padding=0
    )
    var grad_conv1_out = relu_backward(grad_relu1_out, conv1_out)
    var conv1_grads = conv2d_backward(
        grad_conv1_out, input, model.conv1_kernel, stride=1, padding=0
    )

    update_parameters_adamw(
        model,
        optim_state,
        learning_rate,
        t,
        weight_decay,
        conv1_grads.grad_weights^,
        conv1_grads.grad_bias^,
        conv2_grads.grad_weights^,
        conv2_grads.grad_bias^,
        fc1_grads.grad_weights^,
        fc1_grads.grad_bias^,
        fc2_grads.grad_weights^,
        fc2_grads.grad_bias^,
        fc3_grads.grad_weights^,
        fc3_grads.grad_bias^,
    )
    return loss


# ============================================================================
# Helpers: train_acc / test_loss / weight_l2
# ============================================================================


def compute_train_accuracy(
    mut model: LeNet5,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    num_classes: Int,
) raises -> Float32:
    """Top-1 accuracy on the training set (or training subset)."""
    return evaluate_model_simple(
        model,
        train_images,
        train_labels,
        batch_size=batch_size,
        num_classes=num_classes,
        verbose=False,
    )


def compute_test_loss(
    mut model: LeNet5,
    test_images: AnyTensor,
    test_labels: AnyTensor,
    batch_size: Int,
    num_classes: Int,
) raises -> Float32:
    """Mean cross-entropy loss across the full test set."""
    var num_samples = test_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size
    var total = Float32(0.0)
    var count = 0

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)
        var batch_images = test_images.slice(start_idx, end_idx, axis=0)
        var batch_labels_int = test_labels.slice(start_idx, end_idx, axis=0)
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=num_classes
        )

        var logits = model.forward(batch_images)
        var loss_tensor = cross_entropy(logits, batch_labels)
        total += loss_tensor._data.bitcast[Float32]()[0]
        count += 1

    if count == 0:
        return Float32(0.0)
    return total / Float32(count)


def compute_weight_l2_norm(model: LeNet5) raises -> Float64:
    """Sum of per-parameter L2 norms across all 10 weight tensors."""
    var params = model.parameters()
    var total = Float64(0.0)
    for i in range(len(params)):
        total += compute_tensor_l2_norm(params[i])
    return total


# ============================================================================
# Multi-metric checkpointer (inline, scoped to this example)
# ============================================================================


struct MetricMode(Copyable, Movable):
    """A metric to track and its mode of tracking.

    mode is one of "max", "min", or "both".
    """

    var name: String
    var mode: String

    def __init__(out self, name: String, mode: String):
        self.name = name
        self.mode = mode


def parse_track_metric(spec: String) raises -> List[MetricMode]:
    """Parse a comma-separated list like "test_acc:max,test_loss:both"."""
    var out = List[MetricMode]()
    if spec.byte_length() == 0:
        return out^
    var parts = spec.split(",")
    for ref part in parts:
        var trimmed = String(part)
        if trimmed.byte_length() == 0:
            continue
        var kv = trimmed.split(":")
        if len(kv) != 2:
            raise Error(
                "Invalid --track-metric entry '"
                + trimmed
                + "' (expected 'name:mode')"
            )
        var name = String(kv[0])
        var mode = String(kv[1])
        if mode != "max" and mode != "min" and mode != "both":
            raise Error(
                "Invalid --track-metric mode '"
                + mode
                + "' (expected max, min, or both)"
            )
        out.append(MetricMode(name=name, mode=mode))
    return out^


def _write_sidecar(
    path: String,
    metric: String,
    mode_kind: String,
    value: Float64,
    epoch: Int,
) raises:
    """Write a tiny JSON sidecar describing a checkpoint event."""
    var body = (
        '{"metric": "'
        + metric
        + '", "mode_kind": "'
        + mode_kind
        + '", "value": '
        + String(value)
        + ', "epoch": '
        + String(epoch)
        + "}\n"
    )
    with open(path, "w") as f:
        f.write(body)


struct MultiMetricCheckpointer:
    """Track multiple metrics and write a separate checkpoint per (metric, kind).

    For each metric in `metrics`:
      mode == "max": save weights to `{ckpt_dir}/{name}_best/` when value rises
      mode == "min": save weights to `{ckpt_dir}/{name}_best/` when value falls
      mode == "both": save `{name}_best/` (max), `{name}_min/` (min),
                      `{name}_min_after_max/`, `{name}_max_after_min/`
    Each checkpoint directory contains the model's `.weights` files plus a
    `meta.json` sidecar.
    """

    var metrics: List[MetricMode]
    var ckpt_dir: String
    var best_max_value: List[Float64]
    var best_max_epoch: List[Int]
    var best_min_value: List[Float64]
    var best_min_epoch: List[Int]
    var has_seen_max: List[Bool]
    var has_seen_min: List[Bool]
    var min_after_max_value: List[Float64]
    var min_after_max_epoch: List[Int]
    var max_after_min_value: List[Float64]
    var max_after_min_epoch: List[Int]

    def __init__(out self, metrics: List[MetricMode], ckpt_dir: String) raises:
        self.metrics = metrics.copy()
        self.ckpt_dir = ckpt_dir
        self.best_max_value = List[Float64]()
        self.best_max_epoch = List[Int]()
        self.best_min_value = List[Float64]()
        self.best_min_epoch = List[Int]()
        self.has_seen_max = List[Bool]()
        self.has_seen_min = List[Bool]()
        self.min_after_max_value = List[Float64]()
        self.min_after_max_epoch = List[Int]()
        self.max_after_min_value = List[Float64]()
        self.max_after_min_epoch = List[Int]()
        var n = len(metrics)
        for _ in range(n):
            self.best_max_value.append(Float64(-1.0e30))
            self.best_max_epoch.append(-1)
            self.best_min_value.append(Float64(1.0e30))
            self.best_min_epoch.append(-1)
            self.has_seen_max.append(False)
            self.has_seen_min.append(False)
            self.min_after_max_value.append(Float64(1.0e30))
            self.min_after_max_epoch.append(-1)
            self.max_after_min_value.append(Float64(-1.0e30))
            self.max_after_min_epoch.append(-1)

    def _save(
        mut self,
        mut model: LeNet5,
        subdir: String,
        metric_name: String,
        mode_kind: String,
        value: Float64,
        epoch: Int,
    ) raises:
        var path = self.ckpt_dir + "/" + subdir
        model.save_weights(path)
        _write_sidecar(
            path + "/meta.json", metric_name, mode_kind, value, epoch
        )

    def update(
        mut self,
        mut model: LeNet5,
        epoch: Int,
        metric_values: Dict[String, Float64],
    ) raises:
        for i in range(len(self.metrics)):
            # Snapshot name/mode into locals so we don't alias self.metrics
            # while passing `mut self` to `_save`.
            var name = String(self.metrics[i].name)
            var mode = String(self.metrics[i].mode)
            if name not in metric_values:
                continue
            var value = metric_values[name]

            # max-mode tracking (also active for "both")
            if mode == "max" or mode == "both":
                if value > self.best_max_value[i]:
                    self.best_max_value[i] = value
                    self.best_max_epoch[i] = epoch
                    self.has_seen_max[i] = True
                    self._save(
                        model, name + "_best", name, "best", value, epoch
                    )

            # min-mode tracking (also active for "both")
            if mode == "min" or mode == "both":
                if value < self.best_min_value[i]:
                    self.best_min_value[i] = value
                    self.best_min_epoch[i] = epoch
                    self.has_seen_min[i] = True
                    var subdir: String
                    if mode == "min":
                        subdir = name + "_best"
                    else:
                        subdir = name + "_min"
                    self._save(model, subdir, name, "best", value, epoch)

            # both: track min-after-max and max-after-min
            if mode == "both":
                if self.has_seen_max[i] and value < self.min_after_max_value[i]:
                    self.min_after_max_value[i] = value
                    self.min_after_max_epoch[i] = epoch
                    self._save(
                        model,
                        name + "_min_after_max",
                        name,
                        "min_after_max",
                        value,
                        epoch,
                    )
                if self.has_seen_min[i] and value > self.max_after_min_value[i]:
                    self.max_after_min_value[i] = value
                    self.max_after_min_epoch[i] = epoch
                    self._save(
                        model,
                        name + "_max_after_min",
                        name,
                        "max_after_min",
                        value,
                        epoch,
                    )


# ============================================================================
# Training loop
# ============================================================================


def train_epoch(
    mut model: LeNet5,
    mut optim_state: AdamWState,
    mut t: Int,
    train_images: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    learning_rate: Float64,
    weight_decay: Float64,
    num_classes: Int,
    max_batches: Int,
) raises -> Tuple[Float32, Int]:
    """Train for one epoch with AdamW. Returns (avg_loss, new_timestep)."""
    var num_samples = train_images.shape()[0]
    var num_batches = (num_samples + batch_size - 1) // batch_size
    var total_loss = Float32(0.0)
    var processed = 0

    for batch_idx in range(num_batches):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        var start_idx = batch_idx * batch_size
        var end_idx = min(start_idx + batch_size, num_samples)

        var batch_images = train_images.slice(start_idx, end_idx, axis=0)
        var batch_labels_int = train_labels.slice(start_idx, end_idx, axis=0)
        var batch_labels = one_hot_encode(
            batch_labels_int, num_classes=num_classes
        )

        var batch_loss = forward_backward_step(
            model,
            optim_state,
            t,
            batch_images,
            batch_labels,
            learning_rate,
            weight_decay,
        )
        total_loss += batch_loss
        processed += 1
        t += 1

    var avg = total_loss / Float32(max(processed, 1))
    return Tuple[Float32, Int](avg, t)


def main() raises:
    """Main training entry point."""
    print("=" * 60)
    print("LeNet-5 Grokking Experiment (AdamW + EMNIST)")
    print("=" * 60)

    var config = parse_args()
    print("\nConfiguration:")
    print("  Epochs: ", config.epochs)
    print("  Batch Size: ", config.batch_size)
    print("  Learning Rate: ", config.learning_rate)
    print("  Weight Decay: ", config.weight_decay)
    print("  Subset Size: ", config.subset_size, "(0 = full)")
    print("  Max Batches: ", config.max_batches, "(0 = unlimited)")
    print("  Log Every: ", config.log_every)
    print("  Checkpoint Every: ", config.checkpoint_every)
    print("  Track Metric: ", config.track_metric)
    print("  Data Directory: ", config.data_dir)
    print("  Weights Directory: ", config.weights_dir)
    print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var dataset_info = DatasetInfo("emnist_balanced")
    var model = LeNet5(num_classes=dataset_info.num_classes())
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # AdamW optimizer state
    var optim_state = AdamWState(model)
    var t = 1

    # Load dataset
    print("Loading EMNIST dataset...")
    var train_images_path = (
        config.data_dir + "/emnist-balanced-train-images-idx3-ubyte"
    )
    var train_labels_path = (
        config.data_dir + "/emnist-balanced-train-labels-idx1-ubyte"
    )
    var test_images_path = (
        config.data_dir + "/emnist-balanced-test-images-idx3-ubyte"
    )
    var test_labels_path = (
        config.data_dir + "/emnist-balanced-test-labels-idx1-ubyte"
    )

    var train_images_raw = load_idx_images(train_images_path)
    var train_labels = load_idx_labels(train_labels_path)
    var test_images_raw = load_idx_images(test_images_path)
    var test_labels = load_idx_labels(test_labels_path)

    var train_images = normalize_images(train_images_raw)
    var test_images = normalize_images(test_images_raw)

    # Optional subset (memorization regime)
    if config.subset_size > 0 and config.subset_size < train_images.shape()[0]:
        train_images = train_images.slice(0, config.subset_size, axis=0)
        train_labels = train_labels.slice(0, config.subset_size, axis=0)
        print("  Subset: ", train_images.shape()[0], "samples")

    print("  Training samples: ", train_images.shape()[0])
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Parse metric spec, build checkpointer
    var metric_modes = parse_track_metric(config.track_metric)
    var checkpointer = MultiMetricCheckpointer(metric_modes, config.weights_dir)

    print("Starting training...")
    for epoch in range(1, config.epochs + 1):
        var epoch_result = train_epoch(
            model,
            optim_state,
            t,
            train_images,
            train_labels,
            config.batch_size,
            config.learning_rate,
            config.weight_decay,
            model.num_classes,
            config.max_batches,
        )
        var train_loss = epoch_result[0]
        t = epoch_result[1]

        if epoch % config.log_every == 0 or epoch == config.epochs:
            var train_acc = compute_train_accuracy(
                model,
                train_images,
                train_labels,
                100,
                model.num_classes,
            )
            var test_acc = evaluate_model_simple(
                model,
                test_images,
                test_labels,
                batch_size=100,
                num_classes=model.num_classes,
                verbose=False,
            )
            var test_loss = compute_test_loss(
                model, test_images, test_labels, 100, model.num_classes
            )
            var weight_l2 = compute_weight_l2_norm(model)

            print(
                "EPOCH",
                epoch,
                "train_loss=",
                train_loss,
                "train_acc=",
                train_acc * 100.0,
                "test_loss=",
                test_loss,
                "test_acc=",
                test_acc * 100.0,
                "weight_l2=",
                weight_l2,
            )

            if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
                var metric_values = Dict[String, Float64]()
                metric_values["train_loss"] = Float64(train_loss)
                metric_values["train_acc"] = Float64(train_acc)
                metric_values["test_loss"] = Float64(test_loss)
                metric_values["test_acc"] = Float64(test_acc)
                metric_values["weight_l2_norm"] = weight_l2
                checkpointer.update(model, epoch, metric_values)

    # Save final model
    print("Saving final model weights...")
    model.save_weights(config.weights_dir + "/final")
    print("  Model saved to", config.weights_dir + "/final")
    print()

    print("Training complete!")
