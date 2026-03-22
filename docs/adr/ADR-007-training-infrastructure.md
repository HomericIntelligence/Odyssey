# ADR-007: Training Infrastructure

**Status**: Accepted

**Date**: 2025-12-28

**Decision Owner**: Chief Architect

## Executive Summary

This ADR documents ML Odyssey's training infrastructure for model training loops, configuration
management, and metric tracking. The system provides a flexible, configuration-driven approach
to training that consolidates common patterns from all model implementations.

## Context

### Problem Statement

Training ML models involves many interrelated components:

1. **Training Loop**: Forward pass, loss computation, backward pass, weight updates
2. **Configuration**: Hyperparameters, model settings, training options
3. **Metric Tracking**: Loss, accuracy, learning rate, checkpoints
4. **Data Loading**: Batching, shuffling, prefetching
5. **Evaluation**: Validation, testing, early stopping

Without a unified infrastructure, each model reimplements these components, leading to
duplication and inconsistency.

### Requirements

1. **Unified Training Loop**: Single implementation used by all models
2. **Configuration-Driven**: All settings controlled via config files
3. **Metric Tracking**: Automatic loss and accuracy tracking
4. **Callback Support**: Extensible lifecycle hooks
5. **Gradient Management**: Proper zeroing, clipping, accumulation
6. **Checkpoint Support**: Save and resume training

## Decision

### Training Loop Architecture

**Core Training Step**:

```mojo
fn training_step(
    model_forward: fn(AnyTensor) raises -> AnyTensor,
    compute_loss: fn(AnyTensor, AnyTensor) raises -> AnyTensor,
    optimizer_step: fn() raises -> None,
    zero_gradients: fn() raises -> None,
    data: AnyTensor,
    labels: AnyTensor,
) raises -> Float64:
    # 1. Zero gradients from previous step
    zero_gradients()

    # 2. Forward pass
    var predictions = model_forward(data)

    # 3. Compute loss
    var loss_tensor = compute_loss(predictions, labels)
    var loss_value = loss_tensor.item()

    # 4. Backward pass (implicit through autograd)
    # loss_tensor.backward()

    # 5. Update weights
    optimizer_step()

    return loss_value
```

**TrainingLoop Struct**:

```mojo
struct TrainingLoop:
    var log_interval: Int
    var clip_gradients: Bool
    var max_grad_norm: Float64

    fn run_epoch(
        self,
        model_forward: fn(AnyTensor) raises -> AnyTensor,
        compute_loss: fn(AnyTensor, AnyTensor) raises -> AnyTensor,
        optimizer_step: fn() raises -> None,
        zero_gradients: fn() raises -> None,
        mut train_loader: DataLoader,
        mut metrics: TrainingMetrics,
    ) raises

    fn run(
        self,
        # ... same params ...
        num_epochs: Int,
        mut metrics: TrainingMetrics,
    ) raises
```

### Configuration System

**TrainerConfig**:

```mojo
struct TrainerConfig:
    var num_epochs: Int
    var batch_size: Int
    var learning_rate: Float64
    var log_interval: Int
    var validate_interval: Int
    var save_checkpoints: Bool
    var checkpoint_interval: Int
    var use_scheduler: Bool
    var scheduler_type: String
    var use_mixed_precision: Bool
    var precision_dtype: DType
    var loss_scale: Float32
    var gradient_clip_norm: Float32
```

**Configuration Loading**:

```mojo
# Load from YAML/JSON config file
var config = load_experiment_config("lenet5", "baseline")

# Create trainer config from loaded config
var trainer_config = create_trainer_config(config)
```

**Example Config File** (`papers/lenet5/configs/baseline.yaml`):

```yaml
model:
  name: lenet5
  num_classes: 10
  input_shape: [1, 28, 28]
  dropout: 0.0
  dtype: float32

training:
  epochs: 10
  batch_size: 64
  log_interval: 100
  validate_interval: 1
  save_checkpoints: true
  checkpoint_interval: 5

optimizer:
  name: sgd
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
```

### Metric Tracking

**TrainingMetrics**:

```mojo
struct TrainingMetrics:
    var current_epoch: Int
    var current_batch: Int
    var train_loss: Float64
    var train_accuracy: Float64
    var val_loss: Float64
    var val_accuracy: Float64
    var best_val_loss: Float64
    var best_val_accuracy: Float64
    var best_epoch: Int

    fn reset_epoch(mut self)
    fn update_train_metrics(mut self, loss: Float64, accuracy: Float64)
    fn update_val_metrics(mut self, loss: Float64, accuracy: Float64)
    fn print_summary(self)
```

**Loss Tracking**:

```mojo
struct LossTracker:
    var window_size: Int
    var values: List[Float32]

    fn update(mut self, value: Float32)
    fn get_average(self) -> Float32
    fn get_recent_average(self) -> Float32  # Last window_size values
```

### DataLoader Interface

**DataLoader**:

```mojo
struct DataLoader:
    var data: AnyTensor
    var labels: AnyTensor
    var batch_size: Int
    var shuffle: Bool
    var current_index: Int
    var num_batches: Int

    fn reset(mut self)
    fn has_next(self) -> Bool
    fn next(mut self) -> DataBatch

struct DataBatch:
    var data: AnyTensor
    var labels: AnyTensor
```

### Manual Batch Processing

For compatibility with existing examples:

```mojo
fn run_epoch_manual(
    self,
    train_data: AnyTensor,
    train_labels: AnyTensor,
    batch_size: Int,
    compute_batch_loss: fn(AnyTensor, AnyTensor) raises -> Float32,
    epoch: Int,
    total_epochs: Int,
) raises -> Float32:
    # Slice data into batches
    # Call compute_batch_loss for each batch
    # Track and report loss
    # Return average loss
```

## Rationale

### Why Callback-Based Architecture?

The training step accepts functions rather than objects:

- **Flexibility**: Any function signature matching the callback works
- **No Trait Requirements**: Models don't need to implement specific traits
- **Testability**: Easy to mock forward/backward passes
- **Simplicity**: No complex inheritance hierarchies

### Why Configuration-Driven?

All training parameters come from config files:

- **Reproducibility**: Same config produces same training run
- **Experiment Management**: Easy to track different hyperparameters
- **No Code Changes**: Tune hyperparameters without modifying code
- **Version Control**: Config files can be committed with experiments

### Why Separate Training Loop?

Consolidated training loop used by all models:

- **DRY Principle**: Single implementation, many users
- **Consistency**: All models follow same training pattern
- **Maintainability**: Fix once, benefit everywhere
- **Best Practices**: Gradient zeroing, loss tracking built-in

## Consequences

### Positive

- **Unified Interface**: All models use same training infrastructure
- **Configuration-Driven**: Easy experiment management
- **Metric Tracking**: Automatic loss and accuracy tracking
- **Extensible**: Callback architecture allows customization
- **Consistent Patterns**: All training follows same structure

### Negative

- **Abstraction Overhead**: Additional layer between model and training
- **Callback Complexity**: Function signatures must match exactly
- **Configuration Parsing**: YAML/JSON parsing overhead

### Neutral

- **Manual Mode**: Legacy run_epoch_manual supports existing patterns
- **No Autograd Integration**: Backward pass is placeholder until autograd complete

## Alternatives Considered

### Alternative 1: Trainer Class with Model Reference

**Description**: Trainer holds reference to model object.

```mojo
struct Trainer:
    var model: Model
    fn train(mut self)
```

**Pros**:

- Object-oriented pattern familiar to PyTorch users
- Simpler API

**Cons**:

- Requires Model trait/interface
- Less flexible than callbacks
- Harder to test

**Why Rejected**: Callback approach is more flexible for Mojo's type system.

### Alternative 2: No Configuration System

**Description**: Pass all parameters as function arguments.

**Pros**:

- No config file parsing
- Simpler implementation

**Cons**:

- Functions have many parameters
- Hard to reproduce experiments
- Config changes require code changes

**Why Rejected**: Configuration management is essential for ML experiments.

### Alternative 3: Framework-Agnostic Training

**Description**: Don't provide training infrastructure, let each model implement.

**Pros**:

- No abstraction overhead
- Maximum flexibility

**Cons**:

- Duplicated code across models
- Inconsistent implementations
- Harder to maintain

**Why Rejected**: DRY principle requires consolidated implementation.

## Implementation Details

### File Locations

```text
shared/training/
├── __init__.mojo              # Package exports
├── loops/
│   ├── __init__.mojo
│   └── training_loop.mojo     # TrainingLoop, training_step
├── trainer.mojo               # Trainer class
├── trainer_interface.mojo     # TrainerConfig, TrainingMetrics
├── metrics.mojo               # LossTracker, AccuracyMetric
└── callbacks.mojo             # Training callbacks (future)

shared/utils/
├── config.mojo                # Config struct
├── config_loader.mojo         # load_experiment_config
└── arg_parser.mojo            # Command-line parsing

papers/_template/
├── configs/
│   └── baseline.yaml          # Example config
└── examples/
    └── train.mojo             # Example training script
```

### Usage Examples

**Configuration-Driven Training**:

```mojo
fn main() raises:
    # Load configuration
    var config = load_experiment_config("lenet5", "baseline")

    # Create model and trainer configs
    var model_config = create_model_config(config)
    var trainer_config = create_trainer_config(config)

    # Create model (using config)
    var model = LeNet5(model_config)

    # Create training loop
    var loop = TrainingLoop(log_interval=trainer_config.log_interval)

    # Create data loader
    var train_loader = DataLoader(train_data, train_labels, trainer_config.batch_size)

    # Create metrics
    var metrics = TrainingMetrics()

    # Run training
    loop.run(
        model.forward,
        cross_entropy_loss,
        optimizer.step,
        model.zero_grad,
        train_loader,
        trainer_config.num_epochs,
        metrics,
    )

    # Print results
    metrics.print_summary()
```

**Manual Batch Processing** (Legacy Pattern):

```mojo
var loop = TrainingLoop(log_interval=100)

for epoch in range(num_epochs):
    var loss = loop.run_epoch_manual(
        train_images,
        train_labels,
        batch_size=128,
        compute_batch_loss=compute_lenet_loss,
        epoch=epoch + 1,
        total_epochs=num_epochs,
    )
```

## References

### Related Files

- `shared/training/loops/training_loop.mojo`: Core training loop
- `shared/training/trainer_interface.mojo`: TrainerConfig, TrainingMetrics
- `papers/_template/examples/train.mojo`: Example training script
- `examples/*/train.mojo`: Model-specific training examples

### Related ADRs

- [ADR-004](ADR-004-testing-strategy.md): Testing strategy includes training validation

### External Documentation

- [PyTorch Training Loop](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Configuration Management](https://hydra.cc/docs/intro/)

## Revision History

| Version | Date       | Author          | Changes     |
| ------- | ---------- | --------------- | ----------- |
| 1.0     | 2025-12-28 | Chief Architect | Initial ADR |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-007-training-infrastructure.md`
- **Status**: Accepted
- **Review Frequency**: As-needed
- **Next Review**: On training infrastructure changes
- **Supersedes**: None
- **Superseded By**: None
