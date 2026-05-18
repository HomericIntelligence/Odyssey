# Configuration System Migration Guide

This guide helps you migrate existing code to use the ML Odyssey configuration system.

## Overview

The configuration system provides centralized management of:

- Training hyperparameters
- Model architectures
- Data processing parameters
- Experiment variations

## Migration Benefits

- **Reproducibility**: Every experiment fully defined by configuration
- **Version Control**: Track parameter changes in git
- **Reusability**: Share configurations across experiments
- **Flexibility**: Override defaults at paper and experiment level
- **Type Safety**: Leverage Mojo's type system for validation

## Migration Process

### Step 1: Identify Hardcoded Parameters

**Before** (hardcoded parameters):

```mojo
fn train_model() raises:
    var learning_rate = 0.001
    var batch_size = 32
    var epochs = 10
    var optimizer = "sgd"

    # Training logic...
```text

**Action**: List all hardcoded values that should be configurable.

### Step 2: Create Configuration File

Create a YAML configuration in `configs/experiments/<paper>/<experiment>.yaml`:

```yaml
# configs/experiments/lenet5/baseline.yaml
experiment:
  name: "lenet5_baseline"
  description: "Baseline LeNet-5 reproduction"
  paper: "lenet5"
  tags: ["baseline", "reproduction"]

optimizer:
  name: "sgd"
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0001

training:
  epochs: 10
  batch_size: 32
  validation_split: 0.1
```text

### Step 3: Load Configuration in Code

**After** (configuration-driven):

```mojo
from projectodyssey.utils.config_loader import load_experiment_config

fn train_model() raises:
    # Load configuration
    var config = load_experiment_config("lenet5", "baseline")

    # Extract parameters
    var learning_rate = config.get_float("optimizer.learning_rate")
    var batch_size = config.get_int("training.batch_size")
    var epochs = config.get_int("training.epochs")
    var optimizer = config.get_string("optimizer.name")

    # Training logic...
```text

### Step 4: Update Model Creation

### Before

```mojo
fn create_model() -> Model:
    var model = Model()
    model.add_layer(ConvLayer(filters=6, kernel_size=5))
    model.add_layer(PoolLayer(size=2))
    model.add_layer(ConvLayer(filters=16, kernel_size=5))
    # ...
    return model
```text

### After

```mojo
from projectodyssey.utils.config_loader import load_experiment_config

fn create_model(config: Config) raises -> Model:
    var model = Model()

    # Load layer configurations from config
    var layers = config.get_list("model.layers")
    for i in range(len(layers)):
        # Parse layer config and add to model
        # (Actual implementation depends on your model structure)
        pass

    return model

fn main() raises:
    var config = load_experiment_config("lenet5", "baseline")
    var model = create_model(config)
```text

### Step 5: Validate Configuration

Add validation to catch errors early:

```mojo
from projectodyssey.utils.config_loader import validate_experiment_config

fn main() raises:
    var config = load_experiment_config("lenet5", "baseline")

    # Validate config before use
    validate_experiment_config(config)

    # Proceed with training
    train_model(config)
```text

## Common Patterns

### Pattern 1: Default Values

Use default values for optional parameters:

```mojo
var dropout = config.get_float("model.dropout", default=0.0)
var use_batch_norm = config.get_bool("model.batch_norm", default=False)
```text

### Pattern 2: Nested Configuration

Access nested values using dot notation:

```mojo
var lr = config.get_float("optimizer.learning_rate")
var momentum = config.get_float("optimizer.momentum")
var step_size = config.get_int("scheduler.step_size")
```text

### Pattern 3: Environment Variables

Use environment variables for deployment flexibility:

```yaml
# Config file
paths:
  data_dir: "${DATA_DIR:-./data}"
  model_dir: "${MODEL_DIR:-./models}"
```text

```mojo
var config = load_experiment_config("lenet5", "baseline")
var config_with_env = config.substitute_env_vars()
var data_dir = config_with_env.get_string("paths.data_dir")
```text

### Pattern 4: Configuration Inheritance

Create variations without duplication:

```yaml
# configs/experiments/lenet5/augmented.yaml
extends:
  - ../../papers/lenet5/model.yaml
  - ../../papers/lenet5/training.yaml

# Override specific parameters
training:
  epochs: 50
  batch_size: 64

data:
  augmentation:
    enabled: true
    random_rotation: 10
```text

## Complete Example

### Before Migration

```mojo
fn main() raises:
    # Hardcoded parameters
    var lr = 0.001
    var batch_size = 32
    var epochs = 10

    # Create model
    var model = create_model()

    # Train
    for epoch in range(epochs):
        # Training loop
        pass
```text

### After Migration

```mojo
from projectodyssey.utils.config_loader import (
    load_experiment_config,
    validate_experiment_config
)

fn main() raises:
    # Load configuration
    var config = load_experiment_config("lenet5", "baseline")

    # Validate
    validate_experiment_config(config)

    # Create model from config
    var model = create_model(config)

    # Train with config parameters
    var epochs = config.get_int("training.epochs")
    for epoch in range(epochs):
        # Training loop using config parameters
        pass
```text

## Migration Checklist

Use this checklist to ensure complete migration:

- [ ] Identified all hardcoded parameters
- [ ] Created configuration file in `configs/experiments/`
- [ ] Moved parameters to configuration
- [ ] Updated code to load configuration
- [ ] Added configuration validation
- [ ] Tested with new configuration system
- [ ] Removed hardcoded values
- [ ] Added documentation for configuration options
- [ ] Committed configuration to version control

## Troubleshooting

### Issue: Configuration file not found

**Error**: `Failed to load YAML file: configs/experiments/lenet5/baseline.yaml`

### Solution

1. Verify file path is correct
1. Check file exists: `ls -la configs/experiments/lenet5/`
1. Ensure file extension is `.yaml` not `.yml`

### Issue: Type mismatch

**Error**: `Type mismatch for key 'training.epochs': expected int but got string`

### Solution

Check YAML file - remove quotes from numbers:

```yaml
# Wrong
training:
  epochs: "10"

# Correct
training:
  epochs: 10
```text

### Issue: Missing required key

**Error**: `Missing required configuration key: optimizer.learning_rate`

### Solution

Add missing key to configuration file:

```yaml
optimizer:
  name: "sgd"
  learning_rate: 0.001  # Add this line
```text

### Issue: Nested configuration not supported

**Note**: Current `Config.from_yaml()` only supports flat key-value pairs.

**Workaround**: Use flattened keys:

```yaml
# Instead of nested
# optimizer
#   learning_rate: 0.001

# Use flattened
optimizer.learning_rate: 0.001
optimizer.momentum: 0.9
```text

**Future**: Full nested object support planned for future release.

## Best Practices

### 1. Start with Defaults

Create sensible defaults in `configs/defaults/`:

```yaml
# configs/defaults/training.yaml
optimizer:
  name: "sgd"
  learning_rate: 0.001
  momentum: 0.9

training:
  epochs: 100
  batch_size: 32
```text

### 2. Document Configuration Options

Add comments to configuration files:

```yaml
# Learning rate for optimizer
# Typical range: 0.0001 - 0.01
optimizer.learning_rate: 0.001

# Number of training epochs
# Paper uses 10, but 50-100 often works better
training.epochs: 10
```text

### 3. Version Control Configurations

- Commit all configuration files
- Tag configs used for published results
- Never modify configs used in completed experiments
- Create new configs for variations

### 4. Use Meaningful Names

- Descriptive experiment names: `baseline`, `augmented`, `pruned_90`
- Clear parameter names: `learning_rate` not `lr`
- Consistent naming: `batch_size` everywhere, not `batch_size` and `batchsize`

### 5. Validate Early

Add validation at the start of your script:

```mojo
fn main() raises:
    var config = load_experiment_config("lenet5", "baseline")
    validate_experiment_config(config)  # Fail fast if invalid

    # Rest of the code...
```text

## Common Pitfalls

### Pitfall 1: Forgetting Environment Variable Substitution

**Problem**: Environment variables not substituted

**Solution**: Call `substitute_env_vars()`:

```mojo
var config = load_experiment_config("lenet5", "baseline")
config = config.substitute_env_vars()  # Don't forget this!
```text

### Pitfall 2: Modifying Loaded Config

**Problem**: Changes to config don't persist

**Solution**: Save config after modifications:

```mojo
var config = load_experiment_config("lenet5", "baseline")
config.set("training.epochs", 20)
config.to_yaml("configs/experiments/lenet5/modified.yaml")  # Save changes
```text

### Pitfall 3: Hardcoding File Paths

**Problem**: Code breaks when directory structure changes

**Solution**: Use config for paths:

```yaml
paths:
  data_dir: "${DATA_DIR:-./data}"
  model_dir: "${MODEL_DIR:-./models}"
  checkpoint_dir: "${MODEL_DIR:-./models}/checkpoints"
```text

## Getting Help

If you encounter issues during migration:

1. Check this migration guide
1. Review `configs/README.md` for configuration system documentation
1. Look at `configs/templates/` for example configurations
1. Examine `papers/_template/examples/train.mojo` for code examples
1. Check `notes/review/configs-architecture.md` for design details

## Next Steps

After successful migration:

1. Run experiments with new configuration system
1. Create configuration variations for different experiments
1. Share configurations with team
1. Document any paper-specific configuration requirements
1. Consider contributing improvements to the configuration system

## Summary

The configuration system migration provides:

- Centralized parameter management
- Reproducible experiments
- Easy experiment variations
- Version-controlled parameters
- Type-safe configuration

Follow this guide step-by-step to migrate your code successfully!
