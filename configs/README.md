# Configuration System

Centralized configuration management for ML Odyssey experiments.

## Directory Structure

```text
configs/
├── defaults/           # Default configurations for all experiments
│   ├── training.yaml   # Training parameters (optimizer, batch size, etc.)
│   ├── model.yaml      # Generic model settings
│   ├── data.yaml       # Data loading and preprocessing
│   └── paths.yaml      # Path management with environment variables
├── papers/             # Paper-specific configurations
│   └── lenet5/         # LeNet-5 example
│       ├── model.yaml  # LeNet-5 architecture
│       ├── training.yaml # LeNet-5 training setup
│       └── data.yaml   # MNIST data configuration
├── experiments/        # Experiment variations
│   └── lenet5/
│       ├── baseline.yaml    # Exact paper reproduction
│       └── augmented.yaml   # With data augmentation
├── schemas/            # JSON Schema validation (future)
└── templates/          # Templates for new papers/experiments
    ├── paper.yaml
    └── experiment.yaml
```text

## Quick Start

### Load an experiment configuration

```mojo
from odyssey.utils.config_loader import load_experiment_config

fn main() raises:
    # Load with 3-level merge: defaults → paper → experiment
    var config = load_experiment_config("lenet5", "baseline")

    # Access configuration values
    var lr = config.get_float("optimizer.learning_rate")
    var batch_size = config.get_int("training.batch_size")
    var epochs = config.get_int("training.epochs")
```text

### Load paper configuration

```mojo
from odyssey.utils.config_loader import load_paper_config

fn main() raises:
    # Load paper config with defaults merged
    var config = load_paper_config("lenet5", "training")
    var lr = config.get_float("optimizer.learning_rate")
```text

### Load defaults only

```mojo
from odyssey.utils.config_loader import load_default_config

fn main() raises:
    var training_defaults = load_default_config("training")
    var default_lr = training_defaults.get_float("optimizer.learning_rate")
```text

## Configuration Hierarchy

Configurations use a **3-level merge pattern**:

1. **Defaults** (`configs/defaults/*.yaml`) - Base settings for all experiments
1. **Paper** (`configs/papers/<paper>/*.yaml`) - Paper-specific overrides
1. **Experiment** (`configs/experiments/<paper>/<experiment>.yaml`) - Experiment variations

Later levels override earlier levels, allowing fine-grained control.

## Environment Variables

Use environment variables for path configuration:

```yaml
paths:
  data_dir: "${ML_ODYSSEY_DATA:-./data}"
  checkpoint_dir: "${ML_ODYSSEY_CHECKPOINTS:-./checkpoints}"
```text

Format: `${VAR_NAME:-default_value}`

## Creating a New Paper

1. Create paper directory

   ```bash
   mkdir -p configs/papers/my_paper
   mkdir -p configs/experiments/my_paper
   ```text

1. Copy templates

   ```bash
   cp configs/templates/paper.yaml configs/papers/my_paper/model.yaml
   cp configs/templates/experiment.yaml configs/experiments/my_paper/baseline.yaml
   ```text

1. Edit configurations with paper-specific values

1. Load in code

   ```mojo
   var config = load_experiment_config("my_paper", "baseline")
   ```text

## Configuration Format

All configs use YAML format:

- **2-space indentation**
- **Descriptive comments**
- **`extends` field** for inheritance
- **Environment variables** with `${VAR:-default}` syntax

### Example

```yaml
# configs/experiments/lenet5/baseline.yaml
experiment:
  name: "lenet5_baseline"
  description: "Baseline LeNet-5 reproduction"
  paper: "lenet5"
  tags: ["baseline", "reproduction"]

extends:
  - "papers/lenet5/model.yaml"
  - "papers/lenet5/training.yaml"
  - "papers/lenet5/data.yaml"
```text

## Migration Guide

See [MIGRATION.md](MIGRATION.md) for step-by-step migration from hardcoded parameters.

## Validation

Configurations are validated in CI/CD:

```bash
# Validate YAML syntax
yamllint configs/

# Run config loading tests
mojo test tests/configs/
```text

## Schema Validation (Future)

JSON Schema validation will be added in `configs/schemas/` to provide:

- Type checking
- Required field validation
- Value range constraints
- Clear error messages

## Related Files

- [MIGRATION.md](MIGRATION.md) - Migration guide
- [src/odyssey/utils/config_loader.mojo](../shared/utils/config_loader.mojo) - Loading utilities
- [papers/_template/examples/train.mojo](../papers/_template/examples/train.mojo) - Usage example

## Advanced Usage

### Environment Variable Substitution

Configuration files support environment variables with defaults:

```yaml
# In configs/defaults/paths.yaml
checkpoint_dir: "${ML_ODYSSEY_CHECKPOINTS:-./checkpoints}"
log_dir: "${ML_ODYSSEY_LOGS:-./logs}"
data_dir: "${ML_ODYSSEY_DATA:-./data}"
```text

### Configuration Validation

Use the linting tool to validate configurations:

```bash
# Validate all configs
python scripts/lint_configs.py configs/

# Validate specific file
python scripts/lint_configs.py configs/experiments/lenet5/baseline.yaml

# Verbose output for debugging
python scripts/lint_configs.py -v configs/
```text

### Creating New Configurations

1. **For a new paper implementation:**

   ```bash
   # Create directory
   mkdir -p configs/papers/resnet

   # Copy templates
   cp configs/templates/paper.yaml configs/papers/resnet/model.yaml
   cp configs/defaults/training.yaml configs/papers/resnet/training.yaml

   # Edit with paper-specific values
   vim configs/papers/resnet/model.yaml
   ```text

1. **For a new experiment:**

   ```bash
   # Copy template
   cp configs/templates/experiment.yaml configs/experiments/resnet/improved.yaml

   # Edit to override specific values
   vim configs/experiments/resnet/improved.yaml
   ```text

### Performance Tips

1. **Use flat configurations** - The current parser is optimized for flat key-value pairs
1. **Leverage defaults** - Only override what's different from defaults
1. **Cache loaded configs** - Reuse parsed configurations when possible
1. **Validate early** - Run linting before training to catch issues

### Troubleshooting

**Issue**: Configuration not loading

- Check file path exists
- Validate YAML syntax with linter
- Ensure proper indentation (2 spaces)

**Issue**: Environment variable not resolved

- Export the variable: `export ML_ODYSSEY_DATA=/path/to/data`
- Or use the default value after `:-` in the config

**Issue**: Value type mismatch

- Ensure numbers don't have quotes
- Use proper boolean values: `true/false` not `"true"/"false"`

## See Also

- [Configuration Best Practices](BEST_PRACTICES.md) - Guidelines and anti-patterns
- [Configuration Cookbook](COOKBOOK.md) - Ready-to-use recipes
- [Migration Guide](MIGRATION.md) - Migrating from other config systems
