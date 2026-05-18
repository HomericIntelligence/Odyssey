# ML Odyssey Examples

Complete, runnable examples demonstrating ML Odyssey's features and patterns.

## Overview

This directory contains standalone Mojo code examples referenced throughout the documentation. Each example is:

- **Complete**: Includes all necessary imports and code
- **Runnable**: Can be executed directly with `pixi run mojo run`
- **Documented**: Has clear docstrings explaining what it demonstrates
- **Tested**: Designed to compile and run without errors

## Directory Structure

```text
examples/
├── getting_started/     # Beginner tutorials and quickstart examples
├── mojo_patterns/       # Mojo-specific patterns and idioms
├── custom_layers/       # Building custom neural network components
└── performance/         # Optimization and performance techniques
```text

## Getting Started

### Running Examples

```bash
# Run any example
pixi run mojo run examples/getting_started/quickstart_example.mojo

# Or navigate to examples directory first
cd examples
pixi run mojo run getting_started/quickstart_example.mojo
```text

### Example Categories

#### 1. Getting Started

Introduction to ML Odyssey basics:

- **`quickstart_example.mojo`** - Simple neural network creation and training
- **`first_model_model.mojo`** - 3-layer digit classifier model
- **`first_model_train.mojo`** - Complete training script with callbacks
- **`benchmark_quickstart.mojo`** - Benchmarking neural network training with performance metrics

**Documentation**: [Getting Started Guide](../docs/getting-started/quickstart.md)

#### 2. Mojo Patterns

Mojo-specific ML patterns:

- **`trait_example.mojo`** - Trait-based design (Module, Optimizer)
- **`ownership_example.mojo`** - Ownership and borrowing patterns
- **`simd_example.mojo`** - SIMD vectorization for performance

**Documentation**: [Mojo Patterns Guide](../docs/core/mojo-patterns.md)

#### 3. Custom Layers

Building custom neural network components:

- **`prelu_activation.mojo`** - Parametric ReLU with learnable parameters
- **`attention_layer.mojo`** - Multi-head self-attention mechanism
- **`focal_loss.mojo`** - Custom loss function for imbalanced data

**Documentation**: [Custom Layers Guide](../docs/advanced/custom-layers.md)

#### 4. Performance

Optimization techniques and performance benchmarking:

- **`simd_optimization.mojo`** - SIMD for ReLU, batch norm, matmul
- **`memory_optimization.mojo`** - In-place operations and buffer reuse
- **`benchmark_model_inference.mojo`** - Benchmarking neural network inference performance
- **`benchmark_operations.mojo`** - Benchmarking individual ML operations

**Documentation**: [Performance Guide](../docs/advanced/performance.md)

**Benchmarking Guide**: [Benchmarking Examples](performance/BENCHMARKING.md)

## Benchmarking Examples

The Performance section includes comprehensive benchmarking examples using the `shared.benchmarking`
framework. These examples show how to:

- Measure operation latency with statistical analysis
- Perform warmup iterations for accurate measurements
- Compute percentiles (p50, p95, p99) for latency distribution
- Calculate throughput metrics (operations per second)
- Generate formatted reports for performance comparison
- Analyze scaling behavior across different tensor sizes

### Quick Benchmark Example

```mojo
from projectodyssey.benchmarking import benchmark_function, print_benchmark_report

fn my_operation():
    # Your ML operation here
    var output = relu(input_tensor)

var result = benchmark_function(
    my_operation,
    warmup_iters=10,
    measure_iters=100,
)

print_benchmark_report(result, "ReLU Operation")
```

For detailed benchmarking patterns and best practices, see:
[Benchmarking Guide](performance/BENCHMARKING.md)

## Example File Structure

Each example follows this template:

```mojo
"""Example: [Category] - [Topic]

Brief description of what this example demonstrates.

Usage:
    pixi run mojo run examples/[category]/[filename].mojo

See documentation: docs/[path]/[doc-file].md
"""

from projectodyssey.core import ...  # Imports

# Implementation
struct/fn ...

fn main() raises:
    """Entry point with demonstration code."""
    # Example usage with explanatory comments
    print("Example complete!")
```text

## Contributing Examples

When adding new examples:

1. **Follow the template** - Use consistent structure
1. **Add docstrings** - Explain what the example demonstrates
1. **Keep it focused** - One concept per example
1. **Make it runnable** - Include all necessary code
1. **Link to docs** - Reference the related documentation
1. **Test it** - Verify it compiles and runs

## Example Usage Patterns

### Pattern 1: Simple Execution

```bash
pixi run mojo run examples/getting_started/quickstart_example.mojo
```text

### Pattern 2: With Custom Data

Modify examples to use your own data:

```mojo
# In the example file
var X_train = load_your_data()  # Replace placeholder
var y_train = load_your_labels()
```text

### Pattern 3: Building On Examples

Use examples as starting points:

```bash
cp examples/getting_started/quickstart_example.mojo my_model.mojo
# Edit my_model.mojo with your changes
pixi run mojo run my_model.mojo
```text

## Documentation Links

Each example references its documentation:

- Getting Started: [docs/getting-started/](../docs/getting-started/)
- Core Concepts: [docs/core/](../docs/core/)
- Advanced Topics: [docs/advanced/](../docs/advanced/)

## Common Issues

### Import Errors

If you see import errors:

```bash
# Ensure you're in the repository root
cd /path/to/ProjectOdyssey

# Verify shared library exists
ls src/projectodyssey/

# Run from repository root
pixi run mojo run examples/getting_started/quickstart_example.mojo
```text

### Placeholder Data

Many examples use placeholder data (`Tensor.randn()`) for demonstration. Replace with real data for actual use:

```mojo
# Example placeholder (for demonstration)
var X_train = Tensor.randn(1000, 784)

# Replace with real data (for actual use)
var X_train = load_mnist_data()
```text

## See Also

- [Documentation Index](../docs/index.md) - Full documentation
- [Quickstart Guide](../docs/getting-started/quickstart.md) - Get started in 5 minutes
- [Project Structure](../docs/core/project-structure.md) - Repository organization
- [Testing Strategy](../docs/core/testing-strategy.md) - Testing your code

## License

See [LICENSE](../LICENSE) for licensing information.
