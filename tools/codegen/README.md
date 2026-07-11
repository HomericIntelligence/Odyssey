# Code Generation Tools

Boilerplate and pattern generators for common ML implementation code.

## Available Tools

### 1. Mojo Boilerplate Generator (`mojo_boilerplate.py`)

Generate Mojo structs and neural network layers.

**Language**: Python (justified by template processing, string manipulation)

### Usage

```bash
# Generate a simple struct
python tools/codegen/mojo_boilerplate.py struct Point \
    --fields x:Float64 y:Float64

# Generate a linear layer
python tools/codegen/mojo_boilerplate.py layer Linear \
    --params in_features:784 out_features:10

# Generate a Conv2D layer
python tools/codegen/mojo_boilerplate.py layer Conv2D \
    --params in_channels:1 out_channels:32 kernel_size:3
```text

**Output**: Mojo code printed to stdout (redirect to file as needed)

### 2. Training Template Generator (`training_template.py`)

Generate training loop boilerplate with customizable optimizer and loss function.

**Language**: Python (justified by template processing)

### Usage

```bash
# Generate basic training loop
python tools/codegen/training_template.py

# Generate with custom optimizer and loss
python tools/codegen/training_template.py \
    --optimizer Adam \
    --loss MSE \
    --metrics loss accuracy
```text

**Output**: Complete Mojo training loop code

## Design Principles

- **Simple templates**: Focus on common patterns without over-engineering
- **Customizable**: Allow parameter customization while providing sensible defaults
- **Composable**: Generate code that works with other tools
- **Educational**: Generated code includes TODOs and comments

## Language Justification

Per [ADR-001](../../docs/adr/ADR-001-language-selection-tooling.md):

- **Why Python**: Template processing, string manipulation, no performance requirements
- **Conversion blocker**: Mojo regex not production-ready
- **Review**: Quarterly per ADR-001 monitoring strategy

## Future Enhancements

- Data pipeline generators
- Metrics calculation code
- Backward pass auto-generation from forward definitions
- More layer types (RNN, Attention, etc.)

## References

- [Issue #67](https://github.com/HomericIntelligence/Odyssey/issues/67): Tools planning
- [Issue #69](https://github.com/HomericIntelligence/Odyssey/issues/69): Tools implementation
- [ADR-001](../../docs/adr/ADR-001-language-selection-tooling.md): Language strategy
