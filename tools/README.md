# Tools Directory

Development utilities and helper tools for ML paper implementation workflows.

## Overview

The `tools/` directory contains focused development utilities that enhance developer productivity during ML paper
implementation. These tools are distinct from automation scripts in `scripts/` and provide practical solutions for
common development tasks.

## Purpose

**Development Support**: Tools that developers use directly during implementation work.

### Key Distinctions

- **`tools/`**: Development utilities used by developers during implementation
- **`scripts/`**: Automation scripts for repository management and CI/CD
- **Claude Code tools**: Built-in IDE functions and commands

## Tool Categories

### 1. Paper Scaffolding (`paper-scaffold/`)

Generate complete directory structures and boilerplate for new paper implementations.

### Available Now

- `scaffold.py` - CLI tool for generating paper structure
- Template system with variable substitution
- Model, training, and test file stubs
- Documentation scaffolding (README, notes)

### Usage

```bash
python tools/paper-scaffold/scaffold.py \
    --paper lenet5 \
    --title "LeNet-5" \
    --authors "LeCun et al." \
    --year 1998 \
    --url "http://..." \
    --output papers/
```text

### 2. Testing Utilities (`test-utils/`)

Reusable testing components for ML implementations.

### Available Now

- `data_generators.mojo` - Tensor generation utilities (Mojo)
- `fixtures.mojo` - Test model fixtures (Mojo)
- Random, zero, and one-filled tensor generation
- SimpleCNN and LinearModel test fixtures

### Future

- Coverage analysis tools
- More sophisticated data distributions

### 3. Benchmarking (`benchmarking/`)

Performance measurement and tracking tools.

### Available Now

- `benchmark.mojo` - Core benchmarking framework (Mojo)
- `runner.mojo` - Benchmark suite runner (Mojo)
- Inference latency measurement
- Throughput calculation
- Warmup iteration support

### Future

- Memory usage tracking
- Performance report generation (JSON/visualization)

### 4. Code Generation (`codegen/`)

Boilerplate and pattern generators for common ML code.

### Available Now

- `mojo_boilerplate.py` - Struct and layer generators (Python)
- `training_template.py` - Training loop generator (Python)
- Linear and Conv2D layer templates
- Customizable optimizer and loss function

### Future

- Data pipeline generators
- Metrics calculation code
- More layer types (RNN, Attention)

## Language Strategy

Following [ADR-001](../docs/adr/ADR-001-language-selection-tooling.md):

- **Mojo** (default): Performance-critical ML utilities, benchmarking, data generation
- **Python** (when justified): Template processing, external tool integration, string manipulation

Each Python tool includes justification per ADR-001 requirements.

## Quick Start

Tools will be added incrementally as needed. Each tool will include:

- Clear documentation with examples
- Simple CLI interface
- Comprehensive test suite
- Usage examples

## Contributing

### Adding a New Tool

1. **Identify Need**: Document the problem being solved
1. **Choose Language**: Follow ADR-001 decision tree
1. **Create Structure**:

   ```text
   tools/<category>/<tool_name>/
   ├── README.md
   ├── <tool>.[py|mojo]
   ├── tests/
   └── examples/
   ```text

1. **Document**: Write clear usage documentation
1. **Test**: Add comprehensive test coverage

### Design Principles

- **KISS**: Keep tools simple and focused
- **YAGNI**: Build only what's needed now
- **Composable**: Tools should work independently
- **Well-Documented**: Clear docs with examples
- **Maintainable**: Clean code with tests

## Directory Structure

```text
tools/
├── README.md            # This file
├── paper-scaffold/      # Paper implementation scaffolding
├── test-utils/          # Testing utilities
├── benchmarking/        # Performance benchmarking
└── codegen/             # Code generation utilities
```text

## Comparison with scripts/

| Aspect | tools/ | scripts/ |
| --- | --- | --- |
| **Users** | Developers | CI/CD, Maintainers |
| **Usage** | Interactive, on-demand | Automated, scheduled |
| **Focus** | Implementation support | Repository management |
| **Examples** | scaffold.py, benchmark.mojo | create_issues.py, build scripts |

## Status

✅ **Active Development**: Basic tools implemented as part of Issue #69. Tools are functional and ready for use,
with future enhancements planned.

### Current Implementation Status

- ✅ **Paper Scaffolding**: Basic scaffolder with templates (Python)
- ✅ **Testing Utilities**: Data generators and fixtures (Mojo)
- ✅ **Benchmarking**: Performance measurement framework (Mojo)
- ✅ **Code Generation**: Struct and layer generators (Python)

## References

- [Issue #67](https://github.com/HomericIntelligence/Odyssey/issues/67): Planning for tools directory
- [ADR-001](../docs/adr/ADR-001-language-selection-tooling.md): Language selection strategy
- [Scripts Directory](../scripts/README.md): Repository automation scripts
- [Project Guidelines](../AGENTS.md): Overall project documentation

---

**Note**: This is a living document that will evolve as tools are added. Focus is on practical utilities that solve
real problems without over-engineering.
