# Tools Integration Guide

This guide explains how the `tools/` directory integrates with the ML Odyssey repository workflow.

## Overview

The tools directory provides development utilities that complement the repository's automation infrastructure:

- **`tools/`**: Interactive development utilities (this directory)
- **`scripts/`**: Repository automation and CI/CD scripts
- **`.claude/agents/`**: Agent configurations and specifications
- **`.github/workflows/`**: CI/CD pipeline definitions

## Integration Points

### 1. Development Workflow Integration

Tools integrate directly into the paper implementation workflow:

```text
Developer Workflow:
1. Create paper structure → tools/paper-scaffold/
2. Write tests → tests/shared/fixtures/ (fixtures, generators)
3. Implement model → (direct Mojo implementation)
4. Benchmark performance → shared/benchmarking/
5. Generate boilerplate → tools/codegen/
```text

### 2. Scripts Directory Integration

Clear separation of concerns between tools and scripts:

| Aspect | tools/ | scripts/ |
|--------|--------|----------|
| **Purpose** | Development utilities | Repository automation |
| **User** | Developers during implementation | CI/CD, maintainers |
| **Usage Pattern** | Interactive, on-demand | Automated, scheduled |
| **Examples** | scaffold.py, benchmark.mojo | create_issues.py, validate_configs.sh |

### Integration Pattern

- Scripts may invoke tools for validation (e.g., `scripts/validate_configs.sh` could use `tools/validation/`)
- Tools do NOT invoke scripts (separation of concerns)
- Both respect repository conventions (language selection, file structure)

### 3. CI/CD Pipeline Integration

Tools can be integrated into GitHub Actions workflows:

#### Example: Benchmarking in CI

```yaml
# .github/workflows/benchmark.yml
- name: Run performance benchmarks
  run: |
    pixi run mojo test benchmarks/
```text

#### Example: Code Generation Validation

```yaml
# .github/workflows/codegen-validation.yml
- name: Validate generated code
  run: |
    python tools/codegen/mojo_boilerplate.py --validate
```text

#### Current Integration Status

**Existing Workflows** (`.github/workflows/`):

- `pre-commit.yml` - Could integrate validation tools
- `test-agents.yml` - Could use test utilities
- `unit-tests.yml` - Leverages tests/shared/fixtures/
- `benchmark.yml` - Uses shared/benchmarking/
- `validate-configs.yml` - Could use validation utilities

### 4. Agent System Integration

Tools support agent-driven workflows (`.claude/agents/`):

### Agent Tool Usage

- **Planning Agents**: Reference tool capabilities in specifications
- **Implementation Agents**: Use tools for code generation and scaffolding
- **Test Agents**: Leverage test utilities and fixtures
- **Package Agents**: Use tools for distribution preparation

### Example Agent Integration

```markdown
## Skills to Use

- [`generate_boilerplate`](../skills/tier-1/generate-boilerplate/SKILL.md)
  → Wraps `tools/codegen/` for agent use
- [`run_tests`](../skills/tier-1/run-tests/SKILL.md)
  → Uses `tests/shared/fixtures/` for test execution
```text

## Usage Scenarios

### Scenario 1: Creating a New Paper Implementation

### Workflow

```bash
# 1. Scaffold paper structure
python tools/paper-scaffold/scaffold.py \
    --paper "AlexNet" \
    --author "Krizhevsky et al." \
    --year 2012 \
    --output papers/alexnet/

# 2. Generate model boilerplate
python tools/codegen/mojo_boilerplate.py \
    --type layer \
    --name Conv2D \
    --output papers/alexnet/model.mojo

# 3. Use test utilities
# (Import fixtures in test files)
from tests.shared.fixtures import generate_batch, ModelFixture
```text

### Scenario 2: Running Benchmarks

### Workflow

```bash
# 1. Run benchmarks
pixi run mojo test benchmarks/

# 2. Use shared benchmarking module in code
from shared.benchmarking import benchmark_function, print_benchmark_report
```text

### Scenario 3: Test-Driven Development

### Workflow

```mojo
// 1. Use test fixtures
from tests.shared.fixtures import create_test_tensor

fn test_forward_pass() raises:
    var batch = create_test_tensor(List[Int](32, 3, 28, 28))
    var model = MyModel()
    var output = model.forward(batch)
    assert output.shape()[0] == 32

// 2. Use shared benchmarking
from shared.benchmarking import benchmark_function

fn test_inference_speed() raises:
    fn run_inference() raises:
        var model = MyModel()
        _ = model.forward(create_test_tensor(List[Int](1, 3, 28, 28)))

    var result = benchmark_function(run_inference, warmup_iters=10, measure_iters=100)
    assert result.mean_latency_ms < 10.0
```text

## Configuration and Setup

### Environment Detection

Tools respect repository configuration:

```bash
# Tools automatically detect
- Repository root (via git)
- Mojo version (via `mojo --version`)
- Python version (via `python3 --version`)
- Available dependencies
```text

### Installation

See [`INSTALL.md`](./INSTALL.md) for complete setup instructions.

Quick setup:

```bash
# Run setup script
python tools/setup/install_tools.py

# Verify installation
python tools/setup/verify_tools.py
```text

## Tool Discovery

### Finding the Right Tool

### Decision Tree

```text
What do you need?
├── Create new paper structure → paper-scaffold/scaffold.py
├── Generate test data → tests/shared/fixtures/data_generators.mojo
├── Create test fixtures → shared/testing/test_models.mojo
├── Measure performance → shared/benchmarking/
│   ├── Benchmark functions → benchmark_function()
│   └── Print reports → print_benchmark_report()
└── Generate code → codegen/
    ├── Mojo structs → mojo_boilerplate.py
    ├── Training loops → training_template.py
    └── Data pipelines → data_pipeline.py
```text

### Tool Catalog

See [`CATALOG.md`](./CATALOG.md) for complete tool listing with examples.

## Best Practices

### 1. Tool Selection

### DO

- Use tools for repetitive tasks (scaffolding, code generation)
- Use tools for performance measurement (benchmarking)
- Use tools for test data generation (consistent, reproducible)

### DON'T

- Use tools for one-off tasks (write code directly)
- Over-complicate simple tasks (KISS principle)
- Rely on tools for critical logic (implement directly in Mojo)

### 2. Integration Guidelines

### When creating new tools

- Follow ADR-001 language selection
- Document integration points
- Provide usage examples
- Add to tool catalog
- Update this integration guide

### When modifying workflows

- Consider tool integration opportunities
- Update documentation
- Test integration end-to-end
- Document in issue-specific README

### 3. Maintenance

### Tool Health Checks

```bash
# Verify all tools are functional
python tools/setup/verify_tools.py --verbose

# Check for dependency updates
python tools/setup/check_dependencies.py
```text

### Quarterly Reviews

- Assess tool usage (which tools are actually used?)
- Update dependencies
- Review Python tools for Mojo conversion
- Archive unused tools

## Troubleshooting

### Tool Not Found

```bash
# Ensure you're in repository root
cd /path/to/ProjectOdyssey

# Verify tool exists
ls tools/paper-scaffold/scaffold.py
```text

### Import Errors

```bash
# Check Mojo path configuration
export MOJO_PATH=/path/to/ProjectOdyssey

# Verify imports in Mojo REPL
pixi run mojo
>>> from tests.shared.fixtures import create_test_tensor
```text

### Permission Issues

```bash
# Make scripts executable
chmod +x tools/*/\*.py

# Or run with Python explicitly
python3 tools/paper-scaffold/scaffold.py
```text

## Examples

### Complete Paper Implementation Workflow

```bash
# 1. Scaffold new paper
python tools/paper-scaffold/scaffold.py \
    --paper "ResNet" \
    --output papers/resnet/

# 2. Generate layer implementations
python tools/codegen/mojo_boilerplate.py \
    --type layer \
    --name ResidualBlock \
    --output papers/resnet/layers.mojo

# 3. Implement model (manual work)
# Edit papers/resnet/model.mojo

# 4. Generate training loop
python tools/codegen/training_template.py \
    --optimizer SGD \
    --output papers/resnet/train.mojo

# 5. Write tests with fixtures
# Use tests/shared/fixtures/ in test files

# 6. Run benchmarks
pixi run mojo test benchmarks/
```text

## References

- [Tools README](./README.md) - Main tools documentation
- [Scripts README](../scripts/README.md) - Automation scripts
- [ADR-001](../docs/adr/ADR-001-language-selection-tooling.md) - Language selection
- [Agent Hierarchy](../agents/hierarchy.md) - Agent system overview
- [CLAUDE.md](../CLAUDE.md) - Project guidelines

---

**Document**: `/tools/INTEGRATION.md`
**Purpose**: Comprehensive integration guide for tools directory
**Audience**: Developers, maintainers, agents
**Status**: Living document (update as tools and workflows evolve)
