# Supporting Directories Integration Guide

## Overview

The five supporting directories (benchmarks/, docs/, agents/, tools/, configs/) provide critical infrastructure for
ML Odyssey. This guide explains how they work together, common integration patterns, and best practices for using them
in combination.

## The Five Supporting Directories

### Quick Reference

| Directory | Purpose | Primary Users | Key Integration Points |
| --- | --- | --- | --- |
| **benchmarks/** | Performance measurement | Developers, CI/CD | papers/, src/projectodyssey/, configs/ |
| **docs/** | User documentation | All users | All directories |
| **agents/** | AI automation | Claude Code | All directories |
| **tools/** | Development utilities | Developers | papers/, configs/, tests/ |
| **configs/** | Configuration management | All users | papers/, benchmarks/, src/projectodyssey/ |

## Integration Patterns

### Pattern 1: New Paper Implementation

**Complete workflow using all supporting directories**

```bash
```bash

# Step 1: Use tools/ to scaffold paper structure

python tools/paper-scaffold/scaffold.py \
    --paper resnet \
    --title "Deep Residual Learning" \
    --authors "He et al." \
    --year 2015

# Step 2: Create configs/ for the paper

cp configs/templates/paper.yaml configs/papers/resnet/model.yaml
cp configs/templates/experiment.yaml configs/experiments/resnet/baseline.yaml

# Step 3: Implement the paper (in papers/resnet/)

# ... implement model.mojo, train.mojo

# Step 4: Add benchmarks/ for performance tracking

# ... create benchmarks/scripts/resnet_benchmark.mojo

# Step 5: Document in docs/

# ... update docs/api/papers/resnet.md

```text

**Directory Interactions**:

1. **tools/** → **papers/** (scaffolding)
2. **configs/templates/** → **configs/papers/** (configuration)
3. **papers/** + **configs/** → Training
4. **benchmarks/** ← **papers/** (performance measurement)
5. **docs/** ← All directories (documentation)

### Pattern 2: Performance Optimization

**Using benchmarks/ and tools/ together**

```bash

```bash

# Step 1: Run benchmarks/ to identify bottleneck
mojo benchmarks/scripts/run_benchmarks.mojo --paper lenet5

# Step 2: Use tools/benchmarking/ for detailed profiling
mojo tools/benchmarking/runner.mojo --target lenet5 --profile

# Step 3: Implement optimization in papers/lenet5/
# ... optimize code based on profiling results

# Step 4: Re-run benchmarks/ to verify improvement
mojo benchmarks/scripts/run_benchmarks.mojo --paper lenet5 --compare

# Step 5: Document in docs/advanced/
# ... document optimization technique

```text

### Directory Interactions

1. **benchmarks/** → Identify bottleneck
1. **tools/benchmarking/** → Profile specific code
1. **papers/** → Implement optimization
1. **benchmarks/** → Verify improvement
1. **docs/advanced/** → Document technique

### Pattern 3: Experiment Management

### Using configs/ across all directories

```bash
```bash

# Step 1: Create experiment config

cp configs/templates/experiment.yaml configs/experiments/lenet5/augmented.yaml

# Step 2: Edit experiment-specific overrides

# configs/experiments/lenet5/augmented.yaml

#   augmentation

#     enabled: true

#     rotation: 15

#     translation: 0.1

# Step 3: Run training with config

mojo papers/lenet5/train.mojo --config experiments/lenet5/augmented

# Step 4: Benchmark the experiment

mojo benchmarks/scripts/run_benchmarks.mojo --experiment lenet5/augmented

# Step 5: Document results

# docs/research/experiments/lenet5_augmentation.md

```text

**Directory Interactions**:

1. **configs/templates/** → **configs/experiments/** (creation)
2. **configs/experiments/** → **papers/** (training)
3. **configs/experiments/** → **benchmarks/** (performance)
4. **Results** → **docs/research/** (documentation)

### Pattern 4: Documentation Creation

**Using agents/ to automate documentation**

```bash

```bash

# Step 1: Agent reviews code structure
# agents/ → analyzes papers/lenet5/

# Step 2: Generate API documentation
# tools/codegen/ → creates API docs from code

# Step 3: Create user guide
# docs/getting-started/lenet5_tutorial.md

# Step 4: Validate documentation
python scripts/validate_links.py docs/

# Step 5: Integrate with docs/
# Update docs/index.md with new links

```text

### Directory Interactions

1. **agents/** → Analyze code structure
1. **tools/codegen/** → Generate API docs
1. **docs/** → Create and organize documentation
1. **scripts/** → Validate documentation quality

### Pattern 5: CI/CD Integration

### All directories working with CI/CD

```yaml
```yaml

# .github/workflows/paper-validation.yml

name: Paper Validation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      # Use tools/ for setup

      - name: Setup environment
      - name: Setup environment

        run: python scripts/setup.py

      # Use configs/ for test configuration

      - name: Load test config
      - name: Load test config

        run: cp configs/defaults/testing.yaml .test_config.yaml

      # Run tests

      - name: Run tests
      - name: Run tests

        run: mojo test tests/papers/lenet5/

      # Use benchmarks/ for performance check

      - name: Run benchmarks
      - name: Run benchmarks

        run: mojo benchmarks/scripts/run_benchmarks.mojo

      # Use agents/ for code review

      - name: AI code review
      - name: AI code review

        run: python scripts/agents/review_pr.py

      # Validate docs/

      - name: Check documentation
      - name: Check documentation

        run: python scripts/validate_links.py docs/

```text

**Directory Interactions**:

1. **scripts/** → CI/CD setup
2. **configs/** → Test configuration
3. **tests/** → Test execution
4. **benchmarks/** → Performance verification
5. **agents/** → Automated code review
6. **docs/** → Documentation validation

## Cross-Directory Dependencies

### Dependency Map

```text

```text

papers/
  ├─> src/projectodyssey/ (uses layers, optimizers)
  ├─> configs/ (loads configurations)
  └─> tools/ (uses utilities)

src/projectodyssey/
  └─> configs/ (config loading utilities)

benchmarks/
  ├─> papers/ (benchmarks implementations)
  ├─> src/projectodyssey/ (benchmarks components)
  └─> configs/ (benchmark configurations)

tools/
  ├─> papers/ (scaffolds papers)
  ├─> configs/ (generates configs)
  └─> tests/ (generates test templates)

configs/
  (no dependencies - foundation layer)

docs/
  (documents all directories)

agents/
  (coordinates all directories)

```text

### Foundation Layer

**configs/** is the foundation - no dependencies:

- Provides configuration to all other directories
- Used by papers/, benchmarks/, tools/
- Loaded by src/projectodyssey/utils/config_loader

### Integration Layer

**tools/** integrates multiple directories:

- Scaffolds papers/
- Generates configs/
- Creates test templates
- Provides utilities for all

### Coordination Layer

**agents/** coordinates all workflows:

- Automates paper implementation
- Manages code review
- Generates documentation
- Orchestrates multi-step workflows

## Common Usage Scenarios

### Scenario 1: Starting a New Paper

**Goal**: Implement a new research paper efficiently

### Steps

1. **Scaffold** with `tools/paper-scaffold/`
1. **Configure** with `configs/templates/`
1. **Implement** in `papers/{name}/`
1. **Test** with `tests/papers/{name}/`
1. **Benchmark** with `benchmarks/scripts/`
1. **Document** in `docs/`

### Integration

- Tools generate structure → Papers implement → Configs provide parameters → Benchmarks measure → Docs explain

### Scenario 2: Optimizing Performance

**Goal**: Improve performance of existing implementation

### Steps

1. **Measure** with `benchmarks/scripts/`
1. **Profile** with `tools/benchmarking/`
1. **Optimize** in `papers/{name}/` or `src/projectodyssey/`
1. **Verify** with `benchmarks/scripts/`
1. **Document** in `docs/advanced/`

### Integration

- Benchmarks identify issue → Tools profile → Code optimization → Benchmarks verify → Docs preserve knowledge

### Scenario 3: Running Experiments

**Goal**: Test variations of a paper implementation

### Steps

1. **Create** experiment config in `configs/experiments/`
1. **Run** training with `papers/{name}/train.mojo`
1. **Benchmark** with `benchmarks/scripts/`
1. **Compare** results across experiments
1. **Document** findings in `docs/research/`

### Integration

- Configs define variations → Papers execute → Benchmarks measure → Docs record results

### Scenario 4: Contributing Documentation

**Goal**: Add or improve documentation

### Steps

1. **Identify** gap in `docs/`
1. **Write** content following markdown standards
1. **Link** from `docs/index.md`
1. **Validate** with `scripts/validate_links.py`
1. **Review** with `agents/` (optional)

### Integration

- Docs created → Scripts validate → Agents review → Team benefits

### Scenario 5: Automating Workflows

**Goal**: Use agents to automate repetitive tasks

### Steps

1. **Define** agent in `.claude/agents/`
1. **Document** in `agents/`
1. **Integrate** with `tools/` or `scripts/`
1. **Configure** behavior with `configs/`
1. **Document** usage in `docs/`

### Integration

- Agents automate → Tools provide capabilities → Configs control behavior → Docs explain usage

## Best Practices

### Integration Best Practices

**1. Use Configs for All Parameters**

Don't hardcode:

```mojo
```mojo

# Bad

var learning_rate = 0.001
var batch_size = 32

# Good

var config = load_experiment_config("lenet5", "baseline")
var learning_rate = config.get_float("optimizer.learning_rate")
var batch_size = config.get_int("training.batch_size")

```text

**2. Benchmark After Changes**

Always run benchmarks after code changes:

```bash

```bash

# After implementing optimization
mojo benchmarks/scripts/run_benchmarks.mojo --compare

```text

**3. Generate Before Implementing**

Use tools to avoid boilerplate:

```bash
```bash

# Generate structure first

python tools/paper-scaffold/scaffold.py --paper new_paper

# Then implement

cd papers/new_paper/

# ... implementation

```text

**4. Document as You Go**

Update documentation with code:

```bash

```bash

# After implementing feature
vim papers/new_paper/README.md  # Update paper docs
vim docs/api/papers/new_paper.md  # Update API docs

```text

**5. Validate Continuously**

Run validation throughout development:

```bash
```bash

# Validate structure

python scripts/validate_structure.py

# Validate docs

python scripts/validate_links.py

# Run tests

mojo test tests/

```text

### Anti-Patterns to Avoid

**Don't: Bypass Configs**

```mojo

```mojo

# Bad - hardcoded parameters
var lr = 0.001

```text

### Do: Use Config System

```mojo
```mojo

# Good - configurable parameters

var config = load_paper_config("lenet5", "training")
var lr = config.get_float("optimizer.learning_rate")

```text

**Don't: Skip Benchmarking**

```bash

```bash

# Bad - deploy without measuring
git commit -m "optimized code" && git push

```text

### Do: Benchmark Before Committing

```bash
```bash

# Good - verify performance

mojo benchmarks/scripts/run_benchmarks.mojo
git commit -m "optimized code (15% faster)" && git push

```text

**Don't: Duplicate Boilerplate**

```bash

```bash

# Bad - manually create every file
mkdir papers/new_paper
touch papers/new_paper/model.mojo
# ... create 10 more files manually

```text

### Do: Use Scaffolding Tools

```bash
```bash

# Good - generate structure

python tools/paper-scaffold/scaffold.py --paper new_paper

```text

**Don't: Document in Isolation**

```text

```text

# Bad - docs not linked or indexed
papers/my_paper/some_notes.txt

```text

### Do: Integrate with Doc System

```text
```text

# Good - proper location and linking

docs/research/my_paper_analysis.md
docs/index.md (with link to analysis)

```text

## Troubleshooting

### Issue: Config Not Loading

**Symptoms**: Configuration values not found or errors loading config

**Check**:

1. File exists at expected path
2. YAML syntax is valid
3. Environment variables are set
4. Config hierarchy is correct

**Solution**:

```bash

```bash

# Validate config syntax
python scripts/lint_configs.py configs/experiments/my_experiment.yaml

# Check environment
echo $ML_ODYSSEY_DATA

# Test loading
mojo -c "from projectodyssey.utils.config_loader import load_experiment_config; var c = load_experiment_config('paper', 'exp')"

```text

### Issue: Benchmark Regression

**Symptoms**: Benchmarks show performance degradation

### Check

1. Recent code changes
1. Configuration changes
1. Dependency updates
1. System resource contention

### Solution

```bash
```bash

# Compare with baseline

mojo benchmarks/scripts/compare_results.mojo \
  --baseline benchmarks/baselines/baseline_results.json \
  --current benchmarks/results/latest_results.json

# Profile to find bottleneck

mojo tools/benchmarking/runner.mojo --profile --target problem_area

# Review recent changes

git diff HEAD~1 papers/affected_paper/

```python

### Issue: Tool Not Found

**Symptoms**: Tool script not executing or import errors

**Check**:

1. Tool exists in correct directory
2. Python path includes tools/
3. Dependencies installed
4. File permissions correct

**Solution**:

```bash

```bash

# Check tool exists
ls -la tools/paper-scaffold/scaffold.py

# Check permissions
chmod +x tools/paper-scaffold/scaffold.py

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:/home/user/ProjectOdyssey/tools"

# Run with explicit path
python /home/user/ProjectOdyssey/tools/paper-scaffold/scaffold.py

```text

### Issue: Documentation Links Broken

**Symptoms**: Link validation errors or 404s in docs

### Check

1. File paths are correct
1. Relative paths used properly
1. Files not moved or renamed
1. Markdown syntax correct

### Solution

```bash
```bash

# Validate all links

python scripts/validate_links.py docs/

# Check specific file

python scripts/validate_links.py docs/path/to/file.md

# Fix and re-validate

vim docs/path/to/file.md  # Fix links
python scripts/validate_links.py docs/path/to/file.md

```text

## Advanced Integration

### Multi-Directory Workflows

**Complex Workflow Example: Paper + Optimization + Documentation**

```bash

```bash

#!/bin/bash
# Complete workflow using all 5 supporting directories

PAPER="resnet"
EXPERIMENT="baseline"

# 1. Tools: Scaffold paper structure
echo "Step 1: Scaffolding paper structure..."
python tools/paper-scaffold/scaffold.py \
    --paper $PAPER \
    --title "Deep Residual Learning" \
    --authors "He et al." \
    --year 2015

# 2. Configs: Create experiment configuration
echo "Step 2: Creating experiment config..."
cp configs/templates/experiment.yaml configs/experiments/$PAPER/$EXPERIMENT.yaml

# 3. Papers: Implement (manual step)
echo "Step 3: Implement the paper in papers/$PAPER/"
echo "  - Edit papers/$PAPER/model.mojo"
echo "  - Edit papers/$PAPER/train.mojo"
# ... manual implementation

# 4. Benchmarks: Establish baseline
echo "Step 4: Running initial benchmarks..."
mojo benchmarks/scripts/run_benchmarks.mojo --paper $PAPER > benchmarks/baselines/${PAPER}_baseline.json

# 5. Tools: Profile for optimization
echo "Step 5: Profiling for optimization..."
mojo tools/benchmarking/runner.mojo --target $PAPER --profile

# 6. Papers: Optimize based on profiling
echo "Step 6: Implement optimizations..."
# ... manual optimization

# 7. Benchmarks: Verify improvement
echo "Step 7: Verifying performance improvement..."
mojo benchmarks/scripts/run_benchmarks.mojo --paper $PAPER --compare

# 8. Docs: Document everything
echo "Step 8: Creating documentation..."
# Create paper documentation
vim docs/api/papers/$PAPER.md
# Create optimization guide
vim docs/advanced/optimization_${PAPER}.md

# 9. Validate all
echo "Step 9: Running validation..."
python scripts/validate_structure.py
python scripts/validate_links.py
mojo test tests/papers/$PAPER/

echo "Workflow complete!"

```text

### Automated Integration with Agents

### Using agents/ to orchestrate workflows

```yaml
```yaml

# agents/workflows/new-paper.yaml

name: New Paper Implementation
description: Complete workflow for implementing a new research paper

steps:

  - name: scaffold
  - name: scaffold

    agent: implementation-specialist
    action: scaffold_paper
    inputs:
      paper_name: "{{ paper }}"
      template: papers/_template/

  - name: configure
  - name: configure

    agent: configuration-specialist
    action: create_configs
    inputs:
      paper_name: "{{ paper }}"
      config_template: configs/templates/

  - name: implement
  - name: implement

    agent: senior-implementation-engineer
    action: implement_model
    inputs:
      paper_path: "papers/{{ paper }}/"
      spec: "{{ specification }}"

  - name: test
  - name: test

    agent: test-engineer
    action: create_tests
    inputs:
      paper_path: "papers/{{ paper }}/"
      coverage_target: 80

  - name: benchmark
  - name: benchmark

    agent: performance-engineer
    action: create_benchmarks
    inputs:
      paper_name: "{{ paper }}"
      benchmark_path: "benchmarks/scripts/"

  - name: document
  - name: document

    agent: documentation-engineer
    action: create_docs
    inputs:
      paper_name: "{{ paper }}"
      docs_path: "docs/api/papers/"

```text

## Summary

### Key Takeaways

1. **Five Directories Work Together**: Each has a specific role but they integrate seamlessly
2. **Configs are Foundation**: All other directories build on configuration system
3. **Tools Automate**: Use tools/ to avoid manual boilerplate
4. **Benchmarks Verify**: Always measure performance impact
5. **Docs Preserve**: Document knowledge for team and future

### Quick Reference

**Need to...**

- **Start new paper** → tools/paper-scaffold/ + configs/templates/
- **Measure performance** → benchmarks/scripts/ + tools/benchmarking/
- **Run experiments** → configs/experiments/ + papers/
- **Find documentation** → docs/ (user) or notes/review/ (architectural)
- **Automate workflows** → agents/ + scripts/

### Next Steps

1. Review STRUCTURE.md (in repository root) for repository organization
2. Check [getting-started/repository-structure.md](../getting-started/repository-structure.md) for team onboarding
3. Read individual directory READMEs for detailed usage
4. Try the example workflows in this guide

## References

- **STRUCTURE.md** (in repository root): Repository structure overview
- **Issue #80**: Package phase documentation
- **benchmarks/README.md**: Benchmarking infrastructure
- **docs/README.md**: Documentation system
- **agents/README.md**: Agent system
- **tools/README.md**: Development utilities
- **configs/README.md**: Configuration management

---

**Last Updated**: 2025-11-16
**Maintained By**: Documentation Specialist
