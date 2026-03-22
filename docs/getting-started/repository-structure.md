# Repository Structure Guide - Team Onboarding

<!-- markdownlint-disable MD024 -->

## Welcome to ML Odyssey

This guide helps you quickly navigate the ML Odyssey repository and understand where to find what you need.

## Quick Start: Finding Your Way

### "I'm New - Where Do I Start?"

### First Steps

1. **Read**: README.md (in repository root) - Project overview
1. **Install**: [installation.md](installation.md) - Set up environment
1. **Try**: [quickstart.md](quickstart.md) - 5-minute introduction

### "I Want to Implement a Paper"

### Workflow

```bash
# 1. Copy the paper template
cp -r papers/_template papers/{name}

# 2. Implement model and training
cd papers/{name}/
# Edit model.mojo, train.mojo

# 3. Write tests (TDD)
# Add tests to tests/papers/{name}/

# 4. Run tests
just test-mojo

# 5. Build
just build
```

**See**: `papers/_template/` for the starting structure

### "I Want to Find Documentation"

**Documentation Locations**:

| Type | Location | Purpose |
| ------ | ---------- | --------- |
| **User Docs** | `docs/` | Tutorials, guides, API reference |
| **Issue Docs** | GitHub issue comments | Issue-specific implementation notes |
| **Architectural** | `notes/review/` | Design decisions, comprehensive specs |
| **ADRs** | `docs/adr/` | Architecture Decision Records |
| **Agent Docs** | `agents/` | Agent system documentation |
| **Code Comments** | Source files | Inline documentation |

**Quick Links**:

- API Reference → `docs/api/`
- Getting Started → `docs/getting-started/`
- Core Concepts → `docs/core/`
- Advanced Topics → `docs/advanced/`

### "I Want to Run Tests"

**Test Locations**:

```bash
# All Mojo tests
just test-mojo

# Specific test group
just test-group tests/shared/core "test_*.mojo"
just test-group tests/papers/lenet5 "test_*.mojo"

# Single test file
pixi run mojo tests/shared/core/test_creation_part1.mojo
```

**See**: `tests/README.md` for testing guidelines

### "I Want to Measure Performance"

### Benchmarking

```bash
# Run benchmarks
pixi run mojo shared/benchmarking/run_benchmarks.mojo

# Run with release build for accurate numbers
just build-release
```

**See**: `shared/benchmarking/` for benchmark scripts

## Repository Organization

### Top-Level Directories

```text
ProjectOdyssey/
├── papers/          # ML paper implementations
├── shared/          # Reusable ML components
├── docs/            # User documentation
├── agents/          # AI agent system docs
├── tests/           # Test suite
├── scripts/         # Automation scripts
├── notes/           # Planning and architectural docs
├── .claude/         # Claude Code configurations
└── .github/         # CI/CD workflows
```

### Core Directories Explained

#### papers/ - Research Implementations

**What**: ML research paper implementations

### Structure

- Each paper in its own directory
- `_template/` for starting new papers

### Example

```text
papers/
├── _template/
│   ├── model.mojo
│   ├── train.mojo
│   └── README.md
```

**When to Use**: Implementing or studying paper implementations

#### shared/ - Reusable Components

**What**: Core ML components used across papers

**Structure**:

- `core/` - AnyTensor type, creation ops, shape ops
- `training/` - Optimizers, training infrastructure
- `autograd/` - Automatic differentiation
- `data/` - Data loading utilities
- `utils/` - Utilities

**When to Use**: Building models, training, loading data

#### Supporting Directories

##### 1. docs/ - User Documentation

**Purpose**: Comprehensive documentation for all users

### Key Sections

- `getting-started/` - Onboarding
- `core/` - Core concepts
- `advanced/` - Advanced topics
- `dev/` - Developer docs
- `adr/` - Architecture Decision Records

### Common Tasks

- Read guides: Browse `docs/`
- Add docs: Create in appropriate section
- Validate: `just pre-commit-all`

#### 2. agents/ - AI Agent System

**Purpose**: Agent hierarchy documentation

### Key Files

- `hierarchy.md` - Agent levels
- `delegation-rules.md` - Coordination
- `templates/` - Agent templates

### Common Tasks

- Understand agents: Read `hierarchy.md`
- Create agent: Use `templates/`
- Find agent configs: Check `.claude/agents/`

#### 3. scripts/ - Automation Scripts

**Purpose**: Python automation for CI and developer workflows

### Common Tasks

```bash
# Run pre-commit hooks on all files
just pre-commit-all

# Run pre-commit hooks on staged files only
just pre-commit
```

#### 4. tests/ - Test Suite

**Purpose**: Mojo and Python tests for all components

```bash
# Run all Mojo tests
just test-mojo

# Run a specific group
just test-group tests/shared/core "test_*.mojo"
```

## Common Workflows

### Workflow 1: I Want to Start a New Paper

### Steps

1. **Scaffold**:

   ```bash
   cp -r papers/_template papers/resnet
   ```

1. **Implement**:

   ```bash
   cd papers/resnet/
   # Edit model.mojo, train.mojo
   ```

1. **Test**:

   ```bash
   just test-mojo
   ```

1. **Build**:

   ```bash
   just build
   ```

1. **Document**:

   ```bash
   # Update papers/resnet/README.md
   ```

### Workflow 2: I Want to Add a Reusable Component

### Steps

1. **Implement**:

   ```bash
   # Add to appropriate location in shared/
   # e.g., shared/core/layers/attention.mojo
   ```

1. **Test**:

   ```bash
   # Add tests in tests/shared/core/layers/
   just test-mojo
   ```

1. **Document**:

   ```bash
   # Update API docs in docs/api/shared/
   ```

### Workflow 3: I Want to Run an Experiment

### Steps

1. **Train**:

   ```bash
   just train lenet-emnist fp32 10
   ```

1. **Run inference**:

   ```bash
   just infer lenet-emnist lenet5_weights
   ```

1. **Document**:

   ```bash
   # Record results in docs/ or GitHub issue comments
   ```

### Workflow 4: I Want to Optimize Performance

### Steps

1. **Measure**:

   ```bash
   just build-release
   pixi run mojo shared/benchmarking/run_benchmarks.mojo
   ```

1. **Optimize**:

   ```bash
   # Edit model or kernel code
   ```

1. **Verify**:

   ```bash
   just build-release
   pixi run mojo shared/benchmarking/run_benchmarks.mojo
   ```

1. **Document**:

   ```bash
   # Update docs/advanced/ with findings
   ```

## Decision Tree: Where Does This Go

### Content Type Decision Tree

```text
ML Paper Implementation?
├─ Yes → papers/{paper_name}/
└─ No → Continue

Reusable ML Component?
├─ Yes → shared/{core|training|data|utils}/
└─ No → Continue

User Documentation?
├─ Yes → docs/{getting-started|core|advanced|dev}/
└─ No → Continue

Test File?
├─ Yes → tests/{foundation|shared|papers|tools}/
└─ No → Continue

Automation Script?
├─ Yes → scripts/
└─ No → Continue

Agent Documentation?
├─ Yes → agents/ (docs) or .claude/agents/ (configs)
└─ No → Continue

Issue-Specific Notes?
├─ Yes → Post as GitHub issue comment (gh issue comment <number>)
└─ No → Continue

Architectural Decision?
├─ Yes → docs/adr/ (for ADRs) or notes/review/ (for specs)
└─ No → Ask in team channel!
```

## Best Practices

### When Adding New Content

1. **Check Existing Structure** - Don't duplicate
2. **Follow Conventions** - Use established patterns
3. **Add READMEs** - Every directory needs documentation
4. **Update Indexes** - Link from appropriate index files
5. **Run Validation** - Check structure and markdown before committing

### When Writing Code

1. **Use Mojo for ML** - Performance-critical code in Mojo
2. **Use Python for Automation** - Tools and scripts (with justification)
3. **Follow TDD** - Write tests first
4. **Document As You Go** - Don't defer documentation
5. **Benchmark Changes** - Verify performance impact

### When Documenting

1. **Choose Right Location**:
   - User-facing → `docs/`
   - Implementation notes → GitHub issue comments
   - Architecture decisions → `docs/adr/`
   - Comprehensive specs → `notes/review/`

2. **Follow Markdown Standards**:
   - Specify language for code blocks
   - Blank lines around blocks and lists
   - Max 120 character lines

3. **Link Related Content**:
   - Cross-reference related docs
   - Link to source code
   - Reference issues and PRs

## Troubleshooting

### "I Can't Find Where to Put My Code"

**Solution**: Use the decision tree above or ask in team channel

**Quick Check**:

- ML implementation? → `papers/` or `shared/`
- Configuration? → `configs/` (if it exists for your paper)
- Test? → `tests/`

### "My Links Are Broken"

**Solution**: Check relative paths manually or run markdown linting

```bash
# Lint all markdown files
just pre-commit-all
```

**Common Issues**:

- Wrong relative path
- File moved or renamed
- Missing file extension

### "Pre-commit Hooks Fail"

**Solution**: Fix the reported issue, then re-run

```bash
# Run hooks on staged files
just pre-commit

# Run hooks on all files
just pre-commit-all
```

## Quick Reference

### Essential Commands

```bash
# Verify environment
pixi run mojo --version

# Build project
just build

# Run all Mojo tests
just test-mojo

# Run specific test group
just test-group tests/shared/core "test_*.mojo"

# Train LeNet-5
just train

# Run inference
just infer lenet-emnist lenet5_weights

# Format code and validate
just pre-commit-all
```

### Essential Files

| File | Purpose |
| ------ | --------- |
| `README.md` | Project overview |
| `CLAUDE.md` | Claude Code conventions |
| `justfile` | All available build/test/train recipes |
| `pixi.toml` | Dependencies and environment (Mojo 0.26.1) |

### Essential Directories

| Directory | Quick Description |
| ----------- | ------------------- |
| `papers/` | ML implementations |
| `shared/` | Reusable components |
| `docs/` | User documentation |
| `tests/` | Test suite |
| `scripts/` | Automation scripts |

## Next Steps

### For New Contributors

1. **Setup**: Follow [installation guide](installation.md)
2. **Verify**: Run `pixi run mojo --version`
3. **Explore**: Read `shared/README.md` for the shared library
4. **Try**: Run `just test-mojo` to verify everything works
5. **Ask**: Use GitHub issues for questions

### For Implementers

1. **Template**: Copy `papers/_template/` to start
2. **Implement**: Write model in `papers/` using `shared/core/`
3. **Test**: Add tests in `tests/papers/` and run `just test-mojo`
4. **Build**: Run `just build` and `just build-release`

### For Documentation Writers

1. **Identify**: Find documentation gaps
2. **Write**: Create in appropriate `docs/` section
3. **Validate**: Run `just pre-commit-all`
4. **Review**: Submit PR for team review

## Getting Help

### Documentation

- **This Guide**: Repository navigation
- **docs/**: Comprehensive documentation
- **agents/**: Agent system documentation

### Team Resources

- GitHub issues for bugs and features
- Pull request reviews for feedback

## Summary

**Key Takeaways**:

1. **Organization**: Repository is logically organized by purpose
2. **Locations**: Use decision tree to find right location
3. **Build system**: All commands go through `just` or `pixi run`
4. **Validation**: Run `just pre-commit-all` before committing

**Remember**:

- `papers/` - Implementations
- `shared/` - Reusable components
- `docs/` - Documentation
- `tests/` - Test suite

**When in Doubt**: Check `just --list` for available commands, or ask in GitHub issues!

---

**Last Updated**: 2026-03-15
**Maintained By**: Documentation Specialist
