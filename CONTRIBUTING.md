# Contributing to ML Odyssey

Thank you for your interest in contributing to ML Odyssey! We welcome contributions from everyone and are grateful
for your help. This document provides guidelines and instructions for contributing to the project.

## Quick Links

- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-style-guidelines)
- [Changelog Guidelines](#changelog-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Review](#code-review)
- [Testing Guidelines](#testing-guidelines)
- [Getting Help](#questions)

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Mojo compiler (v0.26.3 or later)
- Git
- GitHub CLI (`gh`) for PR workflows

### Environment Setup with Pixi

We use [Pixi](https://pixi.sh/) for environment management. This ensures everyone uses the same dependencies.

```bash
# Install Pixi (if not already installed)
# Visit https://pixi.sh/ for installation instructions

# Create and activate the development environment
pixi shell
```

### Verify Your Setup

```bash
# Check Mojo installation
mojo --version

# Check Python installation
python3 --version

# Verify GitHub CLI authentication
gh auth status

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Find or Create an Issue

Before starting work:

- Browse [existing issues](https://github.com/HomericIntelligence/ProjectOdyssey/issues)
- Comment on an issue to claim it before starting work
- Create a new issue if one doesn't exist for your contribution
- Wait for maintainer approval on significant changes

### 2. Branch Naming Convention

Create a feature branch from `main`:

```bash
# Update your local main branch
git checkout main
git pull origin main

# Create a feature branch
git checkout -b <issue-number>-<short-description>

# Examples:
git checkout -b 2654-add-contributing-guidelines
git checkout -b 1928-consolidate-test-assertions
```

**Branch naming rules:**

- Start with the issue number
- Use lowercase letters
- Use hyphens to separate words
- Keep descriptions short but descriptive

### 3. Make Your Changes

- Follow the [Code Standards](#code-style-guidelines) below
- Write tests for new functionality (see [Testing Guidelines](#testing-guidelines))
- Keep commits focused and atomic
- Update documentation as needed

### 4. Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

| Type | Description |
| --- | --- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code restructuring |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |

**Example:**

```bash
git commit -m "feat(core): add SIMD optimization to matmul

Implements #2588 - optimize matrix multiplication with cache tiling
and SIMD vectorization for 10x speedup on large matrices.

Closes #2588"
```

### 5. Push and Create Pull Request

```bash
# Push your branch
git push -u origin <branch-name>

# Create pull request linked to issue
gh pr create --title "[Type] Brief description" --body "Closes #<issue-number>"

# Enable auto-merge (PR will merge when CI passes)
gh pr merge --auto --rebase
```

**Important:** Always enable auto-merge so PRs merge automatically once CI passes and reviews
are complete.

### Never Push Directly to Main

The `main` branch is protected. All changes must go through pull requests:

- Direct pushes to `main` will be rejected
- Even single-line fixes require a PR
- This ensures code review and CI validation

See [CLAUDE.md](CLAUDE.md#-critical-rules---read-first) for complete branch protection rules.

## Running Tests

We follow Test-Driven Development (TDD) principles. Tests should be written before implementation whenever possible.

```bash
# Run all tests
just test

# Run only Mojo tests
just test-mojo

# Run specific test file
mojo test tests/shared/core/test_tensor.mojo

# Run tests for a specific group
just test-group tests/shared/core "test_*.mojo"

# Format code before testing
just format
```

## Code Style Guidelines

### Mojo Code Style

We use `mojo format` for consistent code formatting. Pre-commit hooks will automatically run this on staged files.

**Key principles for Mojo code:**

- Use `fn` over `def` for performance-critical code
- Use `out self` for constructors, `mut self` for mutating methods
- Use `struct` with `@fieldwise_init` for simple data types
- Leverage SIMD operations for performance-critical code
- Add comprehensive docstrings to all public APIs

**Current Mojo syntax (v0.26.3+):**

| Convention | Use For | Example |
| --- | --- | --- |
| `out self` | Constructors | `fn __init__(out self, value: Int)` |
| `mut self` | Mutating methods | `fn modify(mut self)` |
| `self` | Read-only access | `fn get(self) -> Int` |
| `^` operator | Ownership transfer | `return self._data^` |

**Example:**

```mojo
@fieldwise_init
struct Point(Copyable, Movable):
    var x: Float32
    var y: Float32

    fn distance(self, other: Point) -> Float32:
        """Calculate Euclidean distance to another point."""
        var dx = self.x - other.x
        var dy = self.y - other.y
        return (dx * dx + dy * dy).sqrt()
```

For complete Mojo guidelines, see:

- [Mojo Guidelines](.claude/shared/mojo-guidelines.md) - Current syntax reference
- [Mojo Anti-Patterns](.claude/shared/mojo-anti-patterns.md) - Common mistakes to avoid

### Python Code Style

Python is used for automation scripts where Mojo has limitations (subprocess capture, regex processing).

**Requirements:**

- Python 3.7+ compatibility
- Type hints for all function parameters and return values
- Docstrings using Google style
- Format with `black` (included in pre-commit)

**Example:**

```python
def calculate_mean(values: list[float]) -> float:
    """Calculate the mean of a list of values.

    Args:
        values: List of numeric values

    Returns:
        Mean of the values

    Raises:
        ValueError: If values list is empty
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(values) / len(values)
```

### Documentation Style

All documentation files follow markdown standards and must pass `markdownlint-cli2` linting.

**Key rules:**

- Code blocks must have a language specified (` ```python ` not ` ``` `)
- Code blocks must be surrounded by blank lines (before and after)
- Lists must be surrounded by blank lines
- Headings must be surrounded by blank lines
- Lines should not exceed 120 characters
- Use relative links when possible

See [CLAUDE.md](CLAUDE.md#markdown-standards) for complete markdown standards.

### Security Scanning

Python files in `scripts/`, `tests/`, and `tools/` are scanned for security issues using
[Bandit](https://bandit.readthedocs.io/), an AST-based static analysis tool.

**Configuration**: `.bandit` in the project root controls suppressions. Currently suppressed checks:

- `B310` - URL schemes (safe for internal tooling)
- `B202` - `tarfile.extractall` (extraction paths are controlled)
- `B301` - Skipped for test/dev scripts that do not process untrusted data

**Running Bandit manually**:

```bash
# Scan all Python scripts (same flags as pre-commit)
pixi run bandit -ll --skip B310,B202,B301 scripts/ tests/ tools/

# Check a single file
pixi run bandit -ll scripts/my_script.py
```

Bandit runs automatically on every commit via the `bandit` pre-commit hook.

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits. They run:

- `mojo format` - Auto-format Mojo code
- `bandit` - Security scan for Python files (see [Security Scanning](#security-scanning) above)
- `markdownlint-cli2` - Lint markdown files
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Run hooks on all files
just pre-commit-all

# Run hooks on staged files only
just precommit
```

**Hook Failure Policy:**

Pre-commit hooks exist to enforce code quality. **Never bypass them with `--no-verify`.**

When hooks fail:

1. **Read the error** - Hooks tell you exactly what's wrong
2. **Fix the issue** - Update your code to pass the check
3. **Verify locally** - Run `just pre-commit-all` before committing
4. **Commit properly** - Let hooks validate your changes

If a hook itself is broken (not your code):

- Use `SKIP=hook-id git commit` for that specific hook
- Document reason in commit message
- Create issue to fix the broken hook

#### Known Hook Incompatibility: mojo-format on Debian Buster / glibc < 2.32

The `mojo-format` hook requires **glibc 2.32+** (Debian 12+ or Ubuntu 22.04+).
On Debian 10 (Buster) or other hosts with glibc < 2.32, the hook automatically detects
the incompatibility and **skips with a warning** instead of failing your commit.

You will see output like:

```text
WARNING: mojo-format skipped: host glibc is incompatible with Mojo binary.
         Mojo requires GLIBC_2.32+. Your system has an older glibc.
         Files were NOT reformatted. Run inside Docker for full formatting.
         See docs/dev/mojo-glibc-compatibility.md for details.
```

CI always runs on Ubuntu 24.04 (glibc 2.39) and enforces formatting before merge, so your
code will still be format-checked. To format locally on an incompatible host, use Docker:

```bash
just shell
# Inside container:
pixi run mojo format path/to/file.mojo
```

See [docs/dev/mojo-glibc-compatibility.md](docs/dev/mojo-glibc-compatibility.md) for full details.

## Version Bump Process

The project version is defined in four files that must always stay in sync:

| File | Format | Field |
| --- | --- | --- |
| `pyproject.toml` | TOML | `[project] version` (authoritative source) |
| `pixi.toml` | TOML | top-level `version` |
| `mojo.toml` | TOML | top-level `version` |
| `VERSION` | plain text | entire file content |

A pre-commit hook (`check-version-sync`) enforces consistency automatically.
Any commit that touches one of these files but leaves the others out of sync will be rejected.

### How to Bump the Version

Use the `just bump-version` recipe to update all four files atomically:

```bash
just bump-version 0.2.0
```

This updates all four files and then calls `scripts/check_version_sync.py` to verify the
result before you commit.

### Manual Verification

```bash
just check-version-sync
# or directly:
python3 scripts/check_version_sync.py
```

### Never Edit Version Files Individually

Editing only one file creates version drift that the pre-commit hook will catch and block.
Always use `just bump-version`.

## Changelog Guidelines

[CHANGELOG.md](CHANGELOG.md) is a living document. Contributors are expected to update it as part
of their pull request when the change is user-facing.

### When to Add an Entry

Add a changelog entry for changes that affect users of the project:

| Change type | Add entry? | Example |
| --- | --- | --- |
| `feat` | Yes | New layer, new training feature, new CLI flag |
| `fix` | Yes | Bug fix visible to users or downstream consumers |
| Breaking change | Yes | Renamed API, changed behavior, removed feature |
| `refactor` | Usually no | Internal restructuring with no behavior change |
| `test` | No | New or updated tests only |
| `chore` | No | Dependency bumps, CI configuration, tooling |
| `docs` | No | Documentation-only changes |
| `style` | No | Formatting changes only |

When in doubt, ask: "Would a user of this library notice or care?" If yes, add an entry.

### Where to Place Entries

Always add new entries under the `## [Unreleased]` section at the top of `CHANGELOG.md`.
**Never** add entries under a versioned heading — releases are cut by maintainers.

```markdown
## [Unreleased]

### Added

- Your new entry here

### Fixed

- Your fix entry here
```

Use the standard subsection headings: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.

### How to Write Good Entries

Write entries in **user-facing language** — describe the impact, not the implementation.

**Good entries:**

```text
- `Tensor[dtype]` now supports element-wise comparison operators (`==`, `!=`, `<`, `>`)
- Fixed training loop crash when batch size does not divide dataset evenly
- `AnyTensor.__str__` now renders multi-dimensional shapes correctly
```

**Avoid commit-message style:**

```text
- feat(tensor): add comparison ops              ← too terse, no user context
- fix crash in training loop line 42            ← internal detail, not user-facing
- refactor: move AnyTensor to shared/tensor/    ← internal restructuring, skip entirely
```

**Rules of thumb:**

- Start with the affected component or concept, not the verb
- Use present tense ("supports", "renders", "crashes") not past tense
- Include enough context for a user to understand the impact without reading the diff
- One bullet per logical change; split large PRs into multiple bullets if needed

### Quick Checklist

Before opening your PR, confirm:

- [ ] Change is user-facing → added entry under `[Unreleased]`
- [ ] Entry is in user-facing language (not a commit message)
- [ ] Entry is under the correct subsection (`Added`, `Fixed`, etc.)
- [ ] Internal-only changes (`chore`, `test`, `style`) have no changelog entry

## Pull Request Process

### Before You Start

1. Ensure an issue exists for your work (create one if needed)
2. Create a branch from `main` using naming convention: `<issue-number>-<description>`
3. Write tests first following TDD principles
4. Implement your changes
5. Run tests and pre-commit hooks locally: `just test && just pre-commit-all`

### Creating Your Pull Request

```bash
# Push your branch
git push -u origin <branch-name>

# Create pull request linked to issue
gh pr create --title "[Type] Description" --body "Closes #<issue-number>"

# Enable auto-merge
gh pr merge --auto --rebase
```

**PR Requirements:**

- **One PR per issue** - Each GitHub issue gets exactly ONE pull request
- PR must be linked to a GitHub issue
- PR title should be clear and descriptive
- PR description should summarize changes and reference the issue
- Enable auto-merge (`gh pr merge --auto --rebase`)

## Code Review

### What Reviewers Look For

Reviews focus on:

- **Correctness** - Does the code do what it claims?
- **Test coverage** - Are new features tested?
- **Performance** - Is critical code optimized?
- **Code clarity** - Is the code readable and maintainable?
- **Memory safety** - Are ownership and lifetimes correct?
- **API consistency** - Does it follow existing patterns?

### Responding to Review Comments

Address review comments promptly. Reply to EACH comment individually:

```bash
# Get review comment IDs
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments \
  --jq '.[] | {id: .id, path: .path, body: .body}'

# Reply to a specific comment
gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments/COMMENT_ID/replies \
  --method POST \
  -f body="Fixed - [brief description]"
```

**Response format:**

- Keep responses short (1 line preferred)
- Start with "Fixed -" to indicate resolution
- Examples:
  - `Fixed - Updated conftest.py to use real repository root`
  - `Fixed - Removed deprecated section`

### After Review

1. Make requested changes
2. Verify CI passes: `just test && just pre-commit-all`
3. Push changes to update the PR
4. Request re-review if needed

### Merging

Once approved and CI passes:

1. Ensure auto-merge is enabled (`gh pr merge --auto --rebase`)
2. PR will merge automatically when all checks pass
3. Delete the feature branch after merging
4. Clean up local worktree if used

See [PR Workflow](.claude/shared/pr-workflow.md) for complete details.

## Reporting Issues

Use our [issue templates](.github/ISSUE_TEMPLATE/) for consistent reporting.

### Bug Reports

Include:

- Clear title describing the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- Environment details (Mojo version, OS, Python version)
- Relevant code snippets or logs

### Feature Requests

Include:

- Clear title describing the feature
- Problem it solves or value it provides
- Proposed solution (if you have one)
- Alternatives considered

### Security Issues

**Do not open public issues for security vulnerabilities.**

See [SECURITY.md](SECURITY.md) for responsible disclosure process.

## Testing Guidelines

We follow Test-Driven Development principles.

### Two-Tier Testing Strategy

**Tier 1: Layerwise Unit Tests** (Run on every PR)

- Fast, deterministic tests using special FP-representable values
- Tests each layer independently (forward AND backward passes)
- Small tensor sizes to prevent timeouts
- Runtime target: < 12 minutes

**Tier 2: End-to-End Integration Tests** (Run weekly)

- Full model validation with real datasets
- Tests complete forward-backward pipeline
- Validates training convergence

### Writing Tests

1. Write tests before implementation (TDD)
2. Place tests in `tests/` directory mirroring module structure
3. Use descriptive names: `test_function_name_with_scenario`
4. Include docstrings explaining what is being tested

**Example:**

```mojo
fn test_relu_forward():
    """Test ReLU activation forward pass."""
    var input = Tensor[DType.float32]([1.0, -1.0, 0.5, -0.5])
    var output = relu(input)
    assert_equal(output[0], 1.0)
    assert_equal(output[1], 0.0)
    assert_equal(output[2], 0.5)
    assert_equal(output[3], 0.0)
```

### Test Coverage Requirements

- Test new features before implementation
- Test both happy paths and error cases
- Include edge cases
- Validate gradients for parametric layers

See [Testing Strategy](docs/dev/testing-strategy.md) for comprehensive documentation.

## Questions

If you have questions:

1. Check existing documentation in [CLAUDE.md](CLAUDE.md) and [agents/](agents/)
2. Search existing GitHub issues
3. Create a new discussion or issue with your question
4. Contact the maintainers

## Additional Resources

- [Project Architecture](CLAUDE.md) - Complete project reference
- [Mojo Guidelines](.claude/shared/mojo-guidelines.md) - Mojo syntax reference
- [PR Workflow](.claude/shared/pr-workflow.md) - Pull request process
- [Mojo Documentation](https://docs.modular.com/mojo/manual/) - Official Mojo docs

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. We are committed to providing a
welcoming and inclusive environment for all contributors.

## Security

For reporting security vulnerabilities, see [SECURITY.md](SECURITY.md).

---

Thank you for contributing to ML Odyssey! Your effort helps advance AI research and education.
