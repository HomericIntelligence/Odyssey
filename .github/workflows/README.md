# GitHub Actions Workflows

This directory contains all CI/CD workflows for the ML Odyssey project. The workflows are organized into categories based
on their purpose: testing, validation, security, and performance monitoring.

## Overview

The CI/CD strategy uses GitHub Actions with the following principles:

1. **Pixi-based Setup**: All workflows use Pixi for environment management instead of modular/setup-mojo
2. **Justfile Integration**: Workflows use justfile recipes for consistency between local and CI environments
3. **Parallel Execution**: Test workflows use matrix strategies for parallelization
4. **Fail-Fast Control**: Strategic use of `fail-fast: false` allows complete test runs without early stopping
5. **Artifact Preservation**: Test results and reports are uploaded for 7-30 days for analysis
6. **PR Comments**: Test summaries automatically comment on PRs for quick feedback
7. **Scheduled Runs**: Weekly security and benchmark runs ensure ongoing system health

## Merge Queue Readiness

The repository-side merge queue contract is
[`configs/github/merge-queue-policy.json`](../../configs/github/merge-queue-policy.json).
It is the only source of truth for the required check contexts and queue rule. Inspect it
without copying policy values into another document:

```bash
POLICY=configs/github/merge-queue-policy.json
jq -r '.required_contexts[]' "$POLICY"
jq '.merge_queue_rule' "$POLICY"
```

The workflows that emit required contexts all handle `merge_group` with the
`checks_requested` activity while retaining their existing pull request and push triggers:

- `_required.yml`
- `comprehensive-tests.yml`
- `pre-commit.yml`
- `workflow-smoke-test.yml`

The publishing workflow in `release.yml` remains tag/manual-only and must not run for merge
groups. Focused executable coverage lives in
`tests/smoke/test_merge_queue_workflow_properties.py` and verifies trigger boundaries, exact
policy values, and one-to-one required-context emission.

This repository does not activate or mutate live GitHub rulesets. The Odysseus rollout tracked
by `HomericIntelligence/Odysseus#386` is the sole activation authority. That operator must:

1. Snapshot every active `main` ruleset and retain the pre-edit activation-ruleset snapshot as
   the rollback payload.
2. Refuse activation unless the union of live required contexts exactly equals
   `.required_contexts` and the target ruleset has no existing `merge_queue` rule.
3. Append exactly `.merge_queue_rule`, read it back, and restore the snapshot if read-back differs.
4. Queue a representative pull request and select actual `merge_group` runs for the four workflows.
5. Confirm each policy context appears exactly once and completes successfully before recording
   the live response, run URLs, and queued merge result on Odyssey issue #5624.

Keep issue #5624 open until activation and queued smoke evidence are recorded. A readiness PR
references the issue with `Refs #5624`; it does not close it.

To get the current workflow count:

```bash
ls .github/workflows/*.yml | wc -l
```

## Workflow Summary

| Workflow | Trigger | Purpose | Duration |
| --- | --- | --- | --- |
| **Test Workflows** | | | |
| [comprehensive-tests.yml](#comprehensive-tests) | PR, merge queue, push main, manual | All Mojo tests in 17 groups | < 10 min |
| [test-gradients.yml](#test-gradients) | PR on gradient changes, push main | Backward pass validation | < 5 min |
| [test-data-utilities.yml](#test-data-utilities) | PR/push on data changes | Data loading and processing | < 5 min |
| [coverage.yml](#coverage) | PR, push main, manual | Code coverage tracking | < 5 min |
| **Validation Workflows** | | | |
| [script-validation.yml](#script-validation) | PR on scripts, push main, manual | Python scripts validation | < 5 min |
| [validate-configs.yml](#validate-configs) | PR/push on config changes | YAML and schema validation | < 5 min |
| [test-agents.yml](#test-agents) | PR on agent configs, push main | Agent configuration testing | < 3 min |
| [validate-workflows.yml](#validate-workflows) | PR, push main on .github/ | Validate workflow checkout order | < 3 min |
| [pre-commit.yml](#pre-commit) | PR, merge queue, push main, manual | Code formatting and linting | < 5 min |
| [type-check.yml](#type-check) | PR on scripts, push main, manual | Python type checking | < 3 min |
| [notebook-validation.yml](#notebook-validation) | PR/push on notebooks | Jupyter notebook validation | < 5 min |
| [paper-validation.yml](#paper-validation) | PR/push on papers/, manual | Paper implementation validation | < 15 min |
| [readme-validation.yml](#readme-validation) | Nightly, PR/push on README | README command validation | < 5 min |
| **Security Workflows** | | | |
| [security.yml](#security) | PR, push main, weekly Monday 8 AM UTC | Secret scan, SAST, dep audit | < 10 min |
| [link-check.yml](#link-check) | PR on .md, push main, weekly Monday 9 AM UTC | Broken markdown links detection | < 3 min |
| **Build & Release Workflows** | | | |
| build-validation.yml | PR, push main | Build package and validate compilation | < 5 min |
| [release.yml](#release) | Tag push (v*), manual | Build and publish releases | < 30 min |
| [docker.yml](#docker) | Push main/tags, PR, manual | Docker image build and publish | < 20 min |
| [docs.yml](#docs) | Push main on docs/, PR on docs/ | Deploy documentation site | < 5 min |
| **Performance Workflows** | | | |
| [benchmark.yml](#benchmark) | Manual, scheduled | Performance benchmarks | < 30 min |
| [simd-benchmarks-weekly.yml](#simd-benchmarks-weekly) | Weekly Sunday 2 AM UTC, manual | SIMD performance tracking | < 5 min |
| precommit-benchmark.yml | Push main, manual | Pre-commit hook performance tracking | < 5 min |
| **Maintenance Workflows** | | | |
| [mojo-version-check.yml](#mojo-version-check) | Weekly Sunday 3 AM UTC, manual | Check for new Mojo releases | < 3 min |
| workflow-smoke-test.yml | PR, merge queue, push main, manual | Workflow validation smoke tests | < 2 min |
| worktree-sync-check.yml | PR | Check if PR branch is significantly behind origin/main | < 2 min |
| **AI/Automation Workflows** | | | |
| [claude.yml](#claude) | Issue/PR comments mentioning @claude | Claude Code agent integration | N/A |
| [claude-code-review.yml](#claude-code-review) | PR opened/synchronized | Automated Claude code review | < 5 min |

## Detailed Workflow Documentation

### Testing Workflows

#### comprehensive-tests

**File**: `comprehensive-tests.yml`

**Triggers**: Pull requests, pushes to main, manual dispatch

**Purpose**: Run all Mojo tests organized into 17 logical groups for comprehensive coverage.

**Key Features**:

- **Matrix Strategy**: 17 parallel test groups covering:
  1. Core: Tensors & Operations
  2. Core: Layers & Activations
  3. Core: Advanced Layers
  4. Core: Legacy Tests
  5. Training: Optimizers & Schedulers
  6. Training: Loops & Metrics
  7. Training: Callbacks
  8. Data: Datasets
  9. Data: Loaders & Samplers
  10. Data: Transforms
  11. Integration Tests
  12. Utils & Fixtures
  13. Benchmarks
  14. Configs
  15. Tooling
  16. Top-Level Tests
  17. Debug & Integration

- **Pattern Matching**: Each group uses glob patterns to discover and run relevant test files

- **Individual Test Execution**: Files executed one-by-one for clarity on which tests pass/fail

- **Test Group Summary**: Each group generates result file with test counts and failures

- **Combined Report** (`test-report` job):
  - Aggregates all 17 group results
  - Calculates total statistics
  - Comments on PRs with detailed breakdown

**Cache Strategy**:

- Pixi environments (key: pixi.toml)

**PR Comments**: Yes - Comprehensive test results with per-group breakdown

**Artifacts**:

- `test-results-*` (7 days, one per group)
- `comprehensive-test-report` (30 days)

**Failure Handling**:

- `fail-fast: false` - all groups run even if one fails
- Overall workflow fails if any tests fail

---

#### test-gradients

**File**: `test-gradients.yml`

**Triggers**: PR when backward/activation/arithmetic files change, pushes to main

**Purpose**: Validate that all backward passes produce correct gradients (numerical accuracy).

**Key Features**:

- **Path-Based Triggering**: Only runs when gradient-related files change
- **Gradient Checking Job**: Runs and validates gradient computation tests
- **Coverage Report Job**: Calculates and displays gradient checking coverage

**Success Criteria**:

- All gradient checks pass with exact match
- Coverage >= 80% (warning only)

---

#### test-data-utilities

**File**: `test-data-utilities.yml`

**Triggers**: PR/push when data utilities or tests change

**Purpose**: Validate data loading, sampling, and transformation pipeline.

**Key Features**:

- **Specific Test Execution**: Tests dataset, loader, transform, and sampler implementations
- **All-in-One Runner**: Includes comprehensive data utilities test

---

#### coverage

**File**: `coverage.yml`

**Triggers**: PR targeting main, pushes to main, manual dispatch

**Purpose**: Track test metrics as a proxy for code coverage until Mojo provides native coverage tooling.

**Key Features**:

- Reports test counts and pass rates as coverage proxies
- Comments on PRs with coverage summary
- `pull-requests: write` permission required

**Artifacts**: None (posts directly to PR comments)

---

### Validation Workflows

#### script-validation

**File**: `script-validation.yml`

**Triggers**: PR when Python scripts change, pushes to main, manual dispatch

**Purpose**: Comprehensive validation of Python scripts including syntax, linting, and formatting.

**Validation Steps**:

1. **Script Discovery** - Counts all `.py` files in `scripts/` directory
2. **Syntax Validation** - Uses `python3 -m py_compile`
3. **Linting with Ruff** - GitHub Actions output format
4. **Format Checking** - Validates code formatting
5. **Import Validation** - Checks for proper path setup
6. **Executable Testing** - Runs scripts with `--help` flag
7. **Shebang Validation** - Verifies `#!/usr/bin/env python3` headers
8. **Common Issues Check** - Detects stderr/print and TODO comments

**Summary Output**: Comprehensive validation checklist with all checks performed

**Failure Help**: Displays local fix instructions including pixi commands

---

#### validate-configs

**File**: `validate-configs.yml`

**Triggers**: PR/push when config YAML files change

**Purpose**: Validate configuration files syntax, structure, and consistency.

**Validation Jobs**:

1. **validate-yaml** - YAML syntax and required files
2. **validate-schemas** - Training config structure
3. **test-config-loading** - Placeholder for Mojo tests
4. **validate-experiments** - Experiment-to-paper references

**Failure Conditions**:

- Invalid YAML syntax
- Missing required default config files
- Invalid training config structure

---

#### test-agents

**File**: `test-agents.yml`

**Triggers**: PR when agent configs change, pushes to main

**Purpose**: Validate agent configuration format, loading, and integration patterns.

**Validation Steps**:

1. Configuration syntax and format
2. Agent discovery and loading
3. Delegation pattern validation
4. Workflow integration testing
5. Mojo-specific pattern validation

---

#### validate-workflows

**File**: `validate-workflows.yml`

**Triggers**: PR, pushes to main on `.github/` directory

**Purpose**: Validate workflow file structure and enforce checkout action ordering requirements.

**Validation Steps**:

1. Checks all `.github/workflows/*.yml` files for syntax
2. Enforces that `actions/checkout` appears before any local action references
3. Validates composite action input/output specifications
4. Ensures required environment variables are documented

---

#### pre-commit

**File**: `pre-commit.yml`

**Triggers**: PR, pushes to main, manual dispatch

**Purpose**: Run pre-commit hooks for code formatting and linting.

**Hook Types**:

- `pixi run mojo format` - Auto-format Mojo code (`.mojo`, `.🔥` files)
- `markdownlint-cli2` - Lint markdown files
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax
- `check-added-large-files` - Prevent files > 1MB
- `mixed-line-ending` - Fix mixed line endings

**Failure Help**: Shows how to run locally and fix issues

---

#### type-check

**File**: `type-check.yml`

**Triggers**: PR on Python scripts or `mypy.ini`, pushes to main on those paths, manual dispatch

**Purpose**: Run mypy static type checking on Python scripts.

**Key Features**:

- Path-triggered to avoid unnecessary runs
- Checks `scripts/**/*.py` against mypy configuration in `mypy.ini`

---

#### notebook-validation

**File**: `notebook-validation.yml`

**Triggers**: PR/push when files in `notebooks/` change

**Purpose**: Validate that Jupyter notebooks are well-formed and can be parsed.

**Key Features**:

- Path-triggered: only runs when notebooks change
- Uses `actions/setup-python` (no Pixi dependency)

---

#### paper-validation

**File**: `paper-validation.yml`

**Triggers**: PR/push when files in `papers/` change, manual dispatch

**Purpose**: Validate paper implementations against specifications (structure, implementation, reproducibility).

**Key Features**:

- Path-triggered: only runs when papers/ changes
- Uses `.github/actions/setup-pixi` composite action for Mojo/Pixi environment

---

#### readme-validation

**File**: `readme-validation.yml`

**Triggers**: Nightly at 3 AM UTC, PR/push when `README.md` or validation script changes

**Purpose**: Validate that commands documented in `README.md` are syntactically correct and available.

**Schedule**:

- Nightly (quick): Syntax checking and command availability
- Weekly (comprehensive): Full command execution

---

### Security Workflows

#### security

**File**: `security.yml`

**Triggers**: PR, pushes to main on dependency files, weekly Monday 8 AM UTC, manual dispatch

**Purpose**: Unified security scanning covering secret detection, SAST, supply chain review, and dependency audit.

**Scanning Jobs (PR/push)**:

1. **secret-scan** - Gitleaks for exposed secrets
2. **sast-scan** - Semgrep for source code analysis
3. **supply-chain-review** - Dependency review for critical issues

**Scanning Jobs (scheduled weekly)**:

1. **python-audit** - Safety and pip-audit for vulnerability scanning
2. **pixi-audit** - Pixi/Conda package listing and version tracking
3. **license-audit** - License compliance checking (blocks GPL-3.0, AGPL-3.0)

**Critical Failure Conditions**:

- Secrets detected (exit 1)
- Critical dependency vulnerabilities (exit 1)

**Artifacts**:

- `python-audit-results/` (30 days)
- `license-audit-results/` (30 days)
- `security-report` (90 days)

> **Note**: This workflow replaces the former `security-scan.yml` and `dependency-audit.yml` files, which were removed in
> the #3149 consolidation pass.

---

#### link-check

**File**: `link-check.yml`

**Triggers**: PR on markdown changes, pushes to main, weekly Monday 9 AM UTC, manual dispatch

**Purpose**: Detect broken markdown links and catch link rot over time.

**Features**:

- Uses Lychee link checker
- Cache enabled with 1-day max age
- 3 retries with 5-second wait between attempts
- 15-second timeout per link
- Excludes `notes/plan/` and `file:///` links

**Failure Handling**: Fails workflow if broken links found

---

### Build & Release Workflows

#### build-validation

**File**: `build-validation.yml`

> **Note**: This file does not currently exist on disk. It was referenced in the #3149 consolidation plan but has not yet
> been created. When added, it should build and validate the shared Mojo package on PRs and pushes to main.

---

#### release

**File**: `release.yml`

**Triggers**: Push of tags matching `v*`, manual dispatch with version input

**Purpose**: Automated release workflow for building, testing, and publishing releases.

**Key Features**:

- Tag-triggered: runs on `v*` tags
- Manual dispatch allows specifying version and pre-release flag
- Uses `.github/actions/setup-pixi` composite action for Mojo environment

---

#### docker

**File**: `docker.yml`

**Triggers**: Push to main or tags (`v*`), PR targeting main, manual dispatch

**Purpose**: Build and publish Docker images to GitHub Container Registry (GHCR).

**Key Features**:

- PRs: build only (no push)
- Main/tags: build and push to GHCR
- Manual dispatch includes optional push flag

---

#### docs

**File**: `docs.yml`

**Triggers**: Push to main when `docs/`, `mkdocs.yml`, or workflow file changes; PR on those paths

**Purpose**: Build and deploy the documentation site.

**Permissions**: `contents: write`, `pages: write`, `id-token: write`

---

### Performance Workflows

#### benchmark

**File**: `benchmark.yml`

**Triggers**: Manual dispatch (primary), scheduled

**Purpose**: Execute performance benchmarks across tensor-ops, model-training, and data-loading suites.

**Key Features**:

- Suite selection via `workflow_dispatch` input (all, tensor-ops, model-training, data-loading)
- Uses `.github/actions/setup-pixi` composite action for Mojo environment
- Target duration: < 30 minutes with parallel execution

---

#### simd-benchmarks-weekly

**File**: `simd-benchmarks-weekly.yml`

**Triggers**: Weekly Sunday 2 AM UTC, manual dispatch

**Purpose**: Track SIMD performance trends over time for performance regression detection.

**Key Features**:

- **Schedule**: Cron `0 2 * * 0` (Sunday 2 AM UTC)
- **Branch Protection**: Only runs on `main` for scheduled runs
- **Benchmark Execution**: Runs `benchmarks/bench_simd.mojo`
- **Summary Generation**: Creates human-readable results with metadata
- **Metrics Extraction**: Creates JSON for trend tracking
- **Long-Term Storage**: Artifacts retained for 365 days

**Performance Guidance**:

- Float32: 3-5x speedup
- Float64: 2-3x speedup
- Larger tensors show better speedup

**Artifacts**:

- `simd-benchmark-results-*` (365 days):
  - `simd-output.txt` - Raw benchmark output
  - `summary.md` - Human-readable summary
  - `metrics.json` - Structured metrics

---

### Maintenance Workflows

#### mojo-version-check

**File**: `mojo-version-check.yml`

**Triggers**: Weekly Sunday 3 AM UTC, manual dispatch

**Purpose**: Check for new Mojo releases and create update issues.

**Key Features**:

- Compares pinned version in `pixi.toml` against latest available
- Creates GitHub issue with upgrade checklist when update available
- Updates existing issue comment if one already exists
- Uses `.mojo-version` file as additional version tracking

**Artifacts**: None (creates issues instead)

---

### AI/Automation Workflows

#### claude

**File**: `claude.yml`

**Triggers**: Issue/PR comments or reviews mentioning `@claude`, issues opened/assigned with `@claude`

**Purpose**: Claude Code agent integration — responds to `@claude` mentions in issues and PRs.

**Key Features**:

- Conditional execution: only runs when `@claude` appears in the triggering content
- No Pixi or Mojo dependency

---

#### claude-code-review

**File**: `claude-code-review.yml`

**Triggers**: PR opened or synchronized

**Purpose**: Automated Claude code review on every PR.

**Key Features**:

- Runs on all PRs (can be filtered by author association)
- No Pixi or Mojo dependency

---

## Common Patterns

### Composite Actions

The `.github/actions/setup-pixi/` composite action wraps `prefix-dev/setup-pixi` and an
explicit `actions/cache@v5` step. All 13 Mojo/Pixi workflows use this composite action
via `uses: ./.github/actions/setup-pixi`.

**Migrated workflows** (completed in #3979):

| Workflow | Category | Notes |
| --- | --- | --- |
| `benchmark.yml` | Performance | Manual + scheduled |
| `comprehensive-tests.yml` | Testing | Primary test suite |
| `mojo-version-check.yml` | Maintenance | Weekly scheduled |
| `paper-validation.yml` | Validation | Path-triggered |
| `pre-commit.yml` | Validation | PR + push |
| `readme-validation.yml` | Validation | Nightly scheduled |
| `release.yml` | Release | Tag-triggered |
| `script-validation.yml` | Validation | PR-triggered |
| `simd-benchmarks-weekly.yml` | Performance | Weekly scheduled |
| `test-data-utilities.yml` | Testing | Path-triggered |
| `test-gradients.yml` | Testing | Path-triggered |
| `type-check.yml` | Validation | PR + push |
| `security.yml` | Security | PR + scheduled |

To verify no inline duplication remains:

```bash
grep -rl "prefix-dev/setup-pixi" .github/workflows/*.yml | wc -l
# Expected: 0
```

### Justfile Integration

All CI workflows use justfile recipes for consistent command execution between local development and CI:

```yaml
# Install Just in workflow (using official GitHub Action - more reliable)
- name: Install Just
  uses: extractions/setup-just@v2

# Use justfile recipes
- name: Build package
  run: just build

- name: Run test group
  run: just test-group "tests/odyssey/core" "test_*.mojo"

- name: Run all tests
  run: just test-mojo
```

**Benefits**:

1. **Reproducibility**: Developers can run `just validate` locally to reproduce CI results
2. **Maintainability**: Complex logic lives in justfile, not scattered across workflow YAML
3. **Consistency**: Identical flags and commands between local and CI environments
4. **Documentation**: Justfile is self-documenting with `just --list`

**Available CI Recipes**:

- `just build` - Build shared package with compilation validation
- `just package` - Compile package (validation only, no output artifact)
- `just test-group PATH PATTERN` - Run specific test group
- `just test-mojo` - Run all Mojo tests
- `just validate` - Full validation (build + test)
- `just pre-commit` - Run pre-commit hooks

**See**: `/justfile` for complete implementation and `CLAUDE.md` for developer documentation.

### SHA-Pinned Action References

**Policy**: All third-party GitHub Actions must be pinned to a full SHA commit hash, not a
mutable tag (`@v4`, `@main`). This prevents supply-chain attacks where a tag is silently moved
to a malicious commit.

**Correct pattern** (SHA-pinned):

```yaml
- name: Checkout code
  uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2

- name: Set up Pixi
  uses: prefix-dev/setup-pixi@ba3bb36eb2066252b2363392b7739741bb777659  # v0.9.3
```

**Incorrect pattern** (mutable tag — do not use):

```yaml
- name: Checkout code
  uses: actions/checkout@v4  # ❌ tag can be moved

- name: Set up Pixi
  uses: prefix-dev/setup-pixi@v0.9.3  # ❌ tag can be moved
```

**Finding the correct SHA**: Use `gh release view` on the action's repository or check the
workflow file after Dependabot pins it:

```bash
gh release view v4.2.2 --repo actions/checkout --json tagName,targetCommitish
```

### `path:` Field Does Not Support Glob Patterns

**Constraint**: The `path:` field in `on.push` and `on.pull_request` trigger blocks does
**not** support glob wildcards. Only the `paths:` array field supports patterns.

**Incorrect** (silently ignored):

```yaml
on:
  push:
    path: "tests/**"   # ❌ 'path:' (singular) is not a valid key — treated as unknown field
```

**Correct**:

```yaml
on:
  push:
    paths:             # ✅ 'paths:' (plural) with array supports glob patterns
      - "tests/**"
      - "src/odyssey/**/*.mojo"
```

If you need to trigger on a single path without globs, still use the `paths:` array form:

```yaml
on:
  push:
    paths:
      - "justfile"
```

### Composite Action Checkout Invariant

**Rule**: `actions/checkout` must appear as a step **before** any `uses: ./.github/actions/` reference
within a job.

**Why**: Local composite actions (`uses: ./.github/actions/X`) are resolved from the repository on
disk. If the repository has not been checked out yet, GitHub Actions cannot find the action directory
and the workflow fails with `Cannot find action './.github/actions/X'`.

**Correct pattern**:

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code          # MUST come first
        uses: actions/checkout@v4

      - name: Set up Pixi            # composite action — safe because checkout already ran
        uses: ./.github/actions/setup-pixi
```

**Incorrect pattern** (will fail):

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Set up Pixi            # ERROR: checkout has not run yet
        uses: ./.github/actions/setup-pixi

      - name: Checkout code
        uses: actions/checkout@v4
```

**Run the check locally**:

```bash
python3 scripts/validate_workflow_checkout_order.py .github/workflows/
```

**CI enforcement**: The `validate-workflows.yml` workflow runs this check automatically on all pull
requests that touch `.github/` files.

### Pixi-Based Environment Setup

All Mojo workflows use the shared composite action:

```yaml
- name: Set up Pixi
  uses: ./.github/actions/setup-pixi
```

The composite action at `.github/actions/setup-pixi/action.yml` installs Pixi via
`prefix-dev/setup-pixi@v0.9.4` and caches `.pixi`, `~/.pixi/bin`, and
`~/.cache/rattler/cache` using an explicit `actions/cache@v5` step keyed on `pixi.lock`.
The explicit cache step is intentional — `cache: true` was found unreliable on v0.9.4.

### Matrix Strategies for Parallelization

Workflows use matrix strategies to parallelize test execution:

```yaml
strategy:
  fail-fast: false  # Continue all jobs even if one fails
  matrix:
    test-group:
      - name: "Group 1"
        path: "path/to/tests"
        pattern: "test_*.mojo"
      - name: "Group 2"
        path: "path/to/tests"
        pattern: "test_*.mojo"
```

### PR Comments with Results

Test workflows automatically comment on PRs:

```yaml
- name: Comment on PR
  if: github.event_name == 'pull_request'
  uses: actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd  # v8
  with:
    script: |
      # Update or create bot comment with test results
```

Each workflow checks for existing bot comments and updates them to avoid duplicate comments.

### Artifact Retention Strategy

Different artifact types have different retention periods:

- Test results: 7 days (quick feedback)
- Failure artifacts: 14 days (detailed investigation)
- Reports: 30 days (trend analysis)
- Security reports: 90 days (compliance)
- Benchmark artifacts: 365 days (historical tracking)

### Conditional Job Dependencies

Jobs use `needs` and `if` conditions to manage dependencies:

```yaml
test-report:
  needs: test-mojo-comprehensive
  if: always()  # Run even if upstream fails
```

The `always()` condition allows report generation even when tests fail.

---

## Troubleshooting

### Tests Failing Locally But Passing in CI

**Issue**: Test passes in CI but fails locally

**Solutions**:

1. Ensure you're using the same Python version: `python3 --version`
2. Update Pixi: `pixi self-update`
3. Clear Pixi cache: `rm -rf ~/.pixi && pixi install`
4. Check for uncommitted changes that affect tests

### Workflows Timing Out

**Issue**: Workflow exceeds timeout duration

**Solutions**:

1. Check for hanging tests: `timeout 60 pixi run pytest tests/test_file.py`
2. Split tests across more matrix jobs
3. Increase timeout-minutes in workflow YAML (was designed for specific durations)

### PR Comments Not Appearing

**Issue**: Test results not commented on PR

**Solutions**:

1. Verify workflow has `pull-requests: write` permission
2. Check if bot comment already exists and workflow tried to update non-existent comment
3. Review workflow logs for JavaScript errors in github-script step

### Secret Scan False Positives

**Issue**: Secret scan flags legitimate values as potential secrets

**Solutions**:

1. Add patterns to Gitleaks config
2. Use `# gitleaks:allow` comment in files (temporary)
3. Configure allowlist in `.gitleaksignore` file

### Pixi Cache Not Working

**Issue**: Pixi environments rebuilt despite cache

**Solutions**:

1. Verify `pixi.toml` hasn't changed unexpectedly
2. Check cache action configuration: `uses: actions/cache@v4`
3. Clear cache: Settings > Actions > Clear all caches

---

## Performance Optimization

### Matrix Job Parallelization

The comprehensive test workflow uses 17 parallel jobs to reduce total duration from ~170 minutes to ~10 minutes.

**Before** (sequential):

- 17 test groups x 10 minutes each = 170 minutes

**After** (parallel):

- All 17 groups run simultaneously = 10 minutes

### Caching Strategy

All workflows implement multi-level caching:

1. **Pixi Cache**: Environment directory (`~/.pixi`)
2. **Pip Cache**: Python packages (`~/.cache/pip`)
3. **Pre-commit Cache**: Hook environments (`~/.cache/pre-commit`)
4. **Test Data Cache**: Fixtures directory
5. **Lychee Cache**: Link check results

### Selective Triggering

Workflows use path-based triggers to avoid unnecessary runs:

```yaml
on:
  push:
    paths:
      - 'scripts/**/*.py'  # Only run script validation when scripts change
      - '.github/workflows/script-validation.yml'
```

---

## Maintenance Notes

### Weekly Tasks

- **Monday 8 AM UTC**: Security scan runs automatically (PR/push: always)
- **Monday 9 AM UTC**: Link check runs automatically
- **Sunday 2 AM UTC**: SIMD benchmarks run for performance tracking
- **Sunday 3 AM UTC**: Mojo version check runs

### Monthly Tasks

- Review security report artifacts for trends
- Check benchmark artifacts for performance degradation
- Update workflow dependencies (actions versions)

### Pixi Configuration

All Mojo workflows depend on `pixi.toml` at repository root. Key points:

- Specify `mojo` version (pinned)
- Include Python dependencies for scripts
- Pin pre-commit tool versions

### Handling Workflow Failures

1. Check specific job logs in Actions tab
2. Reproduce locally with same environment
3. Update workflow YAML if infrastructure changed
4. Test changes in feature branch before merging to main

---

## Adding New Workflows

When adding new workflows:

1. **Name**: Use descriptive name with `.yml` extension
2. **Triggers**: Define clear triggering conditions
3. **Caching**: Add appropriate caching (Pixi, pip, tool-specific)
4. **Timeouts**: Set realistic timeout-minutes
5. **Permissions**: Request minimum required (contents: read, pull-requests: write if commenting)
6. **Documentation**: Add entry to this README with workflow details
7. **Testing**: Test in feature branch before merging
8. **Pixi setup**: Use `uses: ./.github/actions/setup-pixi` (not inline `prefix-dev/setup-pixi`)

---

## Quick Reference

### View Workflow Runs

```bash
# List recent runs
gh run list

# View specific workflow
gh run list --workflow script-validation.yml

# View run details
gh run view <run-id>

# Download artifacts
gh run download <run-id> -n artifact-name
```

### Trigger Workflow Manually

```bash
# Trigger workflow_dispatch
gh workflow run script-validation.yml

# With branch specification
gh workflow run script-validation.yml --ref main
```

### Check Workflow Status

```bash
# View all workflows
gh workflow list

# View specific workflow status
gh workflow view script-validation.yml
```

### Audit Composite Action Usage

```bash
# Count workflows using the composite action
grep -rl "setup-pixi" .github/workflows/*.yml | wc -l

# Verify no inline prefix-dev/setup-pixi remains (should be 0)
grep -rl "prefix-dev/setup-pixi" .github/workflows/*.yml | wc -l

# Count total workflows
ls .github/workflows/*.yml | wc -l
```

---

## Related Documentation

- **Pixi Setup**: See `pixi.toml` for environment configuration
- **Pre-commit Hooks**: See `.pre-commit-config.yaml` for local validation
- **Agent System**: See `.claude/agents/` for AI agent configuration testing
- **Security Policy**: See `SECURITY.md` for vulnerability reporting
- **Development Guide**: See `CLAUDE.md` for development workflow and best practices

---

**Last Updated**: 2026-03-07
**Maintained By**: ML Odyssey Team
