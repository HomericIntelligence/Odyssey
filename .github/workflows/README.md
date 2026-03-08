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

## Workflow Summary

| Workflow | Trigger | Purpose | Duration |
|----------|---------|---------|----------|
| **Test Workflows** | | | |
| [unit-tests.yml](#unit-tests) | PR, push main, manual | Fast unit tests for Mojo/Python | < 5 min |
| [integration-tests.yml](#integration-tests) | PR, push main, manual | Component interaction tests | < 8 min |
| [comprehensive-tests.yml](#comprehensive-tests) | PR, push main, manual | All 112 Mojo tests in 17 groups | < 10 min |
| [test-gradients.yml](#test-gradients) | PR on gradient changes, push main | Backward pass validation | < 5 min |
| [test-data-utilities.yml](#test-data-utilities) | PR/push on data changes | Data loading and processing | < 5 min |
| **Validation Workflows** | | | |
| [script-validation.yml](#script-validation) | PR on scripts, push main, manual | 42 Python scripts validation | < 5 min |
| [validate-configs.yml](#validate-configs) | PR/push on config changes | YAML and schema validation | < 5 min |
| [test-agents.yml](#test-agents) | PR on agent configs, push main | Agent configuration testing | < 3 min |
| [pre-commit.yml](#pre-commit) | PR, push main, manual | Code formatting and linting | < 5 min |
| **Security Workflows** | | | |
| [security-scan.yml](#security-scan) | PR, push main, weekly Monday 9 AM UTC | Dependency, secret, and SAST scanning | < 15 min |
| [link-check.yml](#link-check) | PR on .md, push main, weekly Monday 9 AM UTC | Broken markdown links detection | < 3 min |
| **Performance Workflows** | | | |
| [simd-benchmarks-weekly.yml](#simd-benchmarks-weekly) | Weekly Sunday 2 AM UTC, manual | SIMD performance tracking | < 5 min |

## Detailed Workflow Documentation

### Testing Workflows

#### unit-tests

**File**: `unit-tests.yml`

**Triggers**: Pull requests, pushes to main, manual dispatch

**Purpose**: Fast unit tests for both Mojo and Python code, targeting < 5 minutes total duration.

**Key Features**:

- **Mojo Tests** (`test-mojo` job):
  - Runs individual test files from `tests/unit/` directory
  - Uses pixi-based Mojo execution
  - Handles missing tests gracefully during initial development
  - Reports individual test pass/fail status

- **Python Tests** (`test-python` job):
  - Runs pytest on `tests/unit/` with coverage reporting
  - Generates HTML coverage reports
  - Enforces 80% code coverage threshold (allows 0% during initial development)
  - Uploads coverage artifacts for 7 days

- **Coverage Report** (`coverage-report` job):
  - Combines results from Mojo and Python tests
  - Comments on PRs with test summary
  - Preserves combined report as artifact

**Matrix Strategy**: None (parallel jobs instead)

**Cache Strategy**:

- Pixi environments: Key based on `pixi.toml`
- Python pip cache: Standard pip caching

**PR Comments**: Yes - Posts combined test results with pass/fail status

**Artifacts**:

- `mojo-test-results/` (7 days)
- `python-coverage/` (7 days) - includes HTML coverage reports
- `coverage-report` (30 days)

**Failure Handling**:

- Continues running if Mojo tests fail (to see Python results)
- Combined report fails if any test fails
- Graceful handling of missing tests during setup phase

---

#### integration-tests

**File**: `integration-tests.yml`

**Triggers**: Pull requests (non-draft), pushes to main, manual dispatch

**Purpose**: Test component interactions and integration between modules.

**Key Features**:

- **Matrix Strategy**: 3 parallel test suites
  - `mojo-integration`: Tests in `tests/integration/` directory
  - `python-integration`: Agent and foundation structure tests
  - `shared-integration`: Tests in `tests/shared/integration/` directory

- **Smart Discovery**:
  - Case-based pattern matching routes to correct test suite
  - Each suite handles missing tests gracefully
  - Detailed logging of test discovery process

- **Draft PR Skipping**: Uses `github.event.pull_request.draft == false` to skip draft PRs

- **Integration Report** (`integration-report` job):
  - Aggregates results from all 3 suites
  - Counts passed/failed suites
  - Comments on PRs with detailed results
  - Uploads report for 30 days

**Cache Strategy**:

- Pixi environments (key: pixi.toml)
- Test fixtures (key: fixtures directory hash)

**PR Comments**: Yes - Integration test summary

**Artifacts**:

- `integration-results-*` (7 days, one per suite)
- `integration-failures-*` (14 days if failed)
- `integration-report` (30 days)

**Failure Handling**:

- `fail-fast: false` - all suites run even if one fails
- Individual suite failures captured in report
- Overall workflow fails if any suite fails

---

#### comprehensive-tests

**File**: `comprehensive-tests.yml`

**Triggers**: Pull requests, pushes to main, manual dispatch

**Purpose**: Run all 112+ Mojo tests organized into 17 logical groups for comprehensive coverage.

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

### Validation Workflows

#### script-validation

**File**: `script-validation.yml`

**Triggers**: PR when Python scripts change, pushes to main, manual dispatch

**Purpose**: Comprehensive validation of 42 Python scripts including syntax, linting, and formatting.

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

### Security Workflows

#### security-scan

**File**: `security-scan.yml`

**Triggers**: PR, pushes to main, weekly Monday 9 AM UTC, manual dispatch

**Purpose**: Comprehensive security scanning including dependencies, secrets, and code analysis.

**Scanning Jobs**:

1. **dependency-scan** - Safety and OSV Scanner for vulnerabilities
2. **secret-scan** - Gitleaks for exposed secrets
3. **sast-scan** - Semgrep for source code analysis
4. **supply-chain-scan** - Dependency review for critical issues
5. **security-report** - Aggregates all results and comments on PRs

**Critical Failure Conditions**:

- Secrets detected (exit 1)
- Critical dependency vulnerabilities (exit 1)

**Artifacts**:

- `dependency-scan-results/` (30 days)
- `security-report` (90 days)

**PR Comments**: Yes - Security scan status and recommendations

---

#### dependency-audit

**File**: `dependency-audit.yml`

**Triggers**: Weekly Monday 8 AM UTC, push to main when dependency files change, manual dispatch

**Purpose**: Comprehensive dependency audit including Python, Pixi, and license compliance.

**Audit Jobs**:

1. **python-audit** - Safety and pip-audit for vulnerability scanning
2. **pixi-audit** - Pixi/Conda package listing and version tracking
3. **license-audit** - License compliance checking (blocks GPL-3.0, AGPL-3.0)
4. **audit-report** - Aggregates results and creates issues for vulnerabilities

**Artifacts**:

- `python-audit-results/` (30 days)
- `pixi-audit-results/` (30 days)
- `license-audit-results/` (30 days)
- `dependency-audit-report` (90 days)

**Issue Creation**: Automatically creates issue if vulnerabilities found

---

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

### Performance Workflows

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

## Common Patterns

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
  run: just test-group "tests/shared/core" "test_*.mojo"

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

### Pixi-Based Environment Setup

All workflows use the shared composite action for Pixi setup:

```yaml
- name: Set up Pixi
  uses: ./.github/actions/setup-pixi
```

This composite action installs Pixi and caches both `.pixi` and `~/.cache/rattler/cache` using
an explicit `actions/cache@v5` step keyed on `pixi.lock`. This replaces the older inline setup
pattern with `cache: true` (which was unreliable) and the `modular` CLI approach.

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
```text

### PR Comments with Results

Test workflows automatically comment on PRs:

```yaml
- name: Comment on PR
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v8
  with:
    script: |
      # Update or create bot comment with test results
```text

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
```text

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

- 17 test groups × 10 minutes each = 170 minutes

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
```text

---

## Maintenance Notes

### Weekly Tasks

- **Monday 9 AM UTC**: Security scan and link check run automatically
- **Sunday 2 AM UTC**: SIMD benchmarks run for performance tracking

### Monthly Tasks

- Review security report artifacts for trends
- Check benchmark artifacts for performance degradation
- Update workflow dependencies (actions versions)

### Pixi Configuration

All workflows depend on `pixi.toml` at repository root. Key points:

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
```text

### Trigger Workflow Manually

```bash
# Trigger workflow_dispatch
gh workflow run script-validation.yml

# With branch specification
gh workflow run script-validation.yml --ref main
```text

### Check Workflow Status

```bash
# View all workflows
gh workflow list

# View specific workflow status
gh workflow view script-validation.yml
```text

---

## Related Documentation

- **Pixi Setup**: See `pixi.toml` for environment configuration
- **Pre-commit Hooks**: See `.pre-commit-config.yaml` for local validation
- **Agent System**: See `.claude/agents/` for AI agent configuration testing
- **Security Policy**: See `SECURITY.md` for vulnerability reporting
- **Development Guide**: See `CLAUDE.md` for development workflow and best practices

---

**Last Updated**: 2025-11-22
**Maintained By**: ML Odyssey Team
