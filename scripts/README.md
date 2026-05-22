# Scripts Directory

This directory contains Python automation scripts for the Mojo AI Research Repository project.

## Overview

These scripts automate repository management tasks:

- Markdown and link validation
- Repository structure validation
- Agent system utilities and validation
- Shared utilities and validation framework

**Note**: Planning is now done directly through GitHub issues.
See `.claude/shared/github-issue-workflow.md` for the workflow.

### Deprecated Scripts

The following scripts are deprecated and kept for historical reference:

- `create_issues.py` - Was used with notes/plan/ directory (removed)
- `regenerate_github_issues.py` - Was used with notes/plan/ directory (removed)

### Removed Scripts

The following one-time scripts were removed in #3337 (stale audit follow-up to #3148):

- `execute_backward_tests_merge.py` - One-time branch merge workflow; backward-tests branch already merged
- `merge_backward_tests.py` - Duplicate approach to merge backward-tests branch
- `run_bisect_heap.sh` - Shell wrapper for `bisect_heap_test.py`; same reasoning
- `fix-build-errors.py` - One-time autonomous repair script; targeted build errors already fixed
- `batch_planning_docs.py` - One-time batch doc generation; planning now via GitHub issues directly
- `add_delegation_to_agents.py` - One-time bulk agent attribute addition; completed
- `add_examples_to_agents.py` - One-time bulk example injection into agents; completed
- `document_foundation_issues.py` - One-time script to document foundation issues; completed
- `migrate_odyssey_skills.py` - One-time cross-project skill migration to ProjectMnemosyne; completed

## Directory Structure

```text
scripts/
├── README.md                           # This file
├── common.py                           # Repo-specific constants (LABEL_COLORS, EXCLUDE_DIRS)
├── analyze_issues.py                   # GitHub issue complexity analysis
├── analyze_warnings.py                 # Compiler warning analysis
├── audit_shared_links.py               # Audit .claude/shared/ link-backs in CLAUDE.md
├── bench_precommit.py                  # Pre-commit hook performance benchmarking
├── build_pr_comment.py                 # Build PR comment bodies from test results
├── check_coverage.py                   # Test coverage reporting
├── check_note_format.py                # Validate # NOTE comment format consistency
├── check_runtime_output_patterns.py    # Audit for misleading print() patterns (WARNING:, HACK:, XXX:, Not implemented)
├── check_readmes.py                    # README completeness validation
├── check_test_count_badge.py           # Validate README test count badge accuracy
├── check_zero_warnings.sh              # Shell wrapper: fail on compiler warnings
├── convert_image_to_idx.py             # Convert image files to IDX dataset format
├── create_issues.py                    # GitHub issue creation (deprecated - see note above)
├── download_cifar10.py                 # Download CIFAR-10 dataset
├── download_cifar100.py                # Download CIFAR-100 dataset
├── download_datasets.py                # Download all configured datasets
├── download_emnist.py                  # Download EMNIST dataset
├── download_fashion_mnist.py           # Download Fashion-MNIST dataset
├── download_mnist.py                   # Download MNIST dataset
├── generate_changelog.py               # Generate CHANGELOG.md from git log
├── generate_test_metrics.py            # Generate test metrics report
├── get_stats.py                        # Repository statistics collector
├── get_system_info.py                  # System information collector
├── implement_issues.py                 # Automated issue implementation helper
├── lint_configs.py                     # YAML configuration linting
├── merge_prs.py                        # PR merge automation
├── migrate_odyssey_skills.py           # Skill migration tool (Odyssey2 → ProjectMnemosyne)
├── mojo-format-compat.sh               # GLIBC-compatible mojo format wrapper
├── package_papers.py                   # Papers directory packaging
├── package_utils.sh                    # Package utilities shell script
├── plan_issues.py                      # Issue planning helper
├── plot_training.py                    # Training curve visualization
├── rebase-all-branches.sh              # Rebase all branches against main
├── rebase-all-prs.sh                   # Rebase all pull requests against main
├── regenerate_github_issues.py         # Dynamic issue file generation (deprecated)
├── update_version.py                   # Version bump helper
├── validate_configs.sh                 # Validate configuration files
├── validate_installation_anchors.py    # Validate anchor links to installation.md
├── validate_links.py                   # Markdown link validation
├── validate_mojo_syntax_in_docs.py     # Validate Mojo code blocks in documentation
├── validate_readme_commands.py         # Validate commands referenced in README.md
├── validate_structure.py               # Repository structure validation
├── validate_test_coverage.py           # CI test matrix coverage validation
├── validate_test_file_sizes.py         # Enforce test file size limits
├── validate_workflow_checkout_order.py # Validate actions/checkout precedes local actions
├── build_configs_distribution.sh       # Build configs distribution archive
├── build_data_package.sh               # Build data package
├── build_training_package.sh           # Build training package
├── build_utils_package.sh              # Build utils package
├── bump_version.sh                     # Version bump shell helper
├── install_verify_data.sh              # Install and verify data package
├── install_verify_training.sh          # Install and verify training package
├── install_verify_utils.sh             # Install and verify utils package
└── verify_configs_install.sh           # Verify configs package installation
```

## Scripts

### Shared Modules

#### `common.py`

**Purpose**: Shared utilities and constants used across multiple scripts.

### Provides

- `LABEL_COLORS` - Standard GitHub label colors for 5-phase workflow
- `get_repo_root()` - Portable repository root detection
- `get_agents_dir()` - Get .claude/agents directory path

**Usage**: Imported by other scripts to avoid code duplication.

---

> **Note**: the former `validation.py` shared markdown-validation framework
> was removed in issue #5061 — its functionality now lives in the
> `hephaestus` library (`hephaestus.validation.markdown`). Import directly:
> `from hephaestus.validation.markdown import validate_file_exists, …`.

---

### Main Scripts

#### `validate_test_coverage.py`

**Purpose**: Ensure all test_*.mojo files are discovered and included in the CI test matrix.

**Features**

- Finds all test_*.mojo files in the repository (excluding build artifacts and examples)
- Validates they are included in `.github/workflows/comprehensive-tests.yml`
- Quiet success (no output when all tests are covered)
- Detailed report only when tests are missing
- Optional GitHub PR comment posting with `--post-pr` flag

**Usage**

```bash
# Validate test coverage (quiet on success)
python scripts/validate_test_coverage.py

# Validate and post report to PR if tests are missing
python scripts/validate_test_coverage.py --post-pr
```

**Command-line Options**

- `--post-pr`: Post validation report to GitHub PR if tests are missing (only in PR context)

**Exit Codes**

- 0 = All test files are covered by CI matrix
- 1 = Uncovered test files found

**Output Behavior**

- **Success (exit 0)**: No output (quiet mode)
- **Failure (exit 1)**: Prints detailed report with:
  - List of uncovered test files
  - Recommended test groups to add to CI workflow
  - Example YAML configuration

**CI Integration**

The workflow posts a PR comment only when tests are missing:

1. Runs validation (collect exit code)
2. If tests missing AND PR context: Posts detailed report to PR
3. Always fails CI if tests are missing (exit code 1)

**Example Report**

When tests are missing, the script outputs:

```text
======================================================================
Test Coverage Validation
======================================================================

Found 2 uncovered test file(s):

  - tests/new_module/test_feature.mojo
  - tests/another/test_utils.mojo

======================================================================
Recommendations
======================================================================

Add missing test files to .github/workflows/comprehensive-tests.yml
by updating the appropriate test group or creating a new one.

Example test groups to consider:

  - name: "Another"
    path: "tests/another"
    pattern: "test_*.mojo"

  - name: "New Module"
    path: "tests/new_module"
    pattern: "test_*.mojo"
```

---

### `regenerate_github_issues.py`

**Purpose**: Regenerate all github_issue.md files dynamically from their corresponding plan.md files.

#### Features

- Generates github_issue.md files from plan.md sources (task-relative, local only)
- Supports dry-run mode for testing
- Section-by-section processing
- Resume capability with timestamped state files
- Progress tracking and error handling

### Usage

```bash

# Dry-run to preview changes

python3 scripts/regenerate_github_issues.py --dry-run

# Regenerate one section

python3 scripts/regenerate_github_issues.py --section 01-foundation

# Regenerate all files

python3 scripts/regenerate_github_issues.py

# Resume from previous run

python3 scripts/regenerate_github_issues.py --resume
```

### Command-line Options

- `--dry-run`: Show what would be done without making changes
- `--section SECTION`: Process only one section (e.g., 01-foundation)
- `--resume`: Resume from last saved state
- `--plan-dir PATH`: Specify plan directory (default: auto-detected from repository root)

### Output

- Logs to stderr with progress updates
- Saves state to `logs/.issue_creation_state_<timestamp>.json`
- Updates github_issue.md files in place

**Important**: github_issue.md files are dynamically generated and should not be edited manually.
Always regenerate them using this script. These files are local (in `notes/plan/`) and NOT tracked in git.

---

#### `create_issues.py`

**Purpose**: Create GitHub issues from all github_issue.md files in the notes/plan directory

### Features

- Creates GitHub issues using the gh CLI
- Supports dry-run mode for testing
- Section-by-section processing
- Progress tracking and state management
- Automatic retry on failures with exponential backoff
- Updates github_issue.md files with created issue URLs
- Automatic label creation

### Usage

```bash
# Dry-run mode (recommended first - shows what would be created)
python3 scripts/create_issues.py --dry-run

# Test with one section only
python3 scripts/create_issues.py --section 01-foundation

# Test with single file (replaces create_single_component_issues.py)
python3 scripts/create_issues.py --file notes/plan/.../github_issue.md

# Resume from interruption
python3 scripts/create_issues.py --resume

# Create all issues
python3 scripts/create_issues.py

# Disable colored output
python3 scripts/create_issues.py --no-color

# Specify repository explicitly
python3 scripts/create_issues.py --repo username/repo
```

### Command-line Options

- `--dry-run`: Show what would be done without creating issues
- `--section SECTION`: Process only one section (e.g., 01-foundation)
- `--resume`: Resume from last saved state
- `--no-color`: Disable ANSI color output
- `--repo REPO`: Override repository (default: auto-detected from git)

### Output

- Logs to `logs/create_issues_<timestamp>.log`
- Saves state to `logs/.issue_creation_state_<timestamp>.json`
- Updates github_issue.md files with issue URLs

### Prerequisites

- GitHub CLI (`gh`) must be installed and authenticated
- Run `gh auth login` if not already authenticated

---

#### `validate_installation_anchors.py`

**Purpose**: Validate anchor fragments in links from markdown files to `docs/getting-started/installation.md`.

Ensures that every deep-link to `installation.md` (e.g., `installation.md#prerequisites`) resolves
to an actual heading in that file. Catches regressions when headings are renamed or removed.

### Features

- Extracts headings from `installation.md` and derives their GitHub-style anchor slugs
- Scans one or more source markdown files for links pointing to `installation.md`
- Reports any anchor fragment that does not match a valid heading
- Plain links without anchors (e.g., `[Guide](installation.md)`) are always accepted

### Usage

```bash
# Validate links in README.md against installation.md
python3 scripts/validate_installation_anchors.py README.md docs/getting-started/installation.md

# Scan all markdown files in the repo (default when no args given)
python3 scripts/validate_installation_anchors.py

# Verbose output
python3 scripts/validate_installation_anchors.py --verbose
```

### Exit Codes

- 0 = All anchor links valid (or no anchor links found)
- 1 = One or more broken anchor links found

---

#### `validate_links.py`

**Purpose**: Validate all markdown links in the repository.

### Features

- Checks relative file links and anchors
- Validates internal markdown links
- Reports broken links with line numbers
- Supports dry-run mode
- Exit code indicates validation status

### Usage

```bash
# Validate links in all markdown files
python3 scripts/validate_links.py

# Validate links in specific directory
python3 scripts/validate_links.py notes/
```

### Exit Codes

- 0 = All links valid
- 1 = Broken links found

---

#### `validate_structure.py`

**Purpose**: Validate repository directory structure and required files.

### Features

- Checks for required directories
- Validates presence of key files
- Section-by-section validation
- Clear error reporting

### Usage

```bash
# Validate repository structure
python3 scripts/validate_structure.py

# Validate specific section
python3 scripts/validate_structure.py --section 01-foundation
```

### Exit Codes

- 0 = Structure valid
- 1 = Validation errors found

---

#### `check_readmes.py`

**Purpose**: Validate README.md files for completeness and consistency.

### Features

- Checks required sections in README files
- Validates markdown formatting
- Reports missing sections
- Comprehensive validation

### Usage

```bash
# Check all README files
python3 scripts/check_readmes.py

# Check specific directory
python3 scripts/check_readmes.py notes/
```

### Exit Codes

- 0 = All README files valid
- 1 = Issues found

---

#### `lint_configs.py`

**Purpose**: CLI wrapper for linting YAML configuration files. The linting
logic lives in `hephaestus.validation.config_lint.ConfigLinter` (migrated in
issue #5061); this script supplies ML Odyssey's repo-specific deprecated-key
rules and drives the linter over the given paths.

### Usage

```bash
# Lint all config files
python3 scripts/lint_configs.py configs/

# Lint a single file with verbose output
python3 scripts/lint_configs.py --verbose configs/papers/lenet5/model.yaml
```

### Exit Codes

- 0 = All configs valid
- 1 = Linting errors found

---

#### `get_system_info.py`

**Purpose**: Collect system information for bug reports and debugging.

### Features

- Gathers git information
- Collects Mojo version details
- Python environment info
- System details (OS, CPU, etc.)
- Graceful handling of missing tools

### Usage

```bash
# Collect system information
python3 scripts/get_system_info.py

# Output in JSON format
python3 scripts/get_system_info.py --json
```

---

#### `merge_prs.py`

**Purpose**: Merge pull requests using GitHub API.

### Features

- Uses PyGithub library
- Configurable merge method
- Delete branch after merge option
- Requires GITHUB_TOKEN

### Usage

```bash
# Merge PR by number
python3 scripts/merge_prs.py <pr-number>

# Merge with squash
python3 scripts/merge_prs.py <pr-number> --squash

# Delete branch after merge
python3 scripts/merge_prs.py <pr-number> --delete-branch
```

### Requirements

- PyGithub library: `pip install PyGithub`
- GITHUB_TOKEN environment variable

---

#### `package_papers.py`

**Purpose**: Create tarball distribution of papers directory.

### Features

- Creates compressed archive
- Validates directory structure
- Timestamped output files
- Error handling

### Usage

```bash
# Create papers package
python3 scripts/package_papers.py

# Specify output directory
python3 scripts/package_papers.py --output dist/
```

---

#### `convert_image_to_idx.py`

**Purpose**: Convert PNG/JPEG images to IDX format for LeNet-5 inference.

**Features**

- Converts images to 28x28 grayscale
- Optional EMNIST transpose+flip transform
- Requires Pillow (`pip install Pillow`)

**Usage**

```bash
# Convert image to IDX format
python3 scripts/convert_image_to_idx.py input.png output.idx

# Without EMNIST transform
python3 scripts/convert_image_to_idx.py input.jpg output.idx --no-emnist-transform
```

---

---

### Deprecated Scripts

Historical and experimental scripts have been archived (the `scripts/agents/playground/`
directory has been removed).

### Recommended Alternatives

- Instead of `create_single_component_issues.py`, use `create_issues.py --file <path>`
- Agent modification scripts are historical; see playground README for context

---

## Workflow

### Typical Development Flow

**Note**: Plan files are local (in `notes/plan/`) and NOT tracked in git.

1. **Edit plan.md files** - Make changes to local planning documents
1. **Regenerate github_issue.md** - Run `regenerate_github_issues.py` to update local issue files
1. **Test with dry-run** - Run `create_issues.py --dry-run` to preview
1. **Create issues** - Run `create_issues.py` to create GitHub issues from local plans

### Creating All Issues

```bash

# Step 1: Ensure github_issue.md files are up to date

python3 scripts/regenerate_github_issues.py

# Step 2: Dry-run to verify

python3 scripts/create_issues.py --dry-run

# Step 3: Create issues section-by-section (recommended)

python3 scripts/create_issues.py --section 01-foundation
python3 scripts/create_issues.py --section 02-shared-library
python3 scripts/create_issues.py --section 03-tooling
python3 scripts/create_issues.py --section 04-first-paper
python3 scripts/create_issues.py --section 05-ci-cd
python3 scripts/create_issues.py --section 06-agentic-workflows

# OR: Create all at once

python3 scripts/create_issues.py
```

---

## File Locations

### State Files

State files are saved with timestamps in the `logs/` directory:

- `logs/.issue_creation_state_<timestamp>.json`
- Contains processed files list and completion status
- Used for resume capability

### Log Files

Execution logs are saved in the `logs/` directory:

- `logs/create_issues_<timestamp>.log`
- Contains detailed progress and error information

### GitHub Issue Files

github_issue.md files are dynamically generated (local, NOT tracked in git):

- Located in `notes/plan/**/github_issue.md` (task-relative, not in version control)
- Generated from corresponding plan.md files
- Regenerated locally as needed for issue creation
- Each contains 5 issue definitions (Plan, Test, Implementation, Packaging, Cleanup)

---

## Issue Structure

Each component generates 5 GitHub issues following the 5-phase development workflow:

### 5-Phase Hierarchy

**Workflow**: Plan → [Test | Implementation | Packaging] → Cleanup

1. **Plan Issue** - `[Plan] Component Name - Design and Documentation`
   - Labels: `planning`, `documentation`
   - Purpose: Create detailed specifications and design
   - Must complete before other phases

1. **Test Issue** - `[Test] Component Name - Write Tests`
   - Labels: `testing`, `tdd`
   - Purpose: Document and implement test cases
   - Can run in parallel with Implementation and Packaging

1. **Implementation Issue** - `[Impl] Component Name - Implementation`
   - Labels: `implementation`
   - Purpose: Build the main functionality
   - Can run in parallel with Test and Packaging

1. **Packaging Issue** - `[Package] Component Name - Integration and Packaging`
   - Labels: `packaging`, `integration`
   - Purpose: Integrate artifacts and create installer
   - Can run in parallel with Test and Implementation

1. **Cleanup Issue** - `[Cleanup] Component Name - Refactor and Finalize`
   - Labels: `cleanup`, `documentation`
   - Purpose: Collect issues, refactor, and finalize
   - Runs after parallel phases complete

See [notes/review/README.md](../notes/review/README.md) for detailed workflow documentation.

---

## Troubleshooting

### GitHub CLI Not Installed

```bash

# Install GitHub CLI
# See: https://github.com/cli/cli#installation

# Authenticate

gh auth login
```

### State File Issues

If you need to start fresh:

```bash

# State files are in logs/ with timestamps
# Remove specific state file or let script create new one

rm logs/.issue_creation_state_*.json
```

### Permission Errors

Ensure you have:

- Write access to `logs/` directory
- Write access to `notes/plan/` directory (local files, for updating github_issue.md)
- GitHub repository write access

**Note**: `notes/plan/` is local and NOT tracked in git.

### Rate Limiting

GitHub API has rate limits. If you hit them:

- Wait for rate limit to reset
- Use `--resume` to continue from where you left off
- Process sections one at a time with delays between them

### Missing github_issue.md Files

If github_issue.md files are missing:

```bash

# Regenerate all files

python3 scripts/regenerate_github_issues.py
```

---

## Testing

Unit tests for shared modules are located in `/tests/`:

```bash
# Run tests for common.py
python3 tests/test_common.py

# Run all tests (requires pytest)
pytest tests/
```

---

## Script Dependencies

### Python Requirements

- Python 3.7+
- Standard library only for main scripts
- Optional: `pytest` for running unit tests
- Optional: `tqdm` for better progress bars in create_issues.py

### External Tools

- `gh` (GitHub CLI) - Required for creating issues
- `git` - Required for repository detection

### Repository Structure

Scripts expect this structure:

```text
ProjectOdyssey/
├── notes/
│   ├── plan/                # LOCAL ONLY (not in git) - task-relative planning
│   │   ├── 01-foundation/
│   │   │   ├── plan.md
│   │   │   └── (github_issue.md - generated)
│   │   └── ...
│   ├── issues/              # Tracked docs - issue-specific documentation
│   └── review/              # Tracked docs - PR review documentation
├── agents/                  # Tracked docs - agent system documentation
├── scripts/
│   ├── create_issues.py
│   ├── regenerate_github_issues.py
│   └── agents/              # Agent utilities
└── logs/                    # Not tracked
    ├── .issue_creation_state_*.json
    └── create_issues_*.log
```

---

## Best Practices

1. **Always dry-run first**
   - Use `--dry-run` to preview changes before executing
   - Verify output looks correct

1. **Test with one component**
   - Use `create_single_component_issues.py` to test with one component
   - Verify issues are created correctly

1. **Process section-by-section**
   - Use `--section` flag for better control
   - Easier to handle errors and rate limits

1. **Check logs**
   - Review log files in `logs/` directory
   - Contains detailed error messages and progress

1. **Use resume capability**
   - If interrupted, use `--resume` to continue
   - State is saved every 50 files

1. **Keep github_issue.md files in sync**
   - Regenerate after editing plan.md files
   - Don't edit github_issue.md manually
   - Remember: These are local files (not tracked in git)

---

## Script Details

### create_issues.py

- **Lines of Code**: 854
- **Complexity**: High (comprehensive error handling, state management, retry logic)
- **Key Features**:
  - Automatic label creation with predefined colors
  - Exponential backoff retry logic (up to 3 retries)
  - Progress tracking with optional tqdm support
  - Colored terminal output (can be disabled)
  - State saved every 10 issues for resume capability
  - Parses multiple github_issue.md format variations
  - Updates markdown files with created issue URLs

### Label Colors

```python
'planning': 'd4c5f9'       # Light purple
'documentation': '0075ca'  # Blue
'testing': 'fbca04'        # Yellow
'tdd': 'fbca04'           # Yellow
'implementation': '1d76db' # Dark blue
'packaging': 'c2e0c6'      # Light green
'integration': 'c2e0c6'    # Light green
'cleanup': 'd93f0b'        # Red
```

### create_single_component_issues.py

- **Lines of Code**: 198
- **Complexity**: Medium
- **Purpose**: Testing and verification tool for single components
- **Features**:
  - Same label creation as main script
  - Simpler focused implementation for testing
  - Direct markdown file updates
  - Useful for validating changes before bulk creation

### regenerate_github_issues.py

- **Lines of Code**: 450+
- **Complexity**: Medium
- **Purpose**: Dynamic generation of github_issue.md files from plan.md sources
- **Features**:
  - Extracts all sections from plan.md (overview, inputs, outputs, steps, criteria, notes)
  - Generates consistent 5-issue format for each component
  - Supports dry-run, section-by-section, and resume modes
  - Timestamped state files for tracking multiple runs
  - Replaces 4 legacy update scripts with single consolidated tool

### Body Generation

- **Plan**: Objectives, inputs, outputs, success criteria, notes
- **Test**: Testing objectives, what to test, test success criteria, implementation steps
- **Implementation**: Goals, required inputs, outputs, implementation steps, success criteria
- **Packaging**: Objectives, integration requirements, integration steps, success criteria
- **Cleanup**: Objectives, cleanup tasks, success criteria, notes

### migrate_odyssey_skills.py

- **Purpose**: Migrate Odyssey2 `.claude/skills/` skill definitions to ProjectMnemosyne plugin format
- **Complexity**: High (YAML frontmatter parsing, category mapping, directory transformation)
- **Key Features**:
  - Transforms SKILL.md frontmatter and body to Mnemosyne conventions
  - Generates `.claude-plugin/plugin.json` with category, tags, and version metadata
  - Copies auxiliary subdirectories (`scripts/`, `templates/`, `references/`, etc.)
  - Supports `--audit` mode for cross-referencing coverage between repos
  - Dry-run mode (`--dry-run`) to preview changes without writing files

**Required Setup**

Before running, you need a local clone of ProjectMnemosyne. Specify its location using one of:

1. `--target-dir PATH` CLI argument (highest priority)
2. `MNEMOSYNE_DIR` environment variable
3. Default: `/tmp/ProjectMnemosyne`

**Usage Examples**

```bash
# Clone Mnemosyne if not already present
git clone https://github.com/<org>/ProjectMnemosyne /tmp/ProjectMnemosyne

# Migrate all skills (dry run first)
python3 scripts/migrate_odyssey_skills.py --dry-run

# Migrate all skills for real
python3 scripts/migrate_odyssey_skills.py

# Use a custom Mnemosyne path
python3 scripts/migrate_odyssey_skills.py --target-dir /path/to/ProjectMnemosyne

# Or set an environment variable
export MNEMOSYNE_DIR=/path/to/ProjectMnemosyne
python3 scripts/migrate_odyssey_skills.py

# Migrate a single skill
python3 scripts/migrate_odyssey_skills.py --skill gh-create-pr-linked

# Check coverage (audit mode)
python3 scripts/migrate_odyssey_skills.py --audit
```

---

## Related Documentation

- [Repository README](../README.md) - Main project documentation
- [Planning Documentation](../notes/README.md) - GitHub issues plan overview
- [Review Process](../notes/review/README.md) - PR review guidelines and 5-phase workflow
- [Project Conventions](../.clinerules) - Claude Code conventions

---

## Notes

- All scripts use Python 3 standard library only
- github_issue.md files are dynamically generated, not committed (local only)
- plan.md files are task-relative and NOT tracked in git
- State files include timestamps for tracking multiple runs
- Scripts handle errors gracefully with detailed logging
- Resume capability prevents duplicate work if interrupted
- For tracked team documentation, see `docs/dev/` and `agents/`
