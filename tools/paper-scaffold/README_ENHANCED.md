# Enhanced Paper Scaffolding Tool

**Implementation of Issues #744-763**: Directory Generator with validation and comprehensive reporting.

## Overview

The enhanced scaffolding tool generates complete paper implementation directory structures with:

- **Idempotent directory creation** (#744) - Safe to run multiple times
- **Template-based file generation** (#749) - Consistent, customizable files
- **Comprehensive validation** (#754) - Ensures correct structure

## Features

### ✓ Idempotent Operations

- Safe to re-run without errors
- Skips existing files/directories
- Clear reporting of what was created vs. skipped

### ✓ Validation

- Checks required directories exist
- Verifies required files are present
- Validates file content and format
- Provides actionable fix suggestions

### ✓ Comprehensive Reporting

- Detailed progress output
- Summary of created/skipped items
- Validation report with specific errors
- Exit codes for automation

## Usage

### Basic Usage

```bash
python tools/paper-scaffold/scaffold_enhanced.py \
    --paper "LeNet-5" \
    --title "LeNet-5: Gradient-Based Learning Applied to Document Recognition" \
    --authors "LeCun et al." \
    --year 1998 \
    --url "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"
```text

### Dry Run

Preview what would be created without actually creating anything:

```bash
python tools/paper-scaffold/scaffold_enhanced.py \
    --paper "BERT" \
    --dry-run
```text

### Skip Validation

Skip validation step (faster, useful if you'll validate separately):

```bash
python tools/paper-scaffold/scaffold_enhanced.py \
    --paper "GPT-2" \
    --no-validate
```text

### Quiet Mode

Suppress progress output (useful for scripts):

```bash
python tools/paper-scaffold/scaffold_enhanced.py \
    --paper "ResNet" \
    --quiet
```text

## Generated Structure

```text
papers/<paper-name>/
├── README.md                 # Paper documentation
├── src/
│   ├── model.mojo           # Model implementation
│   └── train.mojo           # Training script
├── tests/
│   └── test_model.mojo      # Test file
├── scripts/                 # Download scripts (empty)
├── configs/                 # Configuration files (empty)
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Processed datasets
│   └── cache/               # Cached computations
├── notebooks/               # Jupyter notebooks (empty)
└── examples/                # Usage examples (empty)
```text

## Paper Name Normalization

Paper names are automatically normalized to valid directory names:

| Input | Output |
| --- | --- |
| `LeNet-5` | `lenet-5` |
| `BERT: Pre-training` | `bert-pre-training` |
| `GPT--2` | `gpt-2` |
| `-AlexNet-` | `alexnet` |

### Rules

1. Convert to lowercase
1. Replace spaces/special chars with hyphens
1. Remove consecutive hyphens
1. Trim leading/trailing hyphens

## Exit Codes

- `0` - Success (generation and validation passed)
- `1` - Generation failed
- `2` - Generation succeeded but validation found issues
- `130` - Interrupted by user (Ctrl+C)

## Validation

The validator checks:

### Required Structure

- `src/` - Source code directory
- `tests/` - Test files directory
- `README.md` - Paper documentation

### Recommended Structure (warnings only)

- `docs/` - Additional documentation
- `data/` - Data management
- `configs/` - Configuration files
- `notebooks/` - Jupyter notebooks

### Content Validation

- README.md is not empty and has key sections
- Mojo files have basic structure (functions/structs)
- Files are valid UTF-8

### Validation Report Example

```text
✗ VALIDATION FAILED

Missing Directories (1):
  - papers/lenet-5/tests/

Missing Files (1):
  - papers/lenet-5/README.md

Suggestions:
  - Create missing directories: mkdir -p papers/lenet-5/tests
  - Create README.md from template: cp papers/_template/README.md .
```text

## Template Variables

Templates support variable substitution using `{{VARIABLE}}` syntax:

| Variable | Description | Example |
| --- | --- | --- |
| `{{PAPER_NAME}}` | Normalized paper name | `lenet-5` |
| `{{PAPER_TITLE}}` | Full paper title | `LeNet-5: Gradient-Based...` |
| `{{MODEL_NAME}}` | PascalCase model name | `Lenet5` |
| `{{AUTHORS}}` | Paper authors | `LeCun et al.` |
| `{{YEAR}}` | Publication year | `1998` |
| `{{PAPER_URL}}` | Link to paper | `http://...` |
| `{{DESCRIPTION}}` | Brief description | `CNN for digit recognition` |

## Module API

The tool is also importable as a Python module:

```python
from pathlib import Path
from scaffold_enhanced import DirectoryGenerator

generator = DirectoryGenerator(base_path=Path("papers"), verbose=True)

result = generator.generate(
    paper_name="LeNet-5",
    paper_metadata={
        "PAPER_TITLE": "LeNet-5",
        "AUTHORS": "LeCun et al.",
        "YEAR": "1998",
        "PAPER_URL": "http://...",
        "DESCRIPTION": "CNN for digit recognition"
    },
    validate=True
)

if result.success:
    print(result.summary())
else:
    print("Generation failed:", result.errors)
```text

## Testing

Run comprehensive test suite:

```bash
# Run all tests
pytest tests/tooling/test_paper_scaffold.py -v

# Run specific test class
pytest tests/tooling/test_paper_scaffold.py::TestValidation -v

# Run with coverage
pytest tests/tooling/test_paper_scaffold.py --cov=tools/paper-scaffold
```text

Test coverage includes:

- Paper name normalization
- Directory creation (idempotent, error handling)
- File generation (template rendering, overwrite protection)
- Validation (missing dirs/files, content validation)
- End-to-end workflows

## Design Decisions

### Language: Python

Per [ADR-001](../../docs/adr/ADR-001-language-selection-tooling.md):

- **Justification**: Template processing, file I/O, subprocess limitations in Mojo
- **Alternatives Considered**: Mojo (lacks subprocess output capture), Bash (harder to test)

### Idempotency

Uses `Path.mkdir(parents=True, exist_ok=True)` for safe re-runs.

### Rationale

- Users can recover from interruptions
- Safe to apply template updates
- No race conditions

### Overwrite Protection

Never overwrites existing files without explicit user confirmation.

### Rationale

- Prevents accidental data loss
- Follows principle of least surprise (POLA)
- Supports incremental workflows

### Validation as Separate Stage

Validation runs after generation, never modifies files.

### Rationale

- Clear separation of concerns
- Can be skipped for performance
- Provides actionable error reports

## Related Issues

- **#744-748**: Create Structure (directory creation logic)
- **#749-753**: Generate Files (template-based file generation)
- **#754-758**: Validate Output (structure validation)
- **#759-763**: Directory Generator (parent component)

## References

- [Planning Documentation](https://github.com/HomericIntelligence/Odyssey/issues/759)
- [5-Phase Workflow](../../notes/review/README.md)
- [ADR-001: Language Selection](../../docs/adr/ADR-001-language-selection-tooling.md)
