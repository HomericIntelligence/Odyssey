# Tools Installation Guide

Complete setup guide for ML Odyssey development tools.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or WSL2 (Windows)
- **Python**: 3.8 or higher
- **Mojo**: Latest stable version (check [Modular docs](https://docs.modular.com/mojo/manual/))
- **Git**: For version control
- **Disk Space**: ~500 MB for tools and dependencies

### Verify Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check Mojo installation
mojo --version

# Check Git
git --version

# Check repository root
cd /path/to/Odyssey
git rev-parse --show-toplevel  # Should show repository root
```text

## Quick Start

### Automated Setup

```bash
# Navigate to repository root
cd /path/to/Odyssey

# Run setup script
python3 tools/setup/install_tools.py

# Verify installation
python3 tools/setup/verify_tools.py
```text

The setup script will:

1. Detect your environment (OS, Python, Mojo versions)
1. Check for required dependencies
1. Install Python packages (if needed)
1. Verify Mojo tools are accessible
1. Create necessary directories
1. Run validation tests

## Manual Installation

### Step 1: Python Dependencies

Install Python packages for code generation and reporting:

```bash
# Navigate to tools directory
cd tools/

# Install Python dependencies
pip install -r requirements.txt

# Or install individually
pip install jinja2>=3.0.0    # Template engine
pip install pyyaml>=6.0      # YAML parsing
pip install click>=8.0.0     # CLI framework (optional)
```text

**Note**: Python dependencies are only needed for Python-based tools (scaffolding, code generation, report generation).
Mojo tools have no Python dependencies.

### Step 2: Verify Mojo Tools

Mojo tools should work without additional setup if Mojo is installed:

```bash
# Check Mojo can find tools
mojo -I tools/ -c "from test_utils import generate_batch"

# Or set MOJO_PATH (if needed)
export MOJO_PATH=/path/to/Odyssey
```text

### Step 3: Environment Configuration

Optional environment variables for tool configuration:

```bash
# Add to ~/.bashrc or ~/.zshrc

# Repository root (tools will auto-detect, but can be set explicitly)
export ML_ODYSSEY_ROOT=/path/to/Odyssey

# Mojo tools path
export MOJO_PATH=$ML_ODYSSEY_ROOT

# Python tools path (if not using pip install)
export PYTHONPATH=$ML_ODYSSEY_ROOT/tools:$PYTHONPATH

# Benchmark output directory (default: ./benchmarks)
export BENCHMARK_DIR=$ML_ODYSSEY_ROOT/benchmarks

# Tool verbosity (0=quiet, 1=normal, 2=verbose)
export TOOL_VERBOSITY=1
```text

### Step 4: Verify Installation

```bash
# Run verification script
python3 tools/setup/verify_tools.py --verbose

# Or manually test tools
python3 tools/paper-scaffold/scaffold.py --help
mojo tools/benchmarking/model_bench.mojo --help
```text

## Tool-Specific Setup

### Paper Scaffolding

No additional setup required. Uses Python standard library and optional Jinja2 for templates.

```bash
# Test scaffolding
python3 tools/paper-scaffold/scaffold.py \
    --paper "TestPaper" \
    --output /tmp/test_paper/
```text

### Test Utilities

Mojo tools require Mojo installation. Verify imports work:

```bash
# Test in Mojo REPL
mojo
>>> from tools.test_utils import generate_batch
>>> # Should not error
```text

### Benchmarking

Benchmarking tools require Mojo and optionally Python for report generation:

```bash
# Verify Mojo benchmarks
mojo tools/benchmarking/model_bench.mojo --help

# Verify report generator
python3 tools/benchmarking/report_generator.py --help
```text

For visualization in reports:

```bash
# Install optional visualization dependencies
pip install matplotlib pandas
```text

### Code Generation

Code generators use Python with template engines:

```bash
# Verify Jinja2 is installed
python3 -c "import jinja2; print(jinja2.__version__)"

# Test code generation
python3 tools/codegen/mojo_boilerplate.py --help
```text

## Platform-Specific Notes

### Linux

Standard installation should work. Ensure Python 3.8+ is available:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# Fedora/RHEL
sudo dnf install python3 python3-pip

# Arch
sudo pacman -S python python-pip
```text

### macOS

Use Homebrew for dependencies:

```bash
# Install Python (if needed)
brew install python@3.11

# Install Mojo (follow Modular docs)
curl https://get.modular.com | sh -
modular install mojo
```text

### Windows (WSL2)

Use WSL2 with Ubuntu:

```bash
# Inside WSL2
sudo apt update
sudo apt install python3 python3-pip git

# Install Mojo following Modular docs
```text

## Troubleshooting

### Python Import Errors

**Problem**: `ModuleNotFoundError: No module named 'tools'`

### Solution

```bash
# Option 1: Run from repository root
cd /path/to/Odyssey
python3 tools/paper-scaffold/scaffold.py

# Option 2: Add to PYTHONPATH
export PYTHONPATH=/path/to/Odyssey:$PYTHONPATH
```text

### Mojo Import Errors

**Problem**: `Error: could not find module 'test_utils'`

### Solution

```bash
# Set MOJO_PATH
export MOJO_PATH=/path/to/Odyssey

# Or use -I flag
mojo -I /path/to/Odyssey/tools tools/benchmarking/model_bench.mojo
```text

### Permission Errors

**Problem**: `Permission denied` when running scripts

### Solution

```bash
# Make scripts executable
chmod +x tools/*/\*.py

# Or run with python3 explicitly
python3 tools/paper-scaffold/scaffold.py
```text

### Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named 'jinja2'`

### Solution

```bash
# Install missing Python packages
pip install jinja2 pyyaml

# Or install all requirements
pip install -r tools/requirements.txt
```text

### Mojo Not Found

**Problem**: `mojo: command not found`

### Solution

```bash
# Install Mojo following official docs
curl https://get.modular.com | sh -
modular install mojo

# Add to PATH (usually done automatically)
export PATH=$PATH:~/.modular/bin
```text

## Verification

### Complete Verification

Run the comprehensive verification script:

```bash
python3 tools/setup/verify_tools.py --verbose
```text

### Expected Output

```text
Checking prerequisites...
✓ Python 3.11.5 (required: 3.8+)
✓ Mojo 24.5.0
✓ Repository root: /path/to/Odyssey

Checking Python dependencies...
✓ jinja2 3.1.2
✓ pyyaml 6.0.1
✓ click 8.1.7

Checking tool availability...
✓ paper-scaffold/scaffold.py
✓ test-utils/ (Mojo modules)
✓ benchmarking/ (Mojo modules)
✓ codegen/ (Python scripts)

All checks passed!
```text

### Individual Tool Tests

Test each tool category:

```bash
# Paper scaffolding
python3 tools/paper-scaffold/scaffold.py --help

# Test utilities (Mojo)
mojo -c "from tools.test_utils import generate_batch"

# Benchmarking (Mojo)
mojo tools/benchmarking/model_bench.mojo --help

# Code generation (Python)
python3 tools/codegen/mojo_boilerplate.py --help
```text

## Updating Tools

### Update Python Dependencies

```bash
# Update all dependencies
pip install --upgrade -r tools/requirements.txt

# Update specific package
pip install --upgrade jinja2
```text

### Update Mojo

```bash
# Update Modular CLI
modular update

# Update Mojo
modular install mojo
```text

### Update Tools

Tools are part of the repository. Update via git:

```bash
# Pull latest changes
git pull origin main

# Re-run verification
python3 tools/setup/verify_tools.py
```text

## Uninstallation

### Remove Python Dependencies

```bash
# Uninstall tool dependencies
pip uninstall -y jinja2 pyyaml click matplotlib pandas

# Or use requirements file
pip uninstall -r tools/requirements.txt
```text

**Note**: Uninstalling won't affect Mojo tools (they have no separate dependencies).

### Clean Environment Variables

Remove from `~/.bashrc` or `~/.zshrc`:

```bash
# Remove these lines
export ML_ODYSSEY_ROOT=...
export MOJO_PATH=...
export PYTHONPATH=...
export BENCHMARK_DIR=...
export TOOL_VERBOSITY=...
```text

## Next Steps

After installation:

1. **Read Integration Guide**: See [INTEGRATION.md](./INTEGRATION.md) for workflow integration
1. **Browse Tool Catalog**: See [CATALOG.md](./CATALOG.md) for available tools
1. **Try Examples**: Run example commands from tool READMEs
1. **Configure IDE**: Set up IDE integration (optional, see below)

### IDE Integration (Optional)

#### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.autoComplete.extraPaths": [
    "${workspaceFolder}/tools"
  ],
  "python.analysis.extraPaths": [
    "${workspaceFolder}/tools"
  ],
  "mojo.includePaths": [
    "${workspaceFolder}/tools"
  ]
}
```text

#### PyCharm

1. File → Settings → Project → Project Structure
1. Add `tools/` as "Sources Root"
1. Apply and OK

## References

- [Tools README](./README.md) - Overview and purpose
- [INTEGRATION.md](./INTEGRATION.md) - Integration guide
- [CATALOG.md](./CATALOG.md) - Tool catalog
- [Modular Documentation](https://docs.modular.com/mojo/manual/) - Mojo installation
- [Python Documentation](https://www.python.org/downloads/) - Python installation

---

**Document**: `/tools/INSTALL.md`
**Last Updated**: 2025-11-16
**Audience**: Developers, new team members
**Status**: Living document
