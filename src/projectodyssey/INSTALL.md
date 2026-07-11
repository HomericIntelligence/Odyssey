# Installation Guide - ML Odyssey Shared Library (WIP)

This guide covers installing and setting up the ML Odyssey shared library for use in paper implementations.

## Prerequisites

### Required

- **Mojo 1.0.0b2 or later** - Download from [Modular](https://mojolang.org/)
- **Git** - For cloning the repository

### Optional

- **Python 3.10+** - For development tooling and scripts
- **Pixi** - For managing development environment (recommended)

## Installation Methods

### Option 1: Development Install (Recommended for Contributors)

This method is best for contributors who want to modify the shared library code.

```bash
# Clone the repository
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey

# Install in development mode (creates symlinks)
mojo package shared --install

# Verify installation
mojo run scripts/verify_installation.mojo
```text

### Benefits

- Changes to shared library immediately available
- No need to reinstall after modifications
- Easy to contribute changes back

### Use When

- Contributing to the shared library
- Developing new paper implementations
- Testing experimental features

### Option 2: Package Install

Install from a pre-built package (for users who won't modify the shared library).

```bash
# Download the package (from release or build it)
mojo package shared -o ml_odyssey_shared.mojopkg

# Install the package
mojo install ml_odyssey_shared.mojopkg

# Verify installation
mojo run scripts/verify_installation.mojo
```text

### Benefits

- Fast installation
- No repository needed
- Stable, tested version

### Use When

- Using the library without modifications
- Deploying paper implementations
- Production use

### Option 3: Path-Based Install

Add the shared library to your project via path dependency.

**File**: `your_project/mojo.toml` or `mojoproject.toml`

```toml
[package]
name = "your-paper-implementation"
version = "0.1.0"

[dependencies]
Odyssey-shared = { path = "../Odyssey/shared" }
```text

Then use in your code:

```mojo
from shared import Linear, SGD, Tensor
```text

### Benefits

- Easy setup for related projects
- Automatic updates when shared library changes
- Good for monorepo setups

### Use When

- Working in a monorepo with Odyssey
- Local development across multiple projects
- Quick prototyping

## Build from Source

### Building the Package

```bash
# Build development package (fast, no optimizations)
mojo package shared -o build/ml_odyssey_shared.mojopkg

# Build release package (optimized for performance)
mojo package shared --release -o dist/ml_odyssey_shared-0.1.0.mojopkg

# Build with debug symbols
mojo package shared --debug -o build/ml_odyssey_shared_debug.mojopkg
```text

### Build Options

- `--release` - Enable optimizations, disable debug info (use for production)
- `--debug` - Include debug symbols (use for development/debugging)
- `-o OUTPUT` - Specify output path for package file
- `--install` - Build and install in one step

## Verification

### Quick Verification

Run the verification script to check if installation succeeded:

```bash
mojo run scripts/verify_installation.mojo
```text

Expected output:

```text
✓ Core imports successful
✓ Core functionality working
✓ Training functionality working
✓ Utils functionality working

✅ Shared library installation verified!
```text

### Manual Verification

Test imports manually:

```mojo
from shared import Linear, SGD, DataLoader, Logger

# If this imports without error, installation successful
print("Shared library installed successfully!")
```text

### Detailed Verification

Run the full test suite:

```bash
# Run import validation tests
mojo test tests/projectodyssey/test_imports.mojo

# Run integration tests
mojo test tests/projectodyssey/integration/

# Run all tests
mojo test tests/projectodyssey/
```text

## Usage

### Basic Usage

After installation, import components in your Mojo code:

```mojo
# Import commonly used items from root
from shared import Linear, ReLU, Sequential, SGD, Tensor

# Build model
var model = Sequential([
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
])

# Create optimizer
var optimizer = SGD(learning_rate=0.01, momentum=0.9)
```text

### Advanced Usage

Import from specific modules for less common items:

```mojo
# Import from specific submodules
from projectodyssey.core.layers import MaxPool2D, Dropout, BatchNorm2D
from projectodyssey.training.schedulers import CosineAnnealingLR
from projectodyssey.training.metrics import Precision, Recall
from projectodyssey.data.transforms import RandomCrop, RandomHorizontalFlip
```text

See [EXAMPLES.md](EXAMPLES.md) for complete usage examples.

## Environment Setup

### Using Pixi (Recommended)

If you're using Pixi for environment management:

```bash
# Install dependencies
pixi install

# Run commands in Pixi environment
pixi run mojo --version
pixi run mojo package shared --install
```text

### Manual Setup

Without Pixi:

```bash
# Ensure Mojo is in your PATH
export PATH="/path/to/mojo/bin:$PATH"

# Verify Mojo installation
mojo --version

# Install shared library
mojo package shared --install
```text

## Troubleshooting

### Common Issues and Solutions

#### Import Errors

**Problem**: `Error: Cannot find module 'shared'`

### Solution

1. Verify installation: `mojo list` should show `Odyssey-shared`
1. Reinstall: `mojo package shared --install`
1. Check MOJO_PATH: `echo $MOJO_PATH`
1. Verify using: `mojo run scripts/verify_installation.mojo`

**Problem**: `Error: Module 'shared' found but component 'Linear' not found`

### Solution

- The implementation is not yet complete for that component
- Check Issue #49 for implementation status
- Use placeholder or stub for development
- Check what's actually implemented:

```bash
find src/projectodyssey/ -name "*.mojo" -type f | while read f; do
    if grep -q "^struct\|^fn" "$f" 2>/dev/null; then
        echo "✓ $f (has implementation)"
    else
        echo "○ $f (stub/planning)"
    fi
done
```text

**Problem**: `ImportError: cannot import name 'Linear' from 'shared.core'`

### Solution

```bash
# Verify installation
mojo run scripts/verify_installation.mojo

# Reinstall package
mojo package shared --install

# Check available exports in Mojo REPL
mojo repl
# Then type: from shared import *
# Then type: dir()
```text

#### Build Errors

**Problem**: Build fails with compilation errors

### Solution

1. Check Mojo version: `mojo --version` (requires 24.5+)
1. Clean build artifacts: `rm -rf build/`
1. Update dependencies in `pixi.toml`
1. Report issue if problem persists

### Additional Solutions

```bash
# Update Mojo to latest version
modular update mojo

# Build with verbose output for debugging
mojo package shared --verbose

# Check for syntax errors
mojo check src/projectodyssey/
```text

#### Version Conflicts

**Problem**: Multiple versions of shared library installed

### Solution

```bash
# List installed packages
mojo list

# Uninstall old version
mojo uninstall Odyssey-shared

# Reinstall correct version
mojo package shared --install
```text

**Problem**: API doesn't match documentation

### Solution

```bash
# Check installed version
mojo repl
# Then: from shared import __version__
# Then: print(__version__)

# Reinstall latest version
git pull
mojo package shared --install --force
```text

#### Performance Issues

**Problem**: Code runs slower than expected

### Solutions

```bash
# Build release version for better performance
mojo build --release your_code.mojo

# Check available build flags
mojo build --help

# Profile your code to find bottlenecks
mojo run --profile your_code.mojo
```text

#### Platform-Specific Issues

**Problem**: Installation fails on WSL/Linux

### Solutions

```bash
# Ensure proper permissions on scripts
chmod +x scripts/*.mojo

# Use absolute paths if needed
MOJO_PATH=/path/to/packages mojo run script.mojo

# Check file system case sensitivity
ls -la src/projectodyssey/
```text

**Problem**: Installation fails on macOS

### Solutions

```bash
# Install Xcode Command Line Tools if needed
xcode-select --install

# Check architecture (Apple Silicon vs Intel)
uname -m

# Ensure Mojo is installed for correct architecture
modular install mojo
```text

### Verification Steps

If installation seems successful but imports fail, run these checks:

```bash
# 1. Verify Mojo installation
mojo --version
# Expected: 24.5 or later

# 2. Verify shared library directory structure
ls -la src/projectodyssey/
# Should show __init__.mojo and subdirectories (core, training, data, utils)

# 3. Test basic import
mojo repl
# Then: from shared import __version__
# Then: print(__version__)
# Expected: "0.1.0" or similar

# 4. Run verification script
mojo run scripts/verify_installation.mojo
# Should complete without errors
```text

### Known Limitations

#### Current Implementation Status

The shared library is in active development. Some components may be:

- **Planned**: Documented but not yet implemented
- **Stubbed**: Interface defined, implementation incomplete
- **Beta**: Implemented but not fully tested

#### Platform Support

- **Linux**: Fully supported ✓
- **macOS**: Fully supported ✓
- **Windows**: Via WSL only (native support coming soon)

#### Mojo Version Requirements

- **Minimum**: Mojo 24.5
- **Recommended**: Latest stable release
- **Beta features**: May require nightly builds

### Getting Unstuck

If you're still having issues after trying the solutions above:

**1. Clean Install** - Remove everything and start fresh:

```bash
# Remove old installations
rm -rf ~/.modular/packages/shared*

# Clean build artifacts
rm -rf src/projectodyssey/__pycache__ src/projectodyssey/**/__pycache__

# Reinstall from scratch
git pull
mojo package shared --install
```text

**2. Check Environment** - Print diagnostic information:

```bash
# Print environment info
echo "Mojo: $(mojo --version)"
echo "OS: $(uname -a)"
echo "PATH: $PATH"
echo "MOJO_PATH: $MOJO_PATH"
```text

**3. Minimal Test Case** - Create a minimal test:

```mojo
# test_import.mojo
from shared import __version__

fn main():
    print("Shared library version:", __version__)
```text

Run it: `mojo run test_import.mojo`

**4. File an Issue** - If none of the above works, file a GitHub issue with:

- Mojo version: `mojo --version`
- OS and architecture: `uname -a`
- Full error message (copy/paste complete output)
- Steps to reproduce the issue
- Output from verification steps above

## Uninstallation

### Remove Package

```bash
# Uninstall the package
mojo uninstall Odyssey-shared

# Verify removal
mojo list | grep Odyssey
```text

### Clean Build Artifacts

```bash
# Remove build artifacts
rm -rf build/ dist/

# Remove cached files (if any)
rm -rf .mojo_cache/
```text

## Platform Support

### Supported Platforms

- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (WSL2 required)

### Platform-Specific Notes

### Linux

- Works out of the box on most distributions
- Ensure glibc 2.32+ for optimal performance

**macOS**:

- Works on Intel and Apple Silicon
- May require Xcode Command Line Tools

### Windows

- Use WSL2 (Windows Subsystem for Linux)
- Native Windows support planned

## Development Setup

### For Contributors

Set up a complete development environment:

```bash
# Clone repository
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey

# Install development dependencies
pixi install

# Install pre-commit hooks
pre-commit install

# Install shared library in development mode
pixi run mojo package shared --install

# Run tests to verify setup
pixi run mojo test tests/projectodyssey/

# Format code
pixi run mojo format src/projectodyssey/
```text

### IDE Setup

### VS Code

1. Install Mojo extension
1. Set Mojo path in settings
1. Configure formatter: `pixi run mojo format`

### Other Editors

- Configure LSP for Mojo (if available)
- Set formatter to `pixi run mojo format`
- Configure syntax highlighting

## Next Steps

After installation:

1. **Read Examples**: See [EXAMPLES.md](EXAMPLES.md) for usage patterns
1. **Check API Docs**: Review auto-generated API documentation
1. **Run Tests**: Verify everything works with test suite
1. **Start Building**: Begin implementing your paper

## Support

### Getting Help

- **Documentation**: Check [README.md](README.md) for overview
- **Examples**: See [EXAMPLES.md](EXAMPLES.md) for code samples
- **Issues**: Report bugs on [GitHub Issues](https://github.com/HomericIntelligence/Odyssey/issues)
- **Discussions**: Ask questions in GitHub Discussions

### Reporting Issues

When reporting installation issues:

1. Include Mojo version: `mojo --version`
1. Include OS and architecture: `uname -a`
1. Provide full error message
1. Describe steps to reproduce

## Additional Resources

- [Mojo Documentation](https://docs.modular.com/mojo/manual/)
- [Mojo Packages Guide](https://docs.modular.com/mojo/manual/packages/)
- [Project README](../README.md)
- [Contributing Guide](../CONTRIBUTING.md)

## License

See the main repository [LICENSE](../LICENSE) file.
