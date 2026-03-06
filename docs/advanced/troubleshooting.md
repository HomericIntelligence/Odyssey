# Troubleshooting Guide

This guide provides diagnostic and resolution steps for common issues encountered when working
with ML Odyssey. Follow the quick diagnostic checklist first, then navigate to the specific
section for your issue.

## Quick Diagnostic Checklist

Use this 5-point checklist to identify the problem category:

1. **Can't import modules or run commands?** → See Installation Issues
1. **Compilation errors during `mojo build`?** → See Build Errors
1. **Errors when running code?** → See Runtime Errors
1. **Ownership or constructor errors?** → See Mojo-Specific Issues
1. **Tests failing or CI broken?** → See Testing Issues

---

## Installation Issues

### Symptom: "command not found: mojo"

**Cause**: Mojo is not installed or not in PATH.

**Solution**:

1. Verify Mojo is installed:

```bash
mojo --version
```

1. If not found, activate the Pixi environment:

```bash
cd /path/to/ProjectOdyssey
pixi shell
mojo --version
```

1. If still not found, reinstall Pixi dependencies:

```bash
pixi install --force
pixi shell
mojo --version
```

1. Verify you are in the correct directory with `pixi.toml`:

```bash
ls pixi.toml
```

### Symptom: "Error: module 'shared' not found" or "No module named shared"

**Cause**: Import paths are incorrect or the `-I .` flag is missing.

**Solution**:

When building executables that import from the `shared` package, always use the `-I .` flag:

```bash
# WRONG - Missing -I flag
mojo build examples/train.mojo

# CORRECT - Include current directory in import path
mojo build -I . examples/train.mojo

# CORRECT - Using pixi run (includes -I . automatically)
pixi run mojo build examples/train.mojo
```

For Python scripts, ensure the working directory is the repository root:

```bash
cd /path/to/ProjectOdyssey
python3 scripts/your_script.py
```

### Symptom: "ImportError: No module named 'shared'" in Python

**Cause**: PYTHONPATH is not set or script is running from wrong directory.

**Solution**:

```bash
# Add repository root to PYTHONPATH
cd /path/to/ProjectOdyssey
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 scripts/your_script.py
```

Or run from the repository root without changing PYTHONPATH:

```bash
cd /path/to/ProjectOdyssey
python3 -c "import sys; sys.path.insert(0, '.'); from scripts import your_script"
```

### Symptom: Pixi environment issues or "pixi: command not found"

**Cause**: Pixi is not installed or shell is not configured.

**Solution**:

1. Install Pixi globally:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

1. Activate Pixi in current shell:

```bash
source "$HOME/.local/bin/env"
```

1. Verify installation:

```bash
pixi --version
```

1. Navigate to repository and create environment:

```bash
cd /path/to/ProjectOdyssey
pixi install
pixi shell
```

---

## Build Errors

### Symptom: "module does not contain a 'main' function"

**Cause**: Attempting to build a library file standalone instead of building the package.

**Solution**:

Library files like `shared/core/extensor.mojo` or `shared/training/optimizer.mojo` don't have
a `main()` function. Build the package instead:

```bash
# WRONG - Library files don't have main()
mojo build shared/core/extensor.mojo

# CORRECT - Build the package
mojo package shared -o dist/shared-0.1.0.mojopkg

# CORRECT - Build executables that import from shared
mojo build -I . examples/train.mojo
```

### Symptom: "unable to locate module" or "failed to resolve module"

**Cause**: Import path or package directory structure is incorrect.

**Solution**:

1. Verify the package directory structure exists:

```bash
ls -la shared/**init**.mojo
ls -la shared/core/**init**.mojo
ls -la shared/training/**init**.mojo
```

1. Use absolute import paths from the repository root:

```mojo
# CORRECT - Import from root
from shared.core.extensor import ExTensor
from shared.training.optimizer import Adam

# For relative imports within a package, use ..
# Only valid within the package structure
from ..version import VERSION
```

1. Always build with `-I .` to set the import path:

```bash
mojo build -I . examples/train.mojo
```

1. If using `mojo package`, verify the package path:

```bash
mojo package shared -o dist/shared-0.1.0.mojopkg
mojo package shared/training -o dist/training-0.1.0.mojopkg
```

### Symptom: "error: use of unknown directive" or version mismatch errors

**Cause**: Mojo version is incompatible with code.

**Solution**:

1. Verify Mojo version:

```bash
mojo --version
```

1. Check required version in `pixi.toml` or `CLAUDE.md`:

```bash
grep -A5 "mojo" pixi.toml
```

1. Update Mojo via Pixi:

```bash
pixi update mojo
pixi shell
mojo --version
```

1. If still incompatible, create a new environment:

```bash
pixi remove --all
rm -rf .pixi/
pixi install
pixi shell
```

### Symptom: "error: cannot build package without '**init**.mojo'" or similar package errors

**Cause**: Missing `**init**.mojo` files in package directories.

**Solution**:

Create `**init**.mojo` in all package directories:

```bash
touch shared/**init**.mojo
touch shared/core/**init**.mojo
touch shared/training/**init**.mojo
touch shared/data/**init**.mojo
touch shared/utils/**init**.mojo
```

Each `**init**.mojo` should contain version export (if applicable):

```mojo
# shared/**init**.mojo
from .version import VERSION

**all** = ["VERSION"]
```

---

## Runtime Errors

### Symptom: "Shape mismatch: expected (...) but got (...)"

**Cause**: Tensor dimensions don't match in operation.

**Solution**:

1. Verify input and output shapes in your code:

```mojo
var x = ExTensor(...)
print("Input shape:", x._shape)
var y = forward(x)
print("Output shape:", y._shape)
```

1. Check the layer documentation for expected dimensions:

```bash
# For example, Linear layer expects:
# Input: (batch_size, in_features)
# Output: (batch_size, out_features)
```

1. Ensure batch dimensions are preserved through pipeline:

```mojo
# CORRECT - Batch dimension preserved
var input_shape = List[Int]()
input_shape.append(batch_size)
input_shape.append(input_features)

# WRONG - Batch dimension dropped
var wrong_shape = List[Int]()
wrong_shape.append(input_features)
```

1. For convolutional layers, verify the full 4D shape (batch, channels, height, width):

```mojo
# CORRECT - 4D shape for Conv2D
var conv_input = List[Int]()
conv_input.append(batch_size)  # N
conv_input.append(in_channels)  # C
conv_input.append(height)       # H
conv_input.append(width)        # W
```

### Symptom: "SEGFAULT", "Aborted (core dumped)", or "out of bounds"

**Cause**: Accessing invalid memory or uninitialized tensor data.

**Solution**:

1. Never access uninitialized list indices:

```mojo
# WRONG - Uninitialized list access
var list = List[Int]()
list[0] = 42  # SEGFAULT - index 0 doesn't exist

# CORRECT - Use append to create elements
var list = List[Int]()
list.append(42)
```

1. Verify tensor is initialized before accessing data:

```mojo
# WRONG - Empty shape means 0D scalar (1 element only)
var shape = List[Int]()
var tensor = ExTensor(shape, DType.float32)
tensor._data[1] = 1.0  # SEGFAULT - out of bounds

# CORRECT - Initialize shape with dimensions
var shape = List[Int]()
shape.append(4)
var tensor = ExTensor(shape, DType.float32)
# Now indices 0-3 are valid
```

1. Check bounds before indexing:

```mojo
fn safe_access(tensor: ExTensor, index: Int) -> Float32:
    if index >= 0 and index < tensor.numel():
        return tensor._data[index]
    else:
        # Handle out of bounds
        return 0.0
```

### Symptom: "CUDA out of memory" or "RuntimeError: Cannot allocate X MB"

**Cause**: Tensor operations exceed available memory.

**Solution**:

1. Reduce batch size:

```bash
# WRONG - Too large batch
just train lenet5 fp32 20 batch_size=1024

# CORRECT - Smaller batch
just train lenet5 fp32 20 batch_size=32
```

1. Use lower precision (float16 instead of float32):

```bash
# Uses less memory
just train lenet5 fp16 20
```

1. Reduce model size (remove layers, channels):

```bash
# Modify model configuration to use fewer parameters
```

1. Monitor memory during execution:

```bash
watch -n 1 'nvidia-smi'  # For GPU
free -h  # For system memory
```

1. Check memory requirements in documentation:

```bash
cat docs/MEMORY_REQUIREMENTS.md
```

### Symptom: "TypeError: incompatible argument type" or "expected DType but got Int"

**Cause**: Wrong data type passed to function.

**Solution**:

1. Verify function signature:

```mojo
# WRONG - Passing Int instead of DType
var tensor = ExTensor(shape, 32)

# CORRECT - Use DType enum
var tensor = ExTensor(shape, DType.float32)
```

1. Use correct DType values:

```mojo
DType.int8        # 8-bit integer
DType.int16       # 16-bit integer
DType.int32       # 32-bit integer
DType.float32     # 32-bit float (default)
DType.float16     # 16-bit float
DType.bfloat16    # Brain float 16
```

1. For tensor operations, ensure dtype consistency:

```mojo
# WRONG - Mixing dtypes
var a = ExTensor(shape, DType.float32)
var b = ExTensor(shape, DType.float16)
var c = add(a, b)  # Type mismatch

# CORRECT - Matching dtypes
var a = ExTensor(shape, DType.float32)
var b = ExTensor(shape, DType.float32)
var c = add(a, b)
```

---

## Mojo-Specific Issues

### Symptom: "error: cannot transfer ownership of non-copyable type"

**Cause**: Attempting to use a `List`, `Dict`, or `String` without the transfer operator `^`.

**Solution**:

Always use `^` when returning or passing non-copyable types:

```mojo
# WRONG - Cannot implicitly copy
fn get_params(self) -> List[Float32]:
    return self.params

# CORRECT - Explicit ownership transfer
fn get_params(self) -> List[Float32]:
    return self.params^
```

For List returns:

```mojo
# WRONG
fn get_weights(self) -> List[Float32]:
    return self.weights

# CORRECT
fn get_weights(self) -> List[Float32]:
    return self.weights^
```

### Symptom: "error: cannot pass temporary to var parameter"

**Cause**: Passing a temporary expression to a function parameter marked `var` (owned parameter).

**Solution**:

Create a named variable first, then pass it:

```mojo
# WRONG - Cannot transfer ownership of temporary
var tensor = ExTensor(List[Int](), DType.float32)

# CORRECT - Named variable can be transferred
var shape = List[Int]()
var tensor = ExTensor(shape, DType.float32)
```

For multiple parameters:

```mojo
# WRONG
var model = MyModel(List[Int](), List[Float32]())

# CORRECT
var layer_sizes = List[Int]()
var weights = List[Float32]()
var model = MyModel(layer_sizes, weights)
```

### Symptom: "error: cannot transfer ownership of NoneType"

**Cause**: Empty constructor call `List[Int]()` with nothing inside parentheses.

**Solution**:

Use list initialization or append:

```mojo
# WRONG - Ambiguous
var list = List[Int]()

# CORRECT - Use literal syntax
var list: List[Int] = []

# CORRECT - Use append
var list = List[Int]()
list.append(10)
list.append(20)
```

### Symptom: "error: 'X' does not conform to trait 'ImplicitlyCopyable'"

**Cause**: Adding `ImplicitlyCopyable` to a struct with non-copyable fields.

**Solution**:

Remove `ImplicitlyCopyable` if the struct contains `List`, `Dict`, or `String`:

```mojo
# WRONG - List is NOT ImplicitlyCopyable
struct Model(Copyable, Movable, ImplicitlyCopyable):
    var weights: List[Float32]

# CORRECT - Remove ImplicitlyCopyable
struct Model(Copyable, Movable):
    var weights: List[Float32]

    # Provide explicit transfer for returns
    fn get_weights(self) -> List[Float32]:
        return self.weights^
```

### Symptom: "error: fn **init**(mut self,...) should use out self"

**Cause**: Using `mut self` in constructor instead of `out self`.

**Solution**:

Use `out self` for all constructors:

```mojo
# WRONG - mut self in constructor
fn **init**(mut self, value: Int):
    self.value = value

# CORRECT - out self for constructors
fn **init**(out self, value: Int):
    self.value = value
```

Constructor convention reference:

```mojo
fn **init**(out self, value: Int):
    """Constructor - use out self"""
    self.value = value

fn **moveinit**(out self, owned existing: Self):
    """Move constructor - use out self, owned parameter"""
    self.value = existing.value

fn **copyinit**(out self, existing: Self):
    """Copy constructor - use out self"""
    self.value = existing.value

fn modify(mut self, value: Int):
    """Mutating method - use mut self"""
    self.value = value

fn read(self) -> Int:
    """Read-only method - implicit read"""
    return self.value
```

### Symptom: "error: variable 'X' does not conform to trait 'Copyable'"

**Cause**: Attempting to copy a non-copyable type.

**Solution**:

Use move semantics (`^`) instead of copying:

```mojo
# WRONG - Cannot copy non-copyable type
var a = List[Int]()
a.append(1)
var b = a  # Error: List is not copyable

# CORRECT - Move instead of copy
var a = List[Int]()
a.append(1)
var b = a^  # Transfer ownership with ^
```

---

## Performance Issues

### Symptom: Training is very slow (wall clock time)

**Cause**: Multiple possible causes - suboptimal batch size, missing SIMD optimization, or
wrong precision.

**Solution**:

1. Profile to identify bottleneck:

```bash
# Time individual components
time pixi run mojo test tests/models/test_lenet5_layers.mojo
```

1. Increase batch size (within memory limits):

```bash
# Too small batch size
just train lenet5 fp32 20 batch_size=8

# Optimal batch size (2^N for SIMD efficiency)
just train lenet5 fp32 20 batch_size=32
```

1. Use float16 instead of float32 (2x faster):

```bash
# Slower - 32-bit floats
just train lenet5 fp32 20

# Faster - 16-bit floats
just train lenet5 fp16 20
```

1. Verify SIMD is being used:

```bash
# Check generated assembly
mojo build -I . --emit-mlir examples/train.mojo
```

1. Reduce number of epochs if unnecessary:

```bash
# Too many epochs
just train lenet5 fp32 100

# Fewer epochs for faster testing
just train lenet5 fp32 5
```

### Symptom: Memory usage grows during training (memory leak)

**Cause**: Tensors not being freed or gradients accumulating.

**Solution**:

1. Ensure tensors are properly destructed:

```mojo
# Use scope to force destruction
fn train_epoch() -> Float32:
    {  # Scope for automatic cleanup
        var batch = create_batch()
        var loss = forward_backward(batch)
        return loss
    }  # batch and gradients freed here
```

1. Clear gradients between batches:

```mojo
# CORRECT - Reset gradients
var model = create_model()
for batch in batches:
    zero_gradients(model)  # Important!
    var loss = train_step(batch, model)
```

1. Use `del` to explicitly free large tensors:

```mojo
var x = ExTensor(large_shape, DType.float32)
# ... use x ...
del x  # Explicitly free if needed
```

1. Monitor memory during training:

```bash
watch -n 1 'free -h'
```

### Symptom: SIMD optimization not being used (no vectorization)

**Cause**: Loop structure doesn't match SIMD patterns or using scalar operations.

**Solution**:

1. Structure loops for SIMD:

```mojo
# WRONG - Scalar operations
for i in range(numel):
    result[i] = data[i] * scale

# CORRECT - Use vectorized operations
fn vectorized_scale(data: DTypePointer[DType.float32],
                   scale: Float32, numel: Int):
    for i in range(0, numel, simd_width):  # SIMD width step
        var chunk = simdload[DType.float32, simd_width](
            data + i)
        chunk *= scale
        simdstore[DType.float32, simd_width](
            data + i, chunk)
```

1. Use appropriate SIMD widths:

```mojo
# Typical SIMD widths
# float32: simd_width = 4-8
# float16: simd_width = 8-16

fn get_simd_width() -> Int:
    # Use compile-time known width
    return 4  # For float32
```

1. Check alignment for SIMD operations:

```mojo
# SIMD works best with aligned memory
# ExTensor uses DTypePointer which is automatically aligned
```

---

## Testing Issues

### Symptom: "Test compilation failed" or "error: unknown declaration 'test_*'"

**Cause**: Test syntax is incorrect or test harness is not imported.

**Solution**:

1. Verify test file has correct structure:

```mojo
fn test_basic_operation():
    """Test basic tensor operations."""
    # Test implementation
    pass

fn test_another_case():
    """Another test case."""
    pass
```

1. Import test utilities:

```mojo
from shared.testing.assertions import assert_equal, assert_true
from shared.testing.special_values import get_test_values

fn test_with_assertions():
    """Using test assertions."""
    var result = some_operation()
    assert_true(result == expected, "Operation failed")
```

1. Run test with correct command:

```bash
# Run single test file
pixi run mojo test tests/models/test_lenet5_layers.mojo

# Run all tests in directory
pixi run mojo test tests/models/

# Run specific test (if supported)
pixi run mojo test tests/models/test_lenet5_layers.mojo::test_forward_pass
```

### Symptom: "Test failed: assertion error" with cryptic message

**Cause**: Test assertions are failing due to incorrect implementation or test data.

**Solution**:

1. Add debug output to test:

```mojo
fn test_with_debug():
    """Test with debug output."""
    var x = ExTensor(shape, DType.float32)
    var y = forward(x)

    # Debug output
    print("Input shape:", x.numel())
    print("Output shape:", y.numel())

    assert_true(y.numel() == expected, "Shape mismatch")
```

1. Use special FP-representable values in tests:

```mojo
# These values are exactly representable in all float dtypes
var test_values: List[Float32] = [0.0, 0.5, 1.0, 1.5, -1.0, -0.5]

for value in test_values:
    var result = operation(value)
    assert_true(result == expected_value, "Failed for value: " + str(value))
```

1. For gradient checking, use seeded randomness:

```mojo
from shared.testing.special_values import seeded_rand

fn test_gradients():
    """Test backward pass with seeded random tensors."""
    var tensor = seeded_rand(shape, seed=42)  # Reproducible
    var backward_pass = compute_gradients(tensor)

    # Check against numerical gradients
    var numerical = numerical_gradient(tensor)
    assert_close(backward_pass, numerical, tolerance=1e-2)
```

### Symptom: "Runtime error in test" or "test timeout"

**Cause**: Test is taking too long or running out of memory.

**Solution**:

1. Use smaller tensors in layerwise tests:

```mojo
# WRONG - Too large for unit test
var x = ExTensor(shape=[1024, 1024, 512], DType.float32)

# CORRECT - Smaller for fast unit tests
var x = ExTensor(shape=[32, 64, 64], DType.float32)
```

1. Reduce number of test iterations:

```mojo
# WRONG - Too many iterations
for i in range(1000):
    var result = operation()

# CORRECT - Fewer for unit test
for i in range(10):
    var result = operation()
```

1. Check test timeout settings:

```bash
# See timeout in justfile
just --show test

# Run with extended timeout if needed
timeout 120 pixi run mojo test tests/models/test_lenet5_layers.mojo
```

### Symptom: "CI workflow failed" but tests pass locally

**Cause**: Environment differences between local and CI.

**Solution**:

1. Reproduce CI environment locally:

```bash
# Use same commands as CI workflow
cd /path/to/ProjectOdyssey
pixi install
pixi shell
just validate  # Runs all CI checks locally
```

1. Check CI logs for specific failure:

```bash
# View CI logs
gh workflow view build-validation
gh run view <run-id>
```

1. Ensure all pre-commit hooks pass:

```bash
# Run all pre-commit checks
just pre-commit-all

# Individual checks
pixi run mojo format tests/
pixi run npx markdownlint-cli2 docs/
```

1. Verify Mojo version matches CI:

```bash
# Check pixi.toml for exact version
grep mojo pixi.toml

# Update if needed
pixi update mojo
mojo --version
```

---

## Getting Help

### Filing an Issue

If you encounter a problem not covered in this guide:

1. **Search existing issues first**:

```bash
# Search for similar issues
gh issue list --label bug --state all
gh search issues "your error message"
```

1. **Gather diagnostic information**:

```bash
# System information
mojo --version
pixi --version
python3 --version

# Error output
mojo build -I . examples/train.mojo 2>&1 | head -50

# Relevant file versions
git log -1 --oneline
git status
```

1. **Create detailed issue**:

```bash
gh issue create \
  --title "Mojo compilation error in training loop" \
  --body "$(cat <<'EOF'
## Error
Error message here

## Steps to Reproduce
1. Run pixi run mojo test tests/models/test_lenet5_layers.mojo
1. Observe failure

## Environment
- Mojo: output of mojo --version
- OS: output of uname -a
- Branch: output of git rev-parse --abbrev-ref HEAD

## Expected Behavior
Tests should pass

## Actual Behavior
Compilation fails with ownership error
EOF
)"
```

### Getting Community Help

- **Mojo Discord**: [Mojo Discord](https://discord.gg/modular)
- **ML Odyssey Discussions**: GitHub Discussions on the repository
- **Documentation**: Check `/docs/dev/mojo-test-failure-patterns.md` for detailed patterns
- **Anti-Patterns Reference**: See `.claude/shared/mojo-anti-patterns.md` for 64+ common mistakes

### Common Resources

| Resource | Purpose |
| -------- | ------- |
| `mojo-test-failure-patterns.md` | Comprehensive failure analysis |
| `mojo-anti-patterns.md` | Common mistakes to avoid |
| `mojo-guidelines.md` | Mojo v0.26.1+ syntax and patterns |
| `CLAUDE.md` | Agent system and development workflow |
| `docs/dev/build.md` | Build and package instructions |

### Contact & Escalation

For issues requiring immediate attention:

1. Post to the relevant GitHub issue
1. Use `@mention` for code owners
1. For security issues, use GitHub's private vulnerability reporting
1. For infrastructure issues, contact DevOps team (if applicable)

---

## Quick Reference Links

- Installation: [docs/getting-started/installation.md](../getting-started/installation.md)
- Build Instructions: [docs/dev/build.md](../dev/build.md)

---

**Last Updated**: 2025-12-28
**Related Issues**: #2649
