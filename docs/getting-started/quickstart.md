# Quick Start

Get ML Odyssey running in under 5 minutes.

## Prerequisites

- **Git** — to clone the repository
- **Pixi** — package and environment manager ([install guide](https://pixi.sh/latest/))
- **Mojo 0.26+** — installed automatically via Pixi

## 1. Clone and Install

```bash
git clone https://github.com/HomericIntelligence/ProjectOdyssey.git
cd ProjectOdyssey
pixi install
```

`pixi install` downloads Mojo and all dependencies into an isolated environment.
It may take a few minutes on first run.

## 2. Verify Your Environment

```bash
pixi run mojo --version
```

Expected output:

```text
mojo 0.26.x (... build info ...)
```

## 3. Run Your First Test

Run the tensor creation test suite — a fast, self-contained unit test that exercises
the core `AnyTensor` type from the shared library:

```bash
pixi run mojo tests/shared/core/test_creation_part1.mojo
```

Expected output:

```text
Running AnyTensor creation tests - Part 1: zeros() and ones()...
All Part 1 creation tests completed!
```

If the test completes without errors, your environment is working correctly.

## 4. Minimal Usage Example

The shared library's `core` module provides `AnyTensor` and functional tensor operations.
Create a file called `hello_ml.mojo` at the repository root:

```mojo
from shared.core import AnyTensor, zeros, ones

fn main() raises:
    # Create a 1D tensor of zeros with 5 elements
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)

    print("Tensor created successfully")
    print("  ndim:", t.ndim())
    print("  numel:", t.numel())

    # Create a 2D tensor of ones (3x4)
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    var m = ones(shape2, DType.float32)

    print("Matrix created successfully")
    print("  ndim:", m.ndim())
    print("  numel:", m.numel())
```

Run it from the repository root:

```bash
pixi run mojo run hello_ml.mojo
```

Expected output:

```text
Tensor created successfully
  ndim: 1
  numel: 5
Matrix created successfully
  ndim: 2
  numel: 12
```

## What's Next

- **[Installation Guide](installation.md)** — detailed setup, package builds, and IDE configuration
- **[Building Your First Model](first_model.md)** — train a digit classifier end-to-end
- **[Repository Structure](repository-structure.md)** — navigate the codebase
