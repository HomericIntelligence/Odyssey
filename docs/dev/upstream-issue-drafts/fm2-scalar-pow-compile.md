---
title: "[BUG] `Scalar[dt] ** 0.5` fails to compile for float16/bfloat16/float32 (pass-manager error)"
labels: [bug, mojo]
---

### Bug description

#### Summary

Raising a `Scalar[dt]` to the power `0.5` (`x**0.5`) fails to instantiate for
`DType.float16`, `DType.bfloat16`, and `DType.float32` with a pass-manager error
("constraint failed: unsupported type combination" from `std/builtin/simd.mojo`).
`DType.float64` works. `std.math.sqrt(x)` handles every float dtype and is a valid
workaround.

This **passes on `1.0.0b2` (2cf4d08a)** and **fails on `1.0.0` stable (ed45d567)**.

#### Actual behavior

```text
$ mojo repro_scalar_pow.mojo        # Mojo 1.0.0 stable
error: failed to run the pass manager
note: constraint failed: unsupported type combination   (std/builtin/simd.mojo)
```

#### Expected behavior

```text
$ mojo repro_scalar_pow.mojo        # Mojo 1.0.0b2
float16 => 2.0
bfloat16 => 2.0
float32 => 2.0
float64 => 2.000000000000565
```

#### Steps to reproduce

Save as `repro_scalar_pow.mojo` (also at `docs/dev/reproducers/repro_scalar_pow.mojo`
in Odyssey):

```mojo
def f[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return x**0.5

def main():
    var v1 = Scalar[DType.float16](4.0)
    print("float16 =>", f(v1))
    var v2 = Scalar[DType.float32](4.0)
    print("float32 =>", f(v2))
    var v3 = Scalar[DType.float64](4.0)
    print("float64 =>", f(v3))
```

Run:

```bash
mojo repro_scalar_pow.mojo
```

#### Impact

Code that computes `x**0.5` on `Scalar` values of float16/bfloat16/float32 (e.g. a
generic `sqrt` helper) fails to compile on stable 1.0.0. The Odyssey framework hit this
in `_sqrt_typed` (`normalization.mojo`) and switched to `std.math.sqrt`, which works.

#### Environment

- Mojo version: 1.0.0 (ed45d567) — also reproduced with 1.0.0b2 (2cf4d08a) for comparison
- OS: Linux x86_64 (Ubuntu 24.04 container), glibc 2.39
- Installed via `pip install mojo==1.0.0` / `mojo==1.0.0b2` (PyPI)
