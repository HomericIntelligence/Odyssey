# [FEATURE REQUEST] SIMD[DType.float16, N] support — invalid element type in Mojo 0.26.x

## Environment

- Mojo version: 0.26.3 (dev2026040705)
- OS: Ubuntu 24.04
- GLIBC: 2.39
- CPU: x86_64 (Intel, supports F16C and AVX-512FP16 extensions)

## Description

`SIMD[DType.float16, N]` fails to compile in Mojo 0.26.3 with
`error: invalid SIMD element type 'float16'`. The `DType.float16` enum value is
valid for `UnsafePointer` and scalar operations, but cannot be used as a lane
type in a `SIMD` vector.

This blocks vectorized FP16↔FP32 conversion for mixed-precision training, requiring
scalar element-by-element loops that are ~10-15x slower than the equivalent SIMD
FP32→FP32 path.

## Minimal Reproducer

```mojo
def main():
    var v = SIMD[DType.float16, 8]()
    print(v)
```

```bash
mojo run repro_fp16_simd.mojo
```

**Result (Mojo 0.26.3):**

```text
error: invalid SIMD element type 'float16'
note: SIMD element types must be numeric but float16 is not supported
```

The error is 100% deterministic and reproducible.

## Affected Types

The limitation applies to both 16-bit float types:

- `SIMD[DType.float16, N]` — IEEE 754 half precision
- `SIMD[DType.bfloat16, N]` — Brain float (also fails)

## Expected Behavior

`SIMD[DType.float16, 8]` compiles and generates vectorized FP16 load/store
using F16C or AVX-512FP16 instructions where available, with scalar fallback
on CPUs without those extensions.

## Actual Behavior

Compile-time error: `invalid SIMD element type 'float16'`.

## Impact

Mixed-precision training (a standard technique for reducing memory usage
while maintaining training stability) requires efficient FP16↔FP32 conversion.
Without SIMD support, conversion loops are ~10-15x slower:

```mojo
# Current workaround — scalar loop, no vectorization
var src_ptr = params._data.bitcast[Float16]()
var dst_ptr = result._data.bitcast[Float32]()
for i in range(size):
    dst_ptr[i] = Float32(src_ptr[i])
```

The vectorized version (blocked by this limitation) would be:

```mojo
# Desired — vectorized, matches FP32→FP32 SIMD performance
alias W = simdwidthof[DType.float16]()
for i in range(0, size, W):
    var chunk = (src_ptr + i).load[width=W]()
    var converted = chunk.cast[DType.float32]()
    (dst_ptr + i).store(converted)
```

## Hardware Support

x86_64 CPUs with F16C extension (Intel Ivy Bridge+, AMD Piledriver+) support
`_mm256_cvtph_ps` / `_mm256_cvtps_ph` for vectorized FP16↔FP32 conversion.
AVX-512FP16 (Intel Sapphire Rapids+) supports direct FP16 arithmetic.

Both instruction sets are widely available on modern server-class hardware used
for ML training.

## Workaround

Use scalar element-by-element conversion via `Float32(src_ptr[i])` or
`Float16(src_ptr[i])`. Correct but slow. See
[ADR-010](https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/docs/adr/ADR-010-fp16-simd-mojo-limitation.md)
for the full analysis.
