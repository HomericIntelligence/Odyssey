# Draft comment for modular/modular#6413

**TL;DR**: The SIGILL is `mojo`'s JIT emitting AVX-512 instructions
(`{1to4}` broadcast, `vpternlogd`, `%k1` opmask + masked move) on a GitHub
Actions Azure runner CPU that does not support AVX-512. Captured 4 distinct
faulting-instruction sites across the historic crash signatures; all are
AVX-512.

## Environment

- Mojo: `1.0.0b2.dev2026050805 (ed7c8f0a)` (matches the version pinned in
  ProjectOdyssey `pixi.toml`).
- Runner: standard GitHub Actions `ubuntu-latest`,
  `Linux runnervmeorf1 6.17.0-1010-azure #10~24.04.1-Ubuntu`.
- Container: rootless Podman, `/home/dev/.cache/pixi/envs/.../bin/mojo`.
- Capture method: `gdb -batch` wrapping `mojo` so libKGEN's in-process SIGABRT/
  SIGILL handler doesn't swallow the signal before kernel can dump
  ([gha-mojo-coredump-capture skill v1 in ProjectMnemosyne], plus pipe-handler
  `core_pattern`).

The GHA runner CPU lacks AVX-512 — corroborated by `gdb info all-registers`
on the captured cores showing **no `zmm` and no `k0-k7` opmask registers**,
only `xmm`/`ymm`.

## Captured faulting instructions

Four distinct sites, all reproducing under the same `mojo` build; all
emitting AVX-512 encodings:

### Site A — `assert_almost_equal` / `abs()` SIMD reduce

Backtrace:

```
#0  abs () at math.mojo:3746
#1  assert_almost_equal () at shared/testing/assertions.mojo:170
#2  test_tensor_dataset_negative_indexing ()
```

Faulting instruction:

```
=> 0x7fc4f4158490 <assert_almost_equal+48>: vandps (%r15,%rax,1){1to4},%xmm3,%xmm3
   0x7fc4f4158497 <assert_almost_equal+55>: vucomiss %xmm2,%xmm3
   0x7fc4f415849b <assert_almost_equal+59>: jbe    ...+1006
```

`{1to4}` is AVX-512F embedded broadcast.

### Site B — `_strip` / string slice (`_strip+68`)

Backtrace:

```
#0  _strip () at string_slice.mojo:1035
#1  rstrip () at string_slice.mojo:1104
#2  rstrip () at string.mojo:1654
#3  dirname () at path.mojo:257
#4  _cpython.__init__ () at _cpython.mojo:1432
#5  python.__init__ () at python.mojo:45
#6  closure.0 () at ffi/__init__.mojo:915
#7  KGEN_CompilerRT_GetOrCreateGlobalIndexed () (libKGENCompilerRTShared.so)
#11 import_module () at python.mojo:238
#12 test_substitute_simple_env_var ()
```

Faulting instruction (3 separate core captures, identical site):

```
=> 0x7f1d683e8804 <_strip+68>: vmovdqa64 (%rcx,%rax,1),%zmm0
   0x7f1d683e880b <_strip+75>: movabs $0x240,%rax
```

**This is the clearest evidence of the codegen mismatch.** `%zmm0` is a
512-bit register that **literally only exists in AVX-512**. `vmovdqa64`
loads 64 bytes from memory into it. No SSE / AVX / AVX2 form uses this
register name. The compiler emitted a 64-byte SIMD string-strip load for
a target that has at most 32-byte (`%ymm`) registers.

A related capture in the same job hit `load_config+2857`:

```
=> 0x7fb2783949e9 <load_config+2857>: vmovdqu64 (%r12,%rax,1),%zmm0
   0x7fb2783949f0 <load_config+2864>: vmovdqu64 %zmm0,0x1e0(%rsp)
```

via `_is_valid_utf8_runtime () at _utf8.mojo:173` — again `%zmm0`,
unaligned variant.

### Site D — `_swisstable._insert` / `match_h2` (`_insert+4476`)

Backtrace:

```
#0  match_h2 () at _swisstable.mojo:114
#1  _insert () at _swisstable.mojo (various callers)
```

Faulting instruction (2 separate captures, identical site):

```
=> 0x7f2bbc3a448c <_insert+4476>: vpbroadcastb %eax,%xmm0
   0x7f2bbc3a4492 <_insert+4482>: mov    0x18(%rsp),%rsi
```

`vpbroadcastb` with a **GPR source** (`%eax`) is AVX-512BW + AVX-512VL.
The memory-source form is AVX2, but the register-source form requires
AVX-512BW — and gdb's decoding here shows the reg-to-reg encoding.

### Site C1 — `philox._single_round` / `next_uint32`

Backtrace:

```
#0  _single_round () at philox.mojo:162
#1  step () at philox.mojo:101
#2  next_uint32 () at _rng.mojo:80
#3  next_uint64 () at _rng.mojo:93
#4  next_float64 () at _rng.mojo:104
#5  random_float64 () at _rng.mojo:205
#6  random_float64 () at random.mojo:89
#7  forward () at dropout.mojo:134  /  randn () at tensor_creation.mojo:582
```

Faulting instruction:

```
=> 0x7f8c1c188562 <next_uint32+178>: vpternlogd $0x96,%xmm3,%xmm4,%xmm7
   0x7f8c1c188569 <next_uint32+185>: vpxor  %xmm5,%xmm4,%xmm3
   0x7f8c1c18856d <next_uint32+189>: vpshufd $0x55,%xmm3,%xmm3
```

`VPTERNLOGD` is an AVX-512F instruction. There is no SSE/AVX/AVX2 form.

### Site C2 — `_relu_backward_op` / `dispatch_binary`

Backtrace:

```
#0  _relu_backward_op () at activation.mojo:475
        return grad if x > Scalar[T](0) else Scalar[T](0)
#1  elementwise_binary () at dtype_dispatch.mojo:248
#2  dispatch_binary () at dtype_dispatch.mojo:307
#3  relu_backward () at activation.mojo:508
#4  backward () at layers/relu.mojo:107
#5  test_relu_backward_basic ()
```

Faulting instruction (loop body):

```
=> 0x7f766814bcb0 <dispatch_binary+3024>: vcmpltss (%r10,%rdi,4),%xmm0,%k1
   0x7f766814bcb8 <dispatch_binary+3032>: vmovss (%rsi,%rdi,4),%xmm1{%k1}{z}
   0x7f766814bcbf <dispatch_binary+3039>: vmovss %xmm1,(%r9,%rdi,4)
   0x7f766814bcc5 <dispatch_binary+3045>: inc    %rdi
   0x7f766814bcc8 <dispatch_binary+3048>: cmp    %rdi,%rcx
   0x7f766814bccb <dispatch_binary+3051>: jne    0x7f766814bcb0
```

**This is the smoking-gun site.** The destination of `vcmpltss` is `%k1`, an
AVX-512 opmask register. `k0-k7` literally do not exist outside AVX-512;
there is no AVX2 / AVX1 fallback that uses them. The `vmovss
…{%k1}{z}` immediately after is a masked move with zeroing, also AVX-512F.

The pattern is a vectorized lowering of
`return grad if x > Scalar[T](0) else Scalar[T](0)` — `vcmpltss` produces a
mask, the masked `vmovss{z}` selects `grad` or 0. The compiler picked the
AVX-512 masked-select lowering on a target that doesn't support opmask
registers.

## Hypothesis

**Updated 2026-05-12 after verifying that no host involved has AVX-512:**

- Local dev laptop (Intel Core Ultra 7 258V — Lunar Lake): no `avx512*`
  CPU flags. Local test runs never reproduce the crash (276+ runs, 0 crashes).
- GitHub Actions `ubuntu-latest` build runner (where the CI image is built
  via `container-publish.yml`): Azure VM, no AVX-512.
- GitHub Actions `ubuntu-latest` test runner (where tests actually execute):
  Azure VM, no AVX-512. **Both** runner pools lack AVX-512; both core dumps
  show only `xmm`/`ymm` (no `zmm`, no `k0-k7` opmask) in `info all-registers`.

So the simple "build host had AVX-512, runtime host didn't" story is
**wrong**. Despite no host in the entire pipeline supporting AVX-512,
`mojo` still emitted AVX-512 encodings at multiple call sites in stdlib
and project code.

Candidate mechanisms now narrowed to:

1. **`mojo` defaults to a generic AVX-512-capable target** unless a runtime
   CPU detection routine confidently downgrades to AVX-2. If that detection
   misreads `/proc/cpuinfo`, or if it trusts CPUID directly without
   filtering through Linux's enabled-features mask (`getauxval(AT_HWCAP*)`),
   it could enable AVX-512 codepaths erroneously.
2. **An `--accelerator-arch` / `--target-cpu` value cached into
   `$HOME/.modular`** during `pixi install` on the GHA build runner could
   pin AVX-512. The entrypoint script in our project already mentions
   `getAcceleratorArchOrEmpty()` from `libAsyncRTMojoBindings.so`; if that
   function writes to or reads from `$HOME/.modular` and the value got
   set incorrectly during image build, it would persist into every test
   container that uses the published GHCR image.
3. **A stdlib `.mojopkg` baked into the image at `pixi install` time**
   contains AOT-compiled AVX-512 routines that get linked into JIT output
   regardless of runtime CPU. We see these exact source locations in the
   backtraces: `string_slice.mojo:1035`, `math.mojo:3746`, `philox.mojo:162`,
   `_swisstable.mojo:114`, `activation.mojo:475`. These are stdlib +
   project-code paths that would be candidates for ahead-of-time
   compilation if mojo's stdlib distribution includes precompiled
   variants.

A particularly striking detail: this only manifests on the **GHA-built**
container image. The same source code, the same `pixi.lock` (verified
byte-identical: `mojo-1.0.0b2.dev2026050805-release.conda` sha256
`8b6f080d54b7c53185786a9a928afbfcf2fbb539d89c9d44da3b5b6700a8b6dc`), and
locally-built containers don't reproduce. Whatever step happens during
GHA image build introduces the AVX-512 path; rebuilding the image during
the project's mid-May 2026 rebase incidentally fixed it.

To confirm which mechanism: dumping `mojo --version`, the effective
`--target-cpu` (if such a flag is honored), and the contents of
`$HOME/.modular/` from a failing-era container would tell us
unambiguously.

## Reproducer

Standalone reproducers were filed earlier in this issue. The
[PR #5399 positive-control branch][pc-pr] reproduces both backtrace A and
backtrace B (plus the two new C1/C2 sites) at ~100 % per-run rate on the
Azure runner, using the existing test suite. CI artifact
`crash-bundle-data-utilities` / `crash-bundle-core-layers` from run
[25746055958] contain the 4 ELF cores + symbolicated `info registers` /
`disassemble` logs referenced above.

[pc-pr]: https://github.com/HomericIntelligence/ProjectOdyssey/pull/5399
[25746055958]: https://github.com/HomericIntelligence/ProjectOdyssey/actions/runs/25746055958
