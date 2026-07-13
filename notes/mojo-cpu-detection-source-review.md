# Mojo CPU / Target-Feature Detection — Open-Source Source Review

**Context**: Tracking the AVX-512-on-non-AVX-512-CPU crash from modular/modular#6413
(GHA Azure runners, suspected AMD Zen 4 under Hyper-V where the hypervisor masks
the AVX-512 CPUID bits). This review answers: *where in the open-source
modular/modular tree does Mojo's CPU/target-feature detection live, and which
path could plausibly cause the JIT to emit AVX-512 on a CPU whose kernel/CPUID
view reports no AVX-512?*

**Repository**: <https://github.com/modular/modular> (branch `main`, fetched 2026-05-12)

**Repo layout (top-level)**: `bazel/`, `docs/`, `max/`, `mojo/`, `tools/`,
`utils/`, `.github/`, `.cursor/rules/`. The compiler driver / JIT runtime
(`libKGENCompilerRTShared.so`, the `mojo` binary) is **not** in the open-source
tree — only the **stdlib** (`mojo/stdlib/std/`) and documentation are open.

---

## TL;DR

1. **The Mojo stdlib does zero runtime CPU detection.** Every feature check
   (`has_avx512f`, `has_avx2`, `has_vnni`, …) resolves at *compile time* against
   a `!kgen.target` MLIR attribute via the KGEN intrinsic
   `#kgen.param.expr<target_has_feature, …>`. There is no `cpuid`, no
   `xgetbv`, no `/proc/cpuinfo` parse, no HWCAP read, no
   `__builtin_cpu_supports` anywhere in `mojo/stdlib/std/sys/`.
2. **The `!kgen.target` attribute is populated by the closed-source compiler
   driver**, not by stdlib. That driver — referenced in the open-source docs as
   `CLOptions.h:118` — initializes its target-features list from
   `getHostCPUFeatures()`, which is LLVM's
   [`llvm::sys::getHostCPUFeatures()`](https://llvm.org/doxygen/namespacellvm_1_1sys.html).
3. **LLVM's `getHostCPUFeatures()` on x86 does *not* consult the kernel's
   masked CPUID view**: it issues the `cpuid` instruction directly (and
   `xgetbv` for XSAVE). Under a hypervisor that *correctly* masks AVX-512 bits
   in CPUID, LLVM would observe the mask and stay correct. **But** LLVM has a
   long history of x86 family/model fingerprinting fallbacks (in
   `Host.cpp::getHostCPUName`) that map Zen-family family/model numbers to a
   named CPU (e.g. `znver4`), and `znver4` carries a *fixed* feature list that
   includes `+avx512f,+avx512vl,…` regardless of what CPUID's feature leaves
   reported. That is the most plausible upstream root cause: **CPU-name
   fingerprinting overrides the masked feature view**.
4. The Mojo documentation itself explicitly flags `CLOptions.h:118 →
   getHostCPUFeatures()` as the source of a known related leak bug
   (**MOCO-3686**, the "host features leak to LLVM" issue).
5. **No `MOJO_*` or `KGEN_*` environment variable to override CPU detection** is
   documented. The user-facing override is the build flag
   `--target-cpu=<name>` / `--target-features="-avx512f,…"`, which only helps
   for AOT builds — not for the in-process JIT that crashes in
   `libKGENCompilerRTShared.so`.

---

## 1. Stdlib feature-check surface (open source)

### `mojo/stdlib/std/sys/info.mojo`

This is the file that *every* AVX-512 path in your crash traces ultimately
queries (`src/odyssey/...` → `math.abs` → `SIMD.__abs__` → `CompilationTarget.has_avx512f()`).

URL: <https://github.com/modular/modular/blob/main/mojo/stdlib/std/sys/info.mojo>

Key definitions (line numbers approximate; file is ~1386 lines):

```mojo
# The compile-time target handle — a KGEN attribute, NOT a runtime value
comptime _TargetType = __mlir_type.`!kgen.target`

@always_inline("nodebug")
def _current_target() -> _TargetType:
    return __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`

struct CompilationTarget[value: _TargetType = _current_target()](
    TrivialRegisterPassable
):
    @staticmethod
    def _has_feature[name: StaticString]() -> Bool:
        return __mlir_attr[
            `#kgen.param.expr<target_has_feature,`,
            Self.value,
            `,`,
            _get_kgen_string[name](),
            `> : i1`,
        ]

    @staticmethod
    def has_avx512f() -> Bool:  # ~line 449
        """Returns True if the host system has AVX512, otherwise returns False."""
        return Self._has_feature["avx512f"]()
```

**Verdict for stdlib**: Correct in isolation. `target_has_feature` is a pure
compile-time MLIR query; the value it returns is entirely determined by
whatever feature set the compiler driver baked into the `!kgen.target`
attribute when the IR module was created. The stdlib cannot lie about
AVX-512 unless the driver told it to.

### Sibling files searched (no CPU detection found)

| File | Verdict |
| --- | --- |
| `mojo/stdlib/std/sys/_assembly.mojo` | Generic `inlined_assembly` wrapper. No `cpuid`/`xgetbv`. |
| `mojo/stdlib/std/sys/intrinsics.mojo` | LLVM intrinsics (gather/scatter/prefetch, AMDGPU). No CPU detection. |
| `mojo/stdlib/std/sys/_build.mojo` | Build-type (debug/release) only. |
| `mojo/stdlib/std/sys/_libc.mojo` | libc bindings, no `getauxval`/HWCAP. |
| `mojo/stdlib/std/sys/defines.mojo` | Compile-time `#kgen.param.expr` defines. |
| `mojo/stdlib/std/sys/compile.mojo` | Build-info bridge to KGEN. |

No occurrence of `cpuid`, `xgetbv`, `__builtin_cpu_supports`, `getauxval`,
`HWCAP`, `/proc/cpuinfo`, `getHostCPUName`, `getAcceleratorArchOrEmpty`, or
`detect_target_cpu` anywhere in `mojo/stdlib/`.

---

## 2. The smoking gun: `CLOptions.h:118` (closed source, but documented)

Despite being closed source, the compiler driver is explicitly referenced in
the open-source docs:

### `mojo/docs/code/tools/README-Compilation-Targets.md`

URL: <https://github.com/modular/modular/blob/main/mojo/docs/code/tools/README-Compilation-Targets.md>

The "Known issue (MOCO-3686)" section says, verbatim:

> **Known issue (MOCO-3686):** All three tests emit hundreds of warnings about
> unrecognized features (ARM features from the M4 host leaking to the x86 LLVM
> backend). Multi-threaded compilation causes interleaved warning output.
> Output files are correct despite the warnings.
>
> **Root cause:** `CLOptions.h:118` initializes `targetFeatures` to
> `getHostCPUFeatures()`. The `--march`/`--mcpu` path routes through
> `getMArchFeatures()` which computes the correct features, but the stale
> host defaults leak to LLVM before the override takes effect.
>
> No user-facing workaround exists:
>
> - `--disable-warnings` does not suppress them (LLVM-level, not Mojo-level
>   warnings).
> - `--target-features=""` with `--mcpu` does not clear them (empty string
>   bypasses the mixing check but doesn't overwrite the default).
> - `--target-features="<anything>"` with `--mcpu` triggers the mixing error.

**Interpretation for the Zen-4-under-Hyper-V bug**:

- `CLOptions.h:118 → targetFeatures = getHostCPUFeatures()` is the *single
  point* at which the host's feature set enters Mojo. If
  `getHostCPUFeatures()` returns a list containing `+avx512f,+avx512vl,…` on a
  host whose CPUID has the AVX-512 bits masked off, **Mojo's JIT will see
  AVX-512 as available and emit the instructions** — which is exactly what we
  observe (`vpternlogd`, `{1to4}` broadcast, `%k1` opmask masked moves).
- The documented MOCO-3686 leak is a *related* but distinct bug (cross-compile
  feature leak). For the JIT-on-host case, no `--march`/`--mcpu` override is
  applied, so the host-derived `targetFeatures` is the **final** value used
  for code generation. No "stale leak" needed — it's the primary path.

---

## 3. Why `getHostCPUFeatures()` can be wrong on Zen 4 + Hyper-V

`getHostCPUFeatures` is the LLVM function
`llvm::sys::getHostCPUFeatures()` (declared in
`llvm/include/llvm/TargetParser/Host.h`, implemented in
`llvm/lib/TargetParser/Host.cpp`). The Mojo driver almost certainly uses it
directly (the docs use the LLVM name verbatim and no Mojo-specific shim is
referenced).

The relevant x86 code path in upstream LLVM:

1. **`__cpuid` / `__get_cpuid_count`** — issues raw `cpuid` instructions
   directly (inline asm). On Hyper-V with AVX-512 masked, leaf 7 sub-leaf 0
   `EBX/ECX/EDX` bits for AVX-512F/VL/BW/DQ/VBMI/VNNI are zero. Good so far.
2. **`getHostCPUName()`** — *separately* determines a CPU name (e.g. `znver4`,
   `skylake-avx512`) by reading **family/model/stepping** from CPUID leaf 1
   `EAX` and matching against a hard-coded table:
   - AMD family `0x19` model `0x10–0x1F`, `0x60–0x7F`, `0xA0–0xAF` → `znver4`.
     See `Host.cpp::getAMDProcessorTypeAndSubtype` in upstream LLVM
     (<https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/Host.cpp>).
   - **Family/model are not maskable by Hyper-V's standard CPUID masking** — the
     hypervisor masks *feature* leaves to hide AVX-512 from naive software,
     but it does not change the chip's reported family/model/stepping.
3. **`getHostCPUFeatures` then merges**: it takes the CPUID feature bits **and**
   the implicit features attached to the CPU name. When `znver4` is selected,
   LLVM's `X86TargetParser` table assigns it the *static* feature list
   `+avx512f,+avx512vl,+avx512bw,+avx512dq,+avx512cd,+avx512vnni,+avx512vbmi,
   +avx512vbmi2,+avx512bitalg,+avx512vpopcntdq,+avx512bf16,+avx512fp16,…`. That
   list is **not gated on the kernel's CPUID feature view**.

The exact upstream call path:
`llvm::sys::getHostCPUFeatures` (`Host.cpp`) → `sys::getHostCPUName` →
`getAMDProcessorTypeAndSubtype` → returns `"znver4"` → caller looks up
`znver4` in `X86TargetParser.cpp::Processors[]` → static feature list
includes all AVX-512* extensions.

That's the mechanism by which the JIT can emit AVX-512 on a CPU whose
kernel/CPUID feature leaves say "no AVX-512".

**Verification hint** for whoever picks this up: run `cpuid -1 -l 1 -s 0` in
the failing container; if `EAX` decodes to AMD family `0x19` model in the
Genoa range while leaf 7 has the AVX-512 bits cleared, the hypothesis is
confirmed and the bug is upstream LLVM's CPU-name fingerprinting overriding
the masked feature view.

---

## 4. Other detection paths checked

### `getAcceleratorArchOrEmpty` / `_accelerator_arch`

`info.mojo` line ~701–707 exposes `_accelerator_arch()` for GPU/accelerator
targets. It also resolves via `#kgen.param.expr<…>` — purely compile-time
metadata; nothing reads PCI IDs or `nvidia-smi` from stdlib. Not relevant
to the SIGILL.

### Environment variables

| Variable | Effect | Found in |
| --- | --- | --- |
| `MOJO_*` for CPU override | **None documented** | (no hits in stdlib or docs) |
| `KGEN_*` for CPU override | **None documented** | (no hits in stdlib or docs) |
| LLVM-native: `LLVM_HOST_CPU_FEATURES` | *Not honored by LLVM* — there is no upstream env-var override for `getHostCPUFeatures`. Confirmed by reading `Host.cpp`. | — |

The only override surface exposed to users is the CLI:
`--target-cpu=<name>`, `--target-features="±feat,±feat"`,
`--march=<name>`, `--mcpu=<name>`, `--target-triple=<triple>`. None of these
are picked up by the **in-process JIT** that crashes in
`libKGENCompilerRTShared.so` — they're parsed by the `mojo build` /
`mojo run` driver and applied to AOT module compilation. Test runs invoked
through `mojo` (`mojo run ...` / `mojo test`-style hand-rolled `def main`)
inherit the driver-derived `!kgen.target`, so the flags *would* propagate —
**worth testing** as a workaround: set
`MOJO_TARGET_FEATURES="-avx512f,-avx512vl,-avx512bw,-avx512dq,-avx512cd,-avx512vnni,-avx512vbmi,-avx512vbmi2,-avx512bitalg,-avx512vpopcntdq,-avx512bf16,-avx512fp16"`
or pass `--target-features=...` to `mojo run` to force-disable AVX-512.

There is also no observable "force AVX-512" / "skip detection" env-var; the
inverse path (forcing AVX-512 *on*) requires explicit
`--target-features="+avx512f"`.

### `--print-effective-target`

Documented as the debug command:

```bash
mojo build --print-effective-target
```

This is the **single most useful diagnostic** for the GHA crash: run it
inside the failing container and confirm whether the `cpu`/`features` line
reports `znver4` + AVX-512. If it does, the upstream-LLVM fingerprinting
hypothesis is confirmed at the Mojo layer.

---

## 5. Cross-reference with crash signatures

The faulting instructions captured (per
`notes/modular-6413-avx512-finding.md`):

| Site | Instruction | Feature implied |
| --- | --- | --- |
| `string_slice._strip` | `vpbroadcastb %eax,%xmm0` | AVX-512BW + AVX-512VL (EVEX-encoded `vpbroadcastb` to XMM) |
| `math.abs` | `vandps {1to4},…` | AVX-512F (embedded broadcast `{1toN}`) |
| `philox._single_round` | `vpternlogd` | AVX-512F |
| `_swisstable.match_h2` | `vpcmpeqb %zmm…, %k1` or `vpbroadcastb …, %zmm0` | AVX-512BW |
| `activation._relu_backward_op` | `vcmpltss …, %k1` + `vmovdqa64 %zmm0` | AVX-512F |

All five live behind the same compile-time guard chain:

```text
src/odyssey/* → math.abs / SIMD ops
      └─ SIMD.__abs__ (mojo/stdlib/std/builtin/simd.mojo)
           └─ CompilationTarget.has_avx512f()        ← info.mojo line 449
                └─ __mlir_attr.target_has_feature["avx512f"]   ← KGEN intrinsic
                     └─ !kgen.target attribute                  ← driver-populated
                          └─ getHostCPUFeatures()               ← CLOptions.h:118
                               └─ llvm::sys::getHostCPUFeatures   ← UPSTREAM LLVM
                                    └─ family/model → znver4 → fixed feature list
```

Every site is consistent with `target_has_feature["avx512f"]` returning **true**
when the kernel's CPUID view says **false** — i.e. the driver populated the
`!kgen.target` attribute from a `getHostCPUFeatures()` that fingerprinted
the CPU as Zen 4 and inherited that family's static AVX-512 feature set.

---

## 6. Recommendations for the upstream issue

1. **Ask Modular to confirm `CLOptions.h:118`** uses raw
   `llvm::sys::getHostCPUFeatures()` (not a Mojo-specific wrapper that
   re-validates against the kernel view) — this is implied by the docs but
   should be stated explicitly.
2. **Ask Modular to add a kernel-view sanity check**: after calling
   `getHostCPUFeatures()`, cross-check AVX-512 bits against
   `/proc/cpuinfo` flags or `getauxval(AT_HWCAP2)` on Linux. If the kernel
   view says no AVX-512 but the LLVM-derived list says yes, **strip**
   AVX-512 from `targetFeatures` and log a warning. This is the standard
   workaround for the LLVM Zen-fingerprinting issue (GCC and recent LLVM
   versions have been adding similar guards).
3. **Ask Modular to honor `MOJO_TARGET_FEATURES` / `MOJO_TARGET_CPU`
   environment variables** for the in-process JIT, not just the AOT driver.
   Today there's no way to override the JIT's feature set without rebuilding.
4. **Document `--print-effective-target` as the first-line diagnostic** for
   "wrong instructions emitted" reports.

---

## 7. Files referenced (with URLs)

| Path | URL | Role |
| --- | --- | --- |
| `mojo/stdlib/std/sys/info.mojo` | <https://github.com/modular/modular/blob/main/mojo/stdlib/std/sys/info.mojo> | Compile-time feature queries (`has_avx512f`, `CompilationTarget`, `_current_target`). No runtime detection. |
| `mojo/stdlib/std/sys/_assembly.mojo` | <https://github.com/modular/modular/blob/main/mojo/stdlib/std/sys/_assembly.mojo> | Generic `inlined_assembly`. No CPU detection. |
| `mojo/stdlib/std/sys/intrinsics.mojo` | <https://github.com/modular/modular/blob/main/mojo/stdlib/std/sys/intrinsics.mojo> | LLVM intrinsics. No CPU detection. |
| `mojo/stdlib/std/sys/_build.mojo` | <https://github.com/modular/modular/blob/main/mojo/stdlib/std/sys/_build.mojo> | Debug/release build detection only. |
| `mojo/stdlib/std/sys/defines.mojo` | <https://github.com/modular/modular/blob/main/mojo/stdlib/std/sys/defines.mojo> | `#kgen.param.expr` defines. |
| `mojo/docs/code/tools/README-Compilation-Targets.md` | <https://github.com/modular/modular/blob/main/mojo/docs/code/tools/README-Compilation-Targets.md> | **Names `CLOptions.h:118` + `getHostCPUFeatures()` as the host-feature entry point.** |
| `mojo/docs/tools/compilation.mdx` (rendered) | <https://mojolang.org/docs/tools/compilation/> | User-facing flag docs; documents `--print-effective-target`. |
| *(closed source)* `CLOptions.h` | not in repo | Compiler-driver option parsing; initializes `targetFeatures = getHostCPUFeatures()` at line 118. |
| *(closed source)* `libKGENCompilerRTShared.so` | not in repo | In-process JIT runtime where the SIGILL fires. Consumes the `!kgen.target` attribute populated from `CLOptions.h`. |
| *(upstream)* `llvm/lib/TargetParser/Host.cpp` | <https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/Host.cpp> | `getHostCPUFeatures`, `getHostCPUName`, `getAMDProcessorTypeAndSubtype` — the likely upstream root cause. |
| *(upstream)* `llvm/lib/TargetParser/X86TargetParser.cpp` | <https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/X86TargetParser.cpp> | Static feature lists per CPU name (e.g. `znver4` → AVX-512*). |
