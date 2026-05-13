# CPU detection probes for modular/modular#6413

Three orthogonal probes that together pin down where Mojo's CPU detection is
diverging from the kernel's view on AMD-EPYC-under-Hyper-V GHA runners.

| Probe | What it asks |
| --- | --- |
| `/proc/cpuinfo` flags | What does the kernel allow user-space to see? |
| `cpu_features_probe` (C, direct `cpuid` + `xgetbv`) | What does the silicon report, and what XCR0 has the OS enabled? |
| `__builtin_cpu_supports` | What does compiler-rt say? |
| `probe_cpu.mojo` (`sys.info.has_avx512f` etc.) | What does Mojo's stdlib advertise? |

If `probe_cpu.mojo` reports `has_avx512f() = True` on the GHA runner while `/proc/cpuinfo`
shows no `avx512f` flag, that's the source-side smoking gun: Mojo's detection bypasses
the kernel-mediated view and lights up the AVX-512 codegen path.

## Build & run inside the cached image

```bash
podman run --rm --userns=keep-id \
  -v "$(pwd):/workspace:Z" -w /workspace --ulimit core=-1 \
  projectodyssey:dev \
  bash -c '
    set -e
    echo "════════════════════ /proc/cpuinfo (kernel view) ════════════════════"
    grep -m1 ^flags /proc/cpuinfo | tr " " "\n" | grep -E "^(avx512[a-z0-9_]*|avx_vnni|avx|avx2|fma|f16c|sse4_[12]|vaes|sha_ni)$" | sort -u

    echo
    echo "════════════════════ C probe (cpuid + xgetbv + __builtin_cpu_supports) ════════════════════"
    cd repro/cpuid-probes
    gcc -O2 -o /tmp/cpu_features_probe cpu_features_probe.c
    /tmp/cpu_features_probe

    echo
    echo "════════════════════ Mojo probe (sys.info) ════════════════════"
    cd /workspace
    pixi run mojo run repro/cpuid-probes/probe_cpu.mojo
'
```

## Expected outcomes

| Host | `/proc/cpuinfo` avx512f | `cpuid(7,0).ebx[16]` | `xgetbv(0)[5..7]` | `builtin avx512f` | `mojo has_avx512f()` |
| --- | --- | --- | --- | --- | --- |
| Local Intel without AVX-512 | 0 | 0 | n/a | 0 | **expected 0** |
| Local Intel with AVX-512 (e.g. Tiger Lake) | 1 | 1 | 1 | 1 | **expected 1** |
| GHA EPYC under Hyper-V (suspected bug) | 0 | **1** (silicon!) | **0** (OS not enabled) | **?** | **?** |

The interesting columns are the last three on the GHA row. If `cpuid(7,0).ebx[16]` is 1 but
`xgetbv(0)[5..7]` is 0, the hypervisor is exposing AVX-512 silicon to the guest's `cpuid`
instruction without the OS enabling XSAVE state for it. Whether `__builtin_cpu_supports`
and Mojo see that depends on which API path each consults:

- `cpuid` directly → sees AVX-512 → wrong codegen
- `__builtin_cpu_supports` → modern compiler-rt checks both `cpuid` AND `xgetbv` → safe
- `getauxval(AT_HWCAP*)` → kernel view → safe
- Parsing `/proc/cpuinfo` → kernel view → safe

## Why this matters

If the C probe shows `cpuid(7,0).ebx[16] = 1` AND `xgetbv(0)[5..7] = 0` AND `mojo
has_avx512f() = True`, we have the smoking gun: Mojo's `sys.info.has_avx512f` is
consulting raw CPUID rather than the OS-enabled XCR0. The fix is to gate AVX-512
detection on `xgetbv & (1<<5 | 1<<6 | 1<<7) == 0xe0`, which is what
[the Linux kernel documentation prescribes][1] for AVX-512 use.

[1]: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
     (see Vol. 1 §13.3, "Detection of XSAVE Feature Support")
