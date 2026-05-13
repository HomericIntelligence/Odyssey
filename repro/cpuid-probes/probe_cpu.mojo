# Mojo-side CPU feature probe for modular/modular#6413.
#
# Run inside the cached projectodyssey:dev container alongside the
# cpu_features_probe.c output to compare what Mojo sees vs what
# the kernel/__builtin_cpu_supports/direct cpuid sees.
#
# Goal: identify whether Mojo's stdlib exposes any way to detect
# the CPU features it will use for SIMD codegen, and whether that
# detection routine sees AVX-512 on a host where /proc/cpuinfo
# does not advertise it.
#
# Run:
#   pixi run mojo run repro/cpuid-probes/probe_cpu.mojo


from sys.info import (
    has_avx,
    has_avx2,
    has_avx512f,
    has_fma,
    has_intel_amx_bf16,
    has_intel_amx_int8,
    has_neon,
    has_sse2,
    has_sse3,
    has_sse4_1,
    has_sse4_2,
    has_vnni,
    is_apple_silicon,
    is_neoverse_n1,
    is_x86,
    num_logical_cores,
    num_performance_cores,
    simdbitwidth,
    simdwidthof,
)


def main():
    print("=== Mojo sys.info CPU detection ===")
    print("is_x86()           =", is_x86())
    print("has_sse2()         =", has_sse2())
    print("has_sse3()         =", has_sse3())
    print("has_sse4_1()       =", has_sse4_1())
    print("has_sse4_2()       =", has_sse4_2())
    print("has_avx()          =", has_avx())
    print("has_avx2()         =", has_avx2())
    print("has_fma()          =", has_fma())
    print("has_avx512f()      =", has_avx512f())
    print("has_vnni()         =", has_vnni())
    print("has_intel_amx_bf16=", has_intel_amx_bf16())
    print("has_intel_amx_int8=", has_intel_amx_int8())
    print("has_neon()         =", has_neon())
    print("is_apple_silicon() =", is_apple_silicon())
    print("is_neoverse_n1()   =", is_neoverse_n1())
    print()
    print("=== SIMD width (the JIT will pick this) ===")
    print("simdbitwidth()                =", simdbitwidth())
    print("simdwidthof[DType.float32]()  =", simdwidthof[DType.float32]())
    print("simdwidthof[DType.float64]()  =", simdwidthof[DType.float64]())
    print("simdwidthof[DType.int32]()    =", simdwidthof[DType.int32]())
    print("simdwidthof[DType.int8]()     =", simdwidthof[DType.int8]())
    print()
    print("=== logical cores ===")
    print("num_logical_cores()           =", num_logical_cores())
    print("num_performance_cores()       =", num_performance_cores())
