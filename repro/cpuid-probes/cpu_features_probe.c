// Direct CPU feature probe — written in C to bypass any abstraction layer.
//
// Goal for modular/modular#6413: determine whether the JIT's CPU detection
// would see AVX-512 capabilities on a host where the kernel reports none.
//
// Build:
//   gcc -O2 -o cpu_features_probe cpu_features_probe.c
//
// Run inside the cached Podman image to see what bytes Mojo's underlying
// codegen detection routines would see.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>

#ifdef __x86_64__
#include <cpuid.h>
#include <immintrin.h>
#endif

// HWCAP2 bits relevant to x86 AVX-512 (matches kernel/include/asm/hwcap.h
// or fallback if your headers don't define them). On x86 Linux, AT_HWCAP2
// carries XSAVE / FSGSBASE / etc. AVX-512 itself is advertised via the
// 'flags' field in /proc/cpuinfo more reliably than via HWCAP on x86.
// We print AT_HWCAP / AT_HWCAP2 / AT_PLATFORM regardless and let the
// reader compare.

static void print_auxv(void) {
    printf("--- getauxval (kernel-mediated view) ---\n");
    unsigned long h1 = getauxval(AT_HWCAP);
    unsigned long h2 = getauxval(AT_HWCAP2);
    const char *plat = (const char *)getauxval(AT_PLATFORM);
    printf("  AT_HWCAP:    0x%lx\n", h1);
    printf("  AT_HWCAP2:   0x%lx\n", h2);
    printf("  AT_PLATFORM: %s\n", plat ? plat : "(null)");
}

#ifdef __x86_64__

// CPUID leaf 1: ECX bit 26 = XSAVE, bit 27 = OSXSAVE, bit 28 = AVX
//               EDX bit 25 = SSE, bit 26 = SSE2
// CPUID leaf 7,0: EBX bit 5  = AVX2
//                 EBX bit 16 = AVX-512F
//                 EBX bit 17 = AVX-512DQ
//                 EBX bit 21 = AVX-512IFMA
//                 EBX bit 30 = AVX-512BW
//                 EBX bit 31 = AVX-512VL
//                 ECX bit 1  = AVX-512_VBMI
//                 ECX bit 11 = AVX-512_VNNI
// CPUID leaf 7,1: EAX bit 4  = AVX-VNNI
//                 EAX bit 23 = AVX-IFMA
// CPUID leaf D,0: EAX = XCR0 mask of supported XSAVE features (from silicon, NOT
//                 from kernel's allowed view). Bits 5/6/7 = opmask, hi256, hi16_zmm.

static void print_cpuid_direct(void) {
    unsigned int eax, ebx, ecx, edx;

    printf("--- direct cpuid (bypasses kernel masking, sees silicon) ---\n");

    // Vendor + base info
    __cpuid(0, eax, ebx, ecx, edx);
    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = 0;
    printf("  cpuid(0): max_basic_leaf=%u  vendor=\"%s\"\n", eax, vendor);

    // Brand string (leaves 0x80000002–0x80000004)
    char brand[49] = {0};
    for (int i = 0; i < 3; i++) {
        __cpuid(0x80000002 + i, eax, ebx, ecx, edx);
        memcpy(brand + i * 16 + 0, &eax, 4);
        memcpy(brand + i * 16 + 4, &ebx, 4);
        memcpy(brand + i * 16 + 8, &ecx, 4);
        memcpy(brand + i * 16 + 12, &edx, 4);
    }
    printf("  cpuid(0x80000002..4) brand: \"%s\"\n", brand);

    // Leaf 1
    __cpuid(1, eax, ebx, ecx, edx);
    printf("  cpuid(1): family=%u model=%u stepping=%u  ext_family=%u ext_model=%u\n",
           (eax >> 8) & 0xf, (eax >> 4) & 0xf, eax & 0xf,
           (eax >> 20) & 0xff, (eax >> 16) & 0xf);
    printf("  cpuid(1).ecx: AVX=%d  XSAVE=%d  OSXSAVE=%d  AES=%d  PCLMUL=%d  FMA=%d\n",
           !!(ecx & (1u << 28)), !!(ecx & (1u << 26)), !!(ecx & (1u << 27)),
           !!(ecx & (1u << 25)), !!(ecx & (1u << 1)),  !!(ecx & (1u << 12)));

    // Leaf 7, subleaf 0
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    printf("  cpuid(7,0).ebx: AVX2=%d  AVX512F=%d  AVX512DQ=%d  AVX512IFMA=%d  AVX512CD=%d\n",
           !!(ebx & (1u << 5)),  !!(ebx & (1u << 16)), !!(ebx & (1u << 17)),
           !!(ebx & (1u << 21)), !!(ebx & (1u << 28)));
    printf("  cpuid(7,0).ebx: AVX512BW=%d  AVX512VL=%d  SHA=%d\n",
           !!(ebx & (1u << 30)), !!(ebx & (1u << 31)), !!(ebx & (1u << 29)));
    printf("  cpuid(7,0).ecx: AVX512VBMI=%d  AVX512VBMI2=%d  AVX512VNNI=%d  AVX512BITALG=%d  VAES=%d  VPCLMULQDQ=%d\n",
           !!(ecx & (1u << 1)),  !!(ecx & (1u << 6)),  !!(ecx & (1u << 11)),
           !!(ecx & (1u << 12)), !!(ecx & (1u << 9)),  !!(ecx & (1u << 10)));
    printf("  cpuid(7,0).edx: AVX512_4VNNIW=%d  AVX512_4FMAPS=%d  AVX512_VP2INTERSECT=%d\n",
           !!(edx & (1u << 2)),  !!(edx & (1u << 3)),  !!(edx & (1u << 8)));

    // Leaf 7, subleaf 1
    __cpuid_count(7, 1, eax, ebx, ecx, edx);
    printf("  cpuid(7,1).eax: AVX_VNNI=%d  AVX_IFMA=%d  AVX512_BF16=%d\n",
           !!(eax & (1u << 4)),  !!(eax & (1u << 23)), !!(eax & (1u << 5)));

    // Leaf D, subleaf 0: XCR0 supported by silicon
    __cpuid_count(0xd, 0, eax, ebx, ecx, edx);
    printf("  cpuid(0xd,0).eax (silicon XCR0 mask): 0x%x\n", eax);
    printf("    bit  0 (x87)    = %d\n", !!(eax & (1u << 0)));
    printf("    bit  1 (SSE)    = %d\n", !!(eax & (1u << 1)));
    printf("    bit  2 (AVX)    = %d\n", !!(eax & (1u << 2)));
    printf("    bit  3 (BNDREGS)= %d\n", !!(eax & (1u << 3)));
    printf("    bit  4 (BNDCSR) = %d\n", !!(eax & (1u << 4)));
    printf("    bit  5 (AVX-512 opmask)  = %d\n", !!(eax & (1u << 5)));
    printf("    bit  6 (AVX-512 hi256)   = %d\n", !!(eax & (1u << 6)));
    printf("    bit  7 (AVX-512 hi16_zmm)= %d\n", !!(eax & (1u << 7)));

    // OSXSAVE indicates the OS has done XCR0 setup
    __cpuid(1, eax, ebx, ecx, edx);
    int osxsave = !!(ecx & (1u << 27));
    printf("  OSXSAVE=%d %s\n", osxsave,
           osxsave ? "(OS has enabled XSAVE)" : "(OS has NOT enabled XSAVE — direct AVX use will fault)");

    if (osxsave) {
        // Read actual XCR0 the OS has configured
        unsigned int xcr0_lo, xcr0_hi;
        __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        unsigned long long xcr0 = ((unsigned long long)xcr0_hi << 32) | xcr0_lo;
        printf("  xgetbv(0) (OS-enabled XCR0): 0x%llx\n", xcr0);
        printf("    bit  0 (x87)    = %d\n", !!(xcr0 & (1ull << 0)));
        printf("    bit  1 (SSE)    = %d\n", !!(xcr0 & (1ull << 1)));
        printf("    bit  2 (AVX)    = %d\n", !!(xcr0 & (1ull << 2)));
        printf("    bit  5 (AVX-512 opmask)  = %d\n", !!(xcr0 & (1ull << 5)));
        printf("    bit  6 (AVX-512 hi256)   = %d\n", !!(xcr0 & (1ull << 6)));
        printf("    bit  7 (AVX-512 hi16_zmm)= %d\n", !!(xcr0 & (1ull << 7)));
        printf("    ↑ If silicon (cpuid 0xd,0) has AVX-512 bits set but xgetbv (OS-enabled XCR0)\n");
        printf("    ↑ does NOT, then any code path that consults raw cpuid for codegen decisions\n");
        printf("    ↑ will incorrectly emit AVX-512 — kernel cannot save the registers.\n");
    }
}

// __builtin_cpu_supports: what compiler-rt sees.
// Note: gcc requires a string LITERAL — can't loop. The named features must
// be ones gcc recognizes; older gcc rejects names like "avxvnni".
#define BCS(name) printf("  %-14s = %d\n", name, __builtin_cpu_supports(name))
static void print_builtin_cpu_supports(void) {
    printf("--- __builtin_cpu_supports (compiler-rt) ---\n");
    __builtin_cpu_init();
    BCS("sse");
    BCS("sse2");
    BCS("sse3");
    BCS("ssse3");
    BCS("sse4.1");
    BCS("sse4.2");
    BCS("avx");
    BCS("avx2");
    BCS("fma");
    BCS("avx512f");
    BCS("avx512dq");
    BCS("avx512bw");
    BCS("avx512vl");
    BCS("avx512cd");
    BCS("avx512vnni");
    BCS("avx512vbmi");
    BCS("avx512ifma");
}
#undef BCS

#endif // __x86_64__

int main(void) {
    print_auxv();
    printf("\n");
#ifdef __x86_64__
    print_cpuid_direct();
    printf("\n");
    print_builtin_cpu_supports();
#else
    printf("(non-x86 build — only auxv probe ran)\n");
#endif
    return 0;
}
