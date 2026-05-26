"""Non-deterministic JIT compilation volume crash reproducer.

NO external dependencies. Copy this single file to any Mojo 0.26.3 install.

Environment: Mojo 0.26.3 (dev2026040705), GLIBC 2.39, Linux 6.6.87 x86_64

Reproduces a non-deterministic crash in the Mojo JIT compiler when:
1. A single file has 30+ functions
2. Each function performs heavy alloc/free cycles (many UnsafePointer ops)
3. Multiple imports from different stdlib modules are used

The crash occurs BEFORE any output, indicating JIT compilation failure rather
than a runtime failure. This differs from modular/modular#6187 which is a
runtime use-after-free.

This reproducer is non-deterministic — it may pass or crash depending on
system load, memory fragmentation, and JIT compiler scheduling. Run multiple
times to increase likelihood of triggering the crash.

Stack trace pattern:
  #0 libKGENCompilerRTShared.so  +0x... (varies with each crash)
  #1 libKGENCompilerRTShared.so  +0x...
  #2 libc.so.6                   +0x... (signal handler)
"""

from std.memory import alloc, UnsafePointer, memset_zero
from std.collections import List


# ============================================================================
# Minimal floating-point array operations
# ============================================================================


struct FloatArray(Movable):
    """Simple float array for heavy alloc/free cycles."""
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var size: Int

    def __init__(out self, size: Int) raises:
        self.data = alloc[Float32](size)
        self.size = size
        for i in range(size):
            self.data[i] = Float32(0.0)

    def __init__(out self, *, deinit take: Self):
        self.data = take.data
        self.size = take.size

    def __del__(deinit self):
        self.data.free()

    def fill(self, value: Float32):
        for i in range(self.size):
            self.data[i] = value

    def sum(self) -> Float32:
        var total = Float32(0.0)
        for i in range(self.size):
            total += self.data[i]
        return total


# ============================================================================
# 30+ test functions with heavy alloc/free cycles
# Each allocates 10-20 FloatArrays and performs operations
# ============================================================================


def step_01() raises:
    """Allocate 15 float arrays of size 1024."""
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(1024)
    var arr12 = FloatArray(1024)
    var arr13 = FloatArray(1024)
    var arr14 = FloatArray(1024)
    var arr15 = FloatArray(1024)
    arr1.fill(1.0)
    _ = arr1.sum() + arr2.sum() + arr3.sum()


def step_02() raises:
    var arr1 = FloatArray(2048)
    var arr2 = FloatArray(2048)
    var arr3 = FloatArray(2048)
    var arr4 = FloatArray(2048)
    var arr5 = FloatArray(2048)
    var arr6 = FloatArray(2048)
    var arr7 = FloatArray(2048)
    var arr8 = FloatArray(2048)
    var arr9 = FloatArray(2048)
    var arr10 = FloatArray(2048)
    var arr11 = FloatArray(512)
    var arr12 = FloatArray(512)
    var arr13 = FloatArray(512)
    var arr14 = FloatArray(512)
    var arr15 = FloatArray(512)
    arr1.fill(2.0)
    _ = arr1.sum()


def step_03() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(1024)
    var arr12 = FloatArray(1024)
    var arr13 = FloatArray(1024)
    var arr14 = FloatArray(1024)
    var arr15 = FloatArray(1024)
    arr1.fill(3.0)
    _ = arr2.sum() + arr3.sum()


def step_04() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(4096)
    var arr12 = FloatArray(4096)
    var arr13 = FloatArray(4096)
    var arr14 = FloatArray(4096)
    var arr15 = FloatArray(4096)
    _ = arr1.sum() + arr11.sum()


def step_05() raises:
    var arr1 = FloatArray(768)
    var arr2 = FloatArray(768)
    var arr3 = FloatArray(768)
    var arr4 = FloatArray(768)
    var arr5 = FloatArray(768)
    var arr6 = FloatArray(768)
    var arr7 = FloatArray(768)
    var arr8 = FloatArray(768)
    var arr9 = FloatArray(768)
    var arr10 = FloatArray(768)
    var arr11 = FloatArray(1536)
    var arr12 = FloatArray(1536)
    var arr13 = FloatArray(1536)
    var arr14 = FloatArray(1536)
    var arr15 = FloatArray(1536)
    arr1.fill(5.0)
    _ = arr15.sum()


def step_06() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(2048)
    var arr12 = FloatArray(2048)
    var arr13 = FloatArray(2048)
    var arr14 = FloatArray(2048)
    var arr15 = FloatArray(2048)
    _ = arr1.sum() + arr6.sum()


def step_07() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(8192)
    var arr12 = FloatArray(8192)
    var arr13 = FloatArray(8192)
    var arr14 = FloatArray(8192)
    var arr15 = FloatArray(8192)
    arr1.fill(7.0)
    _ = arr10.sum()


def step_08() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(4096)
    var arr12 = FloatArray(4096)
    var arr13 = FloatArray(4096)
    var arr14 = FloatArray(4096)
    var arr15 = FloatArray(4096)
    _ = arr5.sum() + arr12.sum()


def step_09() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(2048)
    var arr12 = FloatArray(2048)
    var arr13 = FloatArray(2048)
    var arr14 = FloatArray(2048)
    var arr15 = FloatArray(2048)
    arr1.fill(9.0)
    _ = arr1.sum() + arr14.sum()


def step_10() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(768)
    var arr12 = FloatArray(768)
    var arr13 = FloatArray(768)
    var arr14 = FloatArray(768)
    var arr15 = FloatArray(768)
    _ = arr1.sum() + arr10.sum() + arr15.sum()


def step_11() raises:
    var arr1 = FloatArray(2048)
    var arr2 = FloatArray(2048)
    var arr3 = FloatArray(2048)
    var arr4 = FloatArray(2048)
    var arr5 = FloatArray(2048)
    var arr6 = FloatArray(2048)
    var arr7 = FloatArray(2048)
    var arr8 = FloatArray(2048)
    var arr9 = FloatArray(2048)
    var arr10 = FloatArray(2048)
    var arr11 = FloatArray(1024)
    var arr12 = FloatArray(1024)
    var arr13 = FloatArray(1024)
    var arr14 = FloatArray(1024)
    var arr15 = FloatArray(1024)
    arr1.fill(11.0)
    _ = arr2.sum()


def step_12() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(4096)
    var arr12 = FloatArray(4096)
    var arr13 = FloatArray(4096)
    var arr14 = FloatArray(4096)
    var arr15 = FloatArray(4096)
    arr1.fill(12.0)
    _ = arr8.sum() + arr14.sum()


def step_13() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(3072)
    var arr12 = FloatArray(3072)
    var arr13 = FloatArray(3072)
    var arr14 = FloatArray(3072)
    var arr15 = FloatArray(3072)
    _ = arr1.sum() + arr5.sum() + arr11.sum()


def step_14() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(512)
    var arr12 = FloatArray(512)
    var arr13 = FloatArray(512)
    var arr14 = FloatArray(512)
    var arr15 = FloatArray(512)
    arr1.fill(14.0)
    _ = arr9.sum()


def step_15() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(8192)
    var arr12 = FloatArray(8192)
    var arr13 = FloatArray(8192)
    var arr14 = FloatArray(8192)
    var arr15 = FloatArray(8192)
    _ = arr10.sum() + arr11.sum()


def step_16() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(512)
    var arr12 = FloatArray(512)
    var arr13 = FloatArray(512)
    var arr14 = FloatArray(512)
    var arr15 = FloatArray(512)
    arr1.fill(16.0)
    _ = arr3.sum() + arr15.sum()


def step_17() raises:
    var arr1 = FloatArray(2048)
    var arr2 = FloatArray(2048)
    var arr3 = FloatArray(2048)
    var arr4 = FloatArray(2048)
    var arr5 = FloatArray(2048)
    var arr6 = FloatArray(2048)
    var arr7 = FloatArray(2048)
    var arr8 = FloatArray(2048)
    var arr9 = FloatArray(2048)
    var arr10 = FloatArray(2048)
    var arr11 = FloatArray(1536)
    var arr12 = FloatArray(1536)
    var arr13 = FloatArray(1536)
    var arr14 = FloatArray(1536)
    var arr15 = FloatArray(1536)
    _ = arr1.sum() + arr7.sum() + arr13.sum()


def step_18() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(4096)
    var arr12 = FloatArray(4096)
    var arr13 = FloatArray(4096)
    var arr14 = FloatArray(4096)
    var arr15 = FloatArray(4096)
    arr1.fill(18.0)
    _ = arr4.sum() + arr12.sum()


def step_19() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(768)
    var arr12 = FloatArray(768)
    var arr13 = FloatArray(768)
    var arr14 = FloatArray(768)
    var arr15 = FloatArray(768)
    _ = arr2.sum() + arr8.sum() + arr14.sum()


def step_20() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(2048)
    var arr12 = FloatArray(2048)
    var arr13 = FloatArray(2048)
    var arr14 = FloatArray(2048)
    var arr15 = FloatArray(2048)
    arr1.fill(20.0)
    _ = arr6.sum() + arr11.sum()


def step_21() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(3072)
    var arr12 = FloatArray(3072)
    var arr13 = FloatArray(3072)
    var arr14 = FloatArray(3072)
    var arr15 = FloatArray(3072)
    _ = arr1.sum() + arr9.sum()


def step_22() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(1024)
    var arr12 = FloatArray(1024)
    var arr13 = FloatArray(1024)
    var arr14 = FloatArray(1024)
    var arr15 = FloatArray(1024)
    arr1.fill(22.0)
    _ = arr7.sum() + arr15.sum()


def step_23() raises:
    var arr1 = FloatArray(2048)
    var arr2 = FloatArray(2048)
    var arr3 = FloatArray(2048)
    var arr4 = FloatArray(2048)
    var arr5 = FloatArray(2048)
    var arr6 = FloatArray(2048)
    var arr7 = FloatArray(2048)
    var arr8 = FloatArray(2048)
    var arr9 = FloatArray(2048)
    var arr10 = FloatArray(2048)
    var arr11 = FloatArray(512)
    var arr12 = FloatArray(512)
    var arr13 = FloatArray(512)
    var arr14 = FloatArray(512)
    var arr15 = FloatArray(512)
    _ = arr1.sum() + arr10.sum() + arr15.sum()


def step_24() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(8192)
    var arr12 = FloatArray(8192)
    var arr13 = FloatArray(8192)
    var arr14 = FloatArray(8192)
    var arr15 = FloatArray(8192)
    arr1.fill(24.0)
    _ = arr11.sum() + arr13.sum()


def step_25() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(1536)
    var arr12 = FloatArray(1536)
    var arr13 = FloatArray(1536)
    var arr14 = FloatArray(1536)
    var arr15 = FloatArray(1536)
    _ = arr1.sum() + arr12.sum()


def step_26() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(2048)
    var arr12 = FloatArray(2048)
    var arr13 = FloatArray(2048)
    var arr14 = FloatArray(2048)
    var arr15 = FloatArray(2048)
    arr1.fill(26.0)
    _ = arr3.sum() + arr14.sum()


def step_27() raises:
    var arr1 = FloatArray(2048)
    var arr2 = FloatArray(2048)
    var arr3 = FloatArray(2048)
    var arr4 = FloatArray(2048)
    var arr5 = FloatArray(2048)
    var arr6 = FloatArray(2048)
    var arr7 = FloatArray(2048)
    var arr8 = FloatArray(2048)
    var arr9 = FloatArray(2048)
    var arr10 = FloatArray(2048)
    var arr11 = FloatArray(4096)
    var arr12 = FloatArray(4096)
    var arr13 = FloatArray(4096)
    var arr14 = FloatArray(4096)
    var arr15 = FloatArray(4096)
    _ = arr1.sum() + arr11.sum()


def step_28() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(512)
    var arr12 = FloatArray(512)
    var arr13 = FloatArray(512)
    var arr14 = FloatArray(512)
    var arr15 = FloatArray(512)
    arr1.fill(28.0)
    _ = arr9.sum() + arr15.sum()


def step_29() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(2048)
    var arr12 = FloatArray(2048)
    var arr13 = FloatArray(2048)
    var arr14 = FloatArray(2048)
    var arr15 = FloatArray(2048)
    _ = arr1.sum() + arr15.sum()


def step_30() raises:
    var arr1 = FloatArray(512)
    var arr2 = FloatArray(512)
    var arr3 = FloatArray(512)
    var arr4 = FloatArray(512)
    var arr5 = FloatArray(512)
    var arr6 = FloatArray(512)
    var arr7 = FloatArray(512)
    var arr8 = FloatArray(512)
    var arr9 = FloatArray(512)
    var arr10 = FloatArray(512)
    var arr11 = FloatArray(768)
    var arr12 = FloatArray(768)
    var arr13 = FloatArray(768)
    var arr14 = FloatArray(768)
    var arr15 = FloatArray(768)
    arr1.fill(30.0)
    _ = arr1.sum() + arr10.sum() + arr15.sum()


def step_31() raises:
    var arr1 = FloatArray(256)
    var arr2 = FloatArray(256)
    var arr3 = FloatArray(256)
    var arr4 = FloatArray(256)
    var arr5 = FloatArray(256)
    var arr6 = FloatArray(256)
    var arr7 = FloatArray(256)
    var arr8 = FloatArray(256)
    var arr9 = FloatArray(256)
    var arr10 = FloatArray(256)
    var arr11 = FloatArray(4096)
    var arr12 = FloatArray(4096)
    var arr13 = FloatArray(4096)
    var arr14 = FloatArray(4096)
    var arr15 = FloatArray(4096)
    _ = arr5.sum() + arr11.sum() + arr15.sum()


def step_32() raises:
    var arr1 = FloatArray(1024)
    var arr2 = FloatArray(1024)
    var arr3 = FloatArray(1024)
    var arr4 = FloatArray(1024)
    var arr5 = FloatArray(1024)
    var arr6 = FloatArray(1024)
    var arr7 = FloatArray(1024)
    var arr8 = FloatArray(1024)
    var arr9 = FloatArray(1024)
    var arr10 = FloatArray(1024)
    var arr11 = FloatArray(3072)
    var arr12 = FloatArray(3072)
    var arr13 = FloatArray(3072)
    var arr14 = FloatArray(3072)
    var arr15 = FloatArray(3072)
    arr1.fill(32.0)
    _ = arr10.sum() + arr13.sum()


# ============================================================================
# Main: Call all 32 steps sequentially
# ============================================================================


def main() raises:
    """Run all 32 test steps. May crash during JIT compilation (non-deterministic)."""
    print("Running JIT volume crash reproducer (32 heavy alloc/free steps)...", end="")
    step_01()
    print(".", end="")
    step_02()
    print(".", end="")
    step_03()
    print(".", end="")
    step_04()
    print(".", end="")
    step_05()
    print(".", end="")
    step_06()
    print(".", end="")
    step_07()
    print(".", end="")
    step_08()
    print(".", end="")
    step_09()
    print(".", end="")
    step_10()
    print(".", end="")
    step_11()
    print(".", end="")
    step_12()
    print(".", end="")
    step_13()
    print(".", end="")
    step_14()
    print(".", end="")
    step_15()
    print(".", end="")
    step_16()
    print(".", end="")
    step_17()
    print(".", end="")
    step_18()
    print(".", end="")
    step_19()
    print(".", end="")
    step_20()
    print(".", end="")
    step_21()
    print(".", end="")
    step_22()
    print(".", end="")
    step_23()
    print(".", end="")
    step_24()
    print(".", end="")
    step_25()
    print(".", end="")
    step_26()
    print(".", end="")
    step_27()
    print(".", end="")
    step_28()
    print(".", end="")
    step_29()
    print(".", end="")
    step_30()
    print(".", end="")
    step_31()
    print(".", end="")
    step_32()
    print(" OK — no crash")
