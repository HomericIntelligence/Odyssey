"""Probe: does the exact `AtomicStats.update_peak_cached` held-pointer shape
still corrupt on Mojo 1.0.0 stable?

The prior WAR removal for modular/modular#6959 (PR #5804) was validated only
against tests that use single-expression escapes (`_counter(off)[].load()` —
pointer consumed in the same statement).  The standalone repro
(repro/repro_6959_inline.mojo) still FAILS on stable, and it uses the
held-in-local shape (`var ptr = obj._as_atomic()` then `ptr[]` in later
statements).

This probe mirrors `update_peak_cached` exactly:

    var cached = self._counter(_ASTATS_BYTES_CACHED)[].load()   # self last used here?
    var peak_ptr = self._counter(_ASTATS_PEAK_CACHED)            # pointer held in local
    if cached > peak_ptr[].load():
        peak_ptr[].max(Int64(cached))

`bytes_cached` is a RUNNING TOTAL (delta-based: add_bytes_cached can be
negative), so the model here mirrors that.  The regression case is cached
dropping BELOW peak — peak must stay.

Run: mojo run docs/dev/reproducers/probe_update_peak_shape.mojo
"""

from std.atomic import Atomic
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal


struct AtomicStats(Copyable, Movable):
    """Mirror of Odyssey's AtomicStats (src/odyssey/base/memory_pool.mojo)."""

    var _data: Pointer[UInt8, MutUntrackedOrigin]

    def __init__(out self):
        self._data = unsafe_alloc[UInt8](8 * 8)
        for i in range(8 * 8):
            self._data[unsafe_offset=i] = 0

    def _counter(
        self, offset: Int
    ) -> Pointer[Atomic[DType.int64], MutUntrackedOrigin]:
        return self._data.unsafe_offset(offset).unsafe_bitcast[
            Atomic[DType.int64]
        ]()

    def add_bytes_cached(mut self, n: Int):
        _ = self._counter(16)[].fetch_add(Int64(n))

    def add_bytes_allocated(mut self, n: Int):
        _ = self._counter(8)[].fetch_add(Int64(n))

    def update_peak_cached(mut self):
        """Byte-for-byte the production shape (post-#5804 revert)."""
        var cached = self._counter(16)[].load()
        var peak_ptr = self._counter(24)
        if cached > peak_ptr[].load():
            peak_ptr[].max(Int64(cached))

    def peak(mut self) -> Int64:
        return self._counter(24)[].load()

    def __deinit__(deinit self):
        self._data.unsafe_free()


struct Pool(Copyable, Movable):
    """Outer owner, mirroring MemoryPool holding AtomicStats as a field."""

    var stats: AtomicStats

    def __init__(out self):
        self.stats = AtomicStats()

    def update(mut self, cached_delta: Int):
        """Delta update: bytes_cached running total += cached_delta."""
        self.stats.add_bytes_cached(cached_delta)
        self.stats.update_peak_cached()

    def peak(mut self) -> Int64:
        return self.stats.peak()


def main() raises:
    print("=== Probe: update_peak_cached held-pointer shape on stable ===")

    # Running total semantics: cached accumulates; peak = max over time.
    # cached: 0 -> +500 -> 500, peak = 500
    var p = Pool()
    p.update(500)
    var peak1 = Int(p.peak())
    print("  after +500: peak =", peak1, "(expected 500)")
    assert_equal(peak1, 500, "peak must track cached (500)")

    # cached: 500 -> +300 -> 800, peak = 800
    p.update(300)
    var peak2 = Int(p.peak())
    print("  after +300: peak =", peak2, "(expected 800)")
    assert_equal(peak2, 800, "peak must track new high (800)")

    # cached: 800 -> -600 -> 200 (drops below peak) -> peak must stay 800
    p.update(-600)
    var peak3 = Int(p.peak())
    print("  after -600: peak =", peak3, "(expected 800, unchanged)")
    assert_equal(peak3, 800, "peak must not regress (800)")

    # cached: 200 -> +700 -> 900 (new high) -> peak = 900
    p.update(700)
    var peak4 = Int(p.peak())
    print("  after +700: peak =", peak4, "(expected 900)")
    assert_equal(peak4, 900, "peak must track new high (900)")

    print("All assertions passed!")

    # Heavy churn to shake out latent UAF.  `update` applies deltas, so the
    # running total after N=1000 iterations of i*7 is 7 * sum(0..999)
    # = 7 * 999*1000/2 = 3,496,500 — and peak must track that.
    var q = Pool()
    for i in range(1000):
        q.update(i * 7)
    var churn_peak = Int(q.peak())
    var expect = 7 * 999 * 1000 // 2
    print("  churn peak:", churn_peak, "(expected", expect, ")")
    assert_equal(churn_peak, expect, "peak must track max over churn")
    print("All assertions passed (incl. churn)!")
