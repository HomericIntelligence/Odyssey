# ML Odyssey Learnings from Production Fixes

Generalized lessons extracted from root-level fix documents. These inform
best practices for Mojo tensor ops, memory management, and debugging.

## Key Learnings Table

| Theme | Problem | Generalized Solution | Relevant Files | Agent Integration Targets |
| --- | --- | --- | --- | --- |
| **Memory Leaks (reshape/slice)** | Dummy allocations orphaned when overwriting pointers in views. | Free intermediate allocations explicitly before reassigning `_data`; prefer refcounted views. Mark `_is_view=True`. | `BUGFIX_MEMORY_LEAK.md`, `PHASE2_MEMORY_SAFETY_SUMMARY.md` | `.claude/skills/mojo-memory-check/SKILL.md`, performance agents |
| **Broadcasting Crashes** | Incorrect multi-dim index calc (right-to-left strides). Out-of-bounds access. | Precompute row-major strides; extract coords left-to-right via // %; validate shapes pre-op. | `BROADCAST_CRASH_FIX.md` | Safety/review agents (e.g., `mojo-type-safety/SKILL.md`) |
| **Transpose Memory Corruption** | `List[Int](ndim)` wrong size; indexing uninit elems. | Use `List[Int]()` + `.append()`; build/reverse lists safely. | `BUGFIX_TRANSPOSE_MEMORY_CORRUPTION.md` | Mojo review agents (e.g., `mojo-format/SKILL.md`, `mojo-type-safety`) |
| **List Constructor Bugs (8 instances)** | `List[Int](n)` undefined size; index crashes in shape/accuracy/confusion/DataLoader. | ALWAYS `List[Int]()` empty + append; never index constructor. | `LIST_CONSTRUCTOR_FIXES_SUMMARY.md` | Linting agents (e.g., `quality-fix-formatting/SKILL.md`) |
| **Tensor Refcounting (Phase 2)** | Missing lifetime tracking; double-free/use-after-free in views. | Add `_refcount: UnsafePointer[Int]`; `__copyinit__` incr, `__del__` decr/free if 0. | `PHASE2_MEMORY_SAFETY_SUMMARY.md` | `.claude/agents/performance-specialist.md`, memory skills |

## Best Practices

- **TDD Isolation**: Start broad (full train), narrow (forward), isolate layer/op, reproduce minimal test.
- **Mojo Lists**: Append-only; test List behaviors explicitly.
- **Memory Profiling**: Stress loops (10k iters); monitor tcmalloc/Valgrind.
- **Index Calc**: Always precompute strides; print coords/indices in debug.

See `docs/dev/fixes.md` for detailed fix histories. Updated: 2025-11-24
