# Phase G: Corrected Implementation Plan

## Critical Corrections from Prior Plan

The previous plan contained factual errors that have been corrected:

1. **PR #5445 Status**: ✅ Already MERGED on 2026-05-24 — no unblocking work needed. Issue #5061 is resolved.
2. **Open Issue Count**: 30 (not 28) — PR #5445 merging closed one issue post-plan.
3. **File Paths**: Corrected to include `src/projectodyssey/` prefix
   - `src/projectodyssey/tensor/typed/numerical_safety.mojo` (was missing prefix)
   - `src/projectodyssey/core/numerical_safety.mojo` (was missing prefix)
   - `src/projectodyssey/training/gradient_clipping.mojo` (was missing prefix)
4. **Function Location**: `compute_gradient_statistics()` at line 438 in `src/projectodyssey/training/gradient_clipping.mojo`

## Actual Phase G Work (Corrected Scope)

### G1: Triage (COMPLETE)

- ✅ 94 issues classified in prior comment
- ✅ 35 EASY + 33 MEDIUM + 17 HARD + 9 DEFERRED

### G2/G3: EASY/MEDIUM Dispatch (In Progress)

**Current Backlog (30 open issues)**:

| Tier | Count | Issues | Status |
|------|-------|--------|--------|
| EASY | 2 | #3740, #5153 | Ready for dispatch |
| MEDIUM | 6 | #5132, #5134, #5135, #5141, #5157, #3684 | Ready for dispatch |
| HARD | 12 | #3181, #3184, #3187, #5040, #5156, #5159, #5181, #5182, #5183, #5184, #5316, #5391 | Deferred |
| DEFERRED | 6 | #3059, #3067, #3079, #5048, #5191, #5280 | Deferred |
| Epic | 1 | #5354 (this issue) | To close |
| In-Flight (MERGED) | 1 | #5061 (closed by #5445) | Done |
| New tracker issues | 2 | #5449, #5450, #5451, #5454 | Out of scope |

**Total: 30 = 2 EASY + 6 MEDIUM + 12 HARD + 6 DEFERRED + 1 epic + 3 new trackers**

### Files to Modify

For dispatch agents (corrected paths):

- **#5134**: `src/projectodyssey/tensor/typed/numerical_safety.mojo` — SIMD-vectorize
  `_tensor_min_core`, `_tensor_max_core`, `_compute_l2_norm_core`
- **#5135**: `src/projectodyssey/core/numerical_safety.mojo` — SIMD-vectorize `clip_grad_value_`, `clip_grad_norm_`, `clip_grad_global_norm_`
- **#5141**: `src/projectodyssey/training/gradient_clipping.mojo:438-515` — SIMD-vectorize `compute_gradient_statistics()`
- **#5132**: `src/projectodyssey/core/` — Wire `pooled_alloc`/`pooled_free` to `TensorMemoryPool`
- **#5157**: `src/projectodyssey/core/any_tensor.mojo` — Consolidate 5 `__init__` constructors
- **#3684**: `src/projectodyssey/data/dataloader.mojo` — Fix `DataLoader.next()` placeholder
- **#3740**: `tests/shared/model/test_sequential*.mojo` — Verify Sequential2/3 compile in CI
- **#5153**: Audit report (doc-only, no code change)

### Verification Commands (Corrected)

```bash
# Check G1 triage table exists
gh issue view 5354 --comments | grep -E "Total.*94"

# Check current open issue count
gh issue list --state open --limit 200 --json number --jq 'length'

# Verify no duplicates exist for dispatch targets
for issue in 3740 5153 5132 5134 5135 5141 5157 3684; do
  echo "Issue #$issue:"
  gh pr list --json number,title | jq ".[] | select(.title | contains(\"#$issue\")) | .number"
done

# Verify PR #5445 is merged
gh pr view 5445 --json state,title
```

## Next Steps

1. Dispatch Wave 1 (SIMD + pooled_alloc): #5134, #5135, #5141, #5132 (4 agents, parallel)
2. Dispatch Wave 2: #5157, #3684, #3740, #5153 (4 agents, parallel)
3. Monitor/close-out: Watch PRs, close merged issues that haven't auto-closed
4. Post G4 summary: Tier counts, PR count, issues closed, HARD/DEFERRED carry-forward
5. Close epic #5354
