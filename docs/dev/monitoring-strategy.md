# Monitoring & Divergence Detection Strategy

> **Scope**: Training observability for the ProjectOdyssey ML library.
> This is a *library*, not a daemon — there are no service-level health checks
> or Prometheus push targets. All monitoring is done by the training loop and
> surfaced to the caller.

## Metrics That Matter

| Metric | Where Collected | Why |
| --- | --- | --- |
| Training loss (batch + epoch avg) | `src/projectodyssey/training/metrics/loss_tracker.mojo` — `LossTracker` / `Statistics` | Primary signal of learning |
| Validation accuracy | `src/projectodyssey/training/metrics/accuracy.mojo` — `top1_accuracy` | Generalization health |
| Global gradient norm (pre-clip) | `src/projectodyssey/training/gradient_clipping.mojo` — `compute_gradient_norm_list` | Exploding gradient early warning |
| Per-component loss | `src/projectodyssey/training/metrics/loss_tracker.mojo` — multi-component support | Diagnose regularization vs reconstruction |
| CSV metric log | `src/projectodyssey/training/metrics/csv_metrics_logger.mojo` — `CSVMetricsLogger` | Offline analysis / plotting |

## What Is Already Collected

The metrics package (`src/projectodyssey/training/metrics/`) collects the following at training time:

- **`LossTracker`** — circular-buffer moving average, Welford variance, min/max/mean/std
  per component. Supports arbitrary named loss components.
- **`top1_accuracy` / `top_k_accuracy`** — batch-level and accumulated accuracy.
- **`CSVMetricsLogger`** — writes per-step and per-epoch scalars to `<log_dir>/<metric>.csv`;
  integrates via the `Callback` trait so it fires automatically on `on_epoch_end`.
- **`compute_gradient_norm_list`** — returns the global L2 norm across all parameter
  gradients before clipping is applied; already surfaced through `clip_gradients_by_global_norm`.

## Divergence Detection Heuristics

The following heuristics indicate training has diverged. They should be checked
inside the training loop (or a `Callback` implementation) at the end of each epoch.

### NaN / Inf Loss

```text
if isnan(loss) or isinf(loss):
    raise "Training diverged: loss is NaN/Inf at epoch N"
```

Causes: too-high learning rate, missing gradient clipping, bad data normalization.

### Exploding Gradient Norm

```text
GRADIENT_NORM_THRESHOLD = 100.0   # empirical; tune per model

if gradient_norm > GRADIENT_NORM_THRESHOLD:
    warn("Gradient norm {gradient_norm:.1f} exceeds threshold")
```

`gradient_clipping.mojo` returns `total_norm` from `clip_gradients_by_global_norm` — log
this value every epoch. A norm that grows monotonically across epochs is a pre-divergence
signal even before it crosses the threshold.

### Accuracy Plateau

```text
PLATEAU_PATIENCE = 10   # epochs with no improvement

if best_val_acc unchanged for PLATEAU_PATIENCE epochs:
    warn("Validation accuracy has not improved for N epochs")
```

Distinguish plateau (no progress) from oscillation (high variance). Both are actionable.

### Loss Monotonically Increasing

```text
if current_epoch_loss > previous_epoch_loss for 3+ consecutive epochs:
    warn("Loss is increasing — consider reducing learning rate")
```

## Operator Recommendations

### Local Logging (Default)

Use `CSVMetricsLogger` — it requires no external dependencies and produces files
compatible with pandas, Excel, and standard plotting tools.

```text
logs/
  <run_name>/
    train_loss.csv      # (step, value)
    val_accuracy.csv
    gradient_norm.csv
```

Recommended retention: keep the last 5 run directories; archive older runs.

### Export to Prometheus / External Systems

ProjectOdyssey does not ship a Prometheus exporter — it is a library, not a server.
If you need time-series dashboards, read the CSV files with a sidecar scraper
(e.g., `prometheus-csv-exporter`) or write a thin Python wrapper around `CSVMetricsLogger`.

### What NOT to Do

- Do not add HTTP endpoints or background threads to the training loop — this is
  a library; callers control the execution model.
- Do not poll GPU utilization inside the library — use system-level tools
  (e.g., `nvidia-smi`, `nvitop`) alongside the process.

## Integration Pattern

```mojo
from projectodyssey.training.metrics.csv_metrics_logger import CSVMetricsLogger
from projectodyssey.training.gradient_clipping import clip_gradients_by_global_norm

# Setup
var logger = CSVMetricsLogger("logs/run1")

# Inside training loop
var grad_norm = clip_gradients_by_global_norm(gradients, max_norm=1.0)
logger.log_scalar("gradient_norm", Float64(grad_norm))
logger.log_scalar("train_loss", Float64(batch_loss))
logger.step()

# Divergence guard (per epoch)
if isnan(epoch_loss) or grad_norm > 100.0:
    _ = logger.save()
    raise Error("Training diverged")
```

## See Also

- `src/projectodyssey/training/metrics/` — all metric implementations
- `src/projectodyssey/training/gradient_clipping.mojo` — `compute_gradient_norm_list`,
  `compute_gradient_statistics`
- `docs/dev/testing-strategy.md` — tier-1/tier-2 test strategy
- GitHub issue #5318 — original audit finding
