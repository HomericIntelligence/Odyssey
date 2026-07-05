# MobileNetV1 CIFAR-10 — One-Epoch Validation Summary

Validation evidence for issue #5527 (epic #5528, upstream #3187): one full
training epoch of MobileNetV1 on CIFAR-10, executed with
`scripts/run_mobilenetv1_cifar10_epoch.sh`.

## Run provenance

| Field | Value |
| --- | --- |
| Date | 2026-07-04 |
| Command | `pixi run mojo run -I src -I . examples/mobilenetv1_cifar10/train.mojo --epochs 1 --batch-size 128 --lr 0.01 --data-dir datasets/cifar10` |
| Started | 2026-07-04T13:05:59-07:00 |
| Finished | 2026-07-04T13:39:06-07:00 (33 min wall clock, CPU) |
| Mojo exit code | 0 |
| Batches | 391 (full epoch, ceil(50000/128)) |
| Model | MobileNetV1, ~4.2M parameters, 13 depthwise-separable blocks |
| Optimizer | SGD momentum, 110 velocity tensors, lr 0.01 |

## Results

| Checkpoint | Loss |
| --- | --- |
| Batch 100/391 | 2.330883 |
| Batch 200/391 | 2.1634257 |
| Batch 300/391 | 2.054407 |
| Epoch average | 1.9744526 |

Summarizer verdict (`scripts/summarize_epoch_log.py`):

```text
SUMMARY status=SUCCESS parsed=3 first=2.33088 last=2.05441 decreased=True nan_or_inf=False avg=1.9744526
```

Loss decreased monotonically across the epoch with no NaN/Inf values;
weights were saved to `weights/` at completion.

## Evidence

The runner output is committed alongside this file as `epoch1.log`
(force-added past the repo `*.log` ignore rule, as validation evidence is the
deliverable of this issue). The file is a copy of the gitignored operational
log `logs/mobilenetv1-cifar10-epoch-2026-07-04.log` from the execution host
(which remains there for cross-checking), including the `SUMMARY` verdict
lines that `scripts/summarize_epoch_log.py` appends to the log on each
invocation — it appears twice because the summarizer was run twice against
the finished log. No line of program output was added, removed, or edited.
