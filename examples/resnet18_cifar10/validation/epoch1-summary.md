# ResNet-18 CIFAR-10 — One-Epoch Validation

Verbatim training + test-eval output is in [`epoch1.log`](epoch1.log).

## Provenance

- **Run:** the AOT-compiled `train.mojo` binary, executed detached against the
  real CIFAR-10 IDX dataset (`datasets/cifar10`, 50000 train / 10000 test).
- **Duration:** ~17.8 hours wall-clock, single CPU host, exit 0.
- **Config:** 1 epoch, batch size 128, SGD lr=0.01, momentum=0.9 (391 batches).
- **Emitted precision:** full `String(Float32)` (7 significant digits) — the
  losses below are the raw values the code printed, not rounded.

## Measured metrics (verbatim from the run)

| Metric | Value |
| --- | --- |
| Batch 100 loss | `1.7856864` |
| Batch 200 loss | `1.5843451` |
| Batch 300 loss | `1.4652044` |
| Epoch training loss | `1.3781426` |
| Test loss | `1.0901253` |
| Test top-1 accuracy | `60.02%` |

Training loss descends monotonically across the epoch; test loss (`1.09`) is
below the final training loss, consistent with a single under-fitted epoch. 60%
top-1 after one epoch on CIFAR-10 is a plausible ResNet-18-from-scratch result.

## Evidence discipline (ADR-014)

Per Odysseus ADR-014, a committed log is an *artifact*, not a *gate*: it attests
to a run only insofar as the numbers are reproducible by a channel the author
does not control. The reproduction path is the training entrypoint itself
(`examples/resnet18_cifar10/train.mojo`), which the planned CI training-smoke
(ProjectOdyssey #5551) will execute to machine-check that a real run emits
finite, decreasing, full-precision loss. These numbers are the honest output of
one genuine run; a longer or seeded run may vary.
