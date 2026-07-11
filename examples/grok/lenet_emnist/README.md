# Grokking experiment: LeNet-5 on EMNIST subset

A grokking-aimed training regime for LeNet-5 on a small EMNIST subset. Forks
`examples/lenet_emnist/` and adds AdamW with strong weight decay, a subset
sampler, full per-epoch instrumentation (train/test loss + accuracy + weight
L2 norm), and a multi-metric best-checkpoint tracker.

**Architecture**: LeNet-5 (LeCun et al., 1998) — unchanged from
`examples/lenet_emnist/`.

**Dataset**: EMNIST Balanced (47 classes), subsampled at training time.

**Status**: 🧪 **Experimental scaffold** — the training regime is wired and
verified end-to-end. Whether grokking actually occurs on LeNet+EMNIST is a
research question this scaffold is built to investigate; expectations are
calibrated below.

> **Looking for the standard supervised LeNet baseline?** Use
> [`examples/lenet_emnist/`](../../lenet_emnist/). That example uses SGD and the
> full training set and is the right starting point for "I just want LeNet on
> EMNIST to work."

## Why this example exists

A user trained the standard LeNet-5 / EMNIST example for 500 epochs hoping
to observe *grokking* (Power et al. 2022 — delayed generalization where
test accuracy jumps long after training loss plateaus). They observed the
opposite: late-stage overfitting (peak test acc 86.72 % at epoch 246,
slowly regressed to 85.91 % by epoch 500 while training loss kept dropping).
Two skills were captured in ProjectMnemosyne:

- [`training-diagnosis-loss-accuracy-decoupling-overfitting`](https://github.com/HomericIntelligence/ProjectMnemosyne/blob/main/skills/training-diagnosis-loss-accuracy-decoupling-overfitting.md)
  — diagnosing why loss can keep falling while test accuracy plateaus.
- [`training-grokking-preconditions-and-vision-recipe`](https://github.com/HomericIntelligence/ProjectMnemosyne/blob/main/skills/training-grokking-preconditions-and-vision-recipe.md)
  — what grokking actually requires (small subset, AdamW, weight_decay ≈ 1.0,
  10–100× memorization-time of epochs, simple/algorithmic task structure).

This example implements the *recipe* from that second skill.

## Realistic expectations

Even with this scaffold, grokking on LeNet + EMNIST is not guaranteed and
arguably unlikely. Published vision-task grokking (Liu et al. 2023,
*Omnigrok*) used small MLPs on MNIST, not convnets on EMNIST. The convolutional
inductive bias encodes a generalizing prior, which tends to make the model
generalize from epoch 1 rather than going through the memorize-then-grok
sequence. The honest outcome distribution:

| Outcome | Approximate probability |
| --- | --- |
| LeNet (2 conv + FC) on EMNIST shows clear grokking | Very unlikely |
| MLP-only variant on MNIST (closest to Omnigrok) shows grokking | Plausible |
| Any configuration shows interesting *partial* phase transitions | Likely |
| You learn something about the train-loss-vs-test-acc dynamics | Certain |

If you want a known-working grokking recipe, the cleanest single experiment is
the *original* Power et al. setup: modular addition mod 113 with a 2-layer
transformer, 30 % train fraction, `weight_decay=1.0`, AdamW, ~50k epochs. That
is a different example and would live elsewhere in the repo.

## Architecture

LeNet-5 with 47-class output (matches EMNIST Balanced):

```text
Input (1, 1, 28, 28)
  -> Conv2D(6, 5×5)  -> ReLU -> MaxPool(2×2)   (1, 6, 12, 12)
  -> Conv2D(16, 5×5) -> ReLU -> MaxPool(2×2)   (1, 16, 4, 4)
  -> Flatten                                    (1, 256)
  -> Linear(120) -> ReLU
  -> Linear(84)  -> ReLU
  -> Linear(47)                                 logits

~61 k trainable parameters across 10 tensors (conv1/2 kernel+bias,
fc1/2/3 weights+bias).
```

## Quick start

### 1. One-time setup

EMNIST balanced split (~130 MB extracted). If `datasets/emnist/*.idx*` files
already exist you can skip this.

```bash
just download-emnist
```

(If the `just download-emnist` recipe errors after the download — known shell-quoting bug
in the flatten step — manually extract:

```bash
cd datasets/emnist && for f in gzip/emnist-balanced-*.gz; do gunzip -c "$f" > "$(basename "$f" .gz)"; done
```

then verify the four `emnist-balanced-{train,test}-{images,labels}-idx*-ubyte`
files exist.)

### 2. Choose an execution profile and run

```bash
# ~30 s — verify the pipeline runs end-to-end after any change.
./examples/grok/lenet_emnist/train.sh dry-run

# ~5–15 min — verify Phase 1 (train_acc -> 100 %) is reachable on the
# 1k-sample subset. Does NOT wait for grokking.
./examples/grok/lenet_emnist/train.sh smoke

# Hours to a day — real grokking attempt: 30k epochs on 1k samples,
# checkpoints every 500 epochs.
./examples/grok/lenet_emnist/train.sh full
```

Each profile is a single invocation of `pixi run mojo run -I src
examples/grok/lenet_emnist/run_train.mojo` with profile-specific flags
preset. Extra arguments are appended, so e.g. `./train.sh smoke
--weight-decay 0.5` overrides the default for that run.

### 3. Analyze the log

```bash
./examples/grok/lenet_emnist/train.sh full > /tmp/grok_full.log 2>&1
python examples/grok/lenet_emnist/analyze_phases.py --mode full /tmp/grok_full.log
```

The analyzer parses the `EPOCH <n> train_loss=… train_acc=… …` lines and
reports:

- Whether **Phase 1** (memorization complete: train_acc ≥ 99 % for ≥ 3
  consecutive logged epochs) was reached, and at what epoch.
- Whether **Phase 3** (grokking onset: test_acc jumps ≥ 20 pp within a
  200-epoch window while train_acc stays ≥ 99 %) was detected.
- Whether the *loss-vs-accuracy decoupling* fingerprint of late-stage
  overfitting is present (mirroring the diagnostic skill linked above).

There is no separate "Phase 2" detector — Phase 2 (plateau) is simply the
span of logged epochs between Phase 1 reaching and either Phase 3 firing
or the run ending.

## What the training script does differently

Differences from `examples/lenet_emnist/`:

| Aspect | `lenet_emnist/` (baseline) | `grok/lenet_emnist/` (this example) |
| --- | --- | --- |
| Optimizer | SGD | AdamW (decoupled weight decay) |
| Default weight decay | n/a | **1.0** (grokking-aimed; standard regularization values are ~1e-4) |
| Default learning rate | 0.001 | 0.001 |
| Default batch size | 32 | 64 |
| Default epochs | 10 | 1 / 300 / 30000 per profile |
| Training set | Full 112,800 samples | Subsampled to 1,000 (configurable via `--subset-size`) |
| Per-epoch logging | Test accuracy only | train_loss, train_acc, test_loss, test_acc, weight_l2_norm |
| Checkpointing | One save at end of training | Multi-metric best-checkpoint tracker (see below) |

## Multi-metric best-checkpoint tracker

Per-metric tracking modes via `--track-metric NAME:MODE,NAME:MODE,...` (comma-separated, single flag):

| Mode | Files saved | When |
| --- | --- | --- |
| `max` | `{name}_best.bin` | New highest value seen |
| `min` | `{name}_best.bin` | New lowest value seen |
| `both` | `{name}_best.bin`, `{name}_min.bin`, `{name}_min_after_max.bin`, `{name}_max_after_min.bin` | All four; `min_after_max` only after a maximum has been seen, `max_after_min` only after a minimum has been seen |

Each `.bin` file is accompanied by a `.json` sidecar with `{"metric", "mode_kind", "value", "epoch"}`.

Defaults from `train.sh`:

```text
--track-metric "test_acc:max,test_loss:both,train_loss:min,weight_l2_norm:max"
#                ^             ^             ^              ^
#                |             |             |              detect weight-norm growth phase
#                |             |             sanity: training did converge
#                |             detect post-peak loss recovery (grok signal)
#                the headline "best model" checkpoint
```

The `min_after_max` and `max_after_min` distinction matters: a naive
"best loss" tracker is trivially won by epoch 1 (loss starts highest, then
falls monotonically). `min_after_max` only counts once the loss has already
established a maximum and *then* dropped to a new minimum — this is the
grokking-relevant signal (test loss spikes during Phase 2, then collapses
during Phase 3).

## CLI flags (`run_train.mojo`)

| Flag | Default | Purpose |
| --- | --- | --- |
| `--epochs` | 10 | Total training epochs |
| `--batch-size` | 32 | Mini-batch size |
| `--lr` | 0.001 | AdamW learning rate |
| `--weight-decay` | 0.01 | Decoupled weight decay (grokking wants ~1.0) |
| `--precision` | fp32 | Mixed-precision mode (currently always fp32) |
| `--data-dir` | `datasets/emnist` | EMNIST IDX-file directory |
| `--weights-dir` | `lenet5_weights` | Checkpoint output directory |
| `--subset-size` | 0 | If > 0, slice train set to first N samples |
| `--max-batches` | 0 | If > 0, stop training after N batches per epoch (dry-run aid) |
| `--log-every` | 1 | Emit EPOCH structured line every K epochs |
| `--checkpoint-every` | 1 | Run checkpoint update every K epochs |
| `--track-metric` | `test_acc:max` | Comma-separated list of `metric_name:mode` — see table above |

`train.sh` presets a sensible subset of these per profile.

## File structure

```text
examples/grok/lenet_emnist/
├── README.md           # this file
├── model.mojo          # LeNet5 + AdamWState struct + update_parameters_adamw
├── train.mojo          # training loop, instrumentation, MultiMetricCheckpointer
├── run_train.mojo      # CLI wrapper
├── run_infer.mojo      # inference entry point (loads "best" checkpoint)
├── train.sh            # dry-run / smoke / full dispatcher (user-facing)
├── analyze_phases.py   # post-hoc log analyzer
└── checkpoints/        # .gitignore'd; per-metric best checkpoints land here
```

## Implementation notes

- **Manual AdamW, not autograd.** The autograd substrate in
  `src/odyssey/autograd/` is incomplete for convnet training (see
  tracker issue [#5452](https://github.com/HomericIntelligence/Odyssey/issues/5452)):
  `variable_conv2d`, `variable_maxpool2d`, `variable_linear`,
  `variable_cross_entropy`, and `tape.backward()` dispatch are all
  TODO/in-progress per `src/odyssey/autograd/README.md`. So this
  example uses the same manual forward + manual `*_backward` pattern as
  `examples/lenet_emnist/`, but calls
  [`adamw_step`](../../src/odyssey/training/optimizers/adamw.mojo) ten
  times per optimizer step (once per parameter) with its own first/second
  moment state. Once #5452 lands this example will be ported to the autograd
  path along with the others.
- **Weight decay = 1.0 is intentional** and ~100× higher than the standard
  regularization value (1e-4). Grokking depends on weight decay *slowly
  outcompeting* the memorization circuit; smaller decay just produces an
  overfit model.
- **MLP-only mode is on the wishlist but not implemented yet.** The
  published Omnigrok recipe uses a small MLP (no convs) on MNIST. Adding
  a `--mlp-only` flag that bypasses the conv layers would give a stronger
  grokking-attempt baseline. Filed as a follow-up.

## References

- Power, Burns, Smith, Edwards, Misra, Toner — *Grokking: Generalization
  Beyond Overfitting on Small Algorithmic Datasets* (2022).
  <https://arxiv.org/abs/2201.02177>
- Nanda, Chan, Lieberum, Smith, Steinhardt — *Progress Measures for Grokking
  via Mechanistic Interpretability* (ICLR 2023).
  <https://arxiv.org/abs/2301.05217>
- Liu, Michaud, Tegmark — *Omnigrok: Grokking Beyond Algorithmic Data*
  (2023). <https://arxiv.org/abs/2210.01117>
- Thilak, Littwin, Zhai, Saremi, Paiss, Susskind — *The Slingshot
  Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking
  Phenomenon* (2022). <https://arxiv.org/abs/2206.04817>
- Loshchilov, Hutter — *Decoupled Weight Decay Regularization* (ICLR 2019).
  <https://arxiv.org/abs/1711.05101>
- LeCun, Bottou, Bengio, Haffner — *Gradient-based learning applied to
  document recognition* (1998). The LeNet-5 paper. DOI:
  [10.1109/5.726791](https://doi.org/10.1109/5.726791)

## Related issues

- [#5452 — `feat(autograd): complete autograd substrate for convnet training`](https://github.com/HomericIntelligence/Odyssey/issues/5452)
- [#5449 — `feat(optim): add Muon optimizer`](https://github.com/HomericIntelligence/Odyssey/issues/5449)
- [#5450 — `feat(optim): add NorMuon optimizer`](https://github.com/HomericIntelligence/Odyssey/issues/5450)
- [#5451 — `feat(optim): add Lion and Shampoo`](https://github.com/HomericIntelligence/Odyssey/issues/5451)

Once these land, this example can be the natural benchmark for optimizer
comparisons on a controlled small-data regime.
