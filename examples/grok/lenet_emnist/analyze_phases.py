#!/usr/bin/env python3
"""Analyze a grok/lenet_emnist training log for phase transitions and overfitting.

Parses structured per-epoch lines of the form

    EPOCH 50 train_loss=2.3e-4 train_acc=100.000 test_loss=4.21 test_acc=14.7 weight_l2=18.4

emitted by ``examples/grok/lenet_emnist/train.mojo`` and reports:

- Late-stage overfitting (loss-vs-accuracy decoupling): the diagnosis from the
  ProjectMnemosyne ``training-diagnosis-loss-accuracy-decoupling-overfitting``
  skill. Triggers when test accuracy peaks early then regresses while
  training loss keeps falling.

- Phase 1 (memorization complete): train_acc >= 99 % for >= 3 consecutive
  logged epochs.

- Phase 3 (grokking onset): test_acc jumps by >= 20 percentage points within
  any 200-epoch window while train_acc stays >= 99 %.

Modes:
    --mode dry-run   verify log shape only (>= 1 EPOCH line, NaN-free)
    --mode smoke     report Phase 1 reached or explain why not
    --mode full      full Phase 1/2/3 + overfitting report (default)

Usage:
    python analyze_phases.py --mode smoke /tmp/grok_smoke.log
    ./train.sh smoke 2>&1 | tee /tmp/run.log
    python analyze_phases.py --mode smoke /tmp/run.log

Exit codes:
    0 = analysis complete (regardless of whether grokking observed)
    1 = log malformed / no EPOCH lines / NaN detected (test failure)
    2 = argument error
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Match: "EPOCH 1 train_loss= 4.82926 train_acc= 0.0 test_loss= 3.85 test_acc= 2.13 weight_l2= 0.0"
# Mojo's print() inserts a space after each value, so we accept "key= value" (with optional space
# around the =) rather than the stricter "key=value".
EPOCH_RE = re.compile(r"EPOCH\s+(?P<epoch>\d+)(?P<rest>.*)")
KV_RE = re.compile(r"(?P<key>\w+)\s*=\s*(?P<val>[-+]?(?:nan|inf|\d*\.?\d+(?:[eE][-+]?\d+)?))")


@dataclass
class EpochRow:
    epoch: int
    metrics: dict[str, float]


def parse_log(path: Path) -> list[EpochRow]:
    """Extract EPOCH rows from the log file. Tolerates non-EPOCH chatter around them."""
    rows: list[EpochRow] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        epoch = int(m.group("epoch"))
        rest = m.group("rest") or ""
        metrics: dict[str, float] = {}
        for kv in KV_RE.finditer(rest):
            try:
                metrics[kv.group("key")] = float(kv.group("val"))
            except ValueError:
                continue
        rows.append(EpochRow(epoch=epoch, metrics=metrics))
    return rows


def detect_nans(rows: list[EpochRow]) -> list[tuple[int, str, float]]:
    bad: list[tuple[int, str, float]] = []
    for r in rows:
        for k, v in r.metrics.items():
            if math.isnan(v) or math.isinf(v):
                bad.append((r.epoch, k, v))
    return bad


def diagnose_overfitting(rows: list[EpochRow]) -> dict:
    """Loss-vs-accuracy decoupling diagnosis. Mirrors the criteria in the
    ``training-diagnosis-loss-accuracy-decoupling-overfitting`` skill."""
    accs = [(r.epoch, r.metrics.get("test_acc")) for r in rows if "test_acc" in r.metrics]
    losses = {r.epoch: r.metrics.get("train_loss") for r in rows if "train_loss" in r.metrics}
    if len(accs) < 5:
        return {"applicable": False, "reason": "fewer than 5 test_acc points"}

    peak_acc = max(v for _, v in accs if v is not None)
    peak_epoch = next(e for e, v in accs if v is not None and v == peak_acc)
    final_epoch, final_acc = accs[-1]
    loss_at_peak = losses.get(peak_epoch)
    loss_at_end = losses.get(final_epoch)

    drop_ratio = None
    if loss_at_peak and loss_at_end and loss_at_peak > 0:
        drop_ratio = (loss_at_peak - loss_at_end) / loss_at_peak

    post_peak = [v for e, v in accs if e > peak_epoch and v is not None]
    below_peak_fraction = sum(1 for v in post_peak if v < peak_acc - 0.05) / len(post_peak) if post_peak else 0.0

    decoupling = (
        drop_ratio is not None and drop_ratio > 0.10 and (peak_acc - final_acc) > 0.3 and below_peak_fraction > 0.5
    )

    return {
        "applicable": True,
        "decoupling_detected": decoupling,
        "peak_acc": peak_acc,
        "peak_epoch": peak_epoch,
        "final_acc": final_acc,
        "final_epoch": final_epoch,
        "loss_drop_after_peak_ratio": drop_ratio,
        "post_peak_below_peak_fraction": below_peak_fraction,
    }


def detect_phase1(rows: list[EpochRow], threshold: float = 99.0, consecutive: int = 3) -> dict:
    """Phase 1 = memorization complete: train_acc >= threshold for `consecutive` logged epochs.

    Default is 3 (not 5) because real runs use --log-every 50 to keep logs readable; requiring 5
    logged points at 99% would mean 250 actual epochs of memorization, which is too strict."""
    run = 0
    first_epoch: int | None = None
    for r in rows:
        ta = r.metrics.get("train_acc")
        if ta is None:
            continue
        if ta >= threshold:
            if run == 0:
                first_epoch = r.epoch
            run += 1
            if run >= consecutive:
                return {"reached": True, "first_epoch": first_epoch}
        else:
            run = 0
            first_epoch = None
    last_train_acc = next((r.metrics["train_acc"] for r in reversed(rows) if "train_acc" in r.metrics), None)
    return {
        "reached": False,
        "max_train_acc_seen": max((r.metrics["train_acc"] for r in rows if "train_acc" in r.metrics), default=None),
        "last_train_acc": last_train_acc,
        "needed": threshold,
    }


def detect_phase3(rows: list[EpochRow], jump_pp: float = 20.0, window: int = 200) -> dict:
    """Phase 3 = grokking onset: test_acc rises by >= jump_pp inside any `window`-epoch
    span while train_acc remains >= 99 %."""
    rows_with = [r for r in rows if "test_acc" in r.metrics and "train_acc" in r.metrics]
    if len(rows_with) < 2:
        return {"reached": False, "reason": "insufficient data"}

    for i, r_start in enumerate(rows_with):
        if r_start.metrics["train_acc"] < 99.0:
            continue
        for r_end in rows_with[i + 1 :]:
            if r_end.epoch - r_start.epoch > window:
                break
            if r_end.metrics["train_acc"] < 99.0:
                continue
            jump = r_end.metrics["test_acc"] - r_start.metrics["test_acc"]
            if jump >= jump_pp:
                return {
                    "reached": True,
                    "from_epoch": r_start.epoch,
                    "to_epoch": r_end.epoch,
                    "test_acc_before": r_start.metrics["test_acc"],
                    "test_acc_after": r_end.metrics["test_acc"],
                    "jump_pp": jump,
                }
    return {"reached": False, "reason": "no qualifying window found"}


def emit_dry_run(rows: list[EpochRow]) -> int:
    if not rows:
        print("DRY-RUN FAIL: no EPOCH lines found in log", file=sys.stderr)
        return 1
    nans = detect_nans(rows)
    if nans:
        print(f"DRY-RUN FAIL: {len(nans)} NaN/inf metric(s):", file=sys.stderr)
        for e, k, v in nans[:5]:
            print(f"  epoch={e} {k}={v}", file=sys.stderr)
        return 1
    r = rows[0]
    print(f"DRY-RUN OK: {len(rows)} EPOCH line(s) parsed; first epoch = {r.epoch}")
    print(f"  metrics: {sorted(r.metrics)}")
    return 0


def emit_smoke(rows: list[EpochRow]) -> int:
    if not rows:
        print("SMOKE FAIL: no EPOCH lines found", file=sys.stderr)
        return 1
    nans = detect_nans(rows)
    if nans:
        print(f"SMOKE FAIL: {len(nans)} NaN/inf metric(s) — training diverged", file=sys.stderr)
        return 1
    p1 = detect_phase1(rows)
    if p1["reached"]:
        print(f"SMOKE OK: Phase 1 (memorization) reached at epoch {p1['first_epoch']}")
        print("  Model is fitting the training subset. Ready for `full` run to attempt grokking.")
    else:
        print("SMOKE INCOMPLETE: Phase 1 (memorization) NOT reached")
        print(f"  max train_acc seen: {p1['max_train_acc_seen']:.2f}% (need >= {p1['needed']}%)")
        print(f"  last train_acc:     {p1['last_train_acc']}")
        print("  Either: (a) need more epochs in smoke, (b) subset too large to memorize,")
        print("           or (c) optimizer/lr/wd not driving convergence.")
    return 0


def emit_full(rows: list[EpochRow]) -> int:
    if not rows:
        print("FULL FAIL: no EPOCH lines found", file=sys.stderr)
        return 1
    nans = detect_nans(rows)
    if nans:
        print(f"WARN: {len(nans)} NaN/inf metric(s) encountered during run", file=sys.stderr)
        for e, k, v in nans[:5]:
            print(f"  epoch={e} {k}={v}", file=sys.stderr)

    print(f"Parsed {len(rows)} EPOCH lines (epochs {rows[0].epoch}..{rows[-1].epoch})")
    print()

    p1 = detect_phase1(rows)
    if p1["reached"]:
        print(f"Phase 1 (memorization complete):   reached at epoch {p1['first_epoch']}")
    else:
        print(f"Phase 1 (memorization complete):   NOT reached (max train_acc = {p1['max_train_acc_seen']:.2f}%)")

    p3 = detect_phase3(rows)
    if p3["reached"]:
        print(
            f"Phase 3 (grokking onset):          DETECTED — epochs "
            f"{p3['from_epoch']} -> {p3['to_epoch']}, "
            f"test_acc {p3['test_acc_before']:.2f}% -> {p3['test_acc_after']:.2f}% "
            f"(+{p3['jump_pp']:.2f} pp)"
        )
    else:
        print(f"Phase 3 (grokking onset):          NOT detected ({p3.get('reason', '')})")

    print()

    diag = diagnose_overfitting(rows)
    if diag["applicable"]:
        print("Loss-vs-accuracy decoupling diagnosis:")
        print(f"  peak test_acc:           {diag['peak_acc']:.4f}% at epoch {diag['peak_epoch']}")
        print(f"  final test_acc:          {diag['final_acc']:.4f}% at epoch {diag['final_epoch']}")
        if diag["loss_drop_after_peak_ratio"] is not None:
            print(f"  train_loss drop ratio after peak: {diag['loss_drop_after_peak_ratio'] * 100:.1f}%")
        print(f"  fraction of post-peak epochs below peak: {diag['post_peak_below_peak_fraction'] * 100:.1f}%")
        verdict = (
            "OVERFITTING DETECTED (train loss kept falling while test accuracy regressed)"
            if diag["decoupling_detected"]
            else "no overfitting decoupling signal"
        )
        print(f"  verdict: {verdict}")
    else:
        print(f"Loss-vs-accuracy diagnosis:        skipped ({diag.get('reason', '')})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=("dry-run", "smoke", "full"),
        default="full",
        help="analysis depth (default: full)",
    )
    parser.add_argument(
        "log",
        type=Path,
        help="path to the training log (the file ./train.sh writes to)",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"error: log not found: {args.log}", file=sys.stderr)
        return 2

    rows = parse_log(args.log)
    if args.mode == "dry-run":
        return emit_dry_run(rows)
    if args.mode == "smoke":
        return emit_smoke(rows)
    return emit_full(rows)


if __name__ == "__main__":
    sys.exit(main())
