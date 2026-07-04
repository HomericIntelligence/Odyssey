#!/usr/bin/env python3
"""Summarize MobileNetV1 CIFAR-10 epoch log for issue #5526.

Exit codes:
  0  SUCCESS               — mojo passed, ≥3 losses parsed, last < first, all finite
  2  TRAINING_FAILURE      — mojo exited non-zero (training crashed; not our failure)
  3  LOG_FORMAT_MISMATCH   — mojo exited 0 but <3 "Batch N/M - Loss: X" lines found
  4  LOSS_NOT_DECREASING   — mojo exited 0, parse OK, but last_loss >= first_loss
  5  NUMERIC_INSTABILITY   — mojo exited 0, parse OK, but any NaN or inf loss seen
"""

import argparse
import math
import re
import sys

BATCH_RE = re.compile(r"^\s+Batch (\d+)/(\d+) - Loss: (.+)$")
AVG_RE = re.compile(r"^\s+Average Loss: (.+)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--mojo-rc", type=int, required=True)
    args = ap.parse_args()

    if args.mojo_rc != 0:
        print(f"SUMMARY status=TRAINING_FAILURE mojo_rc={args.mojo_rc}")
        return 2

    losses: list[float] = []
    avg: float | None = None
    with open(args.log) as fh:
        for line in fh:
            m = BATCH_RE.match(line.rstrip("\n"))
            if m:
                losses.append(float(m.group(3)))
                continue
            m = AVG_RE.match(line.rstrip("\n"))
            if m:
                avg = float(m.group(1))

    if len(losses) < 3:
        print(
            f"SUMMARY status=LOG_FORMAT_MISMATCH parsed={len(losses)} "
            f"(expected >=3 'Batch N/M - Loss: X' lines from train.mojo:117-126)"
        )
        return 3

    nan_or_inf = any(math.isnan(x) or math.isinf(x) for x in losses) or (
        avg is not None and (math.isnan(avg) or math.isinf(avg))
    )
    first, last = losses[0], losses[-1]
    decreased = last < first

    verdict = (
        "SUCCESS"
        if (decreased and not nan_or_inf)
        else ("NUMERIC_INSTABILITY" if nan_or_inf else "LOSS_NOT_DECREASING")
    )

    # Append machine-parseable summary line (also greppable in CI).
    with open(args.log, "a") as fh:
        fh.write(
            f"SUMMARY status={verdict} parsed={len(losses)} first={first:.6g} "
            f"last={last:.6g} decreased={decreased} nan_or_inf={nan_or_inf} "
            f"avg={avg if avg is not None else 'NA'}\n"
        )
    print(
        f"SUMMARY status={verdict} parsed={len(losses)} first={first:.6g} "
        f"last={last:.6g} decreased={decreased} nan_or_inf={nan_or_inf}"
    )

    if nan_or_inf:
        return 5
    if not decreased:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
