#!/usr/bin/env bash
# train.sh — examples/grok/lenet_emnist/ grokking-experiment dispatcher.
#
# Three execution profiles:
#   dry-run  ~30 s   : compile + 1 batch of 1 epoch; verifies the loop runs
#                      end-to-end and writes a checkpoint set. Use after any
#                      code change before committing.
#   smoke    ~5-15 m : verify the memorization phase (train_acc -> 100%) is
#                      reachable on a 1k-sample subset. Does NOT wait for
#                      grokking. Pair with `analyze_phases.py --mode smoke`.
#   full     hours   : real grokking attempt. 30k epochs on a 1k subset,
#                      checkpoints every 500 epochs, log every 50.
#
# Extra args after the profile are appended to the mojo invocation, so e.g.
#   ./train.sh smoke --weight-decay 0.5
# overrides the default weight_decay for that run.
#
# Logs to stdout. Redirect for persistence:
#   ./train.sh full > /tmp/grok_full.log 2>&1
#
# See README.md for the theory + phase interpretation guide.

set -euo pipefail

PROFILE="${1:?usage: $0 {dry-run|smoke|full} [extra mojo flags...]}"
shift || true

# Operate from the repo root so `-I src` resolves projectodyssey.* correctly.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

case "$PROFILE" in
  dry-run)
    PROFILE_FLAGS=(
      --subset-size 64
      --epochs 1
      --max-batches 1
      --log-every 1
      --checkpoint-every 1
    )
    ;;
  smoke)
    PROFILE_FLAGS=(
      --subset-size 1000
      --epochs 300
      --log-every 10
      --checkpoint-every 50
    )
    ;;
  full)
    PROFILE_FLAGS=(
      --subset-size 1000
      --epochs 30000
      --log-every 50
      --checkpoint-every 500
    )
    ;;
  *)
    echo "error: unknown profile '$PROFILE'" >&2
    echo "usage: $0 {dry-run|smoke|full} [extra mojo flags...]" >&2
    exit 2
    ;;
esac

# Common defaults (grokking-aimed; overridable via "$@").
COMMON_FLAGS=(
  --lr 1e-3
  --weight-decay 1.0
  --batch-size 64
  --track-metric test_acc:max
  --track-metric test_loss:both
  --track-metric train_loss:min
  --track-metric weight_l2_norm:max
  --data-dir datasets/emnist
  --weights-dir "examples/grok/lenet_emnist/checkpoints"
)

echo "==> profile: $PROFILE"
echo "==> command: pixi run mojo run -I src examples/grok/lenet_emnist/run_train.mojo ${COMMON_FLAGS[*]} ${PROFILE_FLAGS[*]} $*"
echo

exec pixi run mojo run -I src examples/grok/lenet_emnist/run_train.mojo \
  "${COMMON_FLAGS[@]}" "${PROFILE_FLAGS[@]}" "$@"
