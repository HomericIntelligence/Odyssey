#!/usr/bin/env bash
# Validation runner for issue #5526. Untracked artifacts under logs/ (gitignored).
set -uo pipefail   # NOTE: no -e — we need the Mojo exit code, not a shell abort.

DATE=$(date +%F)
LOG="logs/mobilenetv1-cifar10-epoch-${DATE}.log"
mkdir -p logs

# Helper function to run command in container (non-interactive)
run_in_container() {
    # Detect whether we're already inside the dev container
    if [ -f /.dockerenv ] || [ -n "${container:-}" ] || grep -q podman /proc/1/cgroup 2>/dev/null; then
        bash -lc "$*"
    else
        # Host path: podman compose exec (not docker)
        podman compose exec -T odyssey-dev bash -lc "$*"
    fi
}

# Header (gitignored file, so free to include full command line + env).
{
    echo "# MobileNetV1 CIFAR-10 — one-epoch validation for issue #5526"
    echo "# Date: ${DATE}"
    echo "# Command: uv run mojo run -I src -I . examples/mobilenetv1_cifar10/train.mojo --epochs 1 --batch-size 128 --lr 0.01 --data-dir datasets/cifar10"
    echo "# Subset: full epoch (num_batches = ceil(50000/128) = 391)"
    echo "# Started: $(date -Iseconds)"
    echo "# ---"
} > "$LOG"

# Ensure dataset is present (CIFAR-10 is ~170MB; download once).
if [ ! -d "datasets/cifar10" ] || [ -z "$(ls -A datasets/cifar10 2>/dev/null)" ]; then
    echo "# Downloading CIFAR-10 to datasets/cifar10/..." | tee -a "$LOG"
    run_in_container "cd /workspace && uv run python3 examples/mobilenetv1_cifar10/download_cifar10.py cifar10 datasets/cifar10" 2>&1 | tee -a "$LOG"
fi

# Run one epoch — tee stdout+stderr, preserve Mojo exit code via PIPESTATUS.
# Note: -I src is required so mojo can find the odyssey package
run_in_container "cd /workspace && uv run mojo run -I src -I . examples/mobilenetv1_cifar10/train.mojo --epochs 1 --batch-size 128 --lr 0.01 --data-dir datasets/cifar10" 2>&1 | tee -a "$LOG"
MOJO_RC=${PIPESTATUS[0]}

echo "# Mojo exit code: ${MOJO_RC}" >> "$LOG"
echo "# Finished: $(date -Iseconds)" >> "$LOG"

# Delegate verdict — summarizer returns distinct exit codes per failure mode.
python3 scripts/summarize_epoch_log.py --log "$LOG" --mojo-rc "${MOJO_RC}"
