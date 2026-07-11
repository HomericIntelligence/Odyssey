# Checkpoint Backup and Disaster Recovery

This document describes the recommended strategy for backing up training
checkpoints and recovering from data loss events. The training infrastructure
under `src/odyssey/training/` writes checkpoints to local disk; persistence and
backup are the operator's responsibility.

## Backup recommendation

For long training runs (≥ 1 hour or ≥ 10 epochs), copy completed checkpoints
to durable storage outside the training host:

```bash
# Example: rsync checkpoints to a backup target after each save
rsync -av --delete checkpoints/ /mnt/backup/odyssey-checkpoints/
```

For high-value models, push checkpoint artifacts to remote object storage
(S3, GCS, Azure Blob, internal artifact registry). The training loop does not
do this automatically — wire it up in your `TrainingConfig` callback or run a
sidecar process.

## Retention

Keep the last `N` checkpoints locally (e.g., `keep_last=5`) plus the
best-validation checkpoint. Backups should retain the best-validation
checkpoint and any checkpoint flagged as a release candidate; everything
else can age out per a policy of your choosing.

## Disaster recovery

If the training host is lost:

1. Identify the most recent checkpoint in your backup target.
2. Restore to a fresh host with the same Mojo / pixi environment.
3. Resume training with `just train --resume <path/to/checkpoint>` (when
   resume support lands — tracked in #5184).

Until #5184 (checkpoint save/load and recovery) ships, "resume" means
restarting training and loading the latest weight file at initialization.
This loses the optimizer state.

## See also

- #5184 — Add training checkpoint save/load and recovery mechanism
- #5183 — Add training operation timeouts and graceful shutdown
- `src/odyssey/training/checkpoint.mojo` — checkpoint serialization primitives
