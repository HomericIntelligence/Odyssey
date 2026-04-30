#!/usr/bin/env bash
# check-symlinks.sh — Verify all symlinks in the repo resolve to existing targets.
set -euo pipefail

broken=()
while IFS= read -r -d '' link; do
  if [ ! -e "$link" ]; then
    broken+=("$link -> $(readlink "$link")")
  fi
done < <(find . -not -path './.git/*' -type l -print0)

if [ ${#broken[@]} -gt 0 ]; then
  echo "::error::Broken symlinks found:"
  for b in "${broken[@]}"; do
    echo "  $b"
  done
  exit 1
fi

echo "All symlinks are valid (checked $(find . -not -path './.git/*' -type l | wc -l) symlink(s))."
