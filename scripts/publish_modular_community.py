#!/usr/bin/env python3
"""Publish projectodyssey's conda recipe to modular/modular-community.

There is no `mojo publish` CLI in Mojo 1.0. Publishing a Mojo library to the
`https://repo.prefix.dev/modular-community` channel works by opening a PR
against `modular/modular-community` that adds a `recipes/<package_name>/`
folder containing the rattler-build recipe.

This script automates that PR via a fork workflow (we have no push access
to `modular/modular-community`):

1. Clones (or refreshes) the HomericIntelligence fork
   (`HomericIntelligence/modular-community`) at
   `$HOME/.cache/projectodyssey/modular-community`. `origin` is the fork;
   an `upstream` remote points at `modular/modular-community` so the fork's
   `main` can be fast-forwarded before branching.
2. Copies `conda.recipe/recipe.yaml` and `conda.recipe/test_import.mojo`
   into `recipes/projectodyssey/`.
3. Rewrites the `source:` block from `path: ..` (local) to a `git:` source
   pinned at the current `HEAD` commit SHA so the recipe is reproducible
   in modular-community's CI.
4. Creates a branch, commits, pushes to the fork, and opens a PR against
   `modular/modular-community` with `--head HomericIntelligence:<branch>`.

Python (not Mojo) per ADR-001: this is GitHub automation involving
subprocess output capture and string templating, both of which Mojo's
runtime is not designed for.

Usage::

    python3 scripts/publish_modular_community.py            # opens PR
    python3 scripts/publish_modular_community.py --dry-run  # prints plan, no push
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGE_NAME = "projectodyssey"
UPSTREAM_REPO_URL = "https://github.com/HomericIntelligence/Odyssey.git"
# We cannot push to modular/modular-community, so PRs come from a fork owned
# by the HomericIntelligence org.
FORK_OWNER = "HomericIntelligence"
MODULAR_COMMUNITY_REPO = "modular/modular-community"
FORK_REPO = f"{FORK_OWNER}/modular-community"
CACHE_DIR = Path.home() / ".cache" / "projectodyssey" / "modular-community"


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return its CompletedProcess (text mode)."""
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def repo_root() -> Path:
    """Resolve the Odyssey repo root from the script's location."""
    return Path(__file__).resolve().parent.parent


def current_commit_sha(root: Path) -> str:
    """Return the current HEAD commit SHA on the source repo."""
    return run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()


def rewrite_source_block(recipe_text: str, commit_sha: str) -> str:
    """Replace the `source:` block in recipe.yaml with a git source.

    The local recipe uses ``source: - path: ..`` for fast iteration during
    development. modular-community needs a reproducible source — a git URL
    pinned to a commit SHA. This rewrite is purely textual; we keep the
    surrounding fields (build, requirements, tests, about) untouched.
    """
    # Match the entire `source:` block up to the next top-level key (`build:`).
    pattern = re.compile(r"^source:\n(?:[ \t]+.*\n|\n)*?(?=^[A-Za-z])", re.MULTILINE)
    replacement = (
        f"source:\n"
        f"  - git: {UPSTREAM_REPO_URL[:-4]}\n"  # strip ".git" — the modular-community recipes don't suffix it
        f"    rev: {commit_sha}\n\n"
    )
    rewritten, n = pattern.subn(replacement, recipe_text, count=1)
    if n != 1:
        raise RuntimeError(
            "could not locate `source:` block in conda.recipe/recipe.yaml — "
            "did the recipe layout change? See scripts/publish_modular_community.py "
            "for the regex this script expects."
        )
    return rewritten


def ensure_clone(dry_run: bool) -> Path:
    """Clone or refresh the fork, syncing its `main` to upstream `modular/main`.

    `origin` is the HomericIntelligence fork (we push branches there).
    `upstream` is `modular/modular-community` (we fast-forward `main` from it
    so each PR branches off current upstream, not a stale fork snapshot).
    """
    if not CACHE_DIR.exists():
        if dry_run:
            print(f"[dry-run] would clone {FORK_REPO} → {CACHE_DIR}")
            return CACHE_DIR
        CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        # `gh repo clone` of a fork auto-adds an `upstream` remote pointing at
        # the parent (modular/modular-community) — no explicit `remote add`.
        run(["gh", "repo", "clone", FORK_REPO, str(CACHE_DIR)])

    if dry_run:
        print(f"[dry-run] would sync {CACHE_DIR} main from upstream/{MODULAR_COMMUNITY_REPO}")
        return CACHE_DIR

    # `upstream` is normally added by `gh repo clone` of a fork; add it only
    # if a pre-fork-workflow cache is missing it.
    remotes = run(["git", "remote"], cwd=CACHE_DIR).stdout.split()
    if "upstream" not in remotes:
        run(
            ["git", "remote", "add", "upstream", f"https://github.com/{MODULAR_COMMUNITY_REPO}.git"],
            cwd=CACHE_DIR,
        )
    run(["git", "fetch", "upstream", "--quiet"], cwd=CACHE_DIR)
    run(["git", "checkout", "main"], cwd=CACHE_DIR)
    run(["git", "reset", "--hard", "upstream/main"], cwd=CACHE_DIR)
    return CACHE_DIR


def copy_recipe(source_root: Path, target_repo: Path, commit_sha: str, dry_run: bool) -> Path:
    """Copy recipe.yaml + test_import.mojo into recipes/projectodyssey/.

    Returns the path to the destination directory.
    """
    src_recipe = source_root / "conda.recipe" / "recipe.yaml"
    src_test = source_root / "conda.recipe" / "test_import.mojo"
    if not src_recipe.exists():
        raise SystemExit(f"missing {src_recipe}; run from Odyssey repo root")
    if not src_test.exists():
        raise SystemExit(f"missing {src_test}")

    dest_dir = target_repo / "recipes" / PACKAGE_NAME
    if dry_run:
        print(f"[dry-run] would write {dest_dir}/recipe.yaml (with git source pinned to {commit_sha})")
        print(f"[dry-run] would write {dest_dir}/test_import.mojo")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    rewritten = rewrite_source_block(src_recipe.read_text(), commit_sha)
    (dest_dir / "recipe.yaml").write_text(rewritten)
    shutil.copy(src_test, dest_dir / "test_import.mojo")
    return dest_dir


def pr_body(commit_sha: str) -> str:
    """Return the body for the modular-community PR."""
    return f"""## Summary

Adds the `projectodyssey` recipe so consumers can install ProjectOdyssey's
Mojo library via `pixi add projectodyssey` from the modular-community
channel.

- **Source**: https://github.com/HomericIntelligence/Odyssey (BSD-3-Clause)
- **Pinned to commit**: `{commit_sha}`

The recipe builds with `mojo package -I src src/projectodyssey -o ${{PREFIX}}/lib/mojo/projectodyssey.mojopkg`
and includes a `tests:` block that imports and instantiates a Tensor and an
AnyTensor from the built `.mojopkg` to prove the package is importable.

## Test Plan

- [ ] `rattler-build build --recipe recipes/projectodyssey/recipe.yaml` succeeds
- [ ] The recipe's `tests:` block passes (smoke import)

Generated by `scripts/publish_modular_community.py` in Odyssey.
"""


def maybe_open_pr(target_repo: Path, branch: str, commit_sha: str, dry_run: bool) -> None:
    """Create a branch, commit, push, and open a PR."""
    title = f"feat: add {PACKAGE_NAME} recipe (pinned at {commit_sha[:7]})"

    if dry_run:
        print()
        print("====== PR title ======")
        print(title)
        print()
        print("====== PR body ======")
        print(pr_body(commit_sha))
        print()
        print(
            f"[dry-run] would: git checkout -b {branch}; git add recipes/{PACKAGE_NAME}; "
            f"commit; push to fork ({FORK_REPO}); "
            f"gh pr create --repo {MODULAR_COMMUNITY_REPO} --head {FORK_OWNER}:{branch}"
        )
        return

    # Branch + commit + push to the fork + PR against upstream.
    run(["git", "checkout", "-B", branch], cwd=target_repo)
    run(["git", "add", f"recipes/{PACKAGE_NAME}"], cwd=target_repo)

    status = run(["git", "status", "--porcelain"], cwd=target_repo).stdout
    if not status.strip():
        print("nothing to commit — recipe already up to date in modular-community")
        return

    run(["git", "commit", "-m", title], cwd=target_repo)
    # `origin` is the fork — push the branch there, then PR cross-repo.
    run(["git", "push", "-u", "--force", "origin", branch], cwd=target_repo)

    result = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            MODULAR_COMMUNITY_REPO,
            "--base",
            "main",
            "--head",
            f"{FORK_OWNER}:{branch}",
            "--title",
            title,
            "--body",
            pr_body(commit_sha),
        ],
        cwd=target_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # Idempotent: PR may already exist.
        if "already exists" in result.stderr:
            print(f"PR already exists for branch {branch}; nothing to do")
            return
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    print(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (recipe rewrite, PR title/body) without pushing or opening a PR.",
    )
    args = parser.parse_args()

    root = repo_root()
    if not (root / "conda.recipe" / "recipe.yaml").exists():
        raise SystemExit(
            f"could not find conda.recipe/recipe.yaml under {root}; run this script from a checkout of ProjectOdyssey"
        )

    sha = current_commit_sha(root)
    print(f"ProjectOdyssey HEAD: {sha}")

    target = ensure_clone(args.dry_run)
    dest = copy_recipe(root, target, sha, args.dry_run)
    print(f"recipe target: {dest}")

    branch = f"projectodyssey-recipe-{sha[:7]}"
    maybe_open_pr(target, branch, sha, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
