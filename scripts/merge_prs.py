#!/usr/bin/env python3
"""
Merge open PRs with successful CI/CD into main using PR.merge(merge_method='rebase').
Always prints check runs. Supports --dry-run and --push-all (push head branches even if checks fail).

Repo: HomericIntelligence/ProjectOdyssey
Requires: PyGithub (pip install PyGithub), Git installed locally

Usage:
  python merge_prs_rebase.py [--dry-run] [--push-all]

Flags:
  --dry-run    Print git and API actions without executing them.
  --push-all   Push every PR head branch to origin before attempting merge, even when CI/CD is not successful.
"""

import os
import sys
import argparse
import subprocess
from github import Github

REPO_NAME = "HomericIntelligence/ProjectOdyssey"


def run(cmd, dry_run=False, cwd=None):
    if dry_run:
        print(f"[DRY-RUN] $ {' '.join(cmd)}")
        return
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def checks_success_and_print(commit):
    """
    Use Checks API when available; print every check run.
    Return (Boolean success, checks_list) or (None, []) if no check runs present.
    """
    checks = list(commit.get_check_runs())
    bad = {"failure", "timed_out", "cancelled", "action_required"}
    any_success = False

    if checks:
        for cr in checks:
            print(f"    - {cr.name}: status={cr.status}, conclusion={cr.conclusion}")
            if cr.status != "completed":
                return False, checks
            if cr.conclusion in bad:
                return False, checks
            if cr.conclusion == "success":
                any_success = True
        return any_success, checks

    return None, []


def legacy_status_and_print(commit):
    combined = commit.get_combined_status()
    for ctx in combined.statuses:
        print(f"    - {ctx.context}: state={ctx.state}, description={ctx.description}")
    return combined.state or "unknown"


def local_branch_exists(branch_name):
    try:
        out = subprocess.check_output(["git", "branch", "--list", branch_name], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def try_push_head_branch(head_branch, dry_run):
    """Push local head branch to origin if it exists locally; otherwise skip (remote assumed present)."""
    if dry_run:
        print(f"[DRY-RUN] Would push local branch '{head_branch}' to origin if it exists locally.")
        return
    if local_branch_exists(head_branch):
        run(["git", "push", "origin", f"{head_branch}:{head_branch}"], dry_run=False)
    else:
        print(f"  Local branch '{head_branch}' not found; assuming remote branch already present.")


def handle_merge_result(result, pr_number, base_branch):
    # PyGithub returns a PullRequestMergeStatus-like object; use attributes
    try:
        merged = getattr(result, "merged", None)
        message = getattr(result, "message", None)
        sha = getattr(result, "sha", None)
    except Exception:
        # Fallback for unexpected types
        merged = False
        message = str(result)
        sha = None

    if merged:
        print(f"  🎉 PR #{pr_number} merged into {base_branch} via rebase. sha={sha}")
    else:
        print(f"  Failed to merge PR #{pr_number}. API message: {message}")


def main():
    parser = argparse.ArgumentParser(description="Merge open PRs with successful CI/CD into main (rebase via PR API).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and API actions without executing.",
    )
    parser.add_argument(
        "--push-all",
        action="store_true",
        help="Push all PR head branches to origin even if CI/CD failed.",
    )
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print(
            "Error: Please set GITHUB_TOKEN environment variable with a token that has 'repo' scope.",
            file=sys.stderr,
        )
        sys.exit(1)

    gh = Github(token)
    try:
        repo = gh.get_repo(REPO_NAME)
    except Exception as e:
        print(f"Error accessing repo {REPO_NAME}: {e}", file=sys.stderr)
        sys.exit(1)

    print("Updating local 'main'...")
    run(["git", "checkout", "main"], dry_run=args.dry_run)
    run(["git", "pull", "origin", "main"], dry_run=args.dry_run)

    for pr in repo.get_pulls(state="open", sort="created"):
        head_branch = pr.head.ref
        base_branch = pr.base.ref
        print(f"\nChecking PR #{pr.number}: {head_branch} -> {base_branch}")

        try:
            commit = repo.get_commit(pr.head.sha)
        except Exception as e:
            print(f"  Unable to retrieve head commit for PR #{pr.number}: {e}")
            continue

        print("  Checks API results:")
        success, checks = checks_success_and_print(commit)

        if success is None:
            print("  No check runs found; falling back to legacy status contexts:")
            state = legacy_status_and_print(commit)
            print(f"  Legacy combined state: {state}")
            ci_success = state == "success"
        else:
            ci_success = bool(success)

        print(f"  CI result interpreted as: {'success' if ci_success else 'not-success'}")

        # If push-all is requested, push the head branch even if CI failed.
        if args.push_all:
            print(f"  --push-all enabled: pushing head branch '{head_branch}' to origin (if local exists).")
            try:
                try_push_head_branch(head_branch, args.dry_run)
            except Exception as e:
                print(f"  Warning: failed to push head branch '{head_branch}': {e}")

        # If CI not successful and not push-all, skip merging
        if not ci_success and not args.push_all:
            print(f"  Skipping PR #{pr.number}: CI/CD not successful and --push-all not enabled.")
            continue

        # At this point either CI succeeded or --push-all was set; attempt merge via PR API (rebase)
        print(f"  ✅ Proceeding to merge PR #{pr.number} via PR API (rebase).")

        if args.dry_run:
            print(f"[DRY-RUN] Would call pr.merge(merge_method='rebase') for PR #{pr.number}")
            continue

        try:
            result = pr.merge(merge_method="rebase")
        except Exception as e:
            print(f"  Error calling pr.merge for PR #{pr.number}: {e}")
            continue

        handle_merge_result(result, pr.number, base_branch)

    print("\nDone." + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
