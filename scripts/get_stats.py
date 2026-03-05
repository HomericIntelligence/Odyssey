#!/usr/bin/env python3
import os
import sys
from github import Github, Auth
from datetime import datetime, timezone
from prettytable import PrettyTable

# --- CONFIG ---
REPO_NAME = "HomericIntelligence/ProjectOdyssey"
AUTHOR = "mvillmow"

# --- Parse command-line arguments ---
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <start-date YYYY-MM-DD> <end-date YYYY-MM-DD>")
    sys.exit(1)

try:
    START = datetime.strptime(sys.argv[1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    END = datetime.strptime(sys.argv[2], "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
except ValueError:
    print("Dates must be in format YYYY-MM-DD")
    sys.exit(1)

# --- Authenticate ---
token = os.getenv("GITHUB_TOKEN")
if not token:
    raise RuntimeError("Please set GITHUB_TOKEN environment variable")
gh = Github(auth=Auth.Token(token))
repo = gh.get_repo(REPO_NAME)

# --- 1 & 2) Issues created/closed ---
issues_created = 0
issues_closed = 0
for issue in repo.get_issues(state="all", creator=AUTHOR, since=START):  # type: ignore[arg-type]
    if START <= issue.created_at <= END:
        issues_created += 1
    if issue.closed_at and START <= issue.closed_at <= END:
        issues_closed += 1

# --- 3 & 4) PRs created/closed ---
prs_created = 0
prs_closed = 0
for pr in repo.get_pulls(state="all"):
    if pr.user.login == AUTHOR:
        if START <= pr.created_at <= END:
            prs_created += 1
        if pr.closed_at and START <= pr.closed_at <= END:
            prs_closed += 1

# --- 5 & 6) Lines added/removed ---
lines_added = 0
lines_removed = 0
for commit in repo.get_commits(author=AUTHOR, since=START, until=END):
    stats = commit.stats
    lines_added += stats.additions
    lines_removed += stats.deletions

# --- 7) Review comments on PRs ---
review_comments = 0
for comment in repo.get_pulls_review_comments(since=START):
    if comment.user.login == AUTHOR and START <= comment.created_at <= END:
        review_comments += 1

# --- OUTPUT TABLE ---
table = PrettyTable()
table.field_names = ["Metric", "Count"]
table.align["Metric"] = "l"
table.align["Count"] = "r"

table.add_row(["Issues Created", issues_created])
table.add_row(["Issues Closed", issues_closed])
table.add_row(["PRs Created", prs_created])
table.add_row(["PRs Closed", prs_closed])
table.add_row(["Lines Added", lines_added])
table.add_row(["Lines Removed", lines_removed])
table.add_row(["Review Comments", review_comments])

print(table)
