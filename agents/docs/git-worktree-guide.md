# Git Worktree Guide for Agent Coordination

<!-- markdownlint-disable MD051 -->

## Table of Contents

- [Overview](#overview)
- [Why Worktrees for Agents](#why-worktrees-for-agents)
- [Basic Worktree Operations](#basic-worktree-operations)
- [Worktree Strategy for 5-Phase Workflow](#worktree-strategy-for-5-phase-workflow)
- [Agent Coordination Across Worktrees](#agent-coordination-across-worktrees)
- [Status Reporting Patterns](#status-reporting-patterns)
- [Common Workflows](#common-workflows)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Git worktrees allow multiple branches to be checked out simultaneously in separate directories. For the ProjectOdyssey
agent hierarchy, worktrees enable:

- **Parallel execution**: Multiple agents work simultaneously on different phases
- **Isolation**: Each phase has its own workspace, preventing conflicts
- **Clean organization**: One worktree per GitHub issue/phase
- **Easy context switching**: Agents can move between phases without git checkout

**Core Concept**: Each GitHub issue gets its own worktree, allowing agents to work in parallel on Test,
Implementation, and Packaging phases after Plan completes.

## Why Worktrees for Agents

### Traditional Approach Problems

```bash
# Traditional: Single working directory
git checkout main
# Start working on implementation
git checkout -b 64-impl-agents
# Need to switch to tests
git stash
git checkout 63-test-agents
# Work on tests
git checkout 64-impl-agents
git stash pop
# CONFLICTS, lost context, inefficient
```text

### Worktree Approach Solution

```bash
# Worktree: Multiple working directories
worktrees/issue-63-test-agents/      # Test Engineer works here
worktrees/issue-64-impl-agents/      # Implementation Engineer works here
worktrees/issue-65-pkg-agents/       # Documentation Engineer works here

# Each agent has dedicated workspace
# No context switching, no conflicts
# Parallel work possible
```text

### Benefits for Agents

1. **Parallel Execution**: Test, Impl, Package phases run simultaneously
1. **No Context Loss**: Each agent maintains state in their worktree
1. **Clear Ownership**: One worktree = one team = clear responsibility
1. **Easy Integration**: Merge branches when phases complete
1. **Isolation**: Changes don't interfere until intentionally merged

## Basic Worktree Operations

### Creating Worktrees

```bash
# Create worktree for new branch
git worktree add worktrees/issue-62-plan-agents 62-plan-agents

# Create worktree from existing branch
git worktree add worktrees/issue-63-test-agents 63-test-agents

# Create worktree at specific commit
git worktree add worktrees/issue-64-impl-agents abc123 -b 64-impl-agents
```text

### Listing Worktrees

```bash
# List all worktrees
git worktree list

# Output
# /home/user/ProjectOdyssey          abc123 [main]
# /home/user/ProjectOdyssey/worktrees/issue-62-plan-agents  def456 [62-plan-agents]
# /home/user/ProjectOdyssey/worktrees/issue-63-test-agents  ghi789 [63-test-agents]
```text

### Removing Worktrees

```bash
# Remove worktree (after merging PR)
git worktree remove worktrees/issue-62-plan-agents

# Force remove (if there are uncommitted changes)
git worktree remove --force worktrees/issue-62-plan-agents

# Prune deleted worktrees
git worktree prune
```text

### Working in Worktrees

```bash
# Navigate to worktree
cd worktrees/issue-64-impl-agents

# All git operations work normally
git status
git add .
git commit -m "Implement Conv2D forward pass"
git push -u origin 64-impl-agents

# Return to main repo
cd ../..
```text

## Worktree Strategy for 5-Phase Workflow

### Phase Mapping to Worktrees

```text
GitHub Issue → Worktree → Branch → Agent Team

Issue #62 (Plan) → worktrees/issue-62-plan-agents/ → 62-plan-agents → Design Agents
Issue #63 (Test) → worktrees/issue-63-test-agents/ → 63-test-agents → Test Engineers
Issue #64 (Impl) → worktrees/issue-64-impl-agents/ → 64-impl-agents → Impl Engineers
Issue #65 (Pkg)  → worktrees/issue-65-pkg-agents/  → 65-pkg-agents → Doc Engineers
Issue #66 (Clean)→ worktrees/issue-66-cleanup-agents/ → 66-cleanup-agents → All Agents
```text

### Worktree Creation Sequence

#### Step 1: Plan Phase (Sequential)

```bash
# Create Plan worktree first
git worktree add worktrees/issue-62-plan-agents -b 62-plan-agents

# Work in Plan worktree
cd worktrees/issue-62-plan-agents

# Architecture Design Agent creates specifications
# ... create component specs, interface definitions, etc

# Commit and push
git add agents/
git commit -m "feat(agents): complete agent architecture design"
git push -u origin 62-plan-agents

# Create PR, get approval, merge to main
# Plan phase complete ✓
```text

#### Step 2: Parallel Phases (Test/Impl/Package)

```bash
# After Plan merges, create parallel worktrees from main
git checkout main
git pull

# Create Test worktree
git worktree add worktrees/issue-63-test-agents -b 63-test-agents

# Create Implementation worktree
git worktree add worktrees/issue-64-impl-agents -b 64-impl-agents

# Create Packaging worktree
git worktree add worktrees/issue-65-pkg-agents -b 65-pkg-agents

# Now three teams can work in parallel
```text

#### Step 3: Cleanup Phase (Sequential)

```bash
# After Test/Impl/Package complete, create Cleanup worktree
git worktree add worktrees/issue-66-cleanup-agents -b 66-cleanup-agents

# Cleanup integrates learnings from all parallel phases
```text

### Directory Structure

```text
ProjectOdyssey/                          # Main repository
├── .git/                            # Git metadata
├── worktrees/                       # Worktree directory
│   ├── issue-62-plan-agents/        # Plan phase
│   │   ├── .git                     # Worktree git link
│   │   ├── agents/                  # Agent docs being created
│   │   └── ... (full repo contents)
│   ├── issue-63-test-agents/        # Test phase (parallel)
│   │   ├── .git
│   │   ├── tests/                   # Test files
│   │   └── ...
│   ├── issue-64-impl-agents/        # Implementation (parallel)
│   │   ├── .git
│   │   ├── .claude/agents/          # Agent config files
│   │   └── ...
│   ├── issue-65-pkg-agents/         # Packaging (parallel)
│   │   ├── .git
│   │   ├── agents/docs/             # Integration docs
│   │   └── ...
│   └── issue-66-cleanup-agents/     # Cleanup (sequential)
│       └── ...
└── ... (main branch files)
```text

## Agent Coordination Across Worktrees

### Coordination Patterns

#### Pattern 1: Specification-Based Coordination

**Best Practice**: Agents coordinate through Plan specifications, not direct file sharing.

```text
Plan Phase (issue-62):
  - Creates detailed specifications
  - Merges to main branch

Parallel Phases (issue-63, 64, 65):
  - All branch from main (includes Plan specs)
  - Work independently using specs
  - Don't need to access each other's worktrees
```text

### Example

```bash
# In issue-64-impl-agents worktree
cd worktrees/issue-64-impl-agents

# Read specification from Plan phase (posted on GitHub issue)
gh issue view 62 --comments

# Implement based on spec
# No need to access issue-63 or issue-65 worktrees
```text

#### Pattern 2: Cherry-Pick Coordination

**Use Case**: Implementation needs test fixture from Test worktree

```bash
# Test Engineer creates fixture
cd worktrees/issue-63-test-agents
# ... create tests/fixtures/conv2d_data.mojo
git add tests/fixtures/conv2d_data.mojo
git commit -m "Add Conv2D test fixture"
# Note commit hash: abc123

# Implementation Engineer needs fixture
cd ../issue-64-impl-agents
git cherry-pick abc123

# Now fixture available in impl worktree
```text

### When to Use

- Need specific file from parallel worktree
- Small, self-contained commits
- No merge conflicts expected

### When NOT to Use

- Large, interdependent changes
- Better to wait and merge in Packaging phase

#### Pattern 3: Temporary Merge for Integration Testing

**Use Case**: Packaging wants to test integration of Test + Implementation

```bash
cd worktrees/issue-65-pkg-agents

# Merge Test branch (no commit)
git merge --no-commit --no-ff 63-test-agents

# Merge Implementation branch (no commit)
git merge --no-commit --no-ff 64-impl-agents

# Run integration tests
mojo test tests/
mojo build src/

# If tests pass, commit the merge
git commit -m "Integrate test and implementation for agents"

# If tests fail, abort and file cleanup issues
git merge --abort
```text

### Benefits

- Validate integration before final merge
- Catch compatibility issues early
- Can test without affecting other worktrees

#### Pattern 4: Status Update Coordination

**Use Case**: Agents communicate progress without file sharing

```bash
# Test Engineer posts status to GitHub issue
gh issue comment 63 --body "$(cat <<'EOF'
## Status Update - 2025-11-08

**Agent**: Test Engineer
**Progress**: 60% complete

### Completed
- Unit tests for Conv2D forward pass
- Integration tests for model composition

### In Progress
- Performance benchmark tests
- Gradient computation tests

### Blockers
- None

### Notes for Implementation Team
- Test fixtures in tests/fixtures/ ready to use
- Gradient test will need backward() method implemented
EOF
)"

# Implementation Engineer reads status from GitHub issue
gh issue view 63 --comments
```text

### Cross-Worktree Communication Protocol

#### Daily Status Updates

Each agent posts status as comments on their GitHub issue:

```markdown
## Status Update - YYYY-MM-DD

**Agent**: [Name/Team]
**Worktree**: worktrees/issue-XX-XXX/
**Progress**: XX%

### Completed
- Item 1
- Item 2

### In Progress
- Item 3

### Blockers
- [None / Description]

### Messages for Other Teams
- [Coordination notes]
```text

#### Handoff Protocol

When one phase depends on another:

```markdown
## Handoff: Test → Implementation

**From**: Test Engineer
**To**: Implementation Engineer
**Date**: 2025-11-08

**Artifacts Delivered**:
- tests/test_conv2d.mojo
- tests/fixtures/conv2d_data.mojo
- tests/fixtures/expected_outputs.mojo

**Integration Point**:
- Tests expect Conv2D class at src/model/layers/conv2d.mojo
- Tests expect forward() method signature:
  fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]

**Notes**:
- Tests will fail until forward() implemented
- SIMD test compares with naive implementation
- See tests/test_conv2d.mojo for all requirements
```text

## Status Reporting Patterns

### Individual Agent Status

```bash
# Each agent posts status to their GitHub issue
gh issue comment 64 --body "$(cat <<'EOF'
# Implementation Progress - Issue #64

## Overall: 75% Complete

### Agents Implemented
- [x] Chief Architect (100%)
- [x] Foundation Orchestrator (100%)
- [x] Architecture Design Agent (100%)
- [x] Implementation Specialist (100%)
- [x] Implementation Engineer (80%)
- [ ] Junior Implementation Engineer (30%)

### Current Focus
- Completing Implementation Engineer examples
- Starting Junior Implementation Engineer

### Blockers
- None

### Next Steps
- Complete remaining agent configs
- Review all agent descriptions for auto-invocation
- Test agent loading with Claude Code
EOF
)"
```text

### Aggregate Status (Section Orchestrator)

Section Orchestrators aggregate status from all GitHub issues:

```bash
# Section Orchestrator checks all parallel issues
# Check Test status
gh issue view 63 --comments | tail -50

# Check Implementation status
gh issue view 64 --comments | tail -50

# Check Packaging status
gh issue view 65 --comments | tail -50

# Post aggregate report to parent issue
gh issue comment 62 --body "$(cat <<'EOF'
# Agents Implementation - Aggregate Status

**Date**: 2025-11-08
**Phase**: Parallel Execution (Test/Impl/Package)

## Phase Progress

| Phase | Issue | Progress | Status |
| --- | --- | --- | --- |
| Test | #63 | 80% | On track |
| Implementation | #64 | 75% | On track |
| Packaging | #65 | 60% | On track |

## Blockers

- None across all phases

## Coordination Notes

- Test and Implementation coordinating well on TDD
- Packaging waiting for API finalization
- Expected completion: End of week

## Next Steps

- Implementation: Complete remaining agents
- Test: Add validation tests for agent configs
- Packaging: Finalize integration documentation
EOF
)"
```text

## Common Workflows

### Workflow 1: Feature Development (5-Phase)

```bash
# === PHASE 1: PLAN ===
git worktree add worktrees/issue-62-plan-agents -b 62-plan-agents
cd worktrees/issue-62-plan-agents

# Architecture Design Agent creates specs
# ... design work, post specs to GitHub issue #62
git add agents/
git commit -m "feat(agents): design agent architecture"
git push -u origin 62-plan-agents

# Create PR, review, merge
# Back to main
cd ../..
git checkout main
git pull

# === PHASE 2-4: PARALLEL ===
# Create three worktrees
git worktree add worktrees/issue-63-test-agents -b 63-test-agents
git worktree add worktrees/issue-64-impl-agents -b 64-impl-agents
git worktree add worktrees/issue-65-pkg-agents -b 65-pkg-agents

# Test Engineer works in issue-63
cd worktrees/issue-63-test-agents
# ... write tests
git commit -am "test(agents): add validation tests"
git push -u origin 63-test-agents

# Implementation Engineer works in issue-64
cd ../issue-64-impl-agents
# ... implement agents
git commit -am "feat(agents): implement agent configs"
git push -u origin 64-impl-agents

# Documentation Engineer works in issue-65
cd ../issue-65-pkg-agents
# ... write docs
git commit -am "docs(agents): add integration guide"
git push -u origin 65-pkg-agents

# All create PRs, get reviewed in parallel

# === PHASE 5: CLEANUP ===
# After parallel PRs merge
cd ../..
git checkout main
git pull

git worktree add worktrees/issue-66-cleanup-agents -b 66-cleanup-agents
cd worktrees/issue-66-cleanup-agents

# All agents review and refactor
# ... cleanup work
git commit -am "refactor(agents): cleanup and finalize"
git push -u origin 66-cleanup-agents

# Final PR, merge, done
```text

### Workflow 2: Bug Fix (Simplified)

```bash
# Bug: Agent description triggers incorrect auto-invocation

# === PLAN (Minimal) ===
# Create plan in main working directory (simple bug, no worktree needed)
# Document bug and fix approach in issue notes

# === PARALLEL (Test + Impl only, no Packaging needed) ===
git worktree add worktrees/issue-100-test-bugfix -b 100-test-bugfix
git worktree add worktrees/issue-101-impl-bugfix -b 101-impl-bugfix

# Test: Reproduce bug
cd worktrees/issue-100-test-bugfix
# ... write failing test
git commit -am "test: reproduce agent invocation bug"
git push -u origin 100-test-bugfix

# Implementation: Fix bug
cd ../issue-101-impl-bugfix
# ... update agent description
git commit -am "fix(agents): correct auto-invocation description"
git push -u origin 101-impl-bugfix

# Both PRs created and merged

# === CLEANUP (Quick review) ===
# No separate worktree needed for simple bug fix
# Final validation in main branch
```text

### Workflow 3: Refactoring (Cleanup-Heavy)

```bash
# Refactoring: Standardize all agent configurations

# === PLAN ===
git worktree add worktrees/issue-200-plan-refactor -b 200-plan-refactor
cd worktrees/issue-200-plan-refactor
# ... create refactoring plan
git commit -am "docs: plan agent config standardization"
git push -u origin 200-plan-refactor
# Merge to main

# === IMPLEMENTATION (No tests needed, docs updated inline) ===
cd ../..
git checkout main
git pull
git worktree add worktrees/issue-201-impl-refactor -b 201-impl-refactor
cd worktrees/issue-201-impl-refactor

# Refactor all agent configs
# ... refactoring work
git commit -am "refactor(agents): standardize all configs"
git push -u origin 201-impl-refactor
# Merge to main

# === CLEANUP ===
git checkout main
git pull
git worktree add worktrees/issue-202-cleanup-refactor -b 202-cleanup-refactor
cd worktrees/issue-202-cleanup-refactor

# Validate refactoring, fix issues
# ... validation and fixes
git commit -am "refactor(agents): final cleanup and validation"
git push -u origin 202-cleanup-refactor
```text

## Best Practices

### Worktree Organization

1. **Consistent Naming**: Always use `issue-{NUMBER}-{PHASE}-{COMPONENT}` format
   - `worktrees/issue-63-test-agents`
   - `worktrees/issue-64-impl-agents`

1. **Centralized Location**: Keep all worktrees in `worktrees/` directory
   - Easier to find
   - Gitignore can exclude if needed
   - Clear separation from main repo

1. **One Issue = One Worktree**: Don't reuse worktrees for multiple issues
   - Prevents confusion
   - Clean history per issue
   - Easier to track

### Branch Management

1. **Branch from Main**: Always create worktrees from main (or latest stable)

   ```bash
   git checkout main
   git pull
   git worktree add worktrees/issue-64-impl-agents -b 64-impl-agents
   ```text

1. **Descriptive Branch Names**: Match issue number and purpose

   ```bash
   62-plan-agents
   63-test-agents
   64-impl-agents
   ```text

1. **Push Immediately**: Set upstream on first push

   ```bash
   git push -u origin 64-impl-agents
   ```text

### Coordination

1. **Specification-First**: Prefer coordination through specs over file sharing

1. **Status Updates**: Regular status reports in issue notes

1. **Minimal Cherry-Picking**: Only when absolutely necessary

1. **Integration Testing**: Test merges before final PR

### Cleanup

1. **Remove After Merge**: Clean up worktrees after PR merges

   ```bash
   git worktree remove worktrees/issue-63-test-agents
   ```text

1. **Prune Regularly**: Remove stale worktree references

   ```bash
   git worktree prune
   ```text

1. **Document Removal**: Note in issue when worktree removed

## Troubleshooting

### Problem: Worktree Already Exists

```bash
$ git worktree add worktrees/issue-64-impl-agents -b 64-impl-agents
fatal: 'worktrees/issue-64-impl-agents' already exists
```text

### Solution

```bash
# Remove existing worktree
git worktree remove worktrees/issue-64-impl-agents

# Or use different path
git worktree add worktrees/issue-64-impl-agents-v2 -b 64-impl-agents
```text

### Problem: Branch Already Checked Out

```bash
$ git worktree add worktrees/issue-64-impl-agents -b 64-impl-agents
fatal: '64-impl-agents' is already checked out at '/home/user/ProjectOdyssey/worktrees/issue-64-impl-agents'
```text

### Solution

```bash
# Remove the worktree first
git worktree remove worktrees/issue-64-impl-agents

# Or check if you already have this worktree
git worktree list
```text

### Problem: Merge Conflicts During Integration

```bash
cd worktrees/issue-65-pkg-agents
git merge 63-test-agents
# CONFLICT in .claude/agents/test-engineer.md
```text

### Solution

```bash
# Resolve conflict manually
vim .claude/agents/test-engineer.md
# ... fix conflict

git add .claude/agents/test-engineer.md
git commit -m "Merge test branch, resolve conflicts"

# Or abort and escalate
git merge --abort
# Document conflict in cleanup issue
```text

### Problem: Lost Worktree Reference

```bash
$ git worktree list
# Worktree missing from list but directory exists
```text

### Solution

```bash
# Prune invalid references
git worktree prune

# Re-add if needed
git worktree add worktrees/issue-64-impl-agents 64-impl-agents
```text

### Problem: Can't Remove Worktree (Uncommitted Changes)

```bash
$ git worktree remove worktrees/issue-64-impl-agents
fatal: 'worktrees/issue-64-impl-agents' contains modified or untracked files, use --force to delete it
```text

### Solution

```bash
# Option 1: Commit changes
cd worktrees/issue-64-impl-agents
git add .
git commit -m "Save work in progress"
cd ../..
git worktree remove worktrees/issue-64-impl-agents

# Option 2: Force remove (lose changes!)
git worktree remove --force worktrees/issue-64-impl-agents
```text

### Problem: Worktree on Different Branch Than Expected

```bash
cd worktrees/issue-64-impl-agents
git branch
# * main  (expected: 64-impl-agents)
```text

### Solution

```bash
# Switch to correct branch
git checkout 64-impl-agents

# If branch doesn't exist
git checkout -b 64-impl-agents
```text

## Advanced Patterns

### Pattern: Shared Worktree for Multiple Agents

For small teams where multiple agents work on same issue:

```bash
# Create worktree
git worktree add worktrees/issue-64-impl-agents -b 64-impl-agents

# Multiple agents work in same worktree
# Agent 1: Works on chief-architect.md
# Agent 2: Works on implementation-specialist.md
# No conflicts since different files

# Coordinate via git
cd worktrees/issue-64-impl-agents
git pull  # Get teammate's changes
# ... do work
git push  # Share your changes
```text

### Pattern: Worktree Snapshots for Experimentation

Test risky changes without affecting main worktree:

```bash
# Create experimental worktree from implementation branch
git worktree add worktrees/issue-64-impl-agents-experiment 64-impl-agents

cd worktrees/issue-64-impl-agents-experiment
# Try experimental approach
# ... make changes

# If successful, merge back to main worktree
cd ../issue-64-impl-agents
git merge --no-ff issue-64-impl-agents-experiment

# If failed, just remove experimental worktree
cd ../..
git worktree remove worktrees/issue-64-impl-agents-experiment
```text

## See Also

- [5-Phase Integration](./5-phase-integration.md) - How workflow phases integrate
- [Common Workflows](./workflows.md) - Complete workflow examples
- [Delegation Rules](../delegation-rules.md) - Agent coordination patterns
- [Worktree Strategy](../../notes/review/worktree-strategy.md) - Complete worktree strategy
- [Git Worktree Documentation](https://git-scm.com/docs/git-worktree) - Official git documentation
