# Git Commit Policy

Shared git commit policy for all agents and developers.

## Core Rule

**`--no-verify` is ABSOLUTELY PROHIBITED with git commits.**

## Policy Statement

All code committed to this repository MUST pass pre-commit hooks. There are no exceptions.

### Why This Policy Exists

1. **Quality gate** - Pre-commit hooks catch errors before they reach CI/CD
2. **Consistency** - All code follows the same quality standards
3. **CI efficiency** - Hooks prevent broken code from entering the pipeline
4. **Developer experience** - Faster feedback loop than waiting for CI

### When Hooks Fail

If pre-commit hooks fail when you try to commit:

1. **Read the error message** - Hooks tell you exactly what's wrong
2. **Fix the code** - Update your code to pass the check
3. **Verify the fix** - Run `just precommit` or `just pre-commit-all`
4. **Commit again** - Let the hooks validate your changes

### Auto-Fix Hooks

Some hooks automatically fix issues (mojo-format, trailing-whitespace, etc.):

```bash
git commit -m "message"
# Hooks run, auto-fix files, abort commit

git add .  # Stage the fixes
git commit -m "message"  # Commit again with fixes
```

### Prohibited Commands

These commands are **strictly forbidden**:

```bash
git commit --no-verify           # ❌ NEVER
git commit -n                    # ❌ NEVER (shorthand for --no-verify)
git commit --no-verify -m "msg"  # ❌ NEVER
```

### Valid Alternative: Skipping Specific Hooks

If a **hook itself** is broken (not your code), you may skip that specific hook:

```bash
# Skip only the broken hook
SKIP=hook-name git commit -m "fix: description (skip hook-name due to issue #123)"
```

**Requirements for using SKIP:**

1. The hook itself must be broken, not your code
2. You must document the reason in the commit message
3. You must create a GitHub issue to fix/remove the broken hook
4. You skip ONLY the broken hook, not all hooks

### Enforcement

- **Local**: Pre-commit hooks block commits
- **CI/CD**: GitHub Actions will reject PRs that bypassed hooks
- **Code review**: Reviewers will reject commits using `--no-verify`

### Getting Help

If you're stuck with a hook failure:

1. Read the hook's error message carefully
2. Check `.pre-commit-config.yaml` for hook configuration
3. Run the hook manually: `pixi run pre-commit hook-name --all-files`
4. Ask for help in issue/PR comments

## References

- Pre-commit configuration: `.pre-commit-config.yaml`
- Pre-commit hooks skill: `.claude/skills/run-precommit/SKILL.md`
- Formatting skill: `.claude/skills/quality-fix-formatting/SKILL.md`
- Main project docs: `AGENTS.md` section "Pre-commit Hooks"
