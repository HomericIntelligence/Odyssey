---
name: blog-writer
description: Write development blog posts in cycle format. Creates engaging, narrative-driven posts about development work with informal tone while maintaining technical accuracy and markdown compliance. Use after completing milestones or phases.
mcp_fallback: none
category: doc
---

# Blog Writer Skill

Write development blog posts in informal "cycle format" that transform development work into
engaging narrative content.

## When to Use

- After completing a development milestone or phase
- Writing daily logs and weekly summaries
- Creating milestone retrospectives
- Documenting discoveries and learnings
- Producing narrative-driven technical content

## Quick Reference

```bash
# Get commits for date range
git log --oneline --since="YYYY-MM-DD" --until="YYYY-MM-DD"

# Get PR details
gh pr view <number>

# Count commits
git log --oneline | wc -l

# Validate markdown
pixi run npx markdownlint-cli2 notes/blog/MM-DD-YYYY.md
```

## Workflow

1. **Gather development work** - Extract commits, PRs, and metrics from git history
2. **Extract key events** - Identify main themes and significant events
3. **Structure narrative** - Organize as "The Plan → Challenges → Solutions → Learnings"
4. **Write content** - Use informal, conversational tone with specific examples
5. **Include metrics** - Add specific numbers (commits, hours, lines of code)
6. **Validate references** - Verify all links and commit hashes are accurate
7. **Ensure markdown compliance** - Follow all markdown linting rules
8. **Submit for review** - Create PR with blog post

## Blog Structure

```markdown
# [Topic]: [Date or Milestone]

## The Plan
[What we set out to do]

## What Happened
[Narrative of the actual work - informal, specific, personal]

## Challenges
[What went wrong or surprised us - "1,100+ errors? Yep."]

## Solutions
[How we fixed it - specific code snippets if useful]

## Learnings
[What we discovered - tied to specific examples]

## Metrics
- Commits: X
- PRs merged: Y
- Tests added: Z
- Lines of code: N

## Next Steps
[What comes next]
```

## Blog Location

- `notes/blog/MM-DD-YYYY.md` - All blog entries (flat structure)

## Writing Style Guidelines

**DO:**

- Write conversationally (informal, personal voice)
- Include specific numbers and metrics
- Link to commits and PRs accurately
- Use parenthetical reactions ("which... worked? Somehow.")
- Reference specific error counts and code changes
- Follow markdown compliance rules

**DO NOT:**

- Slip into formal writing
- Be vague about what happened
- Skip validation of links and hashes
- Skip markdown validation

## Markdown Compliance Checklist

- [ ] All code blocks have a language specified
- [ ] All code blocks have blank lines before and after
- [ ] All lists have blank lines before and after
- [ ] All headings have blank lines before and after
- [ ] No lines exceed 120 characters
- [ ] File ends with newline

## Example

**Task:** Write blog post about implementing LeNet-5 backpropagation.

**Actions:**

1. Extract commits from date range: `git log --oneline --since="2025-01-01"`
2. Identify main theme (gradient computation bugs)
3. Structure as "The Plan → Challenges → Solutions → Learnings"
4. Add conversational tone ("1,100+ errors? Yep.", specific reactions)
5. Include code snippets showing bugs and fixes
6. Add metrics (commits made, hours spent)
7. Validate all links and commit hashes
8. Ensure markdown compliance: `pixi run npx markdownlint-cli2 notes/blog/01-15-2025.md`

**Deliverable:** Engaging blog post combining technical depth with personal narrative,
markdown-compliant, saved to `notes/blog/MM-DD-YYYY.md`.

## Error Handling

| Issue | Fix |
|-------|-----|
| No metrics | Measure before/after with specific numbers |
| Vague accomplishments | List specific implementations or fixes |
| Missing learnings | Add what was discovered/surprised you |
| No next steps | Plan immediate next work |
| Markdown lint fails | Fix formatting issues, re-run linting |
| Broken links | Verify commit hashes, PR numbers, file paths |

## References

- Related skill: `doc-update-blog` for milestone progress updates
- Related skill: `doc-validate-markdown` for markdown validation
- See existing blog posts in `/notes/blog/` for examples
- See `.claude/shared/documentation-rules.md` for documentation rules
