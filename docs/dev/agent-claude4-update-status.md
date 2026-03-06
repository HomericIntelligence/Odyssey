
# Agent Files Claude 4 Update Status

## Overview

This document tracks the progress of updating all 42 agent files with Claude 4-specific sections
following official best practices from:

- [https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices)
- [https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [https://code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents)

## Required Sections

Each agent file must include:

1. **Thinking Guidance** - When to use extended thinking vs standard
2. **Output Preferences** - Format, style, code examples, decisions
3. **Delegation Patterns** - When to use skills vs sub-agents
4. **Sub-Agent Usage** - How to spawn sub-agents with proper context

## Completion Status

### Completed Files (5/42) ✅

**Reference Implementations:**

1. `/.claude/agents/chief-architect.md` ✅
2. `/.claude/agents/implementation-engineer.md` ✅
3. `/.claude/agents/test-engineer.md` ✅
4. `/.claude/agents/foundation-orchestrator.md` ✅

These serve as templates for their respective categories.

### Remaining Files by Category (37/42)

#### Level 1: Orchestrators (5 remaining)

- `.claude/agents/papers-orchestrator.md`
- `.claude/agents/shared-library-orchestrator.md`
- `.claude/agents/tooling-orchestrator.md`
- `.claude/agents/agentic-workflows-orchestrator.md`
- `.claude/agents/cicd-orchestrator.md`

**Pattern**: Use foundation-orchestrator.md as template

#### Level 2: Design Agents (3 remaining)

- `.claude/agents/architecture-design.md`
- `.claude/agents/integration-design.md`
- `.claude/agents/security-design.md`

**Pattern**: Similar to orchestrators but more technical focus

#### Level 3: Specialists (16 remaining)

**Component Specialists (3):**

- `.claude/agents/implementation-specialist.md`
- `.claude/agents/test-specialist.md`
- `.claude/agents/documentation-specialist.md`

**Technical Specialists (3):**

- `.claude/agents/performance-specialist.md`
- `.claude/agents/security-specialist.md`
- `.claude/agents/numerical-stability-specialist.md`

**Review Specialists (10):**

- `.claude/agents/algorithm-review-specialist.md`
- `.claude/agents/implementation-review-specialist.md`
- `.claude/agents/architecture-review-specialist.md`
- `.claude/agents/data-engineering-review-specialist.md`
- `.claude/agents/dependency-review-specialist.md`
- `.claude/agents/documentation-review-specialist.md`
- `.claude/agents/mojo-language-review-specialist.md`
- `.claude/agents/paper-review-specialist.md`
- `.claude/agents/performance-review-specialist.md`
- `.claude/agents/research-review-specialist.md`
- `.claude/agents/safety-review-specialist.md`
- `.claude/agents/test-review-specialist.md`

**Pattern**: Review specialists rarely need sub-agents (they ARE the specialists)

#### Level 4: Engineers (9 remaining)

- `.claude/agents/senior-implementation-engineer.md`
- `.claude/agents/performance-engineer.md`
- `.claude/agents/documentation-engineer.md`
- `.claude/agents/ci-failure-analyzer.md`
- `.claude/agents/log-analyzer.md`
- `.claude/agents/mojo-syntax-validator.md`
- `.claude/agents/test-flakiness-specialist.md`
- `.claude/agents/code-review-orchestrator.md`
  - ~~`.claude/agents/pr-cleanup-specialist.md`~~ — converted to skill
  - ~~`.claude/agents/blog-writer-specialist.md`~~ — converted to skill

**Pattern**: Use implementation-engineer.md or test-engineer.md as templates

#### Level 5: Junior Engineers (2 remaining)

- `.claude/agents/junior-implementation-engineer.md`
- `.claude/agents/junior-test-engineer.md`
- `.claude/agents/junior-documentation-engineer.md`

**Pattern**: No sub-agents, simpler thinking guidance

## Automation Script

A Python script has been created to automate the updates:

**Location:** `/scripts/update_agents_claude4.py`

**Features:**

- Categorizes agents by level and role
- Generates appropriate thinking guidance for each category
- Customizes output preferences and delegation patterns
- Inserts sections before existing references
- Skips files already updated

- Provides progress reporting

**Usage:**

```bash
python3 scripts/update_agents_claude4.py
```text

The script handles:

- Different agent categories (chief-architect, orchestrator, design, specialist, engineer, junior)
- Role-specific thinking tasks and budgets
- Appropriate skill and sub-agent recommendations
- Context examples tailored to agent level

## Section Templates

### Thinking Guidance Template

**Level 0-1 (Strategic):**

- Extended thinking for: Architecture decisions, cross-section conflicts, paper analysis
- Standard thinking for: Routine delegation, status updates

**Level 2-3 (Tactical):**

- Extended thinking for: Complex design trade-offs, algorithm selection, performance optimization
- Standard thinking for: Standard component work, routine reviews

**Level 4-5 (Execution):**

- Extended thinking for: Complex implementations, debugging subtle issues, SIMD optimization
- Standard thinking for: Standard functions, routine maintenance

### Output Preferences Template

**All Levels:**

- Format: Structured Markdown
- Style: Varies by level (strategic → architectural → technical → implementation-focused)
- Code examples: More detailed at lower levels, minimal at higher levels
- Decisions: Clear rationale sections appropriate to level

### Delegation Patterns Template

**Skills:**

- Level-appropriate skill recommendations
- Clear "when to invoke" guidance

**Sub-agents:**

- When appropriate (not for Level 5, rare for Level 4)
- Clear "Do NOT use for" guidance
- Escalation patterns

### Sub-Agent Usage Template (when applicable)

**Context to provide:**

- File paths with line numbers
- Issue numbers
- Clear success criteria
- Scope boundaries

**Example invocations:**

- Tailored to agent's domain
- Specific deliverables
- Measurable success criteria

## Next Steps

1. **Run automation script:**

   ```bash
   cd
   python3 scripts/update_agents_claude4.py
```text

2. **Review generated content:**
   - Check a sample from each category
   - Verify insertion points are correct
   - Ensure examples are appropriate

3. **Manual adjustments:**
   - Review specialists may need lighter sub-agent guidance
   - Junior engineers need simpler language
   - Domain-specific customizations

4. **Update templates:**
   - Update agent templates in `agents/templates/` with new sections
   - Ensure future agents include Claude 4 sections from the start

5. **Validation:**
   - Run `python3 tests/agents/validate_configs.py .claude/agents/`
   - Verify no broken markdown links
   - Check for consistent formatting

6. **Commit and PR:**

   ```bash
   git add .claude/agents/*.md docs/dev/agent-claude4-update-status.md scripts/update_agents_claude4.py
   git commit -m "docs(agents): update agent files for Claude 4 best practices"
   git push
   gh pr create --title "docs(agents): Update all agent files for Claude 4 best practices" \

     --body "Closes #2548" \
     --label "documentation"

```text

## Quality Checklist

Before finalizing:

- [ ] All 42 agent files have Thinking Guidance section
- [ ] All 42 agent files have Output Preferences section
- [ ] All 42 agent files have Delegation Patterns section
- [ ] Appropriate agents have Sub-Agent Usage section
- [ ] Level 5 junior agents do NOT recommend sub-agents
- [ ] Examples are specific to agent's domain
- [ ] File paths use absolute paths
- [ ] Markdown formatting is consistent
- [ ] All agent templates updated
- [ ] Validation tests pass

## References

- **Claude 4 Best Practices**: [https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices)
- **Agent Skills Best Practices**: [https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- **Sub-Agents Guide**: [https://code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents)
- **Output Styles**: [https://code.claude.com/docs/en/output-styles](https://code.claude.com/docs/en/output-styles)
- **Issue #2548**: [https://github.com/HomericIntelligence/ProjectOdyssey/issues/2548](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2548)
