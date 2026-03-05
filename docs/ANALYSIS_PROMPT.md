# Repository Analysis Prompt

Use this prompt to perform a comprehensive per-component quality analysis of this repository.
Paste it into a new Claude Code session at the root of ProjectOdyssey.

---

You are a senior software architect performing a comprehensive code and project quality review
of the ProjectOdyssey repository — a Mojo-based AI research platform for reproducing classic
ML papers.

## Your Task

Perform a structured per-component analysis of this repository. For each component, provide:

1. **Grade**: A-F (with +/- modifiers)
2. **Reasoning**: 2-4 sentences explaining the grade
3. **Issues to Fix**: Concrete, actionable items (not vague suggestions)

## Components to Analyze

Analyze ALL of the following components. Read the relevant files before grading each one.

---

### 1. Project Architecture & Structure

Files: `README.md`, `CLAUDE.md`, `docs/core/project-structure.md`, `docs/dev/architecture.md`,
`justfile`, `pixi.toml`

Evaluate: Directory layout clarity, separation of concerns, whether structure matches stated
goals, over-engineering vs. simplicity.

---

### 2. Agent System Design

Files: `agents/hierarchy.md`, `agents/agent-hierarchy.md`, `agents/delegation-rules.md`,
`.claude/agents/*.md` (sample 5-6 files), `docs/adr/ADR-005-agent-system-architecture.md`,
`docs/dev/skills-architecture.md`

Evaluate: Hierarchy clarity, role definitions, delegation patterns, whether 44 agents is
justified vs. over-engineered, coverage gaps, activation trigger quality.

---

### 3. Skills System

Files: `.claude/skills/` directory (all SKILL.md files and a sample of scripts),
`docs/dev/skills-design.md`, `CLAUDE.md` skills section

Evaluate: Skill decomposition, reusability, whether 82+ skills is justified, script quality,
YAML frontmatter consistency, MCP deprecation handling.

---

### 4. CI/CD Pipeline

Files: `.github/workflows/*.yml` (read all workflow files)

Evaluate: Workflow coverage, redundancy, job efficiency, secrets handling, artifact management,
caching strategy, whether the number of workflows is justified or bloated.

---

### 5. Documentation Quality

Files: `docs/README.md`, `docs/index.md`, `docs/getting-started/`, `docs/adr/`,
`docs/dev/` (sample), `agents/docs/`

Evaluate: Completeness, accuracy vs. stated project status, duplication across locations,
ADR quality, whether docs match actual implementation state.

---

### 6. Mojo Code Quality

Files: All `.mojo` files in `shared/`, `tests/`, `papers/` (read a representative sample
from each directory)

Evaluate: Pattern consistency, use of modern Mojo v0.26.1+ syntax, memory safety, SIMD
usage, test coverage quality, gradient implementations.

---

### 7. Testing Strategy

Files: `docs/dev/testing-strategy.md`, `docs/adr/ADR-004-testing-strategy.md`,
`tests/` directory structure, sample test files, `.github/workflows/comprehensive-tests.yml`,
`.github/workflows/` weekly test workflows

Evaluate: Two-tier strategy soundness, test coverage, gradient checking rigor,
FP-representable value rationale, CI runtime targets.

---

### 8. Python Automation & Scripts

Files: `scripts/*.py`, `docs/adr/ADR-001-language-selection-tooling.md`,
`.claude/hooks/*.py`, `.claude/skills/*/scripts/*.sh`

Evaluate: Code quality, type hint usage, error handling, whether Python vs. Mojo decisions
are justified, script robustness.

---

### 9. Pre-commit & Code Quality Gates

Files: `.pre-commit-config.yaml`, `.github/workflows/pre-commit.yml`,
`.claude/shared/git-commit-policy.md`, `justfile` pre-commit recipes

Evaluate: Hook coverage, hook quality, enforcement rigor, whether hooks would actually
catch real issues, bypass prevention.

---

### 10. Docker & Environment Setup

Files: `Dockerfile` (or `docker/`), `docker-compose.yml` (if present),
`.github/workflows/docker.yml`, `pixi.toml`, `docs/dev/docker.md`

Evaluate: Image layering, size optimization, environment reproducibility, dev vs. prod
image separation, GHCR publishing setup.

---

### 11. Security Posture

Files: `.github/workflows/security-scan.yml`, `.github/workflows/dependency-audit.yml`,
`.github/workflows/security-pr-scan.yml`, `.claude/agents/security-specialist.md`,
`.claude/agents/security-design.md`

Evaluate: Secret scanning, dependency auditing, SAST coverage, whether security workflows
are comprehensive or superficial.

---

### 12. ADR Quality & Decision Documentation

Files: `docs/adr/*.md` (all ADR files)

Evaluate: Decision coverage, rationale quality, whether decisions are well-reasoned,
whether important decisions are missing ADRs, template usage consistency.

---

## Output Format

For each component, use this exact structure:

```text
## [Component Name]

**Grade**: [A/B/C/D/F with +/-]

**Reasoning**:
[2-4 sentences covering what works well and what doesn't]

**Issues to Fix**:
- [Specific actionable issue 1]
- [Specific actionable issue 2]
- [Specific actionable issue 3]
```

After all components, provide:

```text
## Overall Repository Grade

**Grade**: [A/B/C/D/F with +/-]

**Summary**: [3-5 sentences overall assessment]

**Top 5 Priority Fixes**:
1. [Most critical fix]
2. [Second priority]
3. [Third priority]
4. [Fourth priority]
5. [Fifth priority]
```

## Grading Criteria

- **A**: Excellent - production quality, well-reasoned, minimal issues
- **B**: Good - solid work with minor issues that don't block functionality
- **C**: Adequate - works but has significant gaps, inconsistencies, or over-engineering
- **D**: Poor - major issues that need substantial rework before this is useful
- **F**: Failing - broken, missing, or fundamentally flawed

## Important Notes

- Be honest and specific. Vague praise or vague criticism is not helpful.
- If a component has zero Mojo implementation files when they're expected, note that as
  a critical gap.
- Cross-reference between components where relevant (e.g., if docs describe something
  that doesn't exist in code).
- Consider the project's stated phase ("planning phase") when grading - some gaps may be
  intentional but should still be noted.
- Grade based on quality of what exists, not what's planned.
