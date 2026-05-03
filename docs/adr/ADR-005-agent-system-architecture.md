# ADR-005: Agent System Architecture

**Status**: Accepted

**Date**: 2025-12-28

**Decision Owner**: Chief Architect

## Executive Summary

This ADR documents ML Odyssey's 6-level hierarchical agent system with 44 specialized agents.
The system provides structured delegation, clear responsibilities, and automated workflows for
development tasks ranging from strategic architecture decisions to boilerplate generation.

## Context

### Problem Statement

Large-scale ML projects require coordination across many dimensions:

- Strategic decisions (paper selection, architecture)
- Tactical planning (module organization, dependencies)
- Specialized expertise (performance, security, documentation)
- Implementation work (code, tests, documentation)
- Review and quality assurance

Without structure, responsibilities overlap, decisions are inconsistent, and expertise is not
effectively leveraged.

### Requirements

1. **Clear Hierarchy**: Unambiguous delegation chains
2. **Specialization**: Focused agents for specific domains
3. **Scalability**: Support for parallel work across sections
4. **Consistency**: Uniform patterns and guidelines
5. **Flexibility**: Support for various task types and phases

## Decision

### 6-Level Hierarchical Structure

```text
Level 0: Meta-Orchestrator (1 agent)
    Chief Architect - System-wide decisions, paper selection

Level 1: Section Orchestrators (6 agents)
    Foundation, Shared Library, Tooling, Paper Implementation,
    CI/CD, Agentic Workflows

Level 2: Design & Review Agents (4 agents)
    Architecture Design, Integration Design, Security Design,
    Code Review Orchestrator

Level 3: Specialists (24 agents)
    11 Implementation/Execution + 13 Code Review

Level 4: Engineers (6 agents)
    Senior Implementation, Implementation, Test, Documentation,
    Performance, Log Analyzer

Level 5: Junior Engineers (3 agents)
    Junior Implementation, Junior Test, Junior Documentation
```

### Agent Responsibilities by Level

**Level 0 - Chief Architect**:

- Paper selection and prioritization
- System-wide architectural decisions
- Cross-section conflict resolution
- Technology stack decisions
- Mojo vs Python language selection

**Level 1 - Section Orchestrators**:

- Section planning and coordination
- Module organization within section
- Dependency management across modules
- Resource allocation for section work

**Level 2 - Design & Review**:

- Module architecture design
- Interface definitions
- Security design
- Code review routing and coordination

**Level 3 - Specialists**:

- Component implementation approach
- Test strategy and design
- Documentation structure
- Performance optimization strategy
- Code review assessment (13 dimensions)

**Level 4 - Engineers**:

- Function and class implementation
- Test writing and maintenance
- Documentation authoring
- Performance optimization implementation

**Level 5 - Junior Engineers**:

- Boilerplate generation
- Code formatting
- Simple documentation tasks

### 5-Phase Development Workflow

Every component follows a structured workflow:

```text
Plan → [Test | Implementation | Package] → Cleanup
```

1. **Plan**: Design and documentation (must complete first)
2. **Test**: Write tests following TDD (parallel after Plan)
3. **Implementation**: Build functionality (parallel after Plan)
4. **Package**: Create distributable packages (parallel after Plan)
5. **Cleanup**: Refactor and finalize (after parallel phases)

### Agent Configuration Pattern

Agents are defined as markdown files with YAML frontmatter:

```yaml
---
name: Implementation Specialist
level: 3
category: specialist
model: sonnet
phase: [test, implementation]
tools:
  - read
  - edit
  - bash
skills:
  - mojo-format
  - mojo-test-runner
---

# Implementation Specialist

## Role
Specify component implementation approach...

## Responsibilities
...
```

### Skill Delegation Patterns

Agents delegate to skills using five standard patterns:

**Pattern 1: Direct Delegation**

```markdown
Use the `mojo-format` skill to format code:
- **Invoke when**: Before committing Mojo files
- **The skill handles**: Running mojo format command
```

**Pattern 2: Conditional Delegation**

```markdown
If CI is failing:
  - Use the `analyze-ci-failure-logs` skill
Otherwise:
  - Proceed with implementation
```

**Pattern 3: Multi-Skill Workflow**

```markdown
To complete implementation:
1. Use `mojo-format` skill to format code
2. Use `mojo-test-runner` skill to run tests
3. Use `gh-create-pr-linked` skill to create PR
```

**Pattern 4: Skill Selection**

```markdown
Analyze change type:
- If test changes: Use `test-diff-analyzer`
- If performance changes: Use `mojo-simd-optimize`
```

**Pattern 5: Background vs Foreground**

```markdown
Background automation: `run-precommit` (runs automatically)
Foreground tasks: `gh-create-pr-linked` (invoke explicitly)
```

### Code Review Specialists (13 Dimensions)

The Code Review Orchestrator routes PRs to specialized reviewers:

1. **Implementation Review**: Code correctness, logic, structure
2. **Documentation Review**: Clarity, completeness, accuracy
3. **Test Review**: Coverage, assertions, edge cases
4. **Security Review**: Vulnerabilities, input validation
5. **Safety Review**: Memory safety, resource management
6. **Mojo Language Review**: Idioms, syntax, best practices
7. **Performance Review**: Efficiency, SIMD, optimization
8. **Algorithm Review**: Correctness, complexity, numerical stability
9. **Architecture Review**: Design patterns, modularity
10. **Data Engineering Review**: Data flow, preprocessing
11. **Paper Review**: Research faithfulness, citations
12. **Research Review**: Novel contributions, methodology
13. **Dependency Review**: External dependencies, versions

## Rationale

### Why 6 Levels?

The hierarchy balances:

- **Depth**: Enough levels for clear specialization
- **Simplicity**: Not so many levels that delegation is confusing
- **Coverage**: Each level has distinct responsibilities

### Why 44 Agents?

Initial planning estimated 23 agents. Expansion to 44 reflects:

- Emerging needs (blog writing, CI analysis)
- Specialization requirements (numerical stability, test flakiness)
- Review dimension coverage (13 code review specialists)

### Why Mojo-Specific Considerations?

ML Odyssey is a Mojo-first project. Agents at each level need appropriate Mojo expertise:

- **Level 0-2**: Deep understanding of Mojo vs Python trade-offs
- **Level 3**: Proficiency in Mojo syntax and patterns
- **Level 4-5**: Hands-on Mojo coding ability

## Consequences

### Positive

- **Clear Ownership**: Each agent has defined responsibilities
- **Specialization**: Focused expertise per domain
- **Scalability**: Parallel work across sections
- **Consistency**: Uniform patterns via shared guidelines
- **Quality**: Multi-dimensional code review coverage

### Negative

- **Complexity**: 44 agents require management
- **Coordination Overhead**: Delegation chains add latency
- **Learning Curve**: Contributors must understand hierarchy
- **Configuration Maintenance**: Agent files need updates

### Neutral

- **Model Assignment**: Different levels use different Claude models
  (Opus for L0-1, Sonnet for L2-3, Haiku for L4-5)
- **Skill Integration**: 82+ skills available for automation

## Alternatives Considered

### Alternative 1: Flat Agent Structure

**Description**: All agents at same level, no hierarchy.

**Pros**:

- Simple to understand
- No delegation overhead

**Cons**:

- Unclear responsibilities
- No escalation path
- Overlapping decisions

**Why Rejected**: Insufficient structure for complex project.

### Alternative 2: 3-Level Hierarchy

**Description**: Architect, Specialists, Engineers only.

**Pros**:

- Simpler than 6 levels
- Faster delegation

**Cons**:

- Section orchestration missing
- Design phase unclear
- Junior work not distinguished

**Why Rejected**: Insufficient granularity for different task types.

### Alternative 3: No Code Review Specialists

**Description**: Single code review agent handles all dimensions.

**Pros**:

- Simpler review process
- Single point of contact

**Cons**:

- Expertise diluted
- Review quality suffers
- Dimension-specific issues missed

**Why Rejected**: PR quality requires specialized review.

## Implementation Details

### Agent File Location

`.claude/agents/*.md` - 44 agent configuration files

### Shared Guidelines

Agents reference shared files to avoid duplication:

| File | Purpose |
| --- | --- |
| `.claude/shared/common-constraints.md` | Minimal changes principle |
| `.claude/shared/documentation-rules.md` | Output locations |
| `.claude/shared/pr-workflow.md` | PR creation, review |
| `.claude/shared/mojo-guidelines.md` | Mojo v0.26.1+ syntax |
| `.claude/shared/mojo-anti-patterns.md` | 64+ failure patterns |
| `.claude/shared/error-handling.md` | Retry, timeout, escalation |

### Skills Directory

`.claude/skills/` - 82+ skill implementations

Categories:

- GitHub (9 skills): PR, issue, review operations
- Worktree (4 skills): Parallel development
- Phase Workflow (5 skills): Plan, test, implement, package, cleanup
- Mojo (10 skills): Format, test, build, optimize
- Agent System (5 skills): Validate, test, run, coverage
- Documentation (4 skills): ADR, blog, markdown, issue
- CI/CD (6 skills): Pre-commit, validate, fix, analyze
- Quality (5 skills): Lint, format, security, coverage, complexity
- Testing & Analysis (5 skills): Diff, failures, suggestions, progress
- Review (2 skills): Checklist, review

### Delegation Flow

**Top-Down (Task Decomposition)**:

```text
Paper Selection (L0)
    → Section Planning (L1)
        → Module Design (L2)
            → Component Specification (L3)
                → Function Implementation (L4)
                    → Boilerplate Generation (L5)
```

**Bottom-Up (Status Reporting)**:

```text
Code Metrics (L5)
    → Component Health (L4)
        → Module Stability (L3)
            → Section Status (L2)
                → Project Health (L1)
                    → Strategic Alignment (L0)
```

## References

### Related Files

- `/agents/hierarchy.md`: Visual hierarchy diagram
- `/agents/README.md`: Quick start guide
- `/agents/delegation-rules.md`: Coordination patterns
- `/.claude/agents/`: Agent configurations
- `/.claude/skills/`: Skill implementations

### Related ADRs

- [ADR-001](ADR-001-language-selection-tooling.md): Language selection affects agent guidelines

### External Documentation

- [CLAUDE.md](/CLAUDE.md): Project guidelines including agent system

## Revision History

| Version | Date | Author | Changes |
| --- | --- | --- | --- |
| 1.0 | 2025-12-28 | Chief Architect | Initial ADR |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-005-agent-system-architecture.md`
- **Status**: Accepted
- **Review Frequency**: Quarterly
- **Next Review**: 2026-03-28
- **Supersedes**: None
- **Superseded By**: None
