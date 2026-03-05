# Agent System Documentation

## Overview

This directory contains documentation, templates, and reference materials for the ProjectOdyssey multi-level agent
hierarchy. The actual operational agent configurations are in `.claude/agents/` following Claude Code conventions.

## Directory Purpose

### This Directory (`agents/`)

- **Purpose**: Team documentation and reference materials
- **Contents**: READMEs, diagrams, templates, examples
- **Usage**: Read by humans to understand and create agents
- **Version Control**: Committed to repository for team sharing

### Operational Directory (`.claude/agents/`)

- **Purpose**: Working sub-agent configuration files
- **Contents**: Agent .md files that Claude Code executes
- **Usage**: Read by Claude Code to invoke agents
- **Version Control**: Committed to repository for consistency

## Quick Start

### Understanding the System

1. **Read the Hierarchy**: Start with [hierarchy.md](hierarchy.md) to understand the 6 levels
1. **Learn Delegation**: Read [delegation-rules.md](delegation-rules.md) for coordination patterns
1. **Review Templates**: Check [templates/](templates/) for agent configuration examples

### Creating a New Agent

1. **Identify the Level**: Determine which level (0-5) this agent belongs to
1. **Choose a Template**: Use the appropriate template from `templates/`
1. **Fill in Details**: Customize for your specific agent's role
1. **Add to `.claude/agents/`**: Place the config file in `.claude/agents/`
1. **Test**: Verify Claude Code can load and invoke the agent

### Using Agents

Agents can be invoked in two ways:

**Automatic Invocation** (Recommended):

```text
User: "Design the architecture for the authentication module"
→ Claude recognizes task matches Architecture Design Agent
→ Agent invokes automatically
```text

### Explicit Invocation

```text
User: "Use the architecture design agent to plan the auth module"
→ Claude explicitly invokes Architecture Design Agent
```text

## Agent Hierarchy (6 Levels)

### Level 0: Meta-Orchestrator

- **Chief Architect Agent**: System-wide decisions, paper selection, strategic planning

### Level 1: Section Orchestrators

- Foundation, Shared Library, Tooling, Paper, CI/CD, Agentic Workflows Orchestrators
- Manage major repository sections

### Level 2: Module Design Agents & Orchestrators

### Design Agents

- Architecture, Integration, Security Design Agents
- Design module structure and interfaces

### Review Orchestrators

- Code Review Orchestrator
- Coordinates 13 specialized review agents

### Level 3: Component Specialists & Review Specialists

**Implementation/Execution Specialists** (11 agents):

- Implementation, Test, Documentation, Performance, Security Specialists
- Blog Writer, Numerical Stability, Test Flakiness, PR Cleanup Specialists
- Mojo Syntax Validator, CI Failure Analyzer
- Handle specific component aspects and execution planning

**Code Review Specialists** (13 agents):

- Implementation, Documentation, Test Review Specialists
- Security, Safety Review Specialists
- Mojo Language, Performance Review Specialists
- Algorithm, Architecture Review Specialists
- Data Engineering, Paper, Research Review Specialists
- Dependency Review Specialist

### Level 4: Implementation Engineers

- Senior, Standard, Test, Documentation, Performance Engineers
- Write code, tests, documentation

### Level 5: Junior Engineers

- Handle simple tasks, boilerplate, formatting

**See [hierarchy.md](hierarchy.md) for complete details**

## Skills System

Skills are reusable capabilities separate from agents. They're in `.claude/skills/`.

### Key Distinction

- **Agents** = Decision-makers with separate contexts
- **Skills** = Reusable capabilities invoked within agent context

### Skills Taxonomy

- **Tier 1**: Foundational (used by all agents) - code analysis, generation, testing
- **Tier 2**: Domain-specific (specific agent types) - paper analysis, ML ops, documentation
- **Tier 3**: Specialized (narrow use cases) - security scanning, performance profiling

**See [/notes/review/skills-design.md](../notes/review/skills-design.md) for complete skills documentation**

## Delegation Patterns

### Decomposition Delegation

Higher levels break tasks into smaller pieces and delegate down:

```text
Chief Architect → Section Orchestrator → Module Agent → Component Specialist → Engineer
```text

### Specialization Delegation

Orchestrators delegate to specialists based on expertise:

```text
Section Orchestrator
  ├─> Architecture Design (for architecture)
  ├─> Security Design (for security)
  └─> Integration Design (for integration)
```text

### Parallel Delegation

Independent tasks run simultaneously:

```text
Component Specialist
  ├─> Test Engineer (parallel)
  ├─> Implementation Engineer (parallel)
  └─> Documentation Writer (parallel)
```text

**See [delegation-rules.md](delegation-rules.md) for complete patterns**

## Workflow Integration

### 5-Phase Workflow

**Phase 1: Plan** (Sequential)

- Levels 0-2: Orchestrators and designers create specifications

**Phases 2-4: Test/Implementation/Package** (Parallel)

- Levels 3-5: Specialists and engineers execute in parallel
- **Package Phase**: Create distributable artifacts (.mojopkg files, archives, CI/CD workflows)
  - NOT just documenting existing structures
  - Must produce actual installable packages

**Phase 5: Cleanup** (Sequential)

- All levels: Review and refactor

### Git Worktree Strategy

Each GitHub issue gets its own worktree:

- `worktrees/issue-62-plan-agents/` - Plan phase
- `worktrees/issue-63-test-agents/` - Test phase (parallel)
- `worktrees/issue-64-impl-agents/` - Implementation (parallel)
- `worktrees/issue-65-pkg-agents/` - Packaging (parallel)
- `worktrees/issue-66-cleanup-agents/` - Cleanup (sequential)

**See [/notes/review/worktree-strategy.md](../notes/review/worktree-strategy.md) for complete workflow**

## Code Review System

### Overview

A comprehensive code review system with 14 agents (1 orchestrator + 13 specialists) ensures thorough review across all dimensions:

**Code Review Orchestrator** (Level 2):

- Analyzes PRs and routes to appropriate specialists
- Prevents overlap through dimension-based routing
- Consolidates feedback from multiple specialists
- Integrates with CI/CD pipeline

**Review Specialists** (Level 3):

- Implementation: Code correctness, logic, maintainability
- Documentation: Markdown, comments, docstrings, API docs
- Test: Test coverage, quality, assertions
- Security: Vulnerabilities, OWASP top 10, input validation
- Safety: Memory safety, type safety, undefined behavior
- Mojo Language: Ownership, SIMD, fn vs def, traits
- Performance: Algorithmic complexity, cache efficiency
- Algorithm: ML algorithm correctness, numerical stability
- Architecture: System design, modularity, patterns
- Data Engineering: Data pipelines, preprocessing
- Paper: Academic writing, citations
- Research: Experimental design, reproducibility
- Dependency: Version management, licenses

### No-Overlap Strategy

Each specialist reviews one dimension exclusively:

| Dimension | Specialist | Example |
|-----------|-----------|---------|
| Logic Correctness | Implementation | Bug detection, control flow |
| Memory Safety | Safety | Leaks, use-after-free |
| Security Exploits | Security | SQL injection, XSS |
| Language Idioms | Mojo Language | Ownership patterns, SIMD |
| Performance | Performance | Big O complexity |
| ML Correctness | Algorithm | Gradient computation |
| System Design | Architecture | Module structure |
| Test Quality | Test | Coverage, assertions |

### Review Workflow

```text
PR Created
  ↓
Code Review Orchestrator analyzes
  ↓
Routes to specialists (parallel):
  - .mojo files → Mojo Language + Implementation
  - ML algorithms → Algorithm + Implementation
  - Security code → Security + Safety
  - Tests → Test Specialist
  - Dependencies → Dependency + Security
  ↓
Consolidates feedback
  ↓
Comprehensive review report
```text

**See individual agent docs in `.claude/agents/` for detailed checklists and examples**

## Configuration Format

Agents follow Claude Code format with YAML frontmatter:

```text
---
name: agent-name
description: Brief description of when to use this agent (critical for automatic invocation)
tools: Read,Write,Edit,Bash,Grep,Glob
model: sonnet
---

# Agent Name

## Role

[Agent's role in the hierarchy]

## Responsibilities

- Responsibility 1
- Responsibility 2

## Instructions

[Detailed instructions]

## Examples

[Example tasks]

## Constraints

[What NOT to do]
```text

**See [templates/](templates/) for complete examples**

## Available Templates

- [level-0-chief-architect.md](templates/level-0-chief-architect.md) - Chief Architect template
- [level-1-section-orchestrator.md](templates/level-1-section-orchestrator.md) - Section Orchestrator template
- [level-2-module-design.md](templates/level-2-module-design.md) - Module Design Agent template
- [level-2-orchestrator.md](templates/level-2-orchestrator.md) - Generic Level 2 Orchestrator template
- [level-3-component-specialist.md](templates/level-3-component-specialist.md) - Component Specialist template
- [level-3-review-specialist.md](templates/level-3-review-specialist.md) - Generic Review Specialist template
- [level-4-implementation-engineer.md](templates/level-4-implementation-engineer.md) - Implementation Engineer template
- [level-5-junior-engineer.md](templates/level-5-junior-engineer.md) - Junior Engineer template

## Operational Agents

The operational agent configurations are in `.claude/agents/` (44 agents total):

### Level 0: Meta-Orchestrator (1 agent)

- `chief-architect.md` - Strategic decisions, paper selection, system-wide coordination

### Level 1: Section Orchestrators (6 agents)

- `foundation-orchestrator.md` - Section 01 (directory structure, configuration)
- `shared-library-orchestrator.md` - Section 02 (core operations, training utilities)
- `tooling-orchestrator.md` - Section 03 (CLI tools, automation scripts)
- `papers-orchestrator.md` - Section 04 (research paper implementations)
- `cicd-orchestrator.md` - Section 05 (testing, deployment pipelines)
- `agentic-workflows-orchestrator.md` - Section 06 (research assistant, code review, documentation agents)

### Level 2: Module Design Agents & Review Orchestrators (4 agents)

**Design Agents:**

- `architecture-design.md` - Component breakdown, interface design, data flow
- `integration-design.md` - Cross-component APIs, integration testing, Python-Mojo interop
- `security-design.md` - Threat modeling, security requirements, vulnerability prevention

**Review Orchestrators:**

- `code-review-orchestrator.md` - Routes PR changes to 13 specialist reviewers, consolidates feedback

### Level 3: Component Specialists & Review Specialists (24 agents)

**Implementation/Execution Specialists:**

- `implementation-specialist.md` - Break components into functions/classes, coordinate implementation
- `test-specialist.md` - Test planning, test case definition, coverage requirements
- `documentation-specialist.md` - Component READMEs, API docs, usage examples
- `performance-specialist.md` - Performance requirements, benchmarking, optimization
- `security-specialist.md` - Security implementation, testing, vulnerability remediation
- `blog-writer-specialist.md` - Development blog posts in narrative cycle format
- `numerical-stability-specialist.md` - Numerical precision and stability analysis for ML code
- `test-flakiness-specialist.md` - Analyzes and fixes flaky tests
- `pr-cleanup-specialist.md` - Handles PR cleanup and final polish before merge
- `mojo-syntax-validator.md` - Validates Mojo syntax and patterns
- `ci-failure-analyzer.md` - Analyzes CI failure logs and identifies root causes

**Code Review Specialists (13 agents):**

- `implementation-review-specialist.md` - Code correctness, logic, maintainability
- `documentation-review-specialist.md` - Markdown, comments, docstrings, API docs
- `test-review-specialist.md` - Test coverage, quality, assertions
- `security-review-specialist.md` - Vulnerabilities, OWASP top 10, input validation
- `safety-review-specialist.md` - Memory safety, type safety, undefined behavior
- `mojo-language-review-specialist.md` - Ownership, SIMD, fn vs def, traits
- `performance-review-specialist.md` - Algorithmic complexity, cache efficiency
- `algorithm-review-specialist.md` - ML algorithm correctness, numerical stability
- `architecture-review-specialist.md` - System design, modularity, patterns
- `data-engineering-review-specialist.md` - Data pipelines, preprocessing
- `paper-review-specialist.md` - Academic writing, citations
- `research-review-specialist.md` - Experimental design, reproducibility
- `dependency-review-specialist.md` - Version management, licenses

### Level 4: Implementation Engineers (6 agents)

- `senior-implementation-engineer.md` - Complex functions, performance-critical code, SIMD optimization
- `implementation-engineer.md` - Standard functions and classes, following specifications
- `test-engineer.md` - Unit and integration tests, test fixtures, test maintenance
- `documentation-engineer.md` - Docstrings, code examples, README updates
- `performance-engineer.md` - Benchmark code, profiling, optimization implementation
- `log-analyzer.md` - Parses build, test, and execution logs to extract diagnostic information

### Level 5: Junior Engineers (3 agents)

- `junior-implementation-engineer.md` - Simple functions, boilerplate generation, code formatting
- `junior-test-engineer.md` - Simple tests, test boilerplate, test execution
- `junior-documentation-engineer.md` - Docstring templates, formatting, changelog entries

## Best Practices

### Creating Agents

1. **Single Responsibility**: Each agent has one clear role
1. **Clear Description**: Description should trigger appropriate auto-invocation
1. **Tool Minimalism**: Only request tools actually needed
1. **Rich Examples**: Show realistic usage scenarios
1. **Clear Constraints**: Document what agent should NOT do

### Using Agents

1. **Trust the Hierarchy**: Let orchestrators delegate, don't micromanage
1. **Communicate Status**: Report progress clearly
1. **Escalate Blockers**: Don't stay stuck, escalate when needed
1. **Coordinate Horizontally**: Communicate with peer agents
1. **Document Decisions**: Capture rationale for future reference

### Maintenance

1. **Update Templates**: Keep templates current with best practices
1. **Share Learnings**: Document what works and what doesn't
1. **Iterate Configs**: Improve agent configs based on usage
1. **Version Control**: Commit all changes to agent configs
1. **Team Review**: Review new agents with team

## Common Patterns

### Pattern: Implementing a New Feature

1. **Chief Architect** analyzes requirements
1. **Section Orchestrator** breaks into modules
1. **Architecture Design Agent** designs module structure
1. **Component Specialists** plan Test/Impl/Package in parallel
1. **Engineers** execute in parallel worktrees
1. **All Levels** participate in cleanup

### Pattern: Fixing a Bug

1. **Test Engineer** writes failing test
1. **Component Specialist** analyzes root cause
1. **Implementation Engineer** fixes code
1. **Test Engineer** verifies fix
1. **Documentation Writer** updates docs if needed

### Pattern: Refactoring

1. **Component Specialist** identifies refactoring need
1. **Architecture Design Agent** reviews impact
1. **Implementation Engineer** performs refactoring
1. **Test Engineer** ensures tests still pass
1. **Performance Engineer** verifies no performance regression

## Troubleshooting

### Agent Not Invoked Automatically

**Problem**: Claude doesn't automatically invoke your agent

### Solutions

1. Check description - make it specific and clear
1. Add trigger keywords user would naturally say
1. Test with explicit invocation first
1. Review Claude Code sub-agents documentation

### Agent Scope Unclear

**Problem**: Unclear which level agent belongs to

### Solutions

1. Review [hierarchy.md](hierarchy.md) for level definitions
1. Consider decision scope: System → Section → Module → Component → Function → Line
1. Ask: What does this agent decide vs execute?

### Coordination Issues

**Problem**: Agents aren't coordinating effectively

### Solutions

1. Review [delegation-rules.md](delegation-rules.md)
1. Use git worktrees for isolation
1. Establish clear handoff protocols
1. Document interfaces explicitly

## Documentation

### In agents/ (This Directory)

- **README.md** (this file) - Overview and quick start
- **hierarchy.md** - Visual hierarchy diagram, quick reference, and complete agent specifications
- **delegation-rules.md** - Coordination and delegation patterns
- **templates/** - Agent configuration templates
- **docs/** - Integration documentation
  - **5-phase-integration.md** - How agents map to the 5-phase workflow
  - **git-worktree-guide.md** - Using git worktrees with agents
  - **workflows.md** - Common workflow examples (features, bugs, refactoring, etc.)

### In notes/review/

- **agent-architecture-review.md** - Architectural decisions and review
- **skills-design.md** - Skills taxonomy and design
- **orchestration-patterns.md** - Delegation and coordination details
- **worktree-strategy.md** - Git worktree workflow
- **agent-skills-overview.md** - System overview
- **agent-skills-implementation-summary.md** - Implementation summary and lessons learned

### In GitHub Issues

Issue-specific documentation is posted directly to GitHub issues as comments.
See issues #62 through #67 (agents) and #510 through #514 (skills).

## References

- [Claude Code Sub-Agents Documentation](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [Project 5-Phase Workflow](../notes/review/README.md)
- [Complete Agent Hierarchy](hierarchy.md)
- [Skills Design](../notes/review/skills-design.md)
- [Orchestration Patterns](../notes/review/orchestration-patterns.md)

## Contributing

### Adding a New Agent Type

1. Determine appropriate level (0-5)
1. Create configuration from template
1. Test with example tasks
1. Document in this README
1. Submit PR for team review

### Improving Documentation

1. Identify gaps or unclear areas
1. Update relevant documentation
1. Add examples if helpful
1. Submit PR for review

## Support

For questions or issues:

1. Review documentation in this directory
1. Check `/notes/review/` for detailed specs and architectural reviews
1. Check GitHub issues for individual issue documentation
1. Consult Claude Code documentation
1. Ask in team channels

## Version History

- **v1.0** (2025-11-07): Initial agent system design
  - 6-level hierarchy established
  - Skills system designed
  - Documentation created
  - Templates provided
