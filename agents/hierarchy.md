# Agent Hierarchy - Visual Diagram and Quick Reference

## Hierarchy Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                    Level 0: Meta-Orchestrator                │
│                   Chief Architect Agent                      │
│         (System-wide decisions, paper selection)             │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Level 1:       │ │   Level 1:       │ │   Level 1:       │
│   Foundation     │ │ Shared Library   │ │    Tooling       │
│  Orchestrator    │ │  Orchestrator    │ │  Orchestrator    │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                     │
┌────────┴────────┐  ┌────────┴────────┐  ┌────────┴────────┐
│  Level 1: Paper │  │  Level 1: CI/CD │  │ Level 1: Agentic│
│ Implementation  │  │  Orchestrator   │  │    Workflows    │
│  Orchestrator   │  │                 │  │  Orchestrator   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              ▼
            ┌─────────────────────────────────────┐
            │   Level 2: Design & Review Agents   │
            ├─────────────────────────────────────┤
            │  • Architecture Design Agent        │
            │  • Integration Design Agent         │
            │  • Security Design Agent            │
            │  • Code Review Orchestrator         │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │   Level 3: Specialists (22 agents)   │
            ├──────────────────────────────────────┤
            │  • Implementation Specialist         │
            │  • Test Specialist                   │
            │  • Documentation Specialist          │
            │  • Performance Specialist            │
            │  • 13 Code Review Specialists        │
            │  • 4 Additional Specialists          │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │   Level 4: Engineers (6 agents)      │
            ├──────────────────────────────────────┤
            │  • Senior Implementation Engineer    │
            │  • Implementation Engineer           │
            │  • Test Engineer                     │
            │  • Documentation Engineer            │
            │  • Performance Engineer              │
            │  • Log Analyzer                      │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │      Level 5: Junior Engineers       │
            ├──────────────────────────────────────┤
            │  • Junior Implementation Engineer    │
            │  • Junior Test Engineer              │
            │  • Junior Documentation Engineer     │
            └──────────────────────────────────────┘
```

## Level Summaries

### Level 0: Meta-Orchestrator

- **Agents**: 1 (Chief Architect)
- **Scope**: Entire repository
- **Decisions**: Strategic (paper selection, tech stack, architecture)
- **Phase**: Primarily Plan
- **Language Context**: Makes Mojo vs Python decisions for different components

### Level 1: Section Orchestrators

- **Agents**: 6 (one per major section)
- **Scope**: Repository sections
- **Decisions**: Tactical (module organization, dependencies)
- **Phase**: Plan
- **Language Context**: Coordinates Mojo implementation across sections

### Level 2: Module Design & Review Agents

- **Agents**: 4 total (3 design agents + 1 review orchestrator)
  - Design Agents: Architecture, Integration, Security
  - Code Review Orchestrator: Routes PR changes to 13 specialist reviewers
- **Scope**: Modules within sections and overall PR review coordination
- **Decisions**: Module structure, interfaces, security, and code review routing
- **Phase**: Plan (design) and Cleanup (code review)
- **Language Context**: Designs Mojo module structures, leverages Mojo features (SIMD, traits, structs);
  coordinates review across all code dimensions

### Level 3: Specialists & Review Specialists

- **Agents**: 22 total (9 implementation/execution specialists + 13 code review specialists)
- **Scope**: Components within modules and PR review dimensions
- **Decisions**: Component implementation approach and code review assessment
- **Phase**: Plan, Test, Implementation, Package, Cleanup
- **Language Context**: Chooses Mojo patterns (fn vs def, struct vs class, SIMD usage);
  reviews for language correctness, safety, and idioms
- **Package Phase**: Design packaging strategy, specify .mojopkg requirements, plan CI/CD workflows
- **Review Specialties**: Implementation, documentation, test, security, safety, Mojo language,
  performance, algorithm, architecture, data engineering, paper, research, dependency review

### Level 4: Implementation Engineers

- **Agents**: 6 (Senior Implementation, Implementation, Test, Documentation, Performance, Log Analyzer)
- **Scope**: Functions and classes
- **Decisions**: Implementation details
- **Phase**: Test, Implementation, Package
- **Language Context**: Writes Mojo code, uses Mojo standard library, implements algorithms
- **Package Phase**: Build .mojopkg files, create distribution archives, implement packaging scripts

### Level 5: Junior Engineers

- **Agents**: 3 types (Implementation, Test, Documentation)
- **Scope**: Simple functions, boilerplate
- **Decisions**: None (follows instructions)
- **Phase**: Test, Implementation, Package
- **Language Context**: Generates Mojo boilerplate, applies formatting
- **Package Phase**: Run package builds, verify installations, execute packaging commands

## Mojo-Specific Considerations

### Language Expertise by Level

**Level 0-2** (Architects and Designers):

- Deep understanding of Mojo vs Python trade-offs
- Knowledge of Mojo compilation model
- Familiarity with SIMD operations and performance characteristics
- Understanding of MAX platform integration

**Level 3** (Specialists):

- Proficiency in Mojo syntax and idioms
- Knowledge of Mojo traits, structs, and memory management
- Understanding of `fn` vs `def`, owned vs borrowed, etc.
- Ability to design Mojo-specific patterns

**Level 4-5** (Engineers):

- Hands-on Mojo coding ability
- Familiarity with Mojo standard library
- Ability to write performance-critical code
- Knowledge of Mojo testing frameworks

### Mojo-Python Hybrid Considerations

- **Level 0-1**: Decide which components use Mojo vs Python
  - Mojo: Performance-critical ML operations (training, inference)
  - Python: Scripting, tooling, data loading, visualization

- **Level 2-3**: Design interop between Mojo and Python
  - Use Mojo for tensor operations
  - Use Python for data preprocessing
  - Design clean interfaces between languages

- **Level 4-5**: Implement with appropriate language
  - Follow architecture decisions on language choice
  - Ensure proper type annotations
  - Handle conversions between Python and Mojo

## Delegation Flow

### Top-Down (Task Decomposition)

```text
Paper Selection (Level 0)
    ↓
Section Planning (Level 1)
    ↓
Module Design (Level 2)
    ↓
Component Specification (Level 3)
    ↓
Function Implementation (Level 4)
    ↓
Boilerplate Generation (Level 5)
```

### Bottom-Up (Status Reporting)

```text
Code Metrics (Level 5)
    ↑
Component Health (Level 4)
    ↑
Module Stability (Level 3)
    ↑
Section Status (Level 2)
    ↑
Project Health (Level 1)
    ↑
Strategic Alignment (Level 0)
```

## Agent Count

| Level | Name | Current Count |
|-------|------|---|
| 0     | Meta-Orchestrator | 1 |
| 1     | Section Orchestrators | 6 |
| 2     | Module Design & Review Orchestrators | 4 |
| 3     | Specialists (Implementation + Code Review) | 22 |
| 4     | Implementation Engineers | 6 |
| 5     | Junior Engineers | 3 |
| **Total** | **All Agents** | **42** |

**Level 3 Breakdown:**

- Implementation/Execution Specialists: 9 (implementation, test, documentation, performance,
  security, numerical stability, test flakiness, mojo syntax validator, CI failure analyzer)
- Code Review Specialists: 13 (implementation, documentation, test, security, safety,
  mojo language, performance, algorithm, architecture, data engineering, paper, research,
  dependency)

*Historical Note: Initial planning estimated 23 agent types. Actual implementation has expanded
and been refined to 42 specialized agents. Blog writer and PR cleanup were converted to skills
as they perform fixed, repeatable tasks.*

## Quick Reference

### When to Use Each Level

**Use Level 0** when:

- Selecting which research paper to implement
- Making system-wide architectural decisions
- Resolving cross-section conflicts

**Use Level 1** when:

- Planning a major repository section
- Coordinating multiple modules
- Managing section dependencies

**Use Level 2** when:

- Designing module architecture
- Defining component interfaces
- Planning security or integration

**Use Level 3** when:

- Specifying component implementation
- Planning tests for a component
- Designing performance optimization strategy

**Use Level 4** when:

- Writing Mojo code for functions/classes
- Implementing tests
- Writing documentation

**Use Level 5** when:

- Generating Mojo boilerplate
- Formatting code
- Simple documentation tasks

## Coordination Rules

1. **Delegate Down**: When task is too detailed for current level
1. **Escalate Up**: When decision exceeds current authority
1. **Coordinate Laterally**: When sharing resources or dependencies
1. **Report Status**: Keep superior informed of progress
1. **Document Decisions**: Capture rationale for future reference

## Detailed Agent Specifications

### Level 0: Meta-Orchestrator

#### Chief Architect Agent

**Scope**: Entire repository ecosystem

**Responsibilities**:

- Select which AI research papers to implement
- Define repository-wide architectural patterns
- Establish coding standards and conventions
- Coordinate across all 6 major sections
- Resolve conflicts between section orchestrators
- Make technology stack decisions
- Monitor overall project health

**Inputs**: Research papers, user requirements, project goals, industry best practices

**Outputs**: High-level roadmap, ADRs, section assignments, technology selection documents,
cross-section dependency graphs

**Delegates To**: Section Orchestrators (Level 1)

**Coordinates With**: External stakeholders, repository owners

**Decision Scope**: System-wide (multiple sections)

**Workflow Phase**: Primarily Plan phase, oversight in all phases

**Configuration File**: `.claude/agents/chief-architect.md`

---

### Level 1: Section Orchestrators

#### Foundation Orchestrator

**Scope**: Section 01-foundation

**Responsibilities**:

- Coordinate directory structure creation
- Manage configuration file setup
- Oversee initial documentation
- Ensure foundation is ready before other sections proceed

**Delegates To**: Module Design Agents

**Artifacts**: Foundation completion report, configuration baselines

#### Shared Library Orchestrator

**Scope**: Section 02-shared-library

**Responsibilities**:

- Design shared component architecture
- Coordinate core operations, training utilities, data utilities
- Ensure API consistency across modules
- Manage backward compatibility

**Delegates To**: Module Design Agents

**Artifacts**: API documentation, shared library release notes

#### Tooling Orchestrator

**Scope**: Section 03-tooling

**Responsibilities**:

- Coordinate tooling development
- Ensure tools integrate with workflow
- Manage CLI interfaces and automation scripts

**Delegates To**: Module Design Agents

**Artifacts**: Tool documentation, automation scripts

#### Paper Implementation Orchestrator

**Scope**: Section 04-first-paper and future papers

**Responsibilities**:

- Analyze research paper requirements
- Design paper-specific architecture
- Coordinate data preparation, model implementation, training, evaluation
- Ensure paper implementation follows repository patterns

**Delegates To**: Module Design Agents

**Artifacts**: Paper implementation report, evaluation results

#### CI/CD Orchestrator

**Scope**: Section 05-ci-cd

**Responsibilities**:

- Design CI/CD pipeline architecture
- Coordinate testing infrastructure, deployment processes, monitoring
- Ensure quality gates are effective

**Delegates To**: Module Design Agents

**Artifacts**: Pipeline configurations, quality metrics

#### Agentic Workflows Orchestrator

**Scope**: Section 06-agentic-workflows

**Responsibilities**:

- Design agent system architecture
- Coordinate research assistant, code review agent, documentation agent
- Ensure agents follow Claude best practices

**Delegates To**: Module Design Agents

**Artifacts**: Agent configurations, prompt templates

**Configuration Files**: `.claude/agents/foundation-orchestrator.md`, etc.

---

### Level 2: Module Design Agents

#### Architecture Design Agent

**Scope**: Module-level architecture

**Responsibilities**:

- Break down module into components
- Define component interfaces and contracts
- Design data flow within module
- Identify reusable patterns
- Create module architecture documents

**Inputs**: Section requirements from orchestrator

**Outputs**: Component specifications, interface definitions

**Delegates To**: Component Specialists (Level 3)

**Coordinates With**: Other Module Design Agents for cross-module dependencies

**Workflow Phase**: Plan phase

**Configuration File**: `.claude/agents/architecture-design.md`

#### Integration Design Agent

**Scope**: Module-level integration

**Responsibilities**:

- Design integration points between components
- Define module-level APIs
- Create integration test plans
- Manage module dependencies

**Delegates To**: Component Specialists

**Artifacts**: Integration diagrams, API specifications

**Configuration File**: `.claude/agents/integration-design.md`

#### Security Design Agent

**Scope**: Module-level security

**Responsibilities**:

- Threat modeling for module
- Define security requirements
- Design authentication/authorization if needed
- Review for security vulnerabilities

**Delegates To**: Security Implementation Specialist (Level 3)

**Artifacts**: Threat models, security requirements

**Configuration File**: `.claude/agents/security-design.md`

---

### Level 3: Component Specialists

#### Senior Implementation Specialist

**Scope**: Complex components

**Responsibilities**:

- Break component into functions/classes
- Design component architecture
- Create detailed implementation plan
- Review code quality

**Delegates To**: Implementation Engineers (Level 4)

**Coordinates With**: Test Specialist, Documentation Specialist

**Artifacts**: Component design docs, code review reports

**Workflow Phase**: Plan, Implementation, Cleanup

**Configuration File**: `.claude/agents/senior-implementation-specialist.md`

#### Test Design Specialist

**Scope**: Component-level testing

**Responsibilities**:

- Create test plan for component
- Define test cases (unit, integration, edge cases)
- Design test fixtures and mocks
- Specify coverage requirements

**Delegates To**: Test Engineers (Level 4)

**Artifacts**: Test plans, test case specifications

**Workflow Phase**: Plan, Test

**Configuration File**: `.claude/agents/test-design-specialist.md`

#### Documentation Specialist

**Scope**: Component-level documentation

**Responsibilities**:

- Write component README
- Document APIs and interfaces
- Create usage examples
- Write tutorials if needed
- **Package Phase**: Create distribution documentation, installation guides, package metadata

**Delegates To**: Documentation Writers (Level 4)

**Artifacts**: READMEs, API docs, tutorials, package documentation

**Workflow Phase**: Plan, Package, Cleanup

**Configuration File**: `.claude/agents/documentation-specialist.md`

#### Performance Specialist

**Scope**: Component-level performance

**Responsibilities**:

- Define performance requirements
- Design benchmarks
- Identify optimization opportunities
- Profile and analyze performance

**Delegates To**: Performance Engineers (Level 4)

**Artifacts**: Benchmark results, performance reports

**Workflow Phase**: Plan, Implementation, Cleanup

**Configuration File**: `.claude/agents/performance-specialist.md`

#### Security Implementation Specialist

**Scope**: Component-level security implementation

**Responsibilities**:

- Implement security requirements
- Code security best practices
- Perform security testing
- Fix vulnerabilities

**Delegates To**: Implementation Engineers (Level 4)

**Artifacts**: Security test results, vulnerability reports

**Workflow Phase**: Plan, Implementation, Test, Cleanup

**Configuration File**: `.claude/agents/security-implementation-specialist.md`

---

### Level 4: Implementation Engineers

#### Senior Implementation Engineer

**Scope**: Complex functions/classes

**Responsibilities**:

- Write implementation code
- Follow coding standards
- Implement error handling
- Write inline documentation
- Optimize algorithms

**Inputs**: Detailed specifications from specialists

**Outputs**: Implementation code, unit tests

**Delegates To**: Junior Engineers for simple tasks

**Coordinates With**: Test Engineers for TDD

**Workflow Phase**: Implementation

**Skills Used**: code_generation, refactoring, optimization

**Configuration File**: `.claude/agents/senior-implementation-engineer.md`

#### Implementation Engineer

**Scope**: Standard functions/classes

**Responsibilities**:

- Write implementation code
- Follow coding patterns
- Write basic tests
- Document code

**Delegates To**: Junior Engineers for repetitive tasks

**Artifacts**: Source code files, unit tests

**Workflow Phase**: Implementation

**Skills Used**: code_generation, testing, documentation

**Configuration File**: `.claude/agents/implementation-engineer.md`

#### Test Engineer

**Scope**: Test implementation

**Responsibilities**:

- Implement unit tests
- Implement integration tests
- Create test fixtures
- Maintain test suite
- Fix failing tests

**Coordinates With**: Implementation Engineers

**Artifacts**: Test files, test reports

**Workflow Phase**: Test

**Skills Used**: test_generation, test_execution, coverage_analysis

**Configuration File**: `.claude/agents/test-engineer.md`

#### Documentation Writer

**Scope**: Documentation writing

**Responsibilities**:

- Write docstrings
- Create code examples
- Write README sections
- Update documentation as code changes
- **Package Phase**: Write installation guides, package READMEs, distribution documentation

**Artifacts**: Documentation files, docstrings, package documentation

**Workflow Phase**: Package

**Skills Used**: documentation_generation, example_extraction

**Configuration File**: `.claude/agents/documentation-writer.md`

#### Performance Engineer

**Scope**: Performance implementation

**Responsibilities**:

- Write benchmark code
- Profile code execution
- Implement optimizations
- Verify performance improvements

**Artifacts**: Benchmark code, profiling results

**Workflow Phase**: Implementation, Cleanup

**Skills Used**: profiling, benchmarking, optimization

**Configuration File**: `.claude/agents/performance-engineer.md`

---

### Level 5: Junior Engineers

#### Junior Implementation Engineer

**Scope**: Simple functions, boilerplate code

**Responsibilities**:

- Write simple functions
- Generate boilerplate code
- Apply code templates
- Format code
- Run linters

**Inputs**: Clear, detailed instructions

**Outputs**: Simple code implementations

**No Delegation**: Lowest level of hierarchy

**Workflow Phase**: Implementation

**Skills Used**: boilerplate_generation, code_formatting, linting

**Configuration File**: `.claude/agents/junior-implementation-engineer.md`

#### Junior Test Engineer

**Scope**: Simple test cases

**Responsibilities**:

- Write simple unit tests
- Generate test boilerplate
- Update existing tests
- Run test suites

**Artifacts**: Basic test implementations

**Workflow Phase**: Test

**Skills Used**: test_generation, test_execution

**Configuration File**: `.claude/agents/junior-test-engineer.md`

#### Junior Documentation Engineer

**Scope**: Simple documentation

**Responsibilities**:

- Fill in docstring templates
- Format documentation
- Generate changelog entries
- Update simple README sections

**Artifacts**: Basic documentation

**Workflow Phase**: Packaging

**Skills Used**: documentation_generation, formatting

**Configuration File**: `.claude/agents/junior-documentation-engineer.md`

---

## Delegation Rules

1. **Scope Reduction**: Each delegation reduces scope by one level of abstraction:
   System → Section → Module → Component → Function → Line
1. **Specification Detail**: Each level adds more detail to specifications:
   Strategic goals → Tactical plans → Component specs → Implementation details → Code
1. **Autonomy Increase**: Lower levels have more implementation autonomy but less strategic freedom
1. **Review Responsibility**: Each level reviews work of the level below
1. **Escalation Path**: Issues escalate one level up until resolved
1. **Coordination Requirements**: Agents coordinate horizontally when sharing resources or dependencies

## Agent Configuration Template

```text
---
name: agent-name
description: Brief description of when to use this agent
tools: Read,Write,Edit,Bash,Grep,Glob
model: sonnet
---

# Agent Name

## Role

[Agent's role in the hierarchy]

## Responsibilities

- Responsibility 1
- Responsibility 2

## Scope

[What this agent handles]

## Delegation

Delegates to: [Lower level agents]
Coordinates with: [Same level agents]

## Workflow Phase

[Which phases this agent participates in]

## Skills Used

- skill_name_1
- skill_name_2

## Instructions

[Detailed instructions for this agent]

## Examples

[Example tasks this agent handles]

## Constraints

[What this agent should NOT do]
```

## Mapping to Organizational Models

### Traditional Hierarchy

- Level 0 = CTO/VP Engineering
- Level 1 = Engineering Managers
- Level 2 = Principal/Staff Engineers
- Level 3 = Senior Engineers
- Level 4 = Engineers
- Level 5 = Junior Engineers/Interns

### Spotify Model

- Tribes = Level 1 (Section Orchestrators)
- Squads = Level 2 (Module Design Agents)
- Chapters = Cross-cutting specialists
- Guilds = Skills shared across agents

## Integration with 5-Phase Workflow

| Phase | Active Levels | Focus |
|-------|---------------|-------|
| Plan | 0, 1, 2, 3 | Orchestrators and designers create specifications |
| Test | 3, 4, 5 | Specialists and engineers write tests |
| Implementation | 3, 4, 5 | Specialists and engineers build functionality |
| Package | 3, 4, 5 | Specialists and engineers create distributable packages (.mojopkg, archives, CI/CD) |
| Cleanup | All | All levels review and refactor their work |

## See Also

- [README.md](README.md) - Overview and quick start
- [delegation-rules.md](delegation-rules.md) - Detailed coordination patterns
- [templates/](templates/) - Agent configuration templates
- [/notes/review/orchestration-patterns.md](../notes/review/orchestration-patterns.md) - Orchestration patterns
- [Mojo Documentation](https://docs.modular.com/mojo/manual/) - Mojo language reference
