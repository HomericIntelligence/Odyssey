# Agent Catalog - Complete Reference

<!-- markdownlint-disable MD051 -->

## Table of Contents

- [Overview](#overview)
- [Quick Reference Table](#quick-reference-table)
- [Level 0: Meta-Orchestrator](#level-0-meta-orchestrator)
- [Level 1: Section Orchestrators](#level-1-section-orchestrators)
- [Level 2: Module Design Agents](#level-2-module-design-agents)
- [Level 3: Component Specialists](#level-3-component-specialists)
- [Level 4: Implementation Engineers](#level-4-implementation-engineers)
- [Level 5: Junior Engineers](#level-5-junior-engineers)
- [How to Use This Catalog](#how-to-use-this-catalog)

## Overview

This catalog lists all 43 agents in the ML Odyssey hierarchical agent system. Each entry includes:

- **Name**: Agent identifier
- **Description**: What the agent does
- **When to Use**: Situations to invoke this agent
- **Capabilities**: What it can do
- **Example Use Cases**: Concrete scenarios

## Quick Reference Table

Representative sample of 43 agents (see detailed sections below for complete list):

| Level | Agent Name | Primary Focus | Scope |
|-------|-----------|---------------|-------|
| **0** | Chief Architect | Strategic planning | Entire repository |
| **1** | Foundation Orchestrator | Repository structure | Section 01 |
| **1** | Shared Library Orchestrator | Core components | Section 02 |
| **1** | Tooling Orchestrator | Developer tools | Section 03 |
| **1** | Papers Orchestrator | Research papers | Section 04 |
| **1** | CI/CD Orchestrator | Testing/deployment | Section 05 |
| **1** | Agentic Workflows Orchestrator | Automation agents | Section 06 |
| **2** | Architecture Design Agent | Module architecture | Module design |
| **2** | Integration Design Agent | Cross-module integration | Module integration |
| **2** | Security Design Agent | Security architecture | Security design |
| **2** | Code Review Orchestrator | PR review coordination | Code review |
| **3** | Implementation Specialist | Implementation coordination | Component implementation |
| **3** | Test Specialist | Test coordination | Component testing |
| **3** | Documentation Specialist | Documentation coordination | Component docs |
| **3** | Performance Specialist | Performance optimization | Component performance |
| **3** | Blog Writer Specialist | Development blog posts | Narrative content |
| **3** | Implementation Review Specialist | Code correctness review | PR dimensions |
| **3** | Safety Review Specialist | Memory safety review | PR dimensions |
| **3** | Mojo Language Review Specialist | Language patterns review | PR dimensions |
| **3** | *+ 11 more Level 3 specialists* | *Various review dimensions* | *PR reviews* |
| **4** | Senior Implementation Engineer | Complex code | Complex functions |
| **4** | Implementation Engineer | Standard code | Standard functions |
| **4** | Test Engineer | Test implementation | Unit/integration tests |
| **4** | Documentation Engineer | Documentation writing | API docs, examples |
| **4** | Performance Engineer | Performance tuning | Benchmarks, profiling |
| **4** | Log Analyzer | Log parsing & analysis | Diagnostic info |
| **5** | Junior Implementation Engineer | Simple code | Boilerplate, simple functions |
| **5** | Junior Test Engineer | Simple tests | Test boilerplate |

## Level 0: Meta-Orchestrator

### Chief Architect Agent

**Configuration**: `.claude/agents/chief-architect.md`

**Description**: Makes system-wide strategic decisions, selects papers to implement, and coordinates across all
repository sections

### When to Use

- Selecting which research paper to implement next
- Making repository-wide architectural decisions
- Resolving conflicts between section orchestrators
- Deciding technology stack (Mojo vs Python for different components)
- Establishing project-wide standards and conventions
- Evaluating overall project health and direction

### Capabilities

- Paper analysis and selection based on project goals
- System-wide architecture design
- Technology stack evaluation (Mojo, Python, MAX platform)
- Cross-section coordination and conflict resolution
- Strategic planning and roadmap creation
- Repository-wide standard establishment

### Example Use Cases

```text
"Should we implement ResNet-50 or Transformer architecture next?"

"What's our strategy for Mojo vs Python in the data pipeline?"

"Review the overall project architecture and identify improvements"

"Establish repository-wide naming conventions and code standards"

"Evaluate if we should migrate from Python to Mojo for data loading"
```text

**Delegates To**: Section Orchestrators (Level 1)

**Coordinates With**: External stakeholders, repository maintainers

**Workflow Phase**: Primarily Plan, oversight in all phases

---

## Level 1: Section Orchestrators

### Foundation Orchestrator

**Configuration**: `.claude/agents/foundation-orchestrator.md`

**Description**: Manages repository structure, directory organization, and configuration files (Section 01)

### When to Use

- Setting up initial repository structure
- Creating or reorganizing directory hierarchies
- Managing configuration files (pixi.toml, .gitignore, etc.)
- Establishing foundational documentation
- Coordinating infrastructure setup

### Capabilities

- Directory structure design and creation
- Configuration file management
- Initial documentation setup
- Foundation readiness validation
- Prerequisite coordination for other sections

### Example Use Cases

```text
"Set up the directory structure for our new paper implementation"

"Organize configuration files and development environment setup"

"Create the foundational README and documentation structure"

"Ensure the papers/ directory is ready for LeNet-5 implementation"
```text

**Delegates To**: Module Design Agents (Level 2)

**Coordinates With**: Other Section Orchestrators for dependencies

**Workflow Phase**: Plan (primarily), early setup

---

### Shared Library Orchestrator

**Configuration**: `.claude/agents/shared-library-orchestrator.md`

**Description**: Coordinates core reusable components including tensor operations, neural network layers, and
training utilities (Section 02)

### When to Use

- Adding new neural network layer types
- Implementing core tensor operations
- Creating training utilities (optimizers, loss functions)
- Designing shared data structures
- Managing API compatibility across the library

### Capabilities

- Shared component architecture design
- API consistency enforcement
- Backward compatibility management
- Core operations coordination (tensor ops, layers, training utils)
- Cross-paper component reusability

### Example Use Cases

```text
"Add a batch normalization layer to the shared library"

"Implement Adam optimizer for all papers to use"

"Design the tensor API for consistent usage across papers"

"Create shared dropout layer with configurable drop rate"

"Ensure new Conv2D layer is compatible with existing Dense layer"
```text

**Delegates To**: Architecture, Integration, Security Design Agents (Level 2)

**Coordinates With**: Papers Orchestrator (for requirements), CI/CD Orchestrator (for testing)

**Workflow Phase**: Plan, then oversees Test/Impl/Package

---

### Tooling Orchestrator

**Configuration**: `.claude/agents/tooling-orchestrator.md`

**Description**: Manages developer tools, CLI utilities, and automation scripts (Section 03)

### When to Use

- Creating CLI tools for common tasks
- Developing automation scripts
- Building development utilities
- Implementing helper tools for workflows
- Coordinating tool integration

### Capabilities

- CLI tool design and implementation
- Automation script coordination
- Developer workflow enhancement
- Tool integration with existing systems
- Script organization and documentation

### Example Use Cases

```text
"Create a CLI tool for generating new paper scaffolding"

"Build an automation script for running all benchmarks"

"Implement a utility for converting datasets to Mojo format"

"Design a tool for profiling neural network layers"
```text

**Delegates To**: Architecture Design, Integration Design Agents (Level 2)

**Coordinates With**: All orchestrators (tools used by all)

**Workflow Phase**: Plan, then oversees Test/Impl/Package

---

### Papers Orchestrator

**Configuration**: `.claude/agents/papers-orchestrator.md`

**Description**: Coordinates research paper implementations including data preparation, model architecture, training,
and evaluation (Section 04)

### When to Use

- Implementing a new research paper
- Coordinating paper-specific components
- Managing data preparation pipelines
- Overseeing model training and evaluation
- Ensuring paper implementation follows standards

### Capabilities

- Research paper analysis and breakdown
- Paper-specific architecture design
- Data preparation coordination
- Model implementation oversight
- Training and evaluation coordination
- Results validation and documentation

### Example Use Cases

```text
"Implement the LeNet-5 paper end-to-end"

"Coordinate the ResNet-50 implementation across all phases"

"Manage data preparation for MNIST dataset"

"Oversee training pipeline for the current paper"

"Ensure evaluation metrics match the paper's specifications"
```text

**Delegates To**: Architecture, Integration Design Agents (Level 2)

**Coordinates With**: Shared Library Orchestrator (for reusable components)

**Workflow Phase**: Plan, then oversees all phases for paper implementation

---

### CI/CD Orchestrator

**Configuration**: `.claude/agents/cicd-orchestrator.md`

**Description**: Manages continuous integration, testing infrastructure, deployment pipelines, and quality gates
(Section 05)

### When to Use

- Setting up CI/CD pipelines
- Configuring testing infrastructure
- Implementing deployment processes
- Establishing quality gates
- Monitoring build and test health

### Capabilities

- CI/CD pipeline design and configuration
- Testing infrastructure setup (unit, integration, performance tests)
- Deployment process automation
- Quality gate enforcement
- Build and test monitoring
- Pre-commit hook management

### Example Use Cases

```text
"Set up GitHub Actions for automated testing"

"Configure pre-commit hooks for code quality"

"Create deployment pipeline for Mojo models"

"Implement quality gates for pull requests"

"Set up performance regression testing"
```text

**Delegates To**: Architecture Design, Integration Design Agents (Level 2)

**Coordinates With**: All orchestrators (CI/CD serves all)

**Workflow Phase**: Plan, then continuous operation across all phases

---

### Agentic Workflows Orchestrator

**Configuration**: `.claude/agents/agentic-workflows-orchestrator.md`

**Description**: Coordinates Claude-powered automation including research assistants, code review agents, and
documentation generators (Section 06)

### When to Use

- Creating new agent configurations
- Designing automated workflows
- Implementing Claude-powered tools
- Coordinating agent system improvements
- Managing agent templates and documentation

### Capabilities

- Agent system architecture design
- Sub-agent configuration creation
- Prompt template development
- Workflow automation design
- Agent coordination patterns
- Skills system management

### Example Use Cases

```text
"Create a new agent for paper summarization"

"Design workflow for automated code review"

"Implement documentation generation agent"

"Coordinate research assistant agent improvements"

"Update agent templates with new best practices"
```text

**Delegates To**: Architecture Design, Integration Design Agents (Level 2)

**Coordinates With**: All orchestrators (agents support all sections)

**Workflow Phase**: Plan, then enables automation in all phases

---

## Level 2: Module Design Agents

### Architecture Design Agent

**Configuration**: `.claude/agents/architecture-design.md`

**Description**: Designs module architecture, breaks modules into components, defines interfaces and data flow

### When to Use

- Designing a new module's architecture
- Breaking complex modules into components
- Defining component interfaces and contracts
- Designing data flow within modules
- Creating architectural documentation
- Identifying reusable patterns

### Capabilities

- Component decomposition and design
- Interface definition and API design
- Data flow architecture
- Design pattern identification
- Mojo struct and trait design
- Architectural documentation creation

### Example Use Cases

```text
"Design the architecture for the SGD optimizer module"

"Break down the CNN model into components"

"Define the interface between data loader and training loop"

"Create the architecture for tensor operations module"

"Design Mojo struct hierarchy for neural network layers"
```text

**Delegates To**: Component Specialists (Level 3)

**Coordinates With**: Integration Design Agent, Security Design Agent

**Workflow Phase**: Plan

---

### Integration Design Agent

**Configuration**: `.claude/agents/integration-design.md`

**Description**: Designs integration points between modules, defines cross-module APIs, and plans integration testing

### When to Use

- Designing integration between modules
- Defining cross-module APIs
- Planning integration test strategies
- Managing module dependencies
- Coordinating Mojo-Python interop
- Resolving interface conflicts

### Capabilities

- Cross-module API design
- Integration point identification
- Integration test planning
- Dependency management
- Interop design (Mojo-Python, FFI)
- Interface negotiation and conflict resolution

### Example Use Cases

```text
"Design the integration between model and optimizer"

"Define API for Mojo model to call Python data loader"

"Plan integration tests for training pipeline"

"Resolve interface conflicts between two modules"

"Design FFI bindings for Mojo-Python integration"
```text

**Delegates To**: Component Specialists (Level 3)

**Coordinates With**: Architecture Design Agent, other Module Design Agents

**Workflow Phase**: Plan

---

### Security Design Agent

**Configuration**: `.claude/agents/security-design.md`

**Description**: Performs threat modeling, defines security requirements, and designs secure architectures

### When to Use

- Threat modeling for new modules
- Defining security requirements
- Designing authentication/authorization
- Planning security testing strategies
- Reviewing for security vulnerabilities
- Ensuring memory safety in Mojo code

### Capabilities

- Threat modeling and risk assessment
- Security requirements definition
- Secure architecture design
- Memory safety analysis (buffer overflows, leaks)
- Numerical stability review (NaN/Inf handling)
- Security test planning

### Example Use Cases

```text
"Perform threat modeling for the model serving API"

"Define security requirements for model persistence"

"Ensure memory safety in tensor operations"

"Design secure authentication for API endpoints"

"Review Mojo code for potential buffer overflows"
```text

**Delegates To**: Security Specialist (Level 3)

**Coordinates With**: Architecture Design Agent, Integration Design Agent

**Workflow Phase**: Plan, with review in Implementation and Cleanup

---

## Level 3: Component Specialists

### Implementation Specialist

**Configuration**: `.claude/agents/implementation-specialist.md`

**Description**: Coordinates component implementation, breaks components into functions/classes, and manages
implementation work

### When to Use

- Creating detailed implementation plans
- Breaking components into functions and classes
- Coordinating implementation work
- Reviewing code quality
- Managing complex implementations
- Coordinating with Test and Documentation Specialists

### Capabilities

- Component decomposition into functions/classes
- Implementation planning and specification
- Code quality review
- Implementation coordination (parallel Test/Impl/Docs)
- Mojo code organization
- Refactoring guidance

### Example Use Cases

```text
"Create implementation plan for Conv2D layer"

"Break the training loop into manageable functions"

"Coordinate implementation of SGD optimizer"

"Review implementation quality for tensor operations"

"Plan refactoring of model architecture code"
```text

**Delegates To**: Implementation Engineers (Level 4)

**Coordinates With**: Test Specialist, Documentation Specialist, Performance Specialist

**Workflow Phase**: Plan, Implementation, Cleanup

---

### Test Specialist

**Configuration**: `.claude/agents/test-specialist.md`

**Description**: Plans comprehensive testing strategies, defines test cases, and coordinates test implementation

### When to Use

- Creating test plans for components
- Defining test cases (unit, integration, edge cases)
- Planning test coverage requirements
- Designing test fixtures and mocks
- Coordinating TDD workflows
- Reviewing test quality

### Capabilities

- Test plan creation
- Test case definition (unit, integration, property, performance)
- Coverage requirement specification
- Test fixture and mock design
- TDD coordination with Implementation Engineers
- Test quality review

### Example Use Cases

```text
"Create comprehensive test plan for ReLU activation"

"Define test cases for SGD optimizer edge cases"

"Plan integration tests for training pipeline"

"Design test fixtures for MNIST data loading"

"Coordinate TDD workflow for new layer type"
```text

**Delegates To**: Test Engineers (Level 4)

**Coordinates With**: Implementation Specialist, Performance Specialist

**Workflow Phase**: Plan, Test

---

### Documentation Specialist

**Configuration**: `.claude/agents/documentation-specialist.md`

**Description**: Coordinates documentation efforts, plans documentation structure, and ensures comprehensive coverage

### When to Use

- Planning component documentation
- Creating documentation structure
- Writing component READMEs
- Coordinating API documentation
- Creating tutorials and examples
- Reviewing documentation quality

### Capabilities

- Documentation planning and structure
- README creation
- API documentation coordination
- Tutorial and example design
- Documentation quality review
- User guide creation

### Example Use Cases

```text
"Plan documentation for tensor operations module"

"Create README for neural network layers"

"Coordinate API documentation for optimizers"

"Design tutorial for implementing custom layers"

"Review documentation for completeness and clarity"
```text

**Delegates To**: Documentation Writers (Level 4)

**Coordinates With**: Implementation Specialist, Test Specialist

**Workflow Phase**: Plan, Packaging, Cleanup

---

### Performance Specialist

**Configuration**: `.claude/agents/performance-specialist.md`

**Description**: Defines performance requirements, plans optimizations, and coordinates performance engineering

### When to Use

- Defining performance requirements
- Planning performance optimization strategies
- Designing benchmarks
- Identifying optimization opportunities
- Coordinating profiling and analysis
- Reviewing performance characteristics

### Capabilities

- Performance requirement definition
- Benchmark design
- Optimization strategy planning
- Profiling coordination
- SIMD optimization planning
- Performance analysis and review

### Example Use Cases

```text
"Define performance requirements for convolution layer"

"Plan SIMD optimization for matrix multiplication"

"Design benchmarks for optimizer performance"

"Identify bottlenecks in training loop"

"Coordinate profiling of neural network forward pass"
```text

**Delegates To**: Performance Engineers (Level 4)

**Coordinates With**: Implementation Specialist, Senior Implementation Engineer

**Workflow Phase**: Plan, Implementation, Cleanup

---

### Security Specialist

**Configuration**: `.claude/agents/security-specialist.md`

**Description**: Implements security requirements, performs security testing, and fixes vulnerabilities

### When to Use

- Implementing security requirements
- Performing security code reviews
- Coordinating security testing
- Fixing security vulnerabilities
- Ensuring memory safety in implementations
- Validating numerical stability

### Capabilities

- Security implementation
- Security code review
- Security test coordination
- Vulnerability remediation
- Memory safety verification
- Numerical stability validation

### Example Use Cases

```text
"Implement secure model serialization"

"Review tensor operations for memory safety"

"Coordinate security testing for API endpoints"

"Fix buffer overflow vulnerability in data loading"

"Validate numerical stability in loss calculations"
```text

**Delegates To**: Implementation Engineers (Level 4)

**Coordinates With**: Implementation Specialist, Test Specialist

**Workflow Phase**: Plan, Implementation, Test, Cleanup

---

## Level 4: Implementation Engineers

### Senior Implementation Engineer

**Configuration**: `.claude/agents/senior-implementation-engineer.md`

**Description**: Implements complex functions, performance-critical code, and advanced Mojo features like SIMD

### When to Use

- Implementing complex algorithms
- Writing performance-critical code
- Using advanced Mojo features (SIMD, traits, lifetimes)
- Optimizing hot paths
- Implementing low-level operations
- Mentoring junior engineers

### Capabilities

- Complex algorithm implementation
- SIMD vectorization and optimization
- Advanced Mojo features (traits, parametric types, lifetimes)
- Performance-critical code writing
- Low-level memory management
- Code review and mentorship

### Example Use Cases

```text
"Implement SIMD-optimized convolution operation"

"Write the backward pass for custom layer with advanced Mojo"

"Optimize matrix multiplication using SIMD intrinsics"

"Implement complex attention mechanism"

"Create zero-copy tensor operations using Mojo lifetimes"
```text

**Delegates To**: Junior Engineers (Level 5) for boilerplate

**Coordinates With**: Test Engineer, Performance Engineer, Implementation Engineer

**Workflow Phase**: Implementation

---

### Implementation Engineer

**Configuration**: `.claude/agents/implementation-engineer.md`

**Description**: Implements standard functions and classes following specifications

### When to Use

- Implementing standard functions
- Writing typical Mojo classes and structs
- Following established patterns
- Implementing straightforward algorithms
- Writing well-specified code
- Coordinating with test engineers (TDD)

### Capabilities

- Standard function and class implementation
- Mojo code writing (fn, def, struct, class)
- Pattern following and code consistency
- Basic algorithm implementation
- Error handling and validation
- TDD collaboration

### Example Use Cases

```text
"Implement the forward pass for Dense layer"

"Write helper functions for data preprocessing"

"Create struct for optimizer configuration"

"Implement standard activation functions (sigmoid, tanh)"

"Write data validation functions"
```text

**Delegates To**: Junior Engineers (Level 5) for simple tasks

**Coordinates With**: Test Engineer, Documentation Writer

**Workflow Phase**: Implementation

---

### Test Engineer

**Configuration**: `.claude/agents/test-engineer.md`

**Description**: Implements unit tests, integration tests, and maintains test suites

### When to Use

- Writing unit tests
- Implementing integration tests
- Creating test fixtures
- Following TDD workflow
- Maintaining test suites
- Debugging failing tests

### Capabilities

- Unit test implementation
- Integration test implementation
- Test fixture creation
- TDD workflow execution
- Test suite maintenance
- Test debugging and fixing

### Example Use Cases

```text
"Write unit tests for SGD optimizer step function"

"Implement integration tests for training pipeline"

"Create test fixtures for MNIST data loading"

"Debug failing test in convolution backward pass"

"Maintain and update test suite for refactored code"
```text

**Delegates To**: Junior Test Engineers (Level 5) for simple tests

**Coordinates With**: Implementation Engineer (TDD), Test Specialist

**Workflow Phase**: Test

---

### Documentation Writer

**Configuration**: `.claude/agents/documentation-engineer.md`

**Description**: Writes docstrings, API documentation, code examples, and README content

### When to Use

- Writing function and class docstrings
- Creating API documentation
- Writing code examples
- Updating READMEs
- Creating usage guides
- Documenting code changes

### Capabilities

- Docstring writing (Google, NumPy, or custom style)
- API documentation creation
- Code example writing
- README authoring and updates
- Usage guide creation
- Documentation maintenance

### Example Use Cases

```text
"Write docstrings for all tensor operation functions"

"Create API documentation for optimizer module"

"Write code examples showing how to use custom layers"

"Update README with new feature documentation"

"Create usage guide for training pipeline"
```text

**Delegates To**: None (handles formatting directly)

**Coordinates With**: Implementation Engineer, Documentation Specialist

**Workflow Phase**: Packaging

---

### Performance Engineer

**Configuration**: `.claude/agents/performance-engineer.md`

**Description**: Writes benchmarks, performs profiling, implements optimizations, and validates performance

### When to Use

- Writing benchmark code
- Profiling code execution
- Implementing performance optimizations
- Validating performance improvements
- Analyzing performance bottlenecks
- Measuring performance regressions

### Capabilities

- Benchmark implementation
- Profiling and analysis (CPU, memory, cache)
- Performance optimization implementation
- Performance validation and regression testing
- Bottleneck identification
- Performance reporting

### Example Use Cases

```text
"Write benchmarks for convolution layer performance"

"Profile training loop to identify bottlenecks"

"Implement cache-friendly memory access patterns"

"Validate SIMD optimization improvements"

"Measure and report performance regressions in PR"
```text

**Delegates To**: Junior Engineers (Level 5) for simple benchmarks

**Coordinates With**: Senior Implementation Engineer, Performance Specialist

**Workflow Phase**: Implementation, Cleanup

---

## Level 5: Junior Engineers

### Junior Implementation Engineer

**Configuration**: `.claude/agents/junior-implementation-engineer.md`

**Description**: Writes simple functions, generates boilerplate code, formats code, and runs automated tools

### When to Use

- Writing simple, well-specified functions
- Generating boilerplate code
- Creating getter/setter methods
- Formatting code
- Running linters and formatters
- Implementing template-based code

### Capabilities

- Simple function implementation
- Boilerplate code generation
- Code formatting and linting
- Template-based code creation
- Accessor method generation
- Basic code maintenance

### Example Use Cases

```text
"Create getter and setter methods for optimizer hyperparameters"

"Generate boilerplate for new struct definition"

"Format all code in this module using mojo format"

"Run pre-commit hooks and fix linting errors"

"Create simple helper functions following this pattern"
```text

**No Delegation**: Lowest level of hierarchy

**Coordinates With**: Implementation Engineer (receives tasks from)

**Workflow Phase**: Implementation

---

### Junior Test Engineer

**Configuration**: `.claude/agents/junior-test-engineer.md`

**Description**: Writes simple test cases, generates test boilerplate, and runs test suites

### When to Use

- Writing simple unit tests
- Generating test boilerplate
- Running test suites
- Updating existing simple tests
- Creating basic test fixtures
- Executing automated tests

### Capabilities

- Simple unit test implementation
- Test boilerplate generation
- Test suite execution
- Basic test fixture creation
- Test maintenance and updates
- Test automation execution

### Example Use Cases

```text
"Write simple unit test for this getter function"

"Generate test boilerplate for these three functions"

"Run the full test suite and report results"

"Update existing tests after function signature change"

"Create basic test fixture for this data structure"
```text

**No Delegation**: Lowest level of hierarchy

**Coordinates With**: Test Engineer (receives tasks from)

**Workflow Phase**: Test

---

## How to Use This Catalog

### Finding the Right Agent

#### By Task Type

**Planning/Design Tasks** → Levels 0-2

- "Design architecture" → Architecture Design Agent
- "Plan testing strategy" → Test Specialist
- "Select paper to implement" → Chief Architect

**Implementation Tasks** → Levels 3-5

- "Write complex SIMD code" → Senior Implementation Engineer
- "Implement standard function" → Implementation Engineer
- "Generate boilerplate" → Junior Implementation Engineer

**Coordination Tasks** → Levels 1-3

- "Coordinate module development" → Section Orchestrator
- "Manage component implementation" → Component Specialist
- "Oversee parallel work" → Implementation Specialist

#### By Expertise

### Mojo Expertise

- Advanced Mojo: Senior Implementation Engineer
- Standard Mojo: Implementation Engineer
- Simple Mojo: Junior Implementation Engineer

### Testing

- Test strategy: Test Specialist
- Test implementation: Test Engineer
- Simple tests: Junior Test Engineer

### Documentation

- Documentation strategy: Documentation Specialist
- API docs and examples: Documentation Engineer
- Formatting and templates: Documentation Engineer

### Performance

- Performance strategy: Performance Specialist
- Optimization implementation: Performance Engineer, Senior Implementation Engineer
- Benchmarking: Performance Engineer

### Security

- Security architecture: Security Design Agent
- Security implementation: Security Specialist
- Security code review: Security Specialist, Senior Implementation Engineer

#### By Scope

| Scope | Levels | Example Agents |
|-------|--------|----------------|
| Entire repository | 0 | Chief Architect |
| Major section | 1 | Section Orchestrators |
| Module | 2 | Module Design Agents |
| Component | 3 | Component Specialists |
| Function/class | 4 | Implementation Engineers |
| Lines/boilerplate | 5 | Junior Engineers |

### Invocation Patterns

#### Automatic Invocation

Just describe your task naturally:

```text
"I need to optimize the convolution performance" → Performance Specialist
"Write tests for the optimizer" → Test Engineer
"Design the model architecture" → Architecture Design Agent
```text

#### Explicit Invocation

Name the agent specifically:

```text
"Use the senior implementation engineer to write SIMD-optimized code"
"Have the test specialist create a comprehensive test plan"
"Ask the chief architect about technology choices"
```text

### Common Workflows

#### Workflow 1: New Feature

```text
1. Section Orchestrator → coordinates
2. Architecture Design Agent → designs
3. Implementation Specialist → plans
4. Test Engineer + Implementation Engineer + Documentation Writer → execute (parallel)
5. All → cleanup
```text

#### Workflow 2: Bug Fix

```text
1. Test Engineer → writes failing test
2. Implementation Engineer → fixes bug
3. Test Engineer → verifies fix
4. Documentation Writer → updates docs if needed
```text

#### Workflow 3: Performance Optimization

```text
1. Performance Specialist → creates optimization plan
2. Performance Engineer → profiles code
3. Senior Implementation Engineer → implements optimizations
4. Performance Engineer → validates improvements
```text

#### Workflow 4: New Paper Implementation

```text
1. Chief Architect → evaluates and approves
2. Papers Orchestrator → breaks down paper
3. Architecture Design Agent → designs components
4. Component Specialists → create detailed plans
5. Engineers (Levels 4-5) → implement in parallel
6. All → integrate and cleanup
```text

### Configuration Reference

All agent configurations are in:

```bash
.claude/agents/[agent-name].md
```text

Templates are in:

```bash
agents/templates/level-[0-5]-*.md
```text

To view an agent's full configuration:

```bash
cat .claude/agents/architecture-design.md
```text

### Next Steps

- **Quick start**: [quick-start.md](quick-start.md)
- **Complete onboarding**: [onboarding.md](onboarding.md)
- **Troubleshooting**: [troubleshooting.md](troubleshooting.md)
- **Visual hierarchy**: [../hierarchy.md](../hierarchy.md)

---

**Total Agent Types**: 43 across 6 levels (1 L0 + 6 L1 + 4 L2 + 24 L3 + 6 L4 + 2 L5)

**Remember**: Trust the hierarchy, communicate clearly, and let agents work at their appropriate level!
