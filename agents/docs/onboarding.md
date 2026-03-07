# Complete Onboarding Guide - Agent System

<!-- markdownlint-disable MD051 -->

## Table of Contents

- [Introduction](#introduction)
- [System Overview](#system-overview)
- [6-Level Hierarchy Explained](#6-level-hierarchy-explained)
- [Delegation Patterns Walkthrough](#delegation-patterns-walkthrough)
- [Mojo-Specific Capabilities](#mojo-specific-agent-capabilities)
- [Best Practices](#best-practices)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [Step-by-Step Tutorial](#step-by-step-tutorial)
- [Advanced Topics](#advanced-topics)

## Introduction

Welcome to the ML Odyssey multi-level agent system! This guide provides a comprehensive introduction to
understanding and effectively using our hierarchical agent architecture.

### What You'll Learn

- How the 6-level hierarchy works
- When to use each type of agent
- How agents coordinate and delegate
- Mojo-specific capabilities and patterns
- Best practices for effective collaboration
- Common pitfalls and how to avoid them

### Prerequisites

- Basic familiarity with AI research and implementation
- Understanding of Mojo programming language (basics)
- Familiarity with git workflows
- Knowledge of Claude Code (helpful but not required)

### Time Commitment

- **Quick introduction**: 15 minutes (read sections 1-3)
- **Complete walkthrough**: 45 minutes (full document)
- **Hands-on tutorial**: 30 minutes (section 8)

## System Overview

### The Big Picture

The ML Odyssey agent system is a **hierarchical team of AI specialists** designed to implement AI research
papers in Mojo. Think of it as a software engineering organization:

```text
Level 0: CTO/VP Engineering        → Chief Architect
Level 1: Engineering Managers      → Section Orchestrators
Level 2: Principal Engineers       → Module Design Agents
Level 3: Senior Engineers          → Component Specialists
Level 4: Engineers                 → Implementation Engineers
Level 5: Junior Engineers/Interns  → Junior Engineers
```text

### Core Concepts

#### 1. Hierarchical Organization

- **Top-down delegation**: Strategic tasks flow down, becoming more detailed
- **Bottom-up reporting**: Implementation status flows up, becoming more aggregated
- **Horizontal coordination**: Same-level agents collaborate directly

#### 2. Scope Reduction

Each level operates at a different scope:

```text
Level 0: Entire repository (system-wide)
Level 1: Major sections (6 sections)
Level 2: Modules within sections
Level 3: Components within modules
Level 4: Functions and classes
Level 5: Lines and boilerplate
```text

#### 3. Specialization

Agents specialize in different domains:

- **Orchestration**: Coordinating multiple modules
- **Architecture**: Designing system structure
- **Security**: Threat modeling and secure implementation
- **Performance**: Optimization and benchmarking
- **Testing**: Test design and implementation
- **Documentation**: API docs and tutorials

#### 4. Workflow Integration

The system follows a **5-phase workflow**:

1. **Plan** (Sequential): Levels 0-2 create specifications
1. **Test** (Parallel): Levels 3-5 write tests
1. **Implementation** (Parallel): Levels 3-5 write code
1. **Packaging** (Parallel): Levels 3-5 integrate and document
1. **Cleanup** (Sequential): All levels review and refactor

#### 5. Git Worktree Strategy

Each phase gets its own git worktree for parallel work:

```bash
worktrees/issue-62-plan-agents/          # Plan phase
worktrees/issue-63-test-agents/          # Test phase (parallel)
worktrees/issue-64-impl-agents/          # Implementation (parallel)
worktrees/issue-65-pkg-agents/           # Packaging (parallel)
worktrees/issue-66-cleanup-agents/       # Cleanup phase
```text

## 6-Level Hierarchy Explained

### Level 0: Meta-Orchestrator

**Role**: Strategic leadership across the entire repository

**Agent**: Chief Architect (1 agent)

### Responsibilities

- Select which AI research papers to implement
- Define repository-wide architectural patterns
- Establish coding standards and conventions
- Coordinate across all 6 major sections
- Resolve conflicts between section orchestrators
- Make technology stack decisions (Mojo vs Python for different components)
- Monitor overall project health

### When to Invoke

```text
"Which paper should we implement next - ResNet or Transformer?"
"What should our repository-wide testing strategy be?"
"Should we use Mojo or Python for data preprocessing?"
```text

### Example Interaction

```text
User: "We're ready to implement our second paper. Should we do ResNet-50 or VGG-16?"

Chief Architect:
"Based on our current codebase capabilities, I recommend ResNet-50 for these reasons:
1. We already have convolution and pooling layers from LeNet-5
2. ResNet's skip connections will teach us about complex architectures
3. It's more relevant to modern deep learning than VGG
4. The residual patterns will be reusable for future papers

I'll delegate to the Papers Orchestrator to create the implementation plan."
```text

### Level 1: Section Orchestrators

**Role**: Manage major repository sections

**Agents**: 6 orchestrators (one per section)

1. **Foundation Orchestrator**: Directory structure, configuration
1. **Shared Library Orchestrator**: Core reusable components
1. **Tooling Orchestrator**: CLI tools, automation scripts
1. **Papers Orchestrator**: Research paper implementations
1. **CI/CD Orchestrator**: Testing and deployment pipelines
1. **Agentic Workflows Orchestrator**: Claude-powered automation

### Responsibilities

- Break section into modules
- Coordinate module dependencies
- Ensure section quality and consistency
- Manage section-level resources
- Report progress to Chief Architect

### When to Invoke

```text
"Plan the shared library architecture for tensor operations"
"Set up the CI/CD pipeline for our Mojo code"
"Coordinate the LeNet-5 implementation across all modules"
```text

### Example Interaction

```text
User: "We need to add batch normalization to our shared library"

Shared Library Orchestrator:
"I'll coordinate this addition. Batch normalization will be a new module in our
training utilities. Here's my plan:

1. Architecture Design Agent: Design the BatchNorm interface and implementation
2. Integration Design Agent: Ensure compatibility with existing layers
3. Security Design Agent: Review for numerical stability issues

After planning, we'll create issues for Test, Implementation, and Packaging phases.
I'll ensure this integrates smoothly with existing Conv2D and Dense layers."
```text

### Level 2: Module Design Agents

**Role**: Design module architecture and interfaces

**Agents**: 3 core types

1. **Architecture Design Agent**: Component breakdown, interface design, data flow
1. **Integration Design Agent**: Cross-component APIs, integration testing
1. **Security Design Agent**: Threat modeling, security requirements

### Responsibilities

- Break module into components
- Define component interfaces and contracts
- Design data flow within module
- Identify reusable patterns
- Create detailed component specifications

### When to Invoke

```text
"Design the architecture for the optimizer module"
"How should the data loader integrate with the training loop?"
"What security considerations exist for model serialization?"
```text

### Example Interaction

```text
User: "Design the architecture for SGD optimizer"

Architecture Design Agent:
"I'll design the SGD optimizer architecture with Mojo performance in mind:

Components:
1. OptimizerBase trait - defines interface for all optimizers
2. SGD struct - implements basic SGD with momentum
3. ParameterUpdate - handles parameter updates with SIMD

Interface design:
```mojo

trait Optimizer:
    fn step(inout self, parameters: List[Tensor], gradients: List[Tensor])
    fn zero_grad(inout self, parameters: List[Tensor])
    fn get_learning_rate(self) -> Float64

struct SGD(Optimizer):
    var learning_rate: Float64
    var momentum: Float64
    var velocities: List[Tensor]

```text
This design allows easy extension for Adam, RMSprop, etc. I'll delegate to the
Implementation Specialist for detailed function specifications."

### Level 3: Component Specialists

**Role**: Coordinate component-level implementation

**Agents**: 5 specialist types

1. **Implementation Specialist**: Coordinate code implementation
2. **Test Specialist**: Plan and coordinate testing
3. **Documentation Specialist**: Coordinate documentation
4. **Performance Specialist**: Optimize performance
5. **Security Specialist**: Implement security requirements

**Responsibilities**:

- Break components into functions/classes
- Create detailed implementation plans
- Coordinate parallel execution (Test/Impl/Package)
- Review work quality
- Report progress to design agents

**When to Invoke**:

```text

"Create the implementation plan for the SGD optimizer"
"Design comprehensive tests for the convolution layer"
"Plan documentation for the tensor operations API"

```text
**Example Interaction**:

```text

User: "Create comprehensive tests for the SGD optimizer"

Test Specialist:
"I'll create a complete test plan for SGD:

Test Categories:

1. Unit Tests
   - Learning rate application
   - Momentum calculation
   - Parameter updates
   - Zero gradient functionality

1. Integration Tests
   - SGD with Conv2D layers
   - SGD with Dense layers
   - Multi-parameter optimization

1. Property Tests
   - Convergence on simple functions
   - Gradient descent direction
   - Learning rate scaling

1. Performance Tests
   - SIMD vectorization effectiveness
   - Memory allocation patterns

I'll delegate to Test Engineers to implement these tests in parallel with the
implementation work. We'll use TDD approach - tests first, then implementation."

```text
### Level 4: Implementation Engineers

**Role**: Write production-quality code, tests, and documentation

**Agents**: 5 engineer types

1. **Senior Implementation Engineer**: Complex code, SIMD optimization
2. **Implementation Engineer**: Standard functions and classes
3. **Test Engineer**: Unit and integration tests
4. **Documentation Writer**: Docstrings, examples, README
5. **Performance Engineer**: Benchmarks, profiling, optimization

**Responsibilities**:

- Write implementation code following specifications
- Implement tests using TDD approach
- Write documentation and examples
- Optimize performance-critical code
- Review and refactor code

**When to Invoke**:

```text

"Implement the forward pass for the Conv2D layer with SIMD"
"Write unit tests for the SGD parameter update logic"
"Create API documentation with examples for tensor operations"

```text
**Example Interaction**:

```text

User: "Implement the SGD step function with SIMD optimization"

Senior Implementation Engineer:
"I'll implement the SGD step function with Mojo SIMD for optimal performance:

```mojo
fn step(inout self, parameters: List[Tensor], gradients: List[Tensor]):
    """Apply SGD update to parameters.

    Updates parameters using: param = param - lr * (gradient + momentum * velocity)
    Uses SIMD vectorization for performance.
    """
    for i in range(len(parameters)):
        let param = parameters[i]
        let grad = gradients[i]
        let velocity = self.velocities[i]

        # Update velocity with momentum (SIMD optimized)
        @parameter
        fn update_velocity[width: Int](offset: Int):
            let v_vec = velocity.load[width](offset)
            let g_vec = grad.load[width](offset)
            let new_v = self.momentum * v_vec + g_vec
            velocity.store[width](offset, new_v)

        vectorize[update_velocity, simd_width](param.size())

        # Update parameters (SIMD optimized)
        @parameter
        fn update_param[width: Int](offset: Int):
            let p_vec = param.load[width](offset)
            let v_vec = velocity.load[width](offset)
            let new_p = p_vec - self.learning_rate * v_vec
            param.store[width](offset, new_p)

        vectorize[update_param, simd_width](param.size())
```text

This implementation leverages Mojo's SIMD capabilities for performance while
maintaining clarity. I'll coordinate with the Test Engineer to verify correctness."

### Level 5: Junior Engineers

**Role**: Handle simple tasks and boilerplate

**Agents**: 2 junior types

1. **Junior Implementation Engineer**: Simple functions, boilerplate
1. **Junior Test Engineer**: Simple tests, test boilerplate

### Responsibilities

- Write simple, well-specified functions
- Generate boilerplate code from templates
- Format code according to standards
- Run automated tools (linters, formatters)
- Execute simple, repetitive tasks

### When to Invoke

```text
"Create getter and setter functions for this struct"
"Generate test boilerplate for these functions"
"Format all docstrings in this module"
```text

### Example Interaction

```text
User: "Create getter and setter methods for the SGD hyperparameters"

Junior Implementation Engineer:
"I'll create the boilerplate getter and setter methods:

```mojo

fn get_learning_rate(self) -> Float64:
    """Get the learning rate.

    Returns:
        Current learning rate value.
    """
    return self.learning_rate

fn set_learning_rate(inout self, lr: Float64):
    """Set the learning rate.

    Args:
        lr: New learning rate value (must be positive).
    """
    debug_assert(lr > 0, "Learning rate must be positive")
    self.learning_rate = lr

fn get_momentum(self) -> Float64:
    """Get the momentum coefficient.

    Returns:
        Current momentum value.
    """
    return self.momentum

fn set_momentum(inout self, m: Float64):
    """Set the momentum coefficient.

    Args:
        m: New momentum value (must be in range [0, 1]).
    """
    debug_assert(0 <= m <= 1, "Momentum must be in [0, 1]")
    self.momentum = m

```text
All methods follow Mojo conventions and include proper docstrings."

## Delegation Patterns Walkthrough

### Pattern 1: Decomposition Delegation

**Purpose**: Break large tasks into manageable pieces

**Flow**: Top → Down, each level adds more detail

### Example: Implementing a New Layer Type

```text

Chief Architect (Level 0):
"We need to add dropout layers to our neural network library"
    ↓ Delegates to Section Orchestrator

Shared Library Orchestrator (Level 1):
"Dropout will be a new training utility. Create architecture design."
    ↓ Delegates to Module Design Agent

Architecture Design Agent (Level 2):
"Dropout struct with forward/backward passes. Here's the interface:"
    ↓ Delegates to Component Specialist

Implementation Specialist (Level 3):
"Break into 3 functions: forward, backward, and inference mode"
    ↓ Delegates to Engineers

Implementation Engineer (Level 4):
"Implementing forward pass with SIMD random number generation"
    ↓ Delegates boilerplate

Junior Engineer (Level 5):
"Creating struct boilerplate and accessor methods"

```text
**Key Insight**: Each level zooms in, transforming "add dropout" into specific code.

### Pattern 2: Specialization Delegation

**Purpose**: Route tasks to domain experts

**Flow**: Orchestrator → Appropriate Specialist

#### Example: Adding Authentication

```text

User Request: "Add API authentication to the model server"

Foundation Orchestrator:
"This requires security expertise"
    ↓ Delegates to Security Design Agent

Security Design Agent:
"I'll design the authentication mechanism:

 - JWT tokens for stateless auth
 - Role-based access control
 - Rate limiting for API endpoints"
    ↓ Delegates to Security Specialist

Security Implementation Specialist:
"Implementing with these security requirements:

 - bcrypt for password hashing
 - Secure random token generation
 - Token expiration and refresh"
    ↓ Delegates to Implementation Engineer

```text
**Key Insight**: Domain experts handle their specialty, not generalists.

### Pattern 3: Parallel Delegation

**Purpose**: Execute independent tasks simultaneously

**Flow**: Specialist → Multiple Engineers (parallel)

#### Example: Implementing a Component

```text

Implementation Specialist (after planning):
"The Conv2D layer is ready for parallel implementation"

    ├──> Test Engineer (parallel)
    │    "Write unit tests for forward and backward pass"
    │
    ├──> Senior Implementation Engineer (parallel)
    │    "Implement SIMD-optimized convolution operations"
    │
    └──> Documentation Writer (parallel)
         "Create API docs and usage examples"

All three work independently in separate git worktrees:

- worktrees/issue-63-test-conv2d/
- worktrees/issue-64-impl-conv2d/
- worktrees/issue-65-docs-conv2d/

```text
**Key Insight**: Independent work happens simultaneously for faster delivery.

### Pattern 4: TDD Coordination

**Purpose**: Test-Driven Development collaboration

**Flow**: Test Engineer ↔ Implementation Engineer

#### Example: Building a Function

```text

Test Engineer:
   "I'll write failing tests first"

```text
```mojo

fn test_sgd_updates_parameters():
    let optimizer = SGD(learning_rate=0.01)
    # Test implementation here
    assert_equal(...)  # Currently fails

```text
```text

Implementation Engineer:
   "I see the test. Implementing to make it pass..."

```text
```mojo

fn step(inout self, ...):
    # Implementation that satisfies the test

```text
```text

Test Engineer:
   "Tests passing! Adding edge case tests..."

Implementation Engineer:
   "Edge cases failing. Fixing..."

Both:
   "All tests passing. Feature complete!"

```text
**Key Insight**: Tests and implementation evolve together through coordination.

## Mojo-Specific Agent Capabilities

### Language Expertise by Level

#### Levels 0-2: Strategic Mojo Decisions

**Chief Architect & Orchestrators**:

- Decide which components use Mojo vs Python
  - Mojo: Performance-critical ML operations (training, inference)
  - Python: Data loading, visualization, scripting
- Understand Mojo compilation model and performance characteristics
- Design interop patterns between Mojo and Python
- Make decisions about MAX platform integration

**Example Decision**:

```text

Chief Architect:
"For LeNet-5 implementation:

 - Use Mojo for: model layers, forward/backward pass, training loop
 - Use Python for: data loading (MNIST), visualization, hyperparameter tuning
 - Interop via: Mojo functions called from Python with ctypes/FFI"

```text
#### Levels 2-3: Mojo Architecture & Design

**Module Design Agents & Specialists**:

- Design Mojo struct hierarchies and traits
- Leverage Mojo features:
  - `fn` vs `def` for performance vs flexibility
  - `struct` vs `class` for value vs reference semantics
  - SIMD types and operations
  - `@parameter` for compile-time computation
  - Lifetimes and memory ownership
- Design for zero-cost abstractions
- Plan memory layout for cache efficiency

**Example Design**:

```text

Architecture Design Agent:
"The Tensor struct will use:

 - @value decorator for automatic copy/move
 - DType parameter for generic numeric types
 - SIMD-aligned memory layout
 - Traits for operator overloading (+, -, *, /)

trait NumericTensor:
    fn __add__(self, other: Self) -> Self
    fn __mul__(self, scalar: Float64) -> Self

@value
struct Tensor[dtype: DType](NumericTensor):
    var data: DTypePointer[dtype]
    var shape: List[Int]
    # ... methods with SIMD optimization

```text
#### Levels 4-5: Mojo Implementation

**Implementation Engineers**:

- Write production-quality Mojo code
- Implement SIMD vectorization:
  - Use `vectorize[]` for loops
  - Leverage SIMD types (SIMD[Float64, width])
  - Optimize memory access patterns
- Use Mojo standard library effectively
- Handle ownership and lifetimes correctly
- Write Mojo unit tests

**Example Implementation**:

```text

Senior Implementation Engineer:
"SIMD-optimized element-wise addition:

@always_inline
fn add[dtype: DType](
    a: DTypePointer[dtype],
    b: DTypePointer[dtype],
    result: DTypePointer[dtype],
    size: Int
):
    @parameter
    fn vectorized_add[width: Int](offset: Int):
        let a_vec = a.load[width=width](offset)
        let b_vec = b.load[width=width](offset)
        result.store[width=width](offset, a_vec + b_vec)

    vectorize[vectorized_add, simdwidthof[dtype]()](size)
"

```text
### Mojo-Specific Delegation Patterns

#### Pattern: SIMD Optimization

```text

Performance Specialist (Level 3):
"Profile shows convolution is bottleneck. Needs SIMD optimization."
    ↓ Delegates

Senior Implementation Engineer (Level 4):
"I'll vectorize the inner loops using Mojo's SIMD primitives"
    ↓ Uses @parameter and vectorize[]

Performance Engineer (Level 4):
"Benchmarking shows 4x speedup from SIMD. Profiling to verify."

```text
#### Pattern: Type-Safe Generics

```text

Architecture Design Agent (Level 2):
"Tensor should work with Float32, Float64, Int32, etc. Use parametric types."
    ↓ Designs interface

Implementation Specialist (Level 3):
"Use DType parameter, implement for all numeric types"
    ↓ Specifies implementation

Implementation Engineer (Level 4):
"Implementing with DType parameter for compile-time type safety:
struct Tensor[dtype: DType]:
    # Type-safe implementation
"

```text
#### Pattern: Memory Management

```text

Security Design Agent (Level 2):
"Ensure no memory leaks or buffer overflows in tensor operations"
    ↓ Specifies safety requirements

Implementation Specialist (Level 3):
"Use Mojo's ownership system. Ensure all DTypePointers properly freed."
    ↓ Creates implementation plan

Senior Implementation Engineer (Level 4):
"Using __moveinit__ and __del__ for RAII pattern. Memory safety guaranteed."

```text
### Mojo Best Practices by Agent Type

**Orchestrators (Levels 0-1)**:

- Choose Mojo for compute-intensive code
- Allocate Mojo expertise to critical paths
- Balance Mojo complexity vs developer productivity

**Designers (Level 2)**:

- Design Mojo APIs that are ergonomic and fast
- Use traits for polymorphism without virtual dispatch
- Prefer structs over classes for value semantics

**Specialists (Level 3)**:

- Specify SIMD width as parameter
- Plan memory layout for vectorization
- Design benchmarks for Mojo performance

**Engineers (Levels 4-5)**:

- Use `fn` for hot paths (strict type checking)
- Use `def` for prototyping and flexibility
- Apply `@always_inline` for small hot functions
- Leverage `@parameter` for compile-time computation
- Write SIMD code for loops over arrays

## Best Practices

### 1. Trust the Hierarchy

**Do**: Let agents work at their appropriate level

```text

✓ Ask Chief Architect about paper selection
✓ Let Orchestrators break down the work
✓ Trust Engineers to choose implementation details

```text
**Don't**: Micromanage or skip levels

```text

✗ Tell Junior Engineer which algorithm to use (that's Level 3-4)
✗ Skip from Level 0 to Level 5 directly
✗ Second-guess every delegation

```text
### 2. Communicate Clearly

**Do**: Provide context and requirements

```text

✓ "Implement SGD optimizer with momentum support, performance is critical"
✓ "Write tests that cover edge cases, especially numerical stability"
✓ "Document this API for external users, include examples"

```text
**Don't**: Be vague or assume context

```text

✗ "Make it faster"
✗ "Fix the tests"
✗ "Update the docs"

```text
### 3. Escalate Appropriately

**Do**: Escalate blockers to immediate superior

```text

✓ "I need the database schema before I can implement this function" (Engineer → Specialist)
✓ "These two modules have conflicting APIs" (Specialist → Design Agent)
✓ "This section needs more resources" (Orchestrator → Chief Architect)

```text
**Don't**: Stay blocked or escalate prematurely

```text

✗ Stay stuck for days without reporting
✗ Escalate every small decision
✗ Skip levels when escalating

```text
### 4. Coordinate Horizontally

**Do**: Communicate with same-level peers

```text

✓ Test Engineer ↔ Implementation Engineer (TDD coordination)
✓ Architecture Agent ↔ Integration Agent (API negotiation)
✓ Performance Engineer ↔ Implementation Engineer (optimization)

```text
**Don't**: Work in silos

```text

✗ Implement without checking tests
✗ Change interfaces without coordinating
✗ Optimize without benchmarking first

```text
### 5. Document Decisions

**Do**: Capture rationale for future reference

```text

✓ "Chose SGD over Adam because our models are simple and SGD trains faster"
✓ "Using structs instead of classes for zero-copy performance"
✓ "Implemented custom SIMD loop because standard library lacks this operation"

```text
**Don't**: Make undocumented decisions

```text

✗ Change architecture without explanation
✗ Pick approaches randomly
✗ Forget why decisions were made

```text
### 6. Use Git Worktrees Effectively

**Do**: One worktree per issue, clear ownership

```text

✓ issue-63-test-conv2d → Test Engineer
✓ issue-64-impl-conv2d → Implementation Engineer
✓ issue-65-docs-conv2d → Documentation Writer

```text
**Don't**: Mix concerns or share worktrees

```text

✗ Multiple agents editing same worktree simultaneously
✗ Mixing test and implementation in same branch
✗ Unclear ownership of worktrees

```text
### 7. Follow TDD Workflow

**Do**: Tests before implementation

```text

✓ Test Engineer writes failing test
✓ Implementation Engineer makes it pass
✓ Both refactor together
✓ Repeat for next feature

```text
**Don't**: Tests as afterthought

```text

✗ Write all code first, tests later
✗ Skip tests for "simple" code
✗ Ignore failing tests

```text
## Anti-Patterns to Avoid

### 1. Skipping Levels

**Wrong**:

```text

User → Junior Engineer (directly)
"Implement the entire ResNet model"

```text
**Why it's wrong**: Junior Engineer lacks context and authority for such decisions

**Right**:

```text

User → Chief Architect
Chief Architect → Papers Orchestrator
Papers Orchestrator → Architecture Design Agent
Architecture Design Agent → Implementation Specialist
Implementation Specialist → Senior Engineer → Junior Engineer

```text
### 2. Micro-Managing

**Wrong**:

```text

Orchestrator: "Use exactly this variable name: 'num_iterations_for_training_loop'"

```text
**Why it's wrong**: Engineers should choose implementation details

**Right**:

```text

Orchestrator: "Implement a training loop that runs for a configurable number of iterations"
Engineer: Chooses variable names, loop structure, etc.

```text
### 3. Working in Silos

**Wrong**:

```text

Test Engineer writes tests (doesn't share with Implementation Engineer)
Implementation Engineer writes code (doesn't know tests exist)
Integration fails due to mismatched expectations

```text
**Why it's wrong**: Wastes time, creates conflicts

**Right**:

```text

Test Engineer: "I'm writing tests that expect this interface: train(model, data, epochs)"
Implementation Engineer: "Perfect, I'll implement that interface. Let me know if you need changes."
Both: Work together, coordinate in real-time

```text
### 4. Hoarding Information

**Wrong**:

```text

Agent completes work, doesn't document decisions or report status
Superior doesn't know work is done or why decisions were made

```text
**Why it's wrong**: Blocks downstream work, loses institutional knowledge

**Right**:

```text

Agent completes work, writes status report:

- What was done
- Why decisions were made
- What artifacts were created
- What's next

```text
### 5. Premature Optimization

**Wrong**:

```text

Junior Engineer: "I rewrote the entire training loop to be 2% faster using obscure SIMD tricks"

```text
**Why it's wrong**: Outside scope, lacks profiling data, may introduce bugs

**Right**:

```text

Junior Engineer: "Profiling shows training loop is slow. Escalating to Performance Specialist."
Performance Specialist: Analyzes, creates optimization plan
Senior Engineer: Implements optimizations with benchmarks

```text
### 6. Ignoring the 5-Phase Workflow

**Wrong**:

```text

Start implementing before planning phase completes
Mix test, implementation, and packaging in one chaotic sprint
Skip cleanup phase

```text
**Why it's wrong**: Creates confusion, conflicts, technical debt

**Right**:

```text

Phase 1 (Plan): All specs created, reviewed, approved
Phase 2-4 (Test/Impl/Pkg): Parallel execution in separate worktrees
Phase 5 (Cleanup): Review, refactor, finalize

```text
## Step-by-Step Tutorial

### Tutorial: Implementing a Simple Mojo Function

Let's walk through implementing a simple `relu` activation function from start to finish.

#### Step 1: Initiate with High-Level Request

**You say**:

```text

"I need a ReLU activation function for our neural network library, with tests and documentation"

```text
**What happens**: Chief Architect or Shared Library Orchestrator is invoked

#### Step 2: Orchestrator Analyzes and Delegates

**Shared Library Orchestrator**:

```text

"ReLU is a simple activation function. I'll coordinate:

1. Architecture Design Agent: Design the interface
1. After planning, we'll do Test/Impl/Docs in parallel"

```text
#### Step 3: Architecture Design

**Architecture Design Agent**:

```text

"ReLU interface design:

```mojo
fn relu[dtype: DType](
    input: DTypePointer[dtype],
    output: DTypePointer[dtype],
    size: Int
):
    \"\"\"Apply ReLU activation: max(0, x)\"\"\"
    # Implementation will be SIMD-optimized
```text

Specs complete. Delegating to specialists for parallel execution."

#### Step 4: Parallel Execution

Three agents work simultaneously in different worktrees:

**Test Engineer** (in `worktrees/issue-63-test-relu/`):

```text
"Writing tests first (TDD):

```mojo

fn test_relu_basic():
    let input = DTypePointer[DType.float64].alloc(4)
    let output = DTypePointer[DType.float64].alloc(4)

    # Test data: [-2, -1, 0, 1]
    input.store(0, -2.0)
    input.store(1, -1.0)
    input.store(2, 0.0)
    input.store(3, 1.0)

    relu(input, output, 4)

    # Expected: [0, 0, 0, 1]
    assert_equal(output.load(0), 0.0)
    assert_equal(output.load(1), 0.0)
    assert_equal(output.load(2), 0.0)
    assert_equal(output.load(3), 1.0)

fn test_relu_simd_performance():
    # Benchmark test for SIMD effectiveness
    let size = 1000000
    # ... benchmark code

```text
Tests written. Committing to branch 63-test-relu."

**Senior Implementation Engineer** (in `worktrees/issue-64-impl-relu/`):

```text

"Implementing with SIMD optimization:

```mojo
@always_inline
fn relu[dtype: DType](
    input: DTypePointer[dtype],
    output: DTypePointer[dtype],
    size: Int
):
    \"\"\"Apply ReLU activation: output = max(0, input).

    Uses SIMD vectorization for performance.

    Args:
        input: Input tensor data
        output: Output tensor data (pre-allocated)
        size: Number of elements
    \"\"\"
    @parameter
    fn vectorized_relu[width: Int](offset: Int):
        let in_vec = input.load[width=width](offset)
        let zero = SIMD[dtype, width](0)
        let out_vec = max(in_vec, zero)
        output.store[width=width](offset, out_vec)

    vectorize[vectorized_relu, simdwidthof[dtype]()](size)
```text

Implementation complete. This will vectorize across SIMD lanes for performance."

**Documentation Writer** (in `worktrees/issue-65-docs-relu/`):

```text
"Creating API documentation:

# ReLU Activation Function

## Overview

The ReLU (Rectified Linear Unit) activation function is a fundamental building
block of neural networks.

## API Reference

### `relu[dtype: DType]`

Apply ReLU activation function element-wise.

Parameters:
- `dtype`: The data type (Float32, Float64, etc.)

Arguments:
- `input: DTypePointer[dtype]` - Input data pointer
- `output: DTypePointer[dtype]` - Output data pointer (pre-allocated)
- `size: Int` - Number of elements

Returns: None (output written to `output` pointer)

## Example Usage

```mojo

from neural_net.activations import relu

# Create input tensor

let input = DTypePointer[DType.float64].alloc(4)
let output = DTypePointer[DType.float64].alloc(4)

# Fill input

input.store(0, -2.0)
input.store(1, 1.0)

# Apply ReLU

relu(input, output, 2)

# output now contains [0.0, 1.0]

```text
## Performance Notes

This implementation uses SIMD vectorization for optimal performance. On modern
CPUs, it can process 4-8 elements per cycle.

## See Also

- `sigmoid` - Sigmoid activation
- `tanh` - Hyperbolic tangent activation
"

Documentation complete."

### Step 5: Integration in Packaging Phase

**Implementation Specialist**:

```text

"All parallel work complete. Integrating in packaging worktree:

1. Merging test branch → ✓ Tests present
1. Merging implementation branch → ✓ Code present
1. Merging docs branch → ✓ Documentation present

Running full test suite... All tests passing!

ReLU function is complete and ready for use."

```text
#### Step 6: Cleanup Phase

**All Agents Review**:

```text

Test Engineer: "Tests comprehensive, coverage 100%"
Implementation Engineer: "Code follows Mojo best practices, SIMD optimized"
Documentation Writer: "Documentation clear and complete"
Implementation Specialist: "No issues found. Function ready for merge."

```text
#### Step 7: Completion

**Shared Library Orchestrator**:

```text

"ReLU activation function complete:
✓ SIMD-optimized implementation
✓ Comprehensive tests (100% coverage)
✓ Complete documentation with examples

Ready to merge to main branch. Function can now be used in neural network models."

```text
### What You Learned

1. **Automatic delegation**: Request → Orchestrator → Specialists → Engineers
2. **Parallel execution**: Test/Impl/Docs happened simultaneously
3. **TDD approach**: Tests written before/alongside implementation
4. **Mojo optimization**: SIMD vectorization for performance
5. **Git worktrees**: Clean separation of parallel work
6. **Integration**: All artifacts merged in packaging phase
7. **Quality assurance**: Cleanup phase verified everything

## Advanced Topics

### Custom Agent Creation

Want to create a specialized agent? Follow these steps:

1. **Identify the level** (0-5) based on scope
2. **Choose a template** from `agents/templates/`
3. **Customize for your need** (name, description, responsibilities)
4. **Place in `.claude/agents/`**
5. **Test with explicit invocation**

**Example**: Creating a "CUDA Interop Specialist" for GPU operations

```yaml

---
name: cuda-interop-specialist
description: Design and implement CUDA/GPU interop for Mojo neural network operations
tools: Read,Write,Edit,Bash,Grep,Glob
model: sonnet
---

# CUDA Interop Specialist

## Role

Level 3 Component Specialist for GPU acceleration

## Responsibilities

- Design CUDA kernel interfaces for Mojo
- Plan GPU memory management
- Coordinate CPU-GPU data transfers
- Optimize kernel performance

## Delegation

Delegates to: Implementation Engineers
Coordinates with: Performance Specialist, Architecture Design Agent

...

```text
### Multi-Paper Coordination

When implementing multiple papers simultaneously:

1. **Chief Architect assigns papers** to different orchestrators
2. **Shared Library Orchestrator** coordinates common components
3. **Paper Orchestrators** work in parallel
4. **Integration happens in Packaging phase**

**Example**:

```text

Chief Architect:
"We'll implement LeNet-5 and ResNet-50 in parallel:

- LeNet-5 → Papers Orchestrator #1
- ResNet-50 → Papers Orchestrator #2
- Shared components (Conv, Pool, Dense) → Shared Library Orchestrator
- CI/CD for both → CI/CD Orchestrator"

```text
### Cross-Language Coordination (Mojo + Python)

For Mojo-Python hybrid systems:

1. **Level 0-1 decides split**: Which parts Mojo, which Python
2. **Level 2 designs interfaces**: FFI, ctypes bindings
3. **Level 3-4 implements both sides**
4. **Level 4 creates integration tests**

**Example**:

```text

Architecture Design Agent:
"For data loading:

- Python: Use PyTorch DataLoader (mature, flexible)
- Mojo: Expose FFI interface for receiving batches
- Interface: Python calls Mojo train_step(batch) via ctypes

Integration Design Agent:
"Create Python wrapper:

```python
from ctypes import cdll, POINTER, c_float

mojo_lib = cdll.LoadLibrary('./libmodel.so')
mojo_lib.train_step.argtypes = [POINTER(c_float), c_int]

def train_step(batch):
    return mojo_lib.train_step(batch.ctypes.data, batch.size)
```text

"

### Performance Profiling Workflow

When performance is critical:

1. **Performance Specialist creates profiling plan**
1. **Performance Engineer profiles current implementation**
1. **Senior Implementation Engineer optimizes hot paths**
1. **Performance Engineer validates improvements**

### Tools

- Mojo's built-in profiling
- `perf` for CPU profiling
- SIMD intrinsics analysis
- Memory bandwidth analysis

### Security Review Process

For security-critical components:

1. **Security Design Agent** performs threat modeling
1. **Security Specialist** implements security requirements
1. **Test Engineer** creates security-focused tests
1. **Security Specialist** reviews final implementation

### Focus areas

- Memory safety (buffer overflows)
- Numerical stability (NaN/Inf handling)
- Model security (adversarial robustness)
- Data privacy (no data leaks in logs)

## Next Steps

### Immediate Actions

1. **Try the tutorial**: Implement a simple function end-to-end
1. **Browse agent catalog**: See all 43 agents in [agent-catalog.md](agent-catalog.md)
1. **Study real examples**: Check `.claude/agents/` for actual configurations

### Deepening Understanding

1. **Read orchestration patterns**: [/notes/review/orchestration-patterns.md](../../notes/review/orchestration-patterns.md)
1. **Study worktree strategy**: [/notes/review/worktree-strategy.md](../../notes/review/worktree-strategy.md)
1. **Explore skills system**: [/notes/review/skills-design.md](../../notes/review/skills-design.md)

### Getting Help

1. **Quick questions**: [troubleshooting.md](troubleshooting.md)
1. **Visual reference**: [../hierarchy.md](../hierarchy.md)
1. **Quick start**: [quick-start.md](quick-start.md)

## Summary

You've learned:

- ✓ How the 6-level hierarchy organizes work
- ✓ When to use each type of agent
- ✓ How delegation patterns work
- ✓ Mojo-specific capabilities and best practices
- ✓ How to avoid common pitfalls
- ✓ How to implement a function end-to-end

### Key takeaways

1. Trust the hierarchy - let agents work at their level
1. Communicate clearly - provide context and requirements
1. Use parallel execution - leverage git worktrees
1. Follow TDD - tests and implementation together
1. Leverage Mojo - use SIMD, structs, and zero-cost abstractions

**Ready to build?** Start with a simple request and watch the agents coordinate!
