# 5-Phase Workflow Integration with Agents

<!-- markdownlint-disable MD051 -->

## Table of Contents

- [Overview](#overview)
- [The 5-Phase Workflow](#the-5-phase-workflow)
- [Agent Participation by Phase](#agent-participation-by-phase)
- [Phase 1: Plan (Sequential)](#phase-1-plan-sequential)
- [Phase 2-4: Test/Implementation/Packaging (Parallel)](#phase-2-4-testimplementationpackaging-parallel)
- [Phase 5: Cleanup (Sequential)](#phase-5-cleanup-sequential)
- [Workflow Diagrams](#workflow-diagrams)
- [Examples by Phase](#examples-by-phase)
- [Parallel Execution Patterns](#parallel-execution-patterns)
- [Best Practices](#best-practices)

## Overview

The ProjectOdyssey project uses a comprehensive 5-phase development workflow that integrates seamlessly with the
6-level agent hierarchy. This document explains how agents participate in each phase, coordinate work, and execute
in parallel when appropriate.

### Key Principles

- **Plan First**: Design and specifications must complete before implementation begins
- **Parallel Execution**: Test, Implementation, and Packaging run simultaneously after Plan completes
- **Cleanup Last**: Refactoring and finalization happen after parallel phases finish
- **Clear Handoffs**: Each phase produces artifacts consumed by later phases

## The 5-Phase Workflow

```text
┌──────────────────────┐
│   Phase 1: Plan      │ ← Sequential, Must Complete First
└──────────┬───────────┘
           │
           ├────────────────┬────────────────┐
           │                │                │
           ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Phase 2:     │  │ Phase 3:     │  │ Phase 4:     │ ← Parallel Execution
│ Test         │  │ Implement    │  │ Package      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ Phase 5: Cleanup │ ← Sequential, After Parallel
              └──────────────────┘
```text

### Dependencies

- **Plan → Test/Impl/Package**: Parallel phases require Plan specifications
- **Test/Impl/Package → Cleanup**: Cleanup collects issues from all parallel phases
- **Iterative**: Major issues in Cleanup may require revisiting earlier phases

## Agent Participation by Phase

### By Phase

| Phase | Active Levels | Primary Agents | Focus |
|-------|---------------|----------------|-------|
| **Plan** | 0-3 | Orchestrators, Design Agents, Specialists | Create specifications, design architecture, define requirements |
| **Test** | 3-5 | Test Specialists, Test Engineers | Write test cases, implement tests, create fixtures |
| **Implementation** | 3-5 | Implementation Specialists, Engineers | Build functionality, write code, implement features |
| **Packaging** | 3-5 | Documentation Specialists, Integration Engineers | Create docs, integrate components, package deliverables |
| **Cleanup** | All (0-5) | All agents | Refactor, fix issues, finalize, quality assurance |

### By Agent Level

#### Level 0: Chief Architect

- **Plan**: Strategic architecture decisions, paper selection, system-wide coordination
- **Test/Impl/Package**: Oversight and approval of major changes
- **Cleanup**: Final architectural review and validation

#### Level 1: Section Orchestrators

- **Plan**: Section-level planning, module breakdown, resource allocation
- **Test/Impl/Package**: Coordinate parallel work within section
- **Cleanup**: Aggregate cleanup tasks, prioritize refactoring

#### Level 2: Module Design Agents

- **Plan**: Component design, interface definitions, integration planning
- **Test/Impl/Package**: Review implementations for alignment with design
- **Cleanup**: Validate architectural integrity, approve refactoring

#### Level 3: Component Specialists

- **Plan**: Detailed specifications for functions/classes, define success criteria
- **Test/Impl/Package**: Direct execution of work (ACTIVE PHASE)
- **Cleanup**: Identify and fix code smells, refactor implementations

#### Level 4: Implementation Engineers

- **Plan**: Review specifications, ask clarifying questions
- **Test/Impl/Package**: Write code, tests, and documentation (ACTIVE PHASE)
- **Cleanup**: Fix issues, improve code quality

#### Level 5: Junior Engineers

- **Plan**: Limited participation (review templates)
- **Test/Impl/Package**: Simple tasks, boilerplate generation (ACTIVE PHASE)
- **Cleanup**: Format code, update simple documentation

## Phase 1: Plan (Sequential)

### Purpose

Create comprehensive specifications for all subsequent phases. This is the foundation that enables parallel execution.

### Active Agents

```text
Chief Architect (Level 0)
  ↓ Strategic planning
Section Orchestrators (Level 1)
  ↓ Section breakdown
Module Design Agents (Level 2)
  ├─> Architecture Design Agent
  ├─> Integration Design Agent
  └─> Security Design Agent
  ↓ Component specifications
Component Specialists (Level 3)
  ├─> Implementation Specialist
  ├─> Test Specialist
  └─> Documentation Specialist
```text

### Workflow

1. **Chief Architect** analyzes requirements and creates strategic plan
1. **Section Orchestrator** breaks work into modules
1. **Design Agents** create component specifications:
   - Architecture Design: Component structure, interfaces, data flow
   - Integration Design: Cross-component APIs, dependencies
   - Security Design: Threat models, security requirements
1. **Component Specialists** create detailed specifications:
   - Implementation Specialist: Function/class breakdown
   - Test Specialist: Test cases and coverage requirements
   - Documentation Specialist: Documentation outline

### Deliverables

- Strategic architecture documents
- Component specifications
- Interface definitions
- Test plans
- Documentation outlines
- Success criteria for each parallel phase

### Example: LeNet-5 Model Implementation

```text
Chief Architect:
  - Analyzes LeNet-5 paper
  - Delegates to Papers Orchestrator

Papers Orchestrator:
  - Breaks into modules: data_prep, model, training, evaluation
  - Delegates to Architecture Design Agent

Architecture Design Agent:
  - Designs model architecture:
    * Conv2D layer struct
    * MaxPool layer struct
    * Dense layer struct
    * Sequential model orchestrator
  - Creates specifications for each component

Implementation Specialist:
  - Breaks Conv2D into functions:
    * __init__(in_channels, out_channels, kernel_size)
    * forward(input: Tensor) -> Tensor
    * _apply_kernel() with SIMD optimization
  - Creates detailed spec for Implementation Engineers

Test Specialist:
  - Defines test cases:
    * Test Conv2D shape transformation
    * Test gradient computation
    * Test SIMD optimization correctness
  - Creates test plan for Test Engineers
```text

**Completion Criteria**: All specifications reviewed and approved, ready for parallel execution.

## Phase 2-4: Test/Implementation/Packaging (Parallel)

### Purpose

Execute work simultaneously in isolated worktrees, coordinating through specifications created in Plan phase.

### Active Agents

All three phases run in parallel with different agents:

```text
Component Specialist (Level 3) - Coordinates parallel work
  ├─> Test Specialist → Test Engineers (Phase 2)
  ├─> Implementation Specialist → Implementation Engineers (Phase 3)
  └─> Documentation Specialist → Documentation Engineers (Phase 4)
```text

### Phase 2: Test

**Agents**: Test Specialist, Test Engineers (Levels 3-5)

### Work

- Implement unit tests from test plan
- Create integration tests
- Build test fixtures and mock data
- Set up test automation
- Verify coverage requirements

### Deliverables

- Test files (`.mojo`, `.py`)
- Test fixtures
- Test utilities
- Coverage reports

### Mojo Example

```mojo
# tests/test_conv2d.mojo
from testing import assert_equal, assert_raises
from model.layers import Conv2D
from tensor import Tensor

fn test_conv2d_forward():
    # Test: Conv2D transforms input correctly
    var conv = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
    var input = Tensor[DType.float32](1, 1, 28, 28)  # MNIST input
    var output = conv.forward(input)

    # Expected output shape: (1, 6, 24, 24)
    assert_equal(output.shape[0], 1)
    assert_equal(output.shape[1], 6)
    assert_equal(output.shape[2], 24)
    assert_equal(output.shape[3], 24)

fn test_conv2d_simd_optimization():
    # Test: SIMD version produces same results as naive
    var conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    var input = Tensor[DType.float32](1, 3, 32, 32)

    var output_naive = conv.forward_naive(input)
    var output_simd = conv.forward(input)

    # Results should match within floating point tolerance
    assert_tensors_close(output_naive, output_simd, atol=1e-5)
```text

### Phase 3: Implementation

**Agents**: Implementation Specialist, Implementation Engineers (Levels 3-5)

### Work

- Implement functions and classes from specs
- Write performance-critical Mojo code
- Implement error handling
- Add inline documentation
- Coordinate with Test phase for TDD

### Deliverables

- Implementation code (`.mojo`, `.py`)
- Inline documentation
- API implementations
- Performance optimizations

### Mojo Example

```mojo
# src/model/layers/conv2d.mojo
from tensor import Tensor
from algorithm import vectorize
from memory import memset_zero

@value
struct Conv2D:
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int
    ):
        """Initialize Conv2D layer with random weights.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolution kernel (square).
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights: (out_channels, in_channels, kernel_size, kernel_size)
        let weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = Tensor[DType.float32](weight_shape)

        # Initialize bias: (out_channels,)
        self.bias = Tensor[DType.float32](out_channels)

        # TODO: Proper weight initialization (Xavier/He)
        self._init_weights()

    fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply 2D convolution with SIMD optimization.

        Args:
            input: Input tensor of shape (batch, in_channels, height, width).

        Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width).
        """
        # Calculate output dimensions
        let out_height = input.shape[2] - self.kernel_size + 1
        let out_width = input.shape[3] - self.kernel_size + 1

        # Allocate output tensor
        var output = Tensor[DType.float32](
            input.shape[0], self.out_channels, out_height, out_width
        )

        # SIMD-optimized convolution
        self._apply_kernel_simd(input, output)

        return output

    fn _apply_kernel_simd(
        self,
        input: Tensor[DType.float32],
        inout output: Tensor[DType.float32]
    ):
        """SIMD-optimized kernel application."""
        comptime simd_width = simdwidthof[DType.float32]()

        # Vectorized convolution over spatial dimensions
        @parameter
        fn compute_pixel[width: Int](out_y: Int, out_x: Int):
            for out_c in range(self.out_channels):
                var sum = SIMD[DType.float32, width](0)

                # Convolution over kernel and input channels
                for in_c in range(self.in_channels):
                    for ky in range(self.kernel_size):
                        for kx in range(self.kernel_size):
                            let in_y = out_y + ky
                            let in_x = out_x + kx

                            let weight = self.weights.load[width](
                                out_c, in_c, ky, kx
                            )
                            let inp = input.load[width](0, in_c, in_y, in_x)

                            sum += weight * inp

                output.store(0, out_c, out_y, out_x, sum.reduce_add() + self.bias[out_c])

        # Vectorize over output spatial dimensions
        for out_y in range(output.shape[2]):
            vectorize[compute_pixel, simd_width](output.shape[3], out_y)
```text

### Phase 4: Packaging

**Agents**: Documentation Specialist, Documentation Engineers, Integration Engineers (Levels 3-5)

### Work

- Write API documentation
- Create usage examples
- Write installation guides
- Integrate test and implementation artifacts
- Create package configurations

### Deliverables

- README files
- API documentation
- Usage examples
- Integration scripts
- Package configurations

### Example

```markdown
# Conv2D Layer

## Overview

The `Conv2D` layer implements 2D convolution operations optimized with SIMD for performance.

## API

### Constructor

```mojo

Conv2D(in_channels: Int, out_channels: Int, kernel_size: Int)

```text
**Parameters**:

- `in_channels`: Number of input channels
- `out_channels`: Number of output feature maps
- `kernel_size`: Size of the convolution kernel (square)

### Methods

#### forward

```mojo

fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]

```text
Apply 2D convolution to input tensor.

**Parameters**:

- `input`: Input tensor of shape `(batch, in_channels, height, width)`

**Returns**:

- Output tensor of shape `(batch, out_channels, out_height, out_width)`

## Usage Example

```mojo

from model.layers import Conv2D
from tensor import Tensor

# Create layer: 1 input channel, 6 output channels, 5x5 kernel

var conv = Conv2D(in_channels=1, out_channels=6, kernel_size=5)

# Create input: batch=1, channels=1, 28x28 image (MNIST)

var input = Tensor[DType.float32](1, 1, 28, 28)

# Forward pass

var output = conv.forward(input)  # Shape: (1, 6, 24, 24)

```text
## Performance

The implementation uses SIMD vectorization for optimal performance:

- Benchmarked at 2.5ms for MNIST-sized inputs on M1 Mac
- 10x faster than naive Python implementation
- Scales efficiently with batch size

## See Also

- [MaxPool2D](./maxpool2d.md)
- [Dense Layer](./dense.md)
- [Sequential Model](../model.md)

```text

### Coordination Between Phases

### Test ↔ Implementation (TDD)

```text
1. Test Engineer writes failing test
2. Signals Implementation Engineer via status update
3. Implementation Engineer implements feature
4. Test passes, both proceed to next feature
```text

### Implementation ↔ Packaging

```text
1. Implementation Engineer completes function
2. Documentation Engineer reads implementation
3. Extracts API signature and behavior
4. Writes documentation with examples
```text

### Git Worktree Isolation

```bash
# Each phase has its own worktree
worktrees/issue-63-test-agents/      # Test Engineers work here
worktrees/issue-64-impl-agents/      # Implementation Engineers work here
worktrees/issue-65-pkg-agents/       # Documentation Engineers work here
```text

### Cross-Worktree Coordination

- **Option 1**: Cherry-pick commits between worktrees
- **Option 2**: Coordinate through specifications (preferred)
- **Option 3**: Temporary merge for integration testing

## Phase 5: Cleanup (Sequential)

### Purpose

Collect issues discovered during parallel phases, refactor code, and finalize quality.

### Active Agents

**All Levels** (0-5) - Everyone reviews their work

```text
Level 0-3: Identify architectural and design issues
Level 3-5: Fix code quality, refactor, optimize
All: Final quality review
```text

### Workflow

1. **Collect Issues**: Aggregate issues from Test/Impl/Package phases
1. **Prioritize**: Section Orchestrators prioritize cleanup tasks
1. **Execute**: Agents fix issues at appropriate levels
1. **Validate**: Tests still pass, documentation updated
1. **Finalize**: Final sign-off from all levels

### Common Cleanup Tasks

### Level 3-4 (Specialists and Engineers)

- Refactor duplicated code
- Fix code smells (long functions, complex logic)
- Improve error messages
- Optimize performance bottlenecks
- Update documentation based on implementation learnings
- Add missing edge case handling

### Level 2 (Design Agents)

- Validate architectural integrity
- Review interfaces for consistency
- Approve refactoring plans
- Update design documents

### Level 1 (Orchestrators)

- Aggregate cleanup status
- Ensure section-wide consistency
- Coordinate cross-module refactoring

### Level 0 (Chief Architect)

- Final architectural review
- Approve for production
- Document lessons learned

### Example: Cleanup Issues

```markdown
## Cleanup Issues: Conv2D Implementation

### From Test Phase
- [ ] Add edge case test for zero-sized input
- [ ] Test numerical stability with extreme values
- [ ] Add benchmark for performance regression detection

### From Implementation Phase
- [ ] Refactor: _apply_kernel_simd function too long (150 lines)
- [ ] TODO: Implement proper weight initialization (Xavier/He)
- [ ] Performance: Consider tiling for large kernels

### From Packaging Phase
- [ ] Documentation: Add troubleshooting section
- [ ] Example: Add comparison with PyTorch implementation
- [ ] README: Explain SIMD optimization benefits
```text

## Workflow Diagrams

### Complete Workflow

```text
┌─────────────────────────────────────────────────────────────┐
│                     PHASE 1: PLAN                           │
│                                                             │
│  Chief Architect → Section Orchestrator → Design Agents    │
│        ↓                  ↓                     ↓           │
│   Strategic Plan    Module Breakdown    Component Specs    │
│                                                             │
│  Output: Specifications for Test/Impl/Package              │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┬───────────────┐
           │                       │               │
           ▼                       ▼               ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  PHASE 2: TEST   │   │ PHASE 3: IMPL    │   │ PHASE 4: PACKAGE │
│                  │   │                  │   │                  │
│ Test Specialist  │   │ Impl Specialist  │   │  Doc Specialist  │
│       ↓          │   │       ↓          │   │       ↓          │
│ Test Engineers   │   │ Impl Engineers   │   │  Doc Engineers   │
│       ↓          │   │       ↓          │   │       ↓          │
│ - Unit Tests     │   │ - Code           │   │ - Documentation  │
│ - Integration    │   │ - Functions      │   │ - Examples       │
│ - Fixtures       │◄─►│ - Classes        │◄─►│ - Integration    │
│                  │   │ - Optimizations  │   │ - Packaging      │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │   PHASE 5: CLEANUP     │
                   │                        │
                   │  All Agents Review     │
                   │         ↓              │
                   │  - Refactor            │
                   │  - Fix Issues          │
                   │  - Optimize            │
                   │  - Finalize            │
                   │         ↓              │
                   │  Production Ready      │
                   └────────────────────────┘
```text

### Agent Level Participation

```text
PHASE →     Plan        Test        Impl        Package     Cleanup
           ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
Level 0    │ ██  │    │ ░░  │    │ ░░  │    │ ░░  │    │ ██  │
(Chief)    └─────┘    └─────┘    └─────┘    └─────┘    └─────┘
           ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
Level 1    │ ███ │    │ ░░░ │    │ ░░░ │    │ ░░░ │    │ ███ │
(Section)  └─────┘    └─────┘    └─────┘    └─────┘    └─────┘
           ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
Level 2    │ ███ │    │ ░░░ │    │ ░░░ │    │ ░░░ │    │ ███ │
(Design)   └─────┘    └─────┘    └─────┘    └─────┘    └─────┘
           ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
Level 3    │ ███ │    │ ███ │    │ ███ │    │ ███ │    │ ███ │
(Special.) └─────┘    └─────┘    └─────┘    └─────┘    └─────┘
           ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
Level 4    │ ░░░ │    │ ███ │    │ ███ │    │ ███ │    │ ███ │
(Engineer) └─────┘    └─────┘    └─────┘    └─────┘    └─────┘
           ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
Level 5    │ ░░  │    │ ██  │    │ ██  │    │ ██  │    │ ██  │
(Junior)   └─────┘    └─────┘    └─────┘    └─────┘    └─────┘

Legend: ███ = Heavy involvement, ██ = Moderate, ░░ = Light/Oversight
```text

## Examples by Phase

### Example 1: New Feature - MaxPool2D Layer

#### Phase 1: Plan

```text
Papers Orchestrator:
  - Reviews LeNet-5 requirements
  - Identifies need for MaxPool2D layer

Architecture Design Agent:
  - Designs MaxPool2D struct
  - Defines interface: forward(input) -> output
  - Specifies pool_size and stride parameters

Implementation Specialist:
  - Breaks into functions:
    * __init__(pool_size, stride)
    * forward(input) -> output
    * _max_pool_naive(input) -> output
    * _max_pool_simd(input) -> output (optimized)

Test Specialist:
  - Defines test cases:
    * Test output shape calculation
    * Test max value selection
    * Test SIMD correctness vs naive
    * Test gradient computation
```text

#### Phase 2: Test (Parallel with 3 & 4)

```text
Test Engineer:
  - Implements test_maxpool_output_shape()
  - Implements test_max_value_selection()
  - Implements test_simd_correctness()
  - Creates test fixtures with known outputs
```text

#### Phase 3: Implementation (Parallel with 2 & 4)

```text
Implementation Engineer:
  - Implements MaxPool2D struct
  - Implements forward() method
  - Implements _max_pool_simd() with SIMD
  - Coordinates with Test: runs tests, fixes failures
```text

#### Phase 4: Packaging (Parallel with 2 & 3)

```text
Documentation Engineer:
  - Writes MaxPool2D documentation
  - Creates usage examples
  - Adds to model layers guide
  - Updates API reference
```text

#### Phase 5: Cleanup

```text
Issues discovered:
  - [Test] Need stride != pool_size test case
  - [Impl] _max_pool_simd function too complex, refactor
  - [Package] Add performance comparison chart

Implementation Engineer:
  - Refactors _max_pool_simd into smaller functions
  - Adds missing test case
  - Adds performance benchmarks to docs
```text

### Example 2: Bug Fix - Incorrect Gradient Computation

#### Phase 1: Plan

```text
Component Specialist:
  - Analyzes bug report
  - Root cause: incorrect index calculation in backward pass
  - Creates fix specification
  - Plan is minimal: just fix description
```text

#### Phase 2: Test (Parallel with 3 & 4)

```text
Test Engineer:
  - Writes failing test that reproduces bug
  - Test: gradient should match numerical gradient
  - Commits test to test worktree
```text

#### Phase 3: Implementation (Parallel with 2 & 4)

```text
Implementation Engineer:
  - Reads test case
  - Fixes index calculation in backward()
  - Verifies test passes
  - Commits fix to impl worktree
```text

#### Phase 4: Packaging (Parallel with 2 & 4)

```text
Documentation Engineer:
  - Updates CHANGELOG with bug fix
  - No major doc changes needed
```text

#### Phase 5: Cleanup

```text
- Add regression test for related edge cases
- Review other gradient calculations for similar bugs
- Update gradient testing guidelines
```text

## Parallel Execution Patterns

### Pattern 1: TDD (Test-Driven Development)

Test and Implementation phases coordinate closely:

```text
Iteration 1:
  Test Engineer: Write test for feature A → FAIL
  ↓ Status update
  Impl Engineer: Implement feature A → PASS

Iteration 2:
  Test Engineer: Write test for feature B → FAIL
  ↓ Status update
  Impl Engineer: Implement feature B → PASS

... continue until component complete
```text

### Benefits

- Tests guide implementation
- Continuous validation
- High code coverage
- Clear definition of "done"

### Pattern 2: Documentation-Driven Development

Packaging starts early, evolves with implementation:

```text
Early (25% complete):
  Doc Engineer: Draft API documentation from specs

Mid (50% complete):
  Doc Engineer: Update docs based on actual API
  Doc Engineer: Write basic usage examples

Late (75% complete):
  Doc Engineer: Add advanced examples
  Doc Engineer: Write troubleshooting guide

Complete (100%):
  Doc Engineer: Final review and polish
```text

### Benefits

- Documentation stays current
- API design validated early
- Examples guide implementation

### Pattern 3: Cross-Worktree Integration Testing

Periodically integrate work from parallel phases:

```bash
# From packaging worktree, integrate test and impl
cd worktrees/issue-65-pkg-agents

# Merge implementation
git merge --no-ff issue-64-impl-agents

# Merge tests
git merge --no-ff issue-63-test-agents

# Run full integration
mojo test tests/
mojo build src/

# Document any integration issues in cleanup issue
```text

**Frequency**: Weekly or at major milestones

## Best Practices

### Planning Phase

1. **Complete Before Starting Parallel Work**: Don't rush Plan phase
1. **Detailed Specifications**: Parallel phases depend on clear specs
1. **Success Criteria**: Define "done" for each phase
1. **Identify Dependencies**: Document cross-component dependencies
1. **Risk Assessment**: Identify potential issues early

### Parallel Phases

1. **Clear Ownership**: Each worktree has clear owner/team
1. **Status Updates**: Regular communication (daily for active work)
1. **Coordinate Interfaces**: Agree on APIs early
1. **Incremental Progress**: Deliver in small, testable increments
1. **Git Hygiene**: Commit frequently, write clear messages

### TDD Coordination

1. **Test First**: Write failing test before implementation
1. **Red-Green-Refactor**: Test fails → Implement → Test passes → Refactor
1. **Communication**: Test Engineers signal when tests ready
1. **Integration**: Run tests from test worktree against impl worktree

### Cleanup Phase

1. **Continuous Collection**: Add cleanup tasks throughout parallel phases
1. **Prioritize**: High-impact issues first
1. **Validate**: Ensure tests still pass after refactoring
1. **Document**: Update docs to match refactored code
1. **Final Review**: All levels review their respective work

### Worktree Management

1. **One Issue = One Worktree**: Clear isolation
1. **Branch Naming**: Consistent (e.g., `63-test-agents`)
1. **Regular Merges**: Don't let branches diverge too far
1. **Clean Up**: Remove worktrees after PR merges

### Communication

1. **Status Reports**: After major milestones
1. **Blockers**: Escalate immediately
1. **Handoffs**: Clear documentation when passing work
1. **Retrospectives**: Learn from each 5-phase cycle

## Common Pitfalls

### Starting Parallel Work Too Early

**Problem**: Implementation starts before specifications complete

**Solution**: Enforce Plan phase completion criteria

### Poor Cross-Phase Coordination

**Problem**: Test and Implementation don't coordinate, duplicate effort

**Solution**: Establish communication protocols, regular sync meetings

### Cleanup Neglect

**Problem**: Cleanup issues pile up, never addressed

**Solution**: Allocate time for Cleanup, make it non-negotiable

### Merge Conflicts

**Problem**: Parallel worktrees create conflicting changes

**Solution**: Coordinate on interfaces, integrate frequently

### Skipping Levels Inappropriately

**Problem**: Junior Engineer makes architectural decisions

**Solution**: Follow delegation rules, escalate when uncertain

## Related Documentation

- [Git Worktree Guide](./git-worktree-guide.md) - Detailed worktree usage
- [Common Workflows](./workflows.md) - Workflow examples
- [Agent Hierarchy](../hierarchy.md) - Complete agent specification
- [Delegation Rules](../delegation-rules.md) - How agents coordinate
- [Orchestration Patterns](../../notes/review/orchestration-patterns.md) - Detailed coordination patterns
