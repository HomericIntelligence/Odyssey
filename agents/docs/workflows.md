# Common Workflows with ML Odyssey Agents

<!-- markdownlint-disable MD051 -->

## Table of Contents

- [Overview](#overview)
- [Workflow 1: Implementing a New Mojo Feature](#workflow-1-implementing-a-new-mojo-feature)
- [Workflow 2: Fixing a Bug](#workflow-2-fixing-a-bug)
- [Workflow 3: Refactoring Code](#workflow-3-refactoring-code)
- [Workflow 4: Reviewing Pull Requests](#workflow-4-reviewing-pull-requests)
- [Workflow 5: Implementing a Research Paper](#workflow-5-implementing-a-research-paper)
- [Workflow 6: Performance Optimization](#workflow-6-performance-optimization)
- [Workflow 7: Adding Documentation](#workflow-7-adding-documentation)
- [Workflow 8: Security Review](#workflow-8-security-review)
- [Quick Reference](#quick-reference)

## Overview

This document provides concrete examples of common development workflows using the ProjectOdyssey agent hierarchy.
Each workflow shows:

- Which agents are involved
- What each agent does
- How they coordinate
- Expected outputs and artifacts

**Note**: All workflows follow the 5-phase structure (Plan → Test/Impl/Package → Cleanup), but complexity varies
based on the task.

## Workflow 1: Implementing a New Mojo Feature

### Scenario

Implement a new `BatchNorm2D` layer for the shared library to be used across all paper implementations.

### Complexity

**High** - Full 5-phase workflow with all agent levels

### Agent Involvement

| Level | Agents | Phase Participation |
|-------|--------|---------------------|
| 0 | Chief Architect | Plan, Cleanup |
| 1 | Shared Library Orchestrator | Plan, Oversight, Cleanup |
| 2 | Architecture Design Agent | Plan |
| 3 | Implementation Specialist, Test Specialist, Documentation Specialist | Plan, Test/Impl/Package, Cleanup |
| 4 | Senior Implementation Engineer, Test Engineer, Documentation Engineer | Test/Impl/Package, Cleanup |
| 5 | Junior Implementation Engineer | Implementation, Cleanup |

### Workflow Steps

#### Phase 1: Plan (Sequential)

##### Step 1: Strategic Planning (Level 0-1)

```text
Chief Architect:
  - Reviews request for BatchNorm2D
  - Validates: ML papers commonly use batch normalization
  - Delegates to Shared Library Orchestrator

Shared Library Orchestrator:
  - Determines: This belongs in shared library (02-shared-library/core_ops/)
  - Creates GitHub issue #150: [Plan] BatchNorm2D Implementation
  - Delegates to Architecture Design Agent
```text

### Artifacts

- GitHub Issue #150 created
- Initial requirements documented

##### Step 2: Architecture Design (Level 2)

```text
Architecture Design Agent:
  - Analyzes BatchNorm2D requirements:
    * Track running mean and variance
    * Support training and inference modes
    * SIMD-optimized forward and backward passes
  - Designs component structure:
    * BatchNorm2D struct with owned tensors
    * forward() method for inference
    * forward_train() method for training
    * backward() method for gradient computation
  - Defines interfaces:
    * Input: Tensor[dtype](batch, channels, height, width)
    * Output: Same shape as input
  - Documents dependencies:
    * Requires: Tensor operations from core_ops
    * Integration: Will be used by LeNet-5 and future papers
  - Posts detailed specification as comment on GitHub issue #150
```text

### Artifacts

- Component specification: Posted as comment on GitHub issue #150
- Interface definitions: Posted as comment on GitHub issue #150
- Dependency graph

##### Step 3: Component Planning (Level 3)

```text
Implementation Specialist:
  - Breaks BatchNorm2D into implementable units:
    * __init__(num_features, eps, momentum)
    * forward(input) -> output (inference)
    * forward_train(input) -> output (training)
    * backward(grad_output) -> grad_input
    * _update_running_stats(mean, var)
    * _normalize_simd(input, mean, var) -> normalized
  - Creates function specifications for each
  - Assigns complexity:
    * _normalize_simd() → Senior Engineer (SIMD optimization)
    * forward(), forward_train() → Standard Engineer
    * __init__, backward() → Standard Engineer
    * Helper functions → Junior Engineer

Test Specialist:
  - Defines test strategy:
    * Unit tests: Each method individually
    * Integration tests: Within a model
    * Numerical tests: Compare with known implementations
    * Performance tests: SIMD speedup vs naive
  - Creates test cases:
    * Test forward pass correctness
    * Test backward pass gradients
    * Test training mode vs inference mode
    * Test running statistics updates
    * Test numerical stability
  - Specifies test fixtures needed

Documentation Specialist:
  - Plans documentation:
    * API reference for BatchNorm2D
    * Mathematical explanation of batch normalization
    * Usage examples (training and inference)
    * Performance benchmarks
    * Integration guide for papers
  - Creates documentation outline
```text

### Artifacts

- Implementation plan: Posted as comment on GitHub issue #150
- Test plan: Posted as comment on GitHub issue #150
- Documentation outline: Posted as comment on GitHub issue #150

**Plan Phase Complete**: Specifications approved, ready for parallel execution

#### Phase 2-4: Test/Implementation/Packaging (Parallel)

### Git Worktree Setup

```bash
# Create parallel worktrees
git worktree add worktrees/issue-151-test-batchnorm -b 151-test-batchnorm
git worktree add worktrees/issue-152-impl-batchnorm -b 152-impl-batchnorm
git worktree add worktrees/issue-153-pkg-batchnorm -b 153-pkg-batchnorm
```text

#### Phase 2: Test (Parallel)

```text
Test Specialist (Level 3):
  - Reviews test plan
  - Delegates test implementation to Test Engineer

Test Engineer (Level 4):
  - Implements unit tests (tests/test_batchnorm.mojo):

  from testing import assert_equal, assert_close
  from core_ops.layers import BatchNorm2D
  from tensor import Tensor

  fn test_batchnorm_forward_shape():
      """Test BatchNorm2D output shape matches input."""
      var bn = BatchNorm2D(num_features=64)
      var input = Tensor[DType.float32](2, 64, 28, 28)
      var output = bn.forward(input)

      assert_equal(output.shape, input.shape)

  fn test_batchnorm_inference_mode():
      """Test BatchNorm2D uses running stats in inference."""
      var bn = BatchNorm2D(num_features=32)

      # Simulate training updates
      var train_input = Tensor[DType.float32](10, 32, 16, 16)
      _ = bn.forward_train(train_input)

      # Inference should use accumulated running stats
      var test_input = Tensor[DType.float32](1, 32, 16, 16)
      var output = bn.forward(test_input)

      # Running mean/var should affect output
      assert_close(output.mean(), 0.0, atol=0.1)
      assert_close(output.std(), 1.0, atol=0.1)

  fn test_batchnorm_training_mode():
      """Test BatchNorm2D uses batch stats in training."""
      var bn = BatchNorm2D(num_features=16)
      var input = Tensor[DType.float32](8, 16, 32, 32)

      # Initialize with non-zero mean
      input += 10.0

      var output = bn.forward_train(input)

      # Output should be normalized to mean~0, std~1
      assert_close(output.mean(), 0.0, atol=0.1)
      assert_close(output.std(), 1.0, atol=0.1)

  fn test_batchnorm_backward():
      """Test BatchNorm2D gradient computation."""
      var bn = BatchNorm2D(num_features=8)
      var input = Tensor[DType.float32](4, 8, 16, 16)

      # Forward pass
      var output = bn.forward_train(input)

      # Backward pass
      var grad_output = Tensor[DType.float32](4, 8, 16, 16)
      grad_output.fill(1.0)
      var grad_input = bn.backward(grad_output)

      # Gradient should have same shape as input
      assert_equal(grad_input.shape, input.shape)

      # Check gradient numerically (TODO: numerical gradient)

  - Creates test fixtures (tests/fixtures/batchnorm_data.mojo)
  - Commits tests to test worktree
  - Status update: "Tests ready, waiting for implementation"
```text

### Artifacts

- Test files: `tests/test_batchnorm.mojo`
- Test fixtures: `tests/fixtures/batchnorm_data.mojo`
- All tests currently failing (no implementation yet)

#### Phase 3: Implementation (Parallel)

```text
Implementation Specialist (Level 3):
  - Reviews implementation plan
  - Delegates to appropriate engineers based on complexity

Senior Implementation Engineer (Level 4):
  - Implements SIMD-optimized normalization (complex):

  fn _normalize_simd(
      self,
      input: Tensor[DType.float32],
      mean: Tensor[DType.float32],
      var: Tensor[DType.float32]
  ) -> Tensor[DType.float32]:
      """SIMD-optimized batch normalization."""
      comptime simd_width = simdwidthof[DType.float32]()

      var output = Tensor[DType.float32](input.shape)

      let batch = input.shape[0]
      let channels = input.shape[1]
      let height = input.shape[2]
      let width = input.shape[3]

      # Vectorized normalization: (x - mean) / sqrt(var + eps)
      for b in range(batch):
          for c in range(channels):
              let c_mean = mean[c]
              let c_std = sqrt(var[c] + self.eps)

              @parameter
              fn normalize_pixel[simd_w: Int](h: Int, w: Int):
                  let val = input.load[simd_w](b, c, h, w)
                  let normalized = (val - c_mean) / c_std

                  # Apply learned parameters
                  let scaled = normalized * self.gamma[c] + self.beta[c]

                  output.store(b, c, h, w, scaled)

              # Vectorize over spatial dimensions
              for h in range(height):
                  vectorize[normalize_pixel, simd_width](width, h)

      return output

Standard Implementation Engineer (Level 4):
  - Implements BatchNorm2D struct and standard methods:

  from tensor import Tensor
  from math import sqrt

  @value
  struct BatchNorm2D:
      """Batch Normalization for 2D inputs (4D tensors).

      Normalizes inputs across the batch dimension, maintaining
      separate statistics for each channel.
      """
      var num_features: Int
      var eps: Float32
      var momentum: Float32

      # Learnable parameters
      var gamma: Tensor[DType.float32]  # Scale
      var beta: Tensor[DType.float32]   # Shift

      # Running statistics (for inference)
      var running_mean: Tensor[DType.float32]
      var running_var: Tensor[DType.float32]

      fn __init__(
          inout self,
          num_features: Int,
          eps: Float32 = 1e-5,
          momentum: Float32 = 0.1
      ):
          """Initialize BatchNorm2D.

          Args:
              num_features: Number of channels.
              eps: Small constant for numerical stability.
              momentum: Momentum for running statistics.
          """
          self.num_features = num_features
          self.eps = eps
          self.momentum = momentum

          # Initialize parameters
          self.gamma = Tensor[DType.float32](num_features)
          self.gamma.fill(1.0)

          self.beta = Tensor[DType.float32](num_features)
          self.beta.fill(0.0)

          # Initialize running statistics
          self.running_mean = Tensor[DType.float32](num_features)
          self.running_mean.fill(0.0)

          self.running_var = Tensor[DType.float32](num_features)
          self.running_var.fill(1.0)

      fn forward(
          self,
          input: Tensor[DType.float32]
      ) -> Tensor[DType.float32]:
          """Inference mode: use running statistics."""
          return self._normalize_simd(
              input,
              self.running_mean,
              self.running_var
          )

      fn forward_train(
          inout self,
          input: Tensor[DType.float32]
      ) -> Tensor[DType.float32]:
          """Training mode: use batch statistics and update running stats."""
          # Calculate batch statistics
          let batch_mean = self._compute_mean(input)
          let batch_var = self._compute_var(input, batch_mean)

          # Update running statistics
          self._update_running_stats(batch_mean, batch_var)

          # Normalize using batch statistics
          return self._normalize_simd(input, batch_mean, batch_var)

      fn backward(
          self,
          grad_output: Tensor[DType.float32]
      ) -> Tensor[DType.float32]:
          """Compute gradients for backward pass."""
          # TODO: Implement backward pass
          # For now, return gradient as-is
          return grad_output

      fn _compute_mean(
          self,
          input: Tensor[DType.float32]
      ) -> Tensor[DType.float32]:
          """Compute mean across batch and spatial dimensions."""
          # Implementation...

      fn _compute_var(
          self,
          input: Tensor[DType.float32],
          mean: Tensor[DType.float32]
      ) -> Tensor[DType.float32]:
          """Compute variance across batch and spatial dimensions."""
          # Implementation...

      fn _update_running_stats(
          inout self,
          batch_mean: Tensor[DType.float32],
          batch_var: Tensor[DType.float32]
      ):
          """Update running statistics using exponential moving average."""
          for c in range(self.num_features):
              self.running_mean[c] = (
                  (1.0 - self.momentum) * self.running_mean[c] +
                  self.momentum * batch_mean[c]
              )
              self.running_var[c] = (
                  (1.0 - self.momentum) * self.running_var[c] +
                  self.momentum * batch_var[c]
              )

Junior Implementation Engineer (Level 5):
  - Implements helper functions (_compute_mean, _compute_var)
  - Adds boilerplate code
  - Formats code with mojo format

  - Coordinates with Test Engineer:
    * Runs tests: mojo test tests/test_batchnorm.mojo
    * Tests pass: forward_shape, inference_mode, training_mode ✓
    * Tests fail: backward (not implemented yet)
  - Commits implementation to impl worktree
  - Status update: "Core implementation complete, backward pass pending"
```text

### Artifacts

- Implementation: `src/core_ops/layers/batchnorm.mojo`
- Most tests passing
- Backward pass TODO item for cleanup

#### Phase 4: Packaging (Parallel)

```text
Documentation Specialist (Level 3):
  - Reviews documentation outline
  - Delegates to Documentation Engineer

Documentation Engineer (Level 4):
  - Writes API documentation (docs/api/batchnorm.md):

  # BatchNorm2D

  ## Overview

  Batch Normalization normalizes the inputs of each mini-batch, reducing
  internal covariate shift and allowing higher learning rates.

  ## Mathematical Background

  For input x, batch normalization computes:

      y = γ * (x - μ) / √(σ² + ε) + β

  Where:
  - μ: batch mean
  - σ²: batch variance
  - γ: learned scale parameter
  - β: learned shift parameter
  - ε: small constant for numerical stability

  ## API Reference

  ### Constructor

  ```mojo

  BatchNorm2D(num_features: Int, eps: Float32 = 1e-5, momentum: Float32 = 0.1)

  ```

### Methods

#### forward (Inference Mode)

```mojo

  fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]

  ```

#### forward_train (Training Mode)

  ```mojo

  fn forward_train(inout self, input: Tensor[DType.float32]) -> Tensor[DType.float32]

  ```

## Usage Examples

### Basic Usage

```mojo

  from core_ops.layers import BatchNorm2D
  from tensor import Tensor

  # Create BatchNorm for 64 channels
  var bn = BatchNorm2D(num_features=64)

  # Training
  var train_input = Tensor[DType.float32](32, 64, 28, 28)
  var train_output = bn.forward_train(train_input)

  # Inference
  var test_input = Tensor[DType.float32](1, 64, 28, 28)
  var test_output = bn.forward(test_input)

```text
### In a Model

```mojo

  from core_ops.layers import Conv2D, BatchNorm2D, ReLU

  struct ConvBlock:
      var conv: Conv2D
      var bn: BatchNorm2D
      var relu: ReLU

      fn __init__(inout self, in_ch: Int, out_ch: Int):
          self.conv = Conv2D(in_ch, out_ch, 3)
          self.bn = BatchNorm2D(out_ch)
          self.relu = ReLU()

      fn forward(inout self, input: Tensor) -> Tensor:
          var x = self.conv.forward(input)
          x = self.bn.forward_train(x)  # Training mode
          x = self.relu.forward(x)
          return x

```text
## Performance

SIMD-optimized implementation provides significant speedup:

- 8x faster than naive Python implementation
- Benchmarked at 1.2ms for 32x64x28x28 input on M1 Mac
- Scales linearly with batch size

## Integration

To use in your paper implementation:

```mojo

  from core_ops.layers import BatchNorm2D

  # Add to your model
  var bn = BatchNorm2D(num_features=128)

```text
## See Also

- [Conv2D Layer](./conv2d.md)
- [ReLU Activation](./relu.md)
- [Model Architecture Guide](../guides/model-architecture.md)

- Creates integration guide (docs/guides/using-batchnorm.md)
- Updates shared library README
- Commits documentation to packaging worktree
- Status update: "Documentation complete"

```text

### Artifacts

- API documentation: `docs/api/batchnorm.md`
- Usage guide: `docs/guides/using-batchnorm.md`
- Updated README: `src/core_ops/README.md`

### Coordination During Parallel Phases

```text
Week 1:
  - Test: Tests 80% complete
  - Impl: Core implementation 60% complete
  - Package: Documentation 40% complete

Week 2:
  - Test: Tests 100% complete, some failing (waiting for backward())
  - Impl: Implementation 90% complete, backward() pending
  - Package: Documentation 80% complete

Week 3:
  - All phases complete
  - Ready for integration and cleanup
```text

### Phase 5: Cleanup (Sequential)

```text
Git Worktree Setup:
  cd /home/user/ProjectOdyssey
  git checkout main
  git pull  # Gets merged Test/Impl/Package branches
  git worktree add worktrees/issue-154-cleanup-batchnorm -b 154-cleanup-batchnorm

Cleanup Issues Collected:
  From Test:
    - [ ] Add numerical gradient test for backward()
    - [ ] Add performance regression test
    - [ ] Test with extreme values (numerical stability)

  From Implementation:
    - [ ] TODO: Implement backward() method
    - [ ] Refactor: _normalize_simd is 80 lines, split into smaller functions
    - [ ] Performance: Consider caching sqrt(var + eps) computation

  From Packaging:
    - [ ] Add comparison with PyTorch BatchNorm
    - [ ] Add troubleshooting section to docs
    - [ ] Add migration guide from Python

Senior Implementation Engineer (Cleanup):
  - Implements backward() method
  - Refactors _normalize_simd into smaller functions
  - Optimizes by caching repeated computations

Test Engineer (Cleanup):
  - Adds numerical gradient test
  - Adds performance regression test
  - Verifies all tests pass

Documentation Engineer (Cleanup):
  - Adds PyTorch comparison
  - Adds troubleshooting FAQ
  - Updates examples with backward() usage

Architecture Design Agent (Cleanup):
  - Reviews final implementation
  - Validates against original design
  - Approves for production

Shared Library Orchestrator (Cleanup):
  - Final review
  - Merges cleanup PR
  - Updates project roadmap
```text

### Artifacts

- Complete, production-ready BatchNorm2D
- Full test coverage including backward pass
- Comprehensive documentation
- Performance benchmarks

### Expected Outputs

### Code

- `src/core_ops/layers/batchnorm.mojo` - Implementation
- `tests/test_batchnorm.mojo` - Tests
- `tests/fixtures/batchnorm_data.mojo` - Test fixtures

### Documentation

- `docs/api/batchnorm.md` - API reference
- `docs/guides/using-batchnorm.md` - Usage guide
- `src/core_ops/README.md` - Updated with BatchNorm

### GitHub

- 5 Issues (#150-154) - Plan, Test, Impl, Package, Cleanup
- 5 PRs - One per phase
- All merged to main

---

## Workflow 2: Fixing a Bug

### Scenario

Bug: Conv2D layer produces incorrect output for non-square kernels (3x5 kernel fails).

### Complexity

**Low** - Simplified 5-phase workflow, minimal planning, no packaging

### Agent Involvement

| Level | Agents | Phase Participation |
|-------|--------|---------------------|
| 3 | Implementation Specialist, Test Specialist | Plan, Test/Impl, Cleanup |
| 4 | Implementation Engineer, Test Engineer | Test/Impl, Cleanup |

### Workflow Steps

#### Phase 1: Plan (Minimal)

```text
Implementation Specialist:
  - Analyzes bug report
  - Root cause: Kernel dimension calculation assumes square kernels
  - Creates brief fix specification:
    * Update _apply_kernel_simd to handle rectangular kernels
    * Fix: Use kernel_height and kernel_width separately
  - Posts fix plan as comment on GitHub issue #200

Test Specialist:
  - Plans test:
    * Test 3x5 kernel (currently fails)
    * Test 5x3 kernel
    * Test various rectangular kernels
```text

**Time**: 30 minutes

#### Phase 2-3: Test + Implementation (Parallel, No Packaging)

```bash
# Create worktrees
git worktree add worktrees/issue-201-test-conv2d-bug -b 201-test-conv2d-bug
git worktree add worktrees/issue-202-impl-conv2d-bug -b 202-impl-conv2d-bug
```text

```text
Test Engineer:
  - Writes failing test:

  fn test_conv2d_rectangular_kernel():
      """Test Conv2D with non-square kernel."""
      # 3x5 kernel (height=3, width=5)
      var conv = Conv2D(in_channels=1, out_channels=1, kernel_height=3, kernel_width=5)
      var input = Tensor[DType.float32](1, 1, 28, 28)
      var output = conv.forward(input)

      # Expected output shape: (1, 1, 26, 24)
      # height: 28 - 3 + 1 = 26
      # width: 28 - 5 + 1 = 24
      assert_equal(output.shape[2], 26)
      assert_equal(output.shape[3], 24)  # Currently FAILS

  - Test fails (as expected)
  - Commits test

Implementation Engineer:
  - Fixes _apply_kernel_simd:

  # Before (bug):
  for ky in range(self.kernel_size):
      for kx in range(self.kernel_size):  # Assumes square
          ...

  # After (fixed):
  for ky in range(self.kernel_height):
      for kx in range(self.kernel_width):  # Handles rectangular
          ...

  - Runs test: PASSES ✓
  - Commits fix
```text

**Time**: 1-2 hours

#### Phase 5: Cleanup (Quick)

```text
Test Engineer:
  - Adds more edge case tests (1x1, 7x1, 1x7 kernels)
  - Adds regression test to prevent future bugs

Implementation Engineer:
  - Reviews code for similar issues
  - Adds inline comment explaining rectangular kernel support

Documentation Engineer:
  - Updates Conv2D documentation to highlight rectangular kernel support
  - Adds example with rectangular kernel
```text

**Time**: 1 hour

### Expected Outputs

### Code

- Fixed `src/core_ops/layers/conv2d.mojo`
- New test `tests/test_conv2d.mojo::test_rectangular_kernel()`

### Documentation

- Updated `docs/api/conv2d.md` with rectangular kernel examples

### GitHub

- 3 Issues (#200-202) - Plan, Test+Impl combined, Cleanup
- 2 PRs - One for fix, one for cleanup
- Merged to main

---

## Workflow 3: Refactoring Code

### Scenario

Refactor: Multiple layers have duplicated SIMD vectorization code. Extract into shared utilities.

### Complexity

**Medium** - Standard 5-phase, but cleanup-heavy

### Agent Involvement

| Level | Agents | Phase Participation |
|-------|--------|---------------------|
| 1 | Shared Library Orchestrator | Plan, Oversight, Cleanup |
| 2 | Architecture Design Agent | Plan |
| 3 | Implementation Specialist | Plan, Implementation, Cleanup |
| 4 | Implementation Engineer | Implementation, Cleanup |
| 4 | Test Engineer | Test (verify refactoring doesn't break) |

### Workflow Steps

#### Phase 1: Plan

```text
Shared Library Orchestrator:
  - Identifies code duplication across Conv2D, BatchNorm2D, Dense layers
  - Delegates to Architecture Design Agent

Architecture Design Agent:
  - Designs shared utilities module:
    * core_ops/utils/simd_helpers.mojo
    * Functions: vectorize_2d, vectorize_4d, reduce_sum_simd
  - Defines interfaces for each utility
  - Plans migration strategy: Update one layer at a time

Implementation Specialist:
  - Creates refactoring plan:
    1. Create simd_helpers module
    2. Refactor Conv2D to use helpers
    3. Refactor BatchNorm2D
    4. Refactor Dense
    5. Remove duplicated code
  - Identifies potential risks:
    * Performance regression
    * Behavior changes
```text

**Time**: 1 week

#### Phase 2: Test

```text
Test Engineer:
  - Creates comprehensive regression tests:
    * Baseline performance benchmarks
    * Baseline correctness tests for all layers
  - Tests will verify refactoring doesn't change behavior
```text

#### Phase 3: Implementation

```text
Implementation Engineer:
  - Creates simd_helpers.mojo:

  from algorithm import vectorize
  from memory import memset_zero

  fn vectorize_4d[
      func: fn[Int](Int, Int, Int, Int),
      simd_width: Int
  ](
      dim0: Int,
      dim1: Int,
      dim2: Int,
      dim3: Int
  ):
      """Vectorize computation over 4D tensor."""
      for i0 in range(dim0):
          for i1 in range(dim1):
              for i2 in range(dim2):
                  vectorize[func, simd_width](dim3, i0, i1, i2)

  - Refactors Conv2D:

  # Before:
  for b in range(batch):
      for c in range(channels):
          for h in range(height):
              vectorize[compute_pixel, simd_width](width, b, c, h)

  # After:
  vectorize_4d[compute_pixel, simd_width](batch, channels, height, width)

  - Refactors other layers similarly
  - Removes duplicated code
```text

#### Phase 4: Packaging

```text
Documentation Engineer:
  - Documents new simd_helpers module
  - Updates layer documentation to reference helpers
  - Creates migration guide for future layer implementations
```text

#### Phase 5: Cleanup

```text
Test Engineer:
  - Runs regression tests: All pass ✓
  - Runs performance benchmarks: No regression ✓
  - Adds tests for simd_helpers directly

Implementation Engineer:
  - Final code review
  - Ensures consistent usage across all layers
  - Adds inline documentation

Architecture Design Agent:
  - Validates refactoring maintains architectural integrity
  - Approves design
```text

### Expected Outputs

### Code

- New module: `src/core_ops/utils/simd_helpers.mojo`
- Refactored: `conv2d.mojo`, `batchnorm.mojo`, `dense.mojo`
- Removed: ~200 lines of duplicated code

### Tests

- Regression tests confirming no behavior changes
- Unit tests for simd_helpers

### Documentation

- API docs for simd_helpers
- Migration guide for future implementations

---

## Workflow 4: Reviewing Pull Requests

### Scenario

Review PR #300: "Add Dropout layer implementation"

### Complexity

**Low** - Ad-hoc review process, not full 5-phase

### Agent Involvement

| Level | Agents | Responsibilities |
|-------|--------|------------------|
| 2 | Architecture Design Agent | Architectural review |
| 3 | Implementation Specialist | Code quality review |
| 3 | Test Specialist | Test coverage review |
| 4 | Security Specialist | Security review |
| 4 | Performance Engineer | Performance review |

### Workflow Steps

```text
PR Submitted: #300 - Add Dropout layer

Architecture Design Agent:
  - Reviews: Does Dropout fit shared library architecture?
  - Checks: Interface matches other layers (forward, backward)
  - Validates: Proper integration with existing code
  - Feedback: "✓ Architecture looks good, consistent with other layers"

Implementation Specialist:
  - Reviews code quality:
    * Mojo idioms followed?
    * Error handling present?
    * Code readable and maintainable?
  - Checks: Memory management (owned, borrowed, inout)
  - Feedback: "Request: Add error handling for invalid dropout rate (p < 0 or p > 1)"

Test Specialist:
  - Reviews test coverage:
    * Unit tests present?
    * Edge cases covered?
    * Integration tests needed?
  - Checks: Tests actually test the implementation
  - Feedback: "Missing test: Dropout in inference mode (should be pass-through)"

Security Specialist:
  - Reviews for security issues:
    * Buffer overflows?
    * Unsafe memory access?
    * Input validation?
  - Checks: Random number generation (for dropout mask)
  - Feedback: "✓ No security concerns"

Performance Engineer:
  - Reviews performance:
    * SIMD optimization used where appropriate?
    * Performance benchmarks included?
    * No obvious bottlenecks?
  - Runs benchmarks: Compares with PyTorch
  - Feedback: "Performance good, 5x faster than PyTorch. Request: Add benchmark to docs"

PR Author addresses feedback:
  - Adds error handling for invalid dropout rate
  - Adds inference mode test
  - Adds performance benchmark to documentation

Final Review:
  All reviewers approve ✓

Merge:
  Implementation Specialist merges PR
```text

### Expected Outputs

- PR reviewed by multiple specialists
- Feedback addressed
- PR merged with high confidence in quality

---

## Workflow 5: Implementing a Research Paper

### Scenario

Implement LeNet-5 (LeCun et al., 1998) as the first paper in the repository.

### Complexity

**Very High** - Complete 5-phase workflow, full agent hierarchy, multi-module

### Agent Involvement

**All Levels (0-5)**, **Multiple Sections**

### Workflow Steps

#### Phase 1: Plan

```text
Chief Architect (Level 0):
  - Selects LeNet-5 as first paper (proven, foundational, well-documented)
  - Analyzes paper requirements:
    * 7 layers: Conv → Pool → Conv → Pool → FC → FC → FC
    * MNIST dataset
    * Training with gradient descent
  - Delegates to Papers Orchestrator

Papers Orchestrator (Level 1):
  - Breaks into modules:
    1. Data Preparation (dataset loading, preprocessing)
    2. Model Implementation (LeNet-5 architecture)
    3. Training (training loop, optimization)
    4. Evaluation (accuracy, visualization)
  - Creates GitHub issues for each module
  - Delegates each module to Module Design Agents

Architecture Design Agent (Level 2):
  - Designs model architecture module:
    * Sequential model container
    * Layer composition (Conv2D → ReLU → MaxPool → Dense)
    * Forward and backward passes
  - Creates component specifications

Integration Design Agent (Level 2):
  - Designs integration between modules:
    * Data → Model (batch processing)
    * Model → Training (gradient flow)
    * Training → Evaluation (checkpointing)
  - Defines APIs between modules

Component Specialists (Level 3):
  For each module, specialists create detailed plans:
    * Implementation Specialist: Break into functions/classes
    * Test Specialist: Define test cases
    * Documentation Specialist: Plan docs
```text

### Plan Phase Artifacts

- Module specifications: Posted as comments on GitHub issues #400-403
- Interface definitions
- Integration plan

#### Phase 2-4: Test/Implementation/Packaging (Parallel)

Each module (Data, Model, Training, Evaluation) has its own Test/Impl/Package issues running in parallel.

##### Example: Model Module

```text
Test (Issue #411):
  - Test Sequential model container
  - Test layer composition
  - Test forward pass with known inputs
  - Test backward pass (gradient flow)

Implementation (Issue #412):
  from core_ops.layers import Conv2D, MaxPool2D, Dense, ReLU

  struct LeNet5:
      """LeNet-5 architecture for MNIST classification."""
      var conv1: Conv2D
      var pool1: MaxPool2D
      var conv2: Conv2D
      var pool2: MaxPool2D
      var fc1: Dense
      var fc2: Dense
      var fc3: Dense
      var relu: ReLU

      fn __init__(inout self):
          # Layer 1: Conv 1@28x28 → 6@24x24
          self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
          self.pool1 = MaxPool2D(pool_size=2, stride=2)

          # Layer 2: Conv 6@12x12 → 16@8x8
          self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
          self.pool2 = MaxPool2D(pool_size=2, stride=2)

          # Fully connected layers
          self.fc1 = Dense(in_features=16*4*4, out_features=120)
          self.fc2 = Dense(in_features=120, out_features=84)
          self.fc3 = Dense(in_features=84, out_features=10)

          self.relu = ReLU()

      fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
          """Forward pass through LeNet-5."""
          var x = input

          # Conv block 1
          x = self.conv1.forward(x)
          x = self.relu.forward(x)
          x = self.pool1.forward(x)

          # Conv block 2
          x = self.conv2.forward(x)
          x = self.relu.forward(x)
          x = self.pool2.forward(x)

          # Flatten
          x = x.reshape(x.shape[0], -1)

          # Fully connected layers
          x = self.fc1.forward(x)
          x = self.relu.forward(x)
          x = self.fc2.forward(x)
          x = self.relu.forward(x)
          x = self.fc3.forward(x)

          return x

Packaging (Issue #413):
  - Document LeNet-5 architecture
  - Create usage examples
  - Write training guide
  - Benchmark performance vs PyTorch
```text

**Similar for other modules**: Data, Training, Evaluation

#### Phase 5: Cleanup

```text
Cleanup Issues Collected:
  - Optimize training loop (currently slow)
  - Add data augmentation
  - Improve visualization
  - Add model checkpointing
  - Write paper reproduction report (accuracy comparison with original)

All agents participate in cleanup:
  - Performance Engineer optimizes training
  - Implementation Engineer adds features
  - Documentation Engineer writes reproduction report
```text

### Expected Outputs

### Code

- Complete LeNet-5 implementation
- MNIST data loading
- Training loop
- Evaluation scripts

### Documentation

- LeNet-5 architecture guide
- Training tutorial
- Reproduction report comparing with original paper

### Results

- Model achieves ~99% accuracy on MNIST (matching paper)
- Performance benchmarks vs PyTorch

### GitHub

- 20+ issues (Plan, Test, Impl, Package, Cleanup for each module)
- 20+ PRs
- All merged to main

---

## Workflow 6: Performance Optimization

### Scenario

Optimize: Conv2D forward pass is slower than expected. Target: 2x speedup.

### Complexity

**Medium** - Focus on Implementation and Cleanup phases

### Agent Involvement

| Level | Agents | Phase Participation |
|-------|--------|---------------------|
| 3 | Performance Specialist | Plan, Implementation, Cleanup |
| 4 | Performance Engineer | Implementation |
| 4 | Test Engineer | Test (validate correctness maintained) |

### Workflow Steps

#### Phase 1: Plan

```text
Performance Specialist:
  - Profiles current implementation:
    * Bottleneck: Inner convolution loop (60% of time)
    * Opportunity: Better SIMD vectorization
    * Opportunity: Tiling for cache locality
  - Creates optimization plan:
    1. Optimize SIMD vectorization
    2. Add tiling for large inputs
    3. Inline hot functions
  - Defines success criteria:
    * 2x speedup on 32x3x224x224 input
    * No accuracy regression
```text

#### Phase 2: Test

```text
Test Engineer:
  - Creates baseline benchmarks
  - Creates correctness tests (compare optimized vs original)
```text

#### Phase 3: Implementation

```text
Performance Engineer:
  - Implements optimizations:
    * Better SIMD: Vectorize over output channels
    * Tiling: Process 8x8 tiles for cache reuse
    * Inlining: @always_inline for hot paths
  - Benchmarks: 2.3x speedup achieved ✓
  - Validates: All correctness tests pass ✓
```text

#### Phase 5: Cleanup

```text
Test Engineer:
  - Adds performance regression tests
  - Updates benchmarks in CI/CD

Documentation Engineer:
  - Documents optimization techniques
  - Adds performance tips for other layers
```text

### Expected Outputs

- Optimized Conv2D implementation
- 2.3x speedup
- No accuracy regression
- Documentation of optimization techniques

---

## Workflow 7: Adding Documentation

### Scenario

Add: Comprehensive getting started guide for the ProjectOdyssey project.

### Complexity

**Low** - Packaging-focused workflow

### Agent Involvement

| Level | Agents | Phase Participation |
|-------|--------|---------------------|
| 3 | Documentation Specialist | Plan, Packaging |
| 4 | Documentation Engineer | Packaging |

### Workflow Steps

#### Phase 1: Plan

```text
Documentation Specialist:
  - Analyzes documentation needs:
    * Installation guide
    * First model tutorial
    * Repository structure overview
    * Contribution guide
  - Creates documentation outline
```text

#### Phase 4: Packaging (Main Phase)

```text
Documentation Engineer:
  - Writes getting started guide:
    * Installation (pixi, Mojo setup)
    * Quick start (train LeNet-5 in 5 minutes)
    * Repository tour (where to find things)
    * Next steps (how to implement a paper)
  - Creates example code
  - Adds diagrams and screenshots
```text

#### Phase 5: Cleanup

```text
Documentation Specialist:
  - Reviews guide for clarity
  - Tests instructions on fresh system
  - Fixes any issues
```text

### Expected Outputs

- `docs/getting-started.md` - Comprehensive guide
- Example code in `examples/quickstart/`
- Validated on fresh system

---

## Workflow 8: Security Review

### Scenario

Security audit before release: Review all code for security vulnerabilities.

### Complexity

**Medium** - Security-focused review across all code

### Agent Involvement

| Level | Agents | Responsibilities |
|-------|--------|------------------|
| 2 | Security Design Agent | Plan security review |
| 3 | Security Specialist | Execute security review |
| 4 | Implementation Engineers | Fix issues |

### Workflow Steps

#### Phase 1: Plan

```text
Security Design Agent:
  - Plans security audit:
    * Review all tensor operations (buffer overflows)
    * Review all file I/O (path traversal)
    * Review random number generation (predictability)
    * Review dependencies (known CVEs)
  - Defines security standards
```text

#### Phase 2: Test

```text
Security Specialist:
  - Creates security test cases:
    * Fuzz testing for tensor operations
    * Invalid input tests
    * Boundary condition tests
```text

#### Phase 3: Implementation

```text
Security Specialist:
  - Conducts security review:
    * Found: Potential buffer overflow in Conv2D with invalid kernel size
    * Found: Unchecked file path in dataset loader
    * Found: Dependency with known CVE (low severity)

Implementation Engineer:
  - Fixes issues:
    * Add input validation for kernel size
    * Sanitize file paths in dataset loader
    * Update dependency
```text

#### Phase 5: Cleanup

```text
Security Specialist:
  - Re-audits fixes: All resolved ✓
  - Creates security checklist for future code

Documentation Engineer:
  - Documents security practices
  - Adds security section to contribution guide
```text

### Expected Outputs

- Security audit report
- All vulnerabilities fixed
- Security checklist for future development

---

## Quick Reference

### Workflow Complexity Guide

| Complexity | Phases Used | Agent Levels | Example |
|------------|-------------|--------------|---------|
| Very Low | Plan + Impl | 3-4 | Typo fix, simple docs |
| Low | Plan + Test + Impl | 3-4 | Bug fix, simple feature |
| Medium | Full 5-phase | 1-4 | Refactoring, optimization |
| High | Full 5-phase | 0-5 | New component, complex feature |
| Very High | Full 5-phase, multiple modules | 0-5 | Research paper, major subsystem |

### Phase Participation by Agent Level

| Level | Plan | Test | Impl | Package | Cleanup |
|-------|------|------|------|---------|---------|
| 0 | ███ | ░░ | ░░ | ░░ | ███ |
| 1 | ███ | ░░ | ░░ | ░░ | ███ |
| 2 | ███ | ░░ | ░░ | ░░ | ███ |
| 3 | ███ | ███ | ███ | ███ | ███ |
| 4 | ░░ | ███ | ███ | ███ | ███ |
| 5 | ░░ | ██ | ██ | ██ | ██ |

### Common Agent Combinations

### Feature Development

- Chief Architect (strategic) + Section Orchestrator (tactical) + Design Agents (architecture) + Specialists (execution)

### Bug Fixes

- Implementation Specialist + Test Engineer + Implementation Engineer

### Refactoring

- Architecture Design Agent + Implementation Specialist + Implementation Engineers

### Documentation

- Documentation Specialist + Documentation Engineer

### Security

- Security Design Agent + Security Specialist + Implementation Engineers

### Performance

- Performance Specialist + Performance Engineer + Test Engineer

## Related Resources

- [5-Phase Integration](./5-phase-integration.md) - Detailed workflow explanation
- [Git Worktree Guide](./git-worktree-guide.md) - How to use worktrees
- [Agent Hierarchy](../hierarchy.md) - Complete agent reference
- [Delegation Rules](../delegation-rules.md) - Coordination patterns
