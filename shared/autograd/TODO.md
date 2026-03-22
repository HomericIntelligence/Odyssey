# Autograd TODO - Future Features and Missing Components

## Design Philosophy

The autograd module follows **YAGNI** (You Aren't Gonna Need It) and **KISS** (Keep It Simple Stupid) principles:

- **Practical helpers** for common gradient patterns rather than full computation graph autograd
- **Simple function calls** rather than complex graph management
- **Works today** with current Mojo v0.26.1+ constraints
- **Covers 90% of use cases** while keeping implementation maintainable

## Current Status (Issue #2196)

### Completed

- [x] **Unified backward function API** - All 40+ backward functions re-exported from `shared.core`
- [x] **Functional gradient helpers** - `mse_loss_and_grad`, `bce_loss_and_grad`, `ce_loss_and_grad`
- [x] **Parameter update helpers** - `apply_gradient`, `apply_gradients`
- [x] **Scalar operations** - `multiply_scalar`, `add_scalar`, `subtract_scalar`, `divide_scalar`
- [x] **SGD optimizer** - Basic stochastic gradient descent
- [x] **Foundation components** - `Variable`, `GradientTape`, `TapeNode` (for future full autograd)

### Backward Functions Implemented (40 functions)

**Activation (10)**

- relu_backward, leaky_relu_backward, prelu_backward
- sigmoid_backward, tanh_backward, gelu_backward
- softmax_backward, swish_backward, mish_backward, elu_backward

**Loss (3)**

- binary_cross_entropy_backward, mean_squared_error_backward, cross_entropy_backward

**Matrix Operations (2)**

- matmul_backward, transpose_backward

**Arithmetic (4)**

- add_backward, subtract_backward, multiply_backward, divide_backward

**Element-wise (7)**

- exp_backward, log_backward, log10_backward, log2_backward
- sqrt_backward, abs_backward, clip_backward

**Reduction (4)**

- sum_backward, mean_backward, max_reduce_backward, min_reduce_backward

**Network Layers (4)**

- linear_backward, linear_no_bias_backward
- conv2d_backward, conv2d_no_bias_backward

**Pooling (3)**

- maxpool2d_backward, avgpool2d_backward, global_avgpool2d_backward

**Normalization (1)**

- batch_norm2d_backward

**Dropout (2)**

- dropout_backward, dropout2d_backward

## Missing Features

### Issue #2197: Layer Norm Backward

- [ ] `layer_norm_backward` - Backward pass for layer normalization
  - Requires gradient computation with respect to normalized values
  - May need layer statistics (mean, variance) from forward pass
  - Current status: Forward pass exists, backward not yet implemented

### Issue #2198: Global Average Pool Backward Edge Cases

- [x] `global_avgpool2d_backward` - Implemented
- [ ] Edge cases testing (zero gradients, numerical stability)
- [ ] SIMD optimization for large batch sizes

### Issue #2197: Dropout Mask Caching

- [x] `dropout_backward` - Implemented
- [x] `dropout2d_backward` - Implemented
- [ ] Automatic mask caching during forward pass (requires tape infrastructure)
  - Currently user must manually pass mask to backward pass
  - Full autograd would cache masks automatically

### Potential Future Additions

- [ ] `concat_backward` - Concatenation gradient
- [ ] `stack_backward` - Stack operation gradient
- [ ] `reshape_backward` - Reshape gradient
- [ ] `expand_dims_backward` - Dimension expansion gradient
- [ ] `squeeze_backward` - Dimension squeezing gradient
- [ ] `flatten_backward` - Flatten gradient

## Future Features

### Phase 1: Enhanced Functionality (Post-#2196)

- [ ] **Momentum optimizer** - SGD with momentum
- [ ] **Adam optimizer** - Adaptive moment estimation
- [ ] **RMSprop optimizer** - Root mean square prop optimizer
- [ ] **Gradient clipping** - Prevent exploding gradients
- [ ] **Weight decay** - L2 regularization in optimizers
- [ ] **Learning rate scheduling** - Decay schedules (step, exponential, cosine)

### Phase 2: Automatic Differentiation (Future Major Release)

- [ ] **Full tape-based autograd** - Automatic differentiation engine
  - Record all operations in computation graph
  - Automatic backward pass without manual gradient functions
  - Support for custom gradient functions
- [ ] **Variable wrapper** - Enhance Variable to support tape recording
  - Track parent operations
  - Automatic gradient accumulation
  - Mutable gradient attributes

### Phase 3: Advanced Features

- [ ] **Higher-order gradients** - Compute gradients of gradients
  - Hessian computation
  - Laplacian computation
  - Automatic second derivatives
- [ ] **Checkpointing** - Memory-efficient gradient computation
  - Recompute activations during backward pass
  - Trade computation time for memory
  - Critical for large language models
- [ ] **Custom gradient registration** - User-defined backward passes
  - Override gradient computation for custom operations
  - Specify custom chain rule rules

### Phase 4: Performance & Integration

- [ ] **JIT compilation support** - Compile gradient functions
- [ ] **Graph fusion** - Optimize tape-based computation graphs
- [ ] **Distributed gradient computation** - Multi-device gradients
- [ ] **Mixed precision training** - fp16 weights with fp32 gradients
- [ ] **Gradient accumulation** - Simulating larger batches

## Known Limitations

### Current Architecture

1. **Manual backward calls** - Users must explicitly call backward functions
   - Requires knowing which forward function was called
   - Risk of incorrect backward function usage
   - No protection against mismatched operations

2. **No automatic tape recording** - Computation graph is not automatically recorded
   - Cannot compute gradients without explicit backward functions
   - No support for control flow in gradient computation
   - Dynamic architectures require more boilerplate

3. **Dropout requires mask storage** - Must manually store and pass mask to backward
   - Forward pass must be called explicitly before backward
   - Masks must be stored externally (not in Variable)
   - No automatic mask generation on backward

4. **Limited to floating-point gradients** - int8, int16, etc. types not fully supported
   - Quantized networks require custom gradient handling
   - No automatic dtype promotion in gradients
   - Some backward passes only support float16/32/64

### Why These Limitations?

These are intentional design choices:

- **KISS principle** - Full tape-based autograd is complex (~5000+ lines)
- **Mojo constraints** - String matching for grad functions is expensive
- **Performance** - Explicit backward calls are faster than automatic recording
- **Predictability** - Users know exactly when gradients are computed
- **Pragmatism** - 90% of use cases don't need full autograd

## Migration Path to Full Autograd

If the project outgrows the current functional helpers:

1. **Phase 1: Enhance Variable class** - Add tape recording capability

   ```mojo
   struct Variable:
       var tensor: AnyTensor
       var grad: AnyTensor  # Accumulated gradient
       var requires_grad: Bool
       var grad_fn: String  # Operation name for backward dispatch
   ```

2. **Phase 2: Implement TapeNode enhancements** - Track operation parameters

   ```mojo
   struct TapeNode:
       var operation: String  # "relu", "linear", etc.
       var parents: List[TapeNode]
       var params: Dict[String, AnyTensor]  # For functions like conv2d_backward
   ```

3. **Phase 3: Add backward dispatcher** - Automatic backward function selection

   ```mojo
   fn backward(tape_node: TapeNode) -> List[AnyTensor]:
       # Use operation name to dispatch to correct backward function
       # Automatically chain gradients through computation graph
   ```

## Testing TODOs

- [ ] Comprehensive gradient checking for all backward functions
- [ ] Numerical gradient validation (finite differences)
- [ ] Edge case testing (zero inputs, very large/small values)
- [ ] Mixed dtype testing (float16, float32, float64)
- [ ] Performance benchmarks for SIMD-optimized backward passes

## Documentation TODOs

- [ ] API documentation for all backward functions
- [ ] Gradient computation guide for custom operations
- [ ] Performance tuning guide
- [ ] Migration guide from functional helpers to full autograd
- [ ] Examples of gradient-based optimization

## References

- **Current Implementation**: `shared/autograd/__init__.mojo`
- **Backward Functions**: `shared/core/*.mojo`
- **Design Document**: `DESIGN.md`
- **Issues**: #2196 (Unification), #2197 (Dropout), #2198 (Global Pool), #2199 (This TODO)
