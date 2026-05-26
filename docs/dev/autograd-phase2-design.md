# Autograd Phase 2 — Design Doc (post-Phase 0 discovery)

**Goal:** Complete tape-based autograd so convnet training works without manual `*_backward` chains.

## Ground-truth state (after reading tape_types.mojo, tape.mojo, variable.mojo (samples), backward_ops.mojo (samples))

### What works today

| Component | Status | Notes |
|---|---|---|
| `Variable` wrapper | ✅ | `variable.mojo` — 11 ops: add/sub/mul/div/matmul/sum/mean/relu/sigmoid/tanh/neg |
| `GradientTape.record()` | ✅ | Appends TapeNode in execution order |
| `GradientTape.backward()` | ✅ | **Traverses nodes in LINEAR REVERSE ORDER** (tape.mojo:301-303). Works correctly because forward execution = topo order. **No separate topo sort needed.** |
| `VariableRegistry.set_grad()` | ✅ | **Already accumulates** gradients (L189-200): if has_grad, add to existing; else copy. So shared-param gradient summation is solved. |
| `_dispatch_backward_op` | ✅ | Dispatches 10 op types (ADD/SUB/MUL/DIV/SUM/MEAN/MATMUL/RELU/SIGMOID/TANH) via if/elif string match. |
| `SavedTensors` polymorphism | ✅ (sufficient) | Has `tensors: List[AnyTensor]` + `shapes: List[List[Int]]` + `scalars: List[Float64]` — enough to encode all convnet op hyperparams without struct changes. |
| 40+ raw `*_backward` functions | ✅ | In `core/{activation,loss,linear,conv,pooling,normalization,dropout}.mojo`. Confirmed via TODO.md inventory. |

### What's missing (Phase 2 scope)

| Component | Required for | Notes |
|---|---|---|
| `variable_linear` | LeNet, AlexNet, ResNet, VGG, GoogLeNet, MobileNet, MNIST | Wraps core.linear.linear + saves (input, weight) for `linear_backward` |
| `variable_conv2d` | All convnets | Wraps core.conv.conv2d + saves (input, weight, stride, padding) |
| `variable_maxpool2d` | LeNet, AlexNet, VGG, GoogLeNet | Wraps core.pooling.maxpool2d + saves (input, kernel_size, stride, padding) |
| `variable_flatten` / `variable_reshape` | All FC-headed convnets | Save (original_shape) only — backward is reshape-back |
| `variable_cross_entropy` | All classifiers | Wraps core.loss.cross_entropy + saves (logits, labels) |
| `variable_batch_norm2d` | ResNet, MobileNet | Wraps core.normalization.batch_norm2d + saves (input, scale, bias, mean, variance) |
| `variable_avgpool2d`, `variable_global_avgpool2d`, `variable_dropout`, `variable_softmax`, `variable_leaky_relu`, `variable_gelu` | One or more architectures | Same pattern as linear |
| Op-type constants `OP_CONV2D` / `OP_LINEAR` / `OP_MAXPOOL2D` / etc. | Tape dispatch | Add to `tape_types.mojo` or `tape.mojo` |
| Dispatch arms in `_dispatch_backward_op` | Each new op | Add elif branch + `backward_<op>(...)` helper |
| `backward_linear` / `backward_conv2d` / etc. wrappers in `backward_ops.mojo` | Dispatch glue | Each unpacks SavedTensors and calls the existing core `*_backward` function, then routes grads back via `registry.set_grad(input_id, grad_input)` etc. |

### Bugs found in Phase 0 (must fix as part of Phase 2)

1. **`SavedTensors.add_tensor` hardcodes Float32** (tape_types.mojo:51). Breaks fp64/fp16/bf16 silently — returns garbage cast values. Fix: dispatch on `tensor.dtype()` (use the same pattern as `core/numerical_safety.mojo` which has dtype dispatch, or pre-allocate via `zeros_like` + bulk memcpy).
2. **`VariableRegistry.set_grad` hardcodes Float32** (L197-198, 205). Same bug. Same fix.
3. **`VariableRegistry.register` creates a 1-element Float32 placeholder** (L170-173) for every variable. Acceptable but wasteful; could use `Optional[AnyTensor]` once Mojo Optional is stable.

These three are independent of the new ops and should be fixed in Phase 1 alongside the dispatch extensions.

## Architecture decisions

### TapeNode polymorphism: use existing `SavedTensors` triple

No new struct types. Each new op defines its saved-tensors layout as a comment in its `variable_*` function. Example for conv2d:

```
SavedTensors layout for OP_CONV2D:
  tensors[0] = input tensor          (for grad_weight + grad_input)
  tensors[1] = weight tensor         (for grad_input)
  shapes[0]  = bias.shape() if has_bias, else []
  scalars[0] = stride (Float64-cast Int)
  scalars[1] = padding (Float64-cast Int)
```

Each `backward_*` helper unpacks this in the reverse order. Comments adjacent to both forward and backward sites make the contract obvious.

### Op type constants: ordinal-indexed lookup table later, string-eq for now

Current dispatch uses string comparison (`if op_type == "add"`). Mojo string compare is slow but correct. For Phase 2 we keep strings (faster to ship). Phase 3 (out of scope): replace with integer ordinals + dispatch table.

### Variable value semantics: rely on registry, not Variable copies

Variables hold `(data: AnyTensor, id: Int, requires_grad: Bool)` plus optionally a tape reference. The `id` is the source of truth — when a Variable is copied (`var y = x`), both copies have the same `id`. The registry handles grad lookup by id. So Variable copies don't break gradient flow.

**Critical rule:** never reassign `.id` on an existing Variable. Always create a new Variable via `variable_*` op (which calls `tape.register_variable()` to get a fresh id). The optimizer must therefore mutate `Variable.data` in place (not return a new Variable), to preserve id continuity.

### Gradient accumulation: already works

`VariableRegistry.set_grad` already does `has_grad ? accumulate : init`. Multi-use parameters get summed contributions for free. **No change needed.**

### Forward = topo order = reverse iteration

Forward execution naturally appends nodes in topological order. So reverse iteration (current behavior) is correct reverse-topological order. **No DAG topo sort needed.** This is the single biggest simplification vs the audit's framing.

### Backward starting point

`tape.backward(loss_var_id, ones_like(loss_var.data))` already works. For convnets, loss is a scalar from `cross_entropy`, so `ones_like(loss)` is just `[[1.0]]`. Caller provides this.

## Implementation phases (revised after Phase 0)

| Phase | Scope | LOC est | Risk |
|---|---|---|---|
| **Phase 1 (foundation fixes + first new op)** | (a) Fix 3 Float32-hardcoded bugs in tape_types.mojo. (b) Add OP_FLATTEN constant. (c) Implement `variable_flatten` (smallest new op, validates tape→backward pipeline). (d) Wire `backward_flatten` into dispatch. (e) Unit test with finite-difference grad check. | ~150 | Low — no new architecture, just extends existing pattern. Will surface bugs early. |
| **Phase 2 (variable_linear + variable_cross_entropy)** | Enable end-to-end MLP-only training. Linear is the simplest layer with weights; cross_entropy is the simplest loss. With these + flatten + relu (existing) we can train an MLP on EMNIST. | ~300 | Medium — first time saving tensors of arbitrary shape; first time loss-as-backward-start. |
| **Phase 3 (variable_conv2d + variable_maxpool2d)** | Enable LeNet-shaped convnet training. | ~400 | Medium-high — maxpool backward needs the FORWARD INPUT to know which positions were max; tests how `SavedTensors.add_tensor` handles 4D tensors. |
| **Phase 4 (variable_batch_norm2d)** | Enable ResNet/MobileNet. Optional for proof-of-concept; LeNet doesn't use BN. | ~200 | Medium — BN saves multiple things (input, running stats). Skip if proof-of-concept LeNet is the goal. |
| **Phase 5 (port lenet_emnist/train.mojo to autograd)** | The integration test. Rewrite ~280 LOC manual backward chain → ~60 LOC tape-based. Compare loss curve + final test acc to existing manual version (should be within 1pp). | ~80 net | High — this is where compounding bugs surface. Reserve time for iterative debugging. |
| **Phase 6 (tests + docs reconcile)** | Property tests for each new op (finite-diff grad check, tolerance 1e-5). Update autograd/README.md status + autograd/TODO.md. Remove the "Currently demonstrates the interface and manual gradient computation" caveat from examples/autograd/simple_example.mojo. | ~300 | Low — mostly mechanical. |
| **Phase 7 (PR + auto-merge)** | Open PR closing #5452. | — | Low. |

**Phases 4 + post-LeNet other-architecture ports deferred** until LeNet + tests prove the approach works.

**Total estimated LOC: ~1,400** (not 5,000 — TODO.md's estimate was pessimistic because `SavedTensors` and `VariableRegistry` already exist and accumulate; we just need new op wrappers and dispatch arms).

## Mojo-language risks identified in Phase 0

| # | Risk | Mitigation |
|---|---|---|
| R1 | `SavedTensors.add_tensor` hardcoded `bitcast[Float32]` only works for fp32 inputs — for fp64/fp16/bf16 it silently copies garbage. | Fix in Phase 1 by dispatching on dtype. All `variable_*` ops can stay fp32-only until R1 is fixed. |
| R2 | `_dispatch_backward_op` uses if/elif string-equality on `op_type`. As N ops grows this is O(N) per backward node. With ~20 ops + LeNet's ~12-node tape per batch + 1500 batches = ~360k dispatches/epoch. Currently fine; future Phase 3 optimization. | Accept for Phase 2. |
| R3 | Variable identity via `id` requires optimizer to mutate `Variable.data` in place. Need to verify the existing `SGD.step()` in `autograd/optimizers.mojo` does this (line 90 takes `inout parameters: DynamicVector[Variable]` — looks correct). | Verify by reading optimizers.mojo lines 80-150 before Phase 2. |
| R4 | `core/conv.mojo`, `core/linear.mojo`, etc. backward functions return `GradResult` structs (per audit S4 reading). Need to confirm exact signatures (do they take input + weight, or input + saved-cache?). | Read first 60 lines of each `*_backward` in core/ before writing the autograd wrapper. |
| R5 | Mojo 1.0 `out self` vs `mut self` constructor distinctions — must follow project conventions per CLAUDE.md. | Apply to all new `variable_*` ops. |

## Definition of done (Phase 2 milestone)

- `examples/lenet_emnist/train_autograd.mojo` (new file, parallel to existing train.mojo) trains LeNet on EMNIST using `loss.backward(tape)` + `optimizer.step(params, tape)`. Manual `*_backward` chain eliminated.
- Loss curve + final test accuracy within 1 percentage point of the existing manual-grad version after 10 epochs.
- 6+ new `variable_*` ops in `variable.mojo` (flatten, linear, conv2d, maxpool2d, cross_entropy, optionally batch_norm2d).
- Tests pass at `pixi run mojo run tests/projectodyssey/autograd/test_variable_layers.mojo`.
- autograd/README.md status updated, TODO.md Phase 2 items checked off.
- All pre-commit hooks pass (no `--no-verify`).
- PR opened, auto-merge enabled.

## Out of scope for this milestone

- Higher-order gradients (Phase 3 in TODO.md)
- Gradient checkpointing for memory efficiency
- Custom user-registered gradients
- Mixed precision in autograd path (fp16 weights with fp32 grads)
- Distributed autograd
- Replacing string op-type with integer ordinal dispatch (Phase 3 optimization)
- Porting all 7 example architectures (only LeNet for proof-of-concept; rest tracked under #5454)
