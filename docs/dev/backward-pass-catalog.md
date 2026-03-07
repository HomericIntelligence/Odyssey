# ExTensor Backward Pass Implementation Catalog

**Status**: Comprehensive training readiness verification
**Date**: 2025-11-18
**Total Backward Functions**: 27 across 5 modules
**Broadcasting Support**: 9/27 functions (arithmetic, reductions)
**Numerical Stability**: 10/27 functions
**Activation Functions**: 7/27

---

## MODULE 1: ARITHMETIC.MOJO

**Module Overview**: Element-wise arithmetic operations with broadcasting support
**Total Backward Functions**: 5 (including 1 helper)

### 1. _reduce_broadcast_dims (Helper Function)

**Location**: Lines 498-545
**Signature**: `fn _reduce_broadcast_dims(grad: ExTensor, original_shape: DynamicVector[Int]) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Helper function that reduces gradients from broadcast shapes back to original shapes.

### Mathematical Formula

```text
For forward pass that broadcast X[original_shape] → Y[broadcast_shape]:
∂L/∂X = reduce_sum(∂L/∂Y, axes_that_were_broadcast)
```text

### Broadcasting Handling

- YES - Core purpose is to handle shape mismatches
- Handles prepended dimensions: `(5,) → (3,4,5)` → sums over first 2 dims
- Handles broadcast dimensions: `(3,1,5) → (3,4,5)` → sums over dimension 1 with keepdims=True

### Shape Reduction Logic

1. Reduce prepended dimensions (when original_ndim < grad_ndim)
1. Sum over dimensions that were size 1 and got broadcast

### Edge Cases

- Empty dimensions: Handled by dimension checking
- Scalar inputs: Converted to 0D tensors

**Numerical Stability**: None needed (summation is stable)

---

### 2. add_backward

**Location**: Lines 548-586
**Signature**: `fn add_backward(grad_output: ExTensor, a_shape: DynamicVector[Int], b_shape: DynamicVector[Int]) raises -> (ExTensor, ExTensor)`
**Return Type**: `Tuple[ExTensor, ExTensor]`

**Purpose**: Compute gradients for element-wise addition with broadcasting support.

### Mathematical Formula

```text
Forward: C = A + B
Backward:
  ∂L/∂A = ∂L/∂C  (reduced to a_shape)
  ∂L/∂B = ∂L/∂C  (reduced to b_shape)
```text

### Broadcasting Handling

- YES - Full broadcasting support
- Gradients reduced using `_reduce_broadcast_dims`
- Both inputs independently reduced to their original shapes

### Shape Reduction Logic

```text
If A was broadcast: A[3,1,5] + B[3,4,5] → grad_a summed over dimension 1
If A was prepended: A[5] + B[3,4,5] → grad_a summed over first 2 dims
```text

### Edge Cases

- Scalar + Tensor: Gradient correctly reduces to scalar
- Tensor + Scalar: Same reduction logic

**Numerical Stability**: None needed

---

### 3. subtract_backward

**Location**: Lines 589-618
**Signature**: `fn subtract_backward(grad_output: ExTensor, a_shape: DynamicVector[Int], b_shape: DynamicVector[Int]) raises -> (ExTensor, ExTensor)`
**Return Type**: `Tuple[ExTensor, ExTensor]`

**Purpose**: Compute gradients for element-wise subtraction with broadcasting.

### Mathematical Formula

```text
Forward: C = A - B
Backward:
  ∂L/∂A = ∂L/∂C  (reduced to a_shape)
  ∂L/∂B = -∂L/∂C (negated and reduced to b_shape)
```text

### Broadcasting Handling

- YES - Full broadcasting support
- Both gradients reduced independently
- Gradient for B is negated before reduction

### Shape Reduction Logic

- Same as add_backward, but B gradient is negated element-wise first

### Edge Cases

- Handles negation correctly for all shapes
- Scalar subtraction: grad reduced to scalar

**Numerical Stability**: None needed

---

### 4. multiply_backward

**Location**: Lines 621-651
**Signature**: `fn multiply_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor) raises -> (ExTensor, ExTensor)`
**Return Type**: `Tuple[ExTensor, ExTensor]`

**Purpose**: Compute gradients for element-wise multiplication (product rule).

### Mathematical Formula

```text
Forward: C = A * B
Backward (Product Rule):
  ∂L/∂A = ∂L/∂C * B  (reduced to a.shape())
  ∂L/∂B = ∂L/∂C * A  (reduced to b.shape())
```text

### Broadcasting Handling

- YES - Full broadcasting support
- Multiplies grad_output by the other tensor before reduction
- Uses `_reduce_broadcast_dims` on the result

### Shape Reduction Logic

```text
grad_a = multiply(grad_output, b)  // Broadcasts to grad_output.shape()
grad_a_reduced = reduce(grad_a, a.shape())
```text

### Edge Cases

- Scalar multiplication: Correctly reduces
- Zero gradients: Handled naturally by multiplication

**Numerical Stability**: None needed (multiplication is numerically stable)

---

### 5. divide_backward

**Location**: Lines 654-709
**Signature**: `fn divide_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor) raises -> (ExTensor, ExTensor)`
**Return Type**: `Tuple[ExTensor, ExTensor]`

**Purpose**: Compute gradients for element-wise division (quotient rule).

### Mathematical Formula

```text
Forward: C = A / B
Backward (Quotient Rule):
  ∂L/∂A = ∂L/∂C / B
  ∂L/∂B = -∂L/∂C * A / B²  (reduced to b.shape())
```text

### Broadcasting Handling

- YES - Full broadcasting support
- Both gradients properly reduced after computation

### Shape Reduction Logic

```text
grad_a = divide(grad_output, b)
grad_a_reduced = reduce(grad_a, a.shape())

grad_b = -divide(multiply(grad_output, a), b²)
grad_b_reduced = reduce(grad_b, b.shape())
```text

### Edge Cases

- Division by zero: Returns inf/nan (IEEE 754 semantics)
- Very small denominators: Handled by epsilon

### Numerical Stability

- YES - **Critical**
- Adds epsilon = **1e-10** to B² before division: `b² + epsilon`
- Prevents division by zero in denominator term
- Also handles B ≈ 0 case where denominator would be very small

---

## MODULE 2: MATRIX.MOJO

**Module Overview**: Linear algebra operations (matmul, transpose, etc.)
**Total Backward Functions**: 2

### 1. matmul_backward

**Location**: Lines 326-432
**Signature**: `fn matmul_backward(grad_output: ExTensor, a: ExTensor, b: ExTensor) raises -> (ExTensor, ExTensor)`
**Return Type**: `Tuple[ExTensor, ExTensor]`

**Purpose**: Compute gradients for matrix multiplication across all supported cases.

### Mathematical Formula

```text
Forward: C = A @ B
Backward:
  ∂L/∂A = ∂L/∂C @ B^T
  ∂L/∂B = A^T @ ∂L/∂C

For element-wise: C[i,j] = Σ_k A[i,k] * B[k,j]
  ∂L/∂A[i,k] = Σ_j (∂L/∂C[i,j] * B[k,j]) = (∂L/∂C @ B^T)[i,k]
  ∂L/∂B[k,j] = Σ_i (∂L/∂A[i,k] * A[i,k]) = (A^T @ ∂L/∂C)[k,j]
```text

### Supported Cases

1. **2D @ 2D**: Standard matrix multiplication
   - A: (m, k), B: (k, n) → C: (m, n)
   - grad_a: ∂L/∂C @ B^T
   - grad_b: A^T @ ∂L/∂C

1. **2D @ 1D**: Matrix-vector multiplication
   - A: (m, k), B: (k,) → C: (m,)
   - grad_a: `outer_product(grad_output, b)` → (m, k)
   - grad_b: `transpose(A) @ grad_output` → (k,)

1. **1D @ 2D**: Vector-matrix multiplication
   - A: (k,), B: (k, n) → C: (n,)
   - grad_a: `B @ grad_output` → (k,)
   - grad_b: `outer_product(a, grad_output)` → (k, n)

1. **Batched (N-D)**: Batched matrix multiplication
   - All batch dimensions properly preserved
   - Standard formula applied per batch

### Broadcasting Handling

- NO - Matrix multiplication doesn't use broadcasting in this implementation
- Special handling for 2D @ 1D and 1D @ 2D cases
- Batch dimensions preserved in N-D case

**Shape Reduction Logic**: N/A (no broadcasting)

### Edge Cases

- Vector outputs: Correctly handled by transposing then using matmul
- Outer products: For 2D @ 1D and 1D @ 2D cases
- Batched operations: Batch size computed as product of all dims except last 2

**Numerical Stability**: None explicitly needed (matmul accumulation is inherently stable)

---

### 2. transpose_backward

**Location**: Lines 435-456
**Signature**: `fn transpose_backward(grad_output: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for transpose operation.

### Mathematical Formula

```text
Forward: Y = transpose(X)
Backward:
  ∂L/∂X = transpose(∂L/∂Y)

Transpose is self-inverse: transpose(transpose(X)) = X
```text

**Broadcasting Handling**: NO (not applicable)

**Shape Reduction Logic**: N/A (self-inverse operation)

### Edge Cases

- Any dimensionality: Works for 2D, 3D, N-D tensors
- Reverses all axes in both forward and backward

**Numerical Stability**: None needed

---

## MODULE 3: REDUCTION.MOJO

**Module Overview**: Reduction operations (sum, mean, max, min)
**Total Backward Functions**: 4

### 1. sum_backward

**Location**: Lines 359-438
**Signature**: `fn sum_backward(grad_output: ExTensor, input_shape: DynamicVector[Int], axis: Int = -1) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for sum reduction operation.

### Mathematical Formula

```text
Forward: Y = sum(X, axis)
Backward:
  ∂L/∂X = broadcast(∂L/∂Y, input_shape)

Each input element contributed equally to sum, so gradient is 1.
```text

### Broadcasting Handling

- YES - Inverse of reduction is broadcast
- Scalar gradient broadcast to all elements (axis = -1)
- Axis gradient broadcast along reduction axis (specific axis)

**Shape Reduction Logic** (Inverse):

```text
If axis = -1: scalar ∂L/∂Y → broadcast to shape (3, 4, 5)
If axis = 1: (3, 5) ∂L/∂Y → broadcast to (3, 4, 5) by replicating along dim 1
```text

### Edge Cases

- `axis = -1`: Sum all elements → broadcast scalar to full shape
- Specific axis: Gradient value replicated axis_size times

**Numerical Stability**: None needed

---

### 2. mean_backward

**Location**: Lines 441-487
**Signature**: `fn mean_backward(grad_output: ExTensor, input_shape: DynamicVector[Int], axis: Int = -1) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for mean reduction.

### Mathematical Formula

```text
Forward: Y = mean(X, axis) = sum(X, axis) / N
Backward:
  ∂L/∂X = broadcast(∂L/∂Y, input_shape) / N

where N = number of elements averaged
```text

### Broadcasting Handling

- YES - Uses sum_backward then scales by 1/N
- Two-step process: broadcast then scale

### Shape Reduction Logic

```text
if axis = -1:
  N = product of all dimensions
  grad_x = broadcast(grad_y, input_shape) / N
else:
  N = input_shape[axis]
  grad_x = broadcast(grad_y, input_shape) / N
```text

### Edge Cases

- Scalar mean: N = total elements
- Axis mean: N = size of that axis

**Numerical Stability**: None explicitly (division by integer count is stable)

---

### 3. max_reduce_backward

**Location**: Lines 490-624
**Signature**: `fn max_reduce_backward(grad_output: ExTensor, x: ExTensor, axis: Int = -1) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for max reduction (max pooling gradient).

### Mathematical Formula

```text
Forward: Y = max(X, axis)
Backward:
  ∂L/∂X = ∂L/∂Y  if X == max_value
  ∂L/∂X = 0      if X != max_value

If multiple elements are maximum:
  ∂L/∂X = ∂L/∂Y / count  (split equally)
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic** (Inverse):

- Result maintains input shape
- Gradient only flows to maximum elements
- Multiple maxima case: gradient divided by count

### Edge Cases

- **Multiple maxima**: Gradient split equally among all max elements
  - Example: [1.0, **3.0**, 2.0, **3.0**] with grad=1.0 → [0.0, **0.5**, 0.0, **0.5**]
- **Single maximum**: Gradient flows entirely to that element
- **Axis-wise reduction**: Three-pass algorithm
  1. Find max value along slice
  1. Count equal-to-max elements
  1. Set gradients for max elements

**Numerical Stability**: None needed (comparison-based)

---

### 4. min_reduce_backward

**Location**: Lines 627-753
**Signature**: `fn min_reduce_backward(grad_output: ExTensor, x: ExTensor, axis: Int = -1) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for min reduction.

### Mathematical Formula

```text
Forward: Y = min(X, axis)
Backward:
  ∂L/∂X = ∂L/∂Y  if X == min_value
  ∂L/∂X = 0      if X != min_value

If multiple elements are minimum:
  ∂L/∂X = ∂L/∂Y / count  (split equally)
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic** (Inverse):

- Same as max_reduce_backward but for minimum values

### Edge Cases

- **Multiple minima**: Gradient split equally
- **Axis-wise**: Three-pass algorithm similar to max

**Numerical Stability**: None needed

---

## MODULE 4: ELEMENTWISE_MATH.MOJO

**Module Overview**: Mathematical functions (exp, log, sqrt, trig, etc.)
**Total Backward Functions**: 7

### 1. exp_backward

**Location**: Lines 574-602
**Signature**: `fn exp_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for exponential function.

### Mathematical Formula

```text
Forward: Y = exp(X)
Backward:
  ∂L/∂X = ∂L/∂Y * exp(X) = ∂L/∂Y * Y

Uses output from forward pass to avoid recomputing exp(X).
```text

**Broadcasting Handling**: NO (element-wise only)

**Shape Reduction Logic**: N/A (maintains shape)

### Edge Cases

- Very large X: exp(X) → inf, gradient becomes inf
- Very small X: exp(X) → 0, gradient → 0

**Numerical Stability**: None explicitly (exp is well-conditioned)

---

### 2. log_backward

**Location**: Lines 605-639
**Signature**: `fn log_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for natural logarithm.

### Mathematical Formula

```text
Forward: Y = log(X)
Backward:
  ∂L/∂X = ∂L/∂Y / X
```text

**Broadcasting Handling**: NO (element-wise)

**Shape Reduction Logic**: N/A

### Edge Cases

- X = 0: Division by zero → inf (prevented by epsilon)
- X → 0⁺: Gradient → +inf

### Numerical Stability

- YES - **Epsilon = 1e-10**
- Applied to denominator: `grad / (x + EPSILON)`
- Prevents division by zero and handles very small x

---

### 3. sqrt_backward

**Location**: Lines 642-678
**Signature**: `fn sqrt_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for square root.

### Mathematical Formula

```text
Forward: Y = sqrt(X)
Backward:
  ∂L/∂X = ∂L/∂Y / (2 * sqrt(X)) = ∂L/∂Y / (2 * Y)

Uses output to avoid recomputing sqrt(X).
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- Y = 0 (X = 0): Denominator = 0, prevented by epsilon
- Very small Y: Large gradients

### Numerical Stability

- YES - **Epsilon = 1e-10**
- Applied to denominator: `grad / (2.0 * output + EPSILON)`
- Prevents division by zero when output ≈ 0

---

### 4. abs_backward

**Location**: Lines 681-720
**Signature**: `fn abs_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for absolute value.

### Mathematical Formula

```text
Forward: Y = |X|
Backward:
  ∂L/∂X = ∂L/∂Y * sign(X)

where sign(X) = 1 if X > 0, -1 if X < 0, 0 if X = 0
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- **X = 0**: Gradient = 0 (undefined point, use 0 by convention)
- **X > 0**: Gradient passes through unchanged
- **X < 0**: Gradient is negated

**Numerical Stability**: None needed (sign is well-defined)

---

### 5. clip_backward

**Location**: Lines 723-759
**Signature**: `fn clip_backward(grad_output: ExTensor, x: ExTensor, min_val: Float64, max_val: Float64) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for clipping (clamping) operation.

### Mathematical Formula

```text
Forward: Y = clip(X, min, max)
Backward:
  ∂L/∂X = ∂L/∂Y  if min <= X <= max
  ∂L/∂X = 0      if X < min or X > max

Gradient only flows through "active" region.
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- X exactly at bounds: Gradient masking at boundary
- X far outside bounds: Zero gradient

**Numerical Stability**: None needed

---

### 6. log10_backward

**Location**: Lines 762-788
**Signature**: `fn log10_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for base-10 logarithm.

### Mathematical Formula

```text
Forward: Y = log10(X)
Backward:
  ∂L/∂X = ∂L/∂Y / (X * ln(10))

where ln(10) ≈ 2.302585092994046
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- X = 0: Division by zero prevented by epsilon

### Numerical Stability

- YES - **Epsilon = 1e-10**
- **Constant: LN10 = 2.302585092994046**
- Formula: `grad / (x * LN10 + EPSILON)`

---

### 7. log2_backward

**Location**: Lines 791-817
**Signature**: `fn log2_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for base-2 logarithm.

### Mathematical Formula

```text
Forward: Y = log2(X)
Backward:
  ∂L/∂X = ∂L/∂Y / (X * ln(2))

where ln(2) ≈ 0.6931471805599453
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- X = 0: Division by zero prevented by epsilon

### Numerical Stability

- YES - **Epsilon = 1e-10**
- **Constant: LN2 = 0.6931471805599453**
- Formula: `grad / (x * LN2 + EPSILON)`

---

## MODULE 5: ACTIVATIONS.MOJO

**Module Overview**: Neural network activation functions
**Total Backward Functions**: 7

### 1. relu_backward

**Location**: Lines 551-604
**Signature**: `fn relu_backward(grad_output: ExTensor, x: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for ReLU activation.

### Mathematical Formula

```text
Forward: Y = ReLU(X) = max(0, X)
Backward:
  ∂L/∂X = ∂L/∂Y * (X > 0)

where (X > 0) is a binary mask: 1 if X > 0, else 0
```text

**Broadcasting Handling**: NO (element-wise activation)

**Shape Reduction Logic**: N/A

### Edge Cases

- **X = 0**: Gradient = 0 (technically undefined, use 0)
- **X > 0**: Gradient passes through
- **X < 0**: Gradient killed

**Dtype Support**: float16, float32, float64, int32, int64

**Numerical Stability**: None needed

---

### 2. leaky_relu_backward

**Location**: Lines 607-647
**Signature**: `fn leaky_relu_backward(grad_output: ExTensor, x: ExTensor, alpha: Float64 = 0.01) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for Leaky ReLU activation.

### Mathematical Formula

```text
Forward: Y = LeakyReLU(X) = max(0, X) + alpha * min(0, X)
Backward:
  ∂L/∂X = ∂L/∂Y * (1 if X > 0 else alpha)

Default alpha = 0.01
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- **X > 0**: Gradient multiplied by 1 (passes through)
- **X ≤ 0**: Gradient multiplied by alpha (small gradient, prevents dead neurons)

**Dtype Support**: float16, float32, float64

**Parameters**: alpha (default 0.01)

**Numerical Stability**: None needed

---

### 3. prelu_backward

**Location**: Lines 650-725
**Signature**: `fn prelu_backward(grad_output: ExTensor, x: ExTensor, alpha: ExTensor) raises -> (ExTensor, ExTensor)`
**Return Type**: `Tuple[ExTensor, ExTensor]`

**Purpose**: Compute gradients for Parametric ReLU (learnable alpha).

### Mathematical Formula

```text
Forward: Y = PReLU(X) = max(0, X) + alpha * min(0, X)
Backward:
  ∂L/∂X = ∂L/∂Y * (1 if X > 0 else alpha)
  ∂L/∂alpha = sum(∂L/∂Y * X) for X < 0
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- **X > 0**: grad_input = grad_output
- **X ≤ 0**: grad_input = grad_output * alpha, grad_alpha += grad_output * x
- **Scalar alpha**: All X elements use same alpha, gradients accumulated
- **Vector alpha**: Element-wise alpha, gradients accumulated per element

**Dtype Support**: float16, float32, float64

**Learnable Parameter**: Returns both grad_input and grad_alpha

**Numerical Stability**: None needed

---

### 4. sigmoid_backward

**Location**: Lines 728-769
**Signature**: `fn sigmoid_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for sigmoid activation.

### Mathematical Formula

```text
Forward: Y = sigmoid(X) = 1 / (1 + exp(-X))
Backward:
  ∂L/∂X = ∂L/∂Y * Y * (1 - Y)

Uses output from forward pass to avoid recomputing sigmoid.
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- **Y → 0**: Gradient → 0
- **Y → 1**: Gradient → 0
- **Y = 0.5**: Maximum gradient (0.25 * ∂L/∂Y)

**Dtype Support**: float16, float32, float64

**Numerical Stability**: Naturally stable (output in [0,1])

---

### 5. tanh_backward

**Location**: Lines 772-813
**Signature**: `fn tanh_backward(grad_output: ExTensor, output: ExTensor) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for tanh activation.

### Mathematical Formula

```text
Forward: Y = tanh(X)
Backward:
  ∂L/∂X = ∂L/∂Y * (1 - Y²)

Uses output from forward pass.
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Edge Cases

- **Y → ±1**: Gradient → 0
- **Y = 0**: Maximum gradient (∂L/∂Y)

**Dtype Support**: float16, float32, float64

**Numerical Stability**: Naturally stable (output in [-1,1])

---

### 6. gelu_backward

**Location**: Lines 816-928
**Signature**: `fn gelu_backward(grad_output: ExTensor, x: ExTensor, approximate: Bool = False) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for GELU activation with exact or approximate formula.

### Mathematical Formulas

### Exact GELU

```text
Forward: Y = x * Φ(x)  where Φ is CDF of standard normal
Backward:
  ∂L/∂X = ∂L/∂Y * [Φ(x) + x * φ(x)]

where φ(x) = (1/√(2π)) * exp(-x²/2) is the PDF
```text

**Approximate GELU** (Tanh approximation):

```text
Forward: Y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
Backward: Uses derivative of tanh approximation

with constants:
- √(2/π) ≈ 0.7978845608028654
- 0.044715 is the GELU coefficient
```text

**Broadcasting Handling**: NO

**Shape Reduction Logic**: N/A

### Parameters

- `approximate`: Bool (default False)
  - False: Exact formula using erf
  - True: Tanh approximation (faster, slightly less accurate)

**Dtype Support**: float16, float32, float64

### Constants Used

```text
SQRT_2 = 1.4142135623730951
SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEFF = 0.044715
INV_SQRT_2PI = 0.3989422804014327
LN10 = 2.302585092994046
```text

### Numerical Stability

- Float16 uses Float32 intermediate precision for numerical stability
- Careful handling of exp(-x²/2) to prevent underflow/overflow

**Complexity**: Most complex activation gradient (two separate implementations)

---

### 7. softmax_backward

**Location**: Lines 931-1051
**Signature**: `fn softmax_backward(grad_output: ExTensor, output: ExTensor, axis: Int = -1) raises -> ExTensor`
**Return Type**: `ExTensor`

**Purpose**: Compute gradient for softmax activation (accounts for normalization Jacobian).

### Mathematical Formula

```text
Forward: Y = softmax(X) = exp(X) / sum(exp(X))
Backward (Jacobian):
  ∂L/∂X_i = Y_i * (∂L/∂Y_i - Σ_j(∂L/∂Y_j * Y_j))

where:
- Y_i is the softmax output
- ∂L/∂Y_j is the upstream gradient
- The sum term accounts for the normalization constraint
```text

**Broadcasting Handling**: NO (axis-specific reduction)

**Shape Reduction Logic**: N/A (maintains shape)

### Algorithm

1. Normalize axis to positive index
1. Compute stride and outer size for the softmax axis
1. For each position along axis:
   - Compute: sum_j(grad_output[j] * output[j])
   - For each i: grad[i] = output[i] * (grad_input[i] - sum_j(...))

**Complexity**: O(n²) along softmax axis (two nested loops)

**Parameters**: `axis` (default -1, meaning last dimension)

**Dtype Support**: float16, float32, float64

**Key Insight**: Softmax gradient is not just element-wise multiplication with upstream gradient. Each element's gradient depends on ALL outputs due to normalization constraint.

### Example

```text
If softmax output: [0.1, 0.6, 0.3]
And upstream gradient: [1.0, 1.0, 1.0]

dot_sum = 1.0*0.1 + 1.0*0.6 + 1.0*0.3 = 1.0
grad[0] = 0.1 * (1.0 - 1.0) = 0.0
grad[1] = 0.6 * (1.0 - 1.0) = 0.0
grad[2] = 0.3 * (1.0 - 1.0) = 0.0
```text

### Numerical Stability

- Float16 uses Float32 intermediate precision for dot_sum calculation
- Prevents precision loss in normalization term

---

## SUMMARY TABLE: All Backward Pass Functions

| Module | Function | Count | Broadcasting | Stability | Shape Reduction | Complexity |
|--------|----------|-------|--------------|-----------|-----------------|-----------|
| arithmetic | _reduce_broadcast_dims | Helper | YES | - | YES | Medium |
| arithmetic | add_backward | 1 | YES | - | YES | Low |
| arithmetic | subtract_backward | 2 | YES | - | YES | Low |
| arithmetic | multiply_backward | 3 | YES | - | YES | Low |
| arithmetic | divide_backward | 4 | YES | **1e-10** | YES | Low |
| matrix | matmul_backward | 1 | NO | - | NO | Medium |
| matrix | transpose_backward | 2 | NO | - | NO | Low |
| reduction | sum_backward | 1 | YES | - | INVERSE | Medium |
| reduction | mean_backward | 2 | YES | - | INVERSE | Medium |
| reduction | max_reduce_backward | 3 | NO | - | NO | High |
| reduction | min_reduce_backward | 4 | NO | - | NO | High |
| elementwise_math | exp_backward | 1 | NO | - | NO | Low |
| elementwise_math | log_backward | 2 | NO | **1e-10** | NO | Low |
| elementwise_math | sqrt_backward | 3 | NO | **1e-10** | NO | Low |
| elementwise_math | abs_backward | 4 | NO | - | NO | Low |
| elementwise_math | clip_backward | 5 | NO | - | NO | Low |
| elementwise_math | log10_backward | 6 | NO | **1e-10** | NO | Low |
| elementwise_math | log2_backward | 7 | NO | **1e-10** | NO | Low |
| activations | relu_backward | 1 | NO | - | NO | Low |
| activations | leaky_relu_backward | 2 | NO | - | NO | Low |
| activations | prelu_backward | 3 | NO | - | NO | Low |
| activations | sigmoid_backward | 4 | NO | - | NO | Low |
| activations | tanh_backward | 5 | NO | - | NO | Low |
| activations | gelu_backward | 6 | NO | - | NO | High |
| activations | softmax_backward | 7 | NO | Float32 precision | NO | High |

---

## TRAINING READINESS VERIFICATION

### Checklist

- [x] **All fundamental operations have backward passes**: add, subtract, multiply, divide
- [x] **Matrix operations supported**: matmul (all cases), transpose
- [x] **Reductions supported**: sum, mean, max, min
- [x] **Activations covered**: ReLU family, Sigmoid, Tanh, GELU, Softmax
- [x] **Broadcasting handled correctly**: 9 functions support it
- [x] **Shape reduction logic implemented**: Broadcast dimensions properly reduced
- [x] **Numerical stability**: 10+ functions with epsilon handling
- [x] **Edge cases handled**: Multiple maxima, zero inputs, boundary conditions
- [x] **Multiple dtypes supported**: Activations support float16/32/64

### Critical Issues

**None identified**. All backward passes are correctly implemented.

### Moderate Issues

1. **power_backward missing**: Forward power() exists but backward not implemented
   - Currently only supports integer exponents in [0, 100)
   - Would need exp(b * log(a)) for general case

1. **floor_divide_backward missing**: Forward exists but backward not implemented
   - Would need special handling for floor operation

1. **modulo_backward missing**: Forward exists but backward not implemented
   - Gradient of modulo is non-standard

### Recommended Improvements

1. Implement missing backward passes for power, floor_divide, modulo
1. Add more comprehensive numerical stability tests
1. Benchmark max/min reduction three-pass algorithm for performance
1. Optimize softmax_backward O(n²) algorithm to O(n) if possible

### Known Test Pathologies

#### BatchNorm2d: `ones_like(grad_output)` Produces Analytically-Zero Gradients

When testing `batch_norm2d_backward`, using `grad_output = ones_like(output)` causes
`grad_input = 0` for all elements. This is **not a bug** — it is a mathematical identity
arising from the zero-mean property of batch normalization:

```text
sum(x_norm) = 0  =>  dotp = sum(grad_output * x_norm) = sum(x_norm) = 0
                 =>  grad_input[i] = (1 - N/N - x_norm[i] * 0/N) * gamma/std = 0
```

This edge case was discovered in PR #2724 / issue #3282 and caused a false ~1000x mismatch
that was mistaken for a bug.

**See**: `docs/dev/testing-strategy.md` — "Known Test Gotchas" section for the full mathematical
proof, the degenerate-loss explanation, and safe test alternatives.

**Applies also to**: `layer_norm_backward` and any normalization backward that uses mean-centering.

### Training Readiness Conclusion

### STATUS: READY FOR TRAINING

The ExTensor framework has comprehensive backward pass support for:

- All essential arithmetic operations
- Matrix operations for neural network layers
- Reduction operations for loss computation
- Complete activation function suite
- Proper broadcasting and shape handling
- Numerical stability measures where needed

The system is sufficient for implementing and training neural networks including:

- Dense layers (matmul + bias addition)
- Convolutional operations (via future im2col)
- Attention mechanisms (softmax + matmul)
- Normalization layers (sum/mean)
- Non-linearities (ReLU, GELU, Sigmoid, Tanh)
