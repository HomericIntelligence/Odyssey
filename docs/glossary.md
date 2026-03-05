# Glossary

This glossary defines technical terms used throughout the ML Odyssey codebase. Use this
reference to understand unfamiliar concepts and standardize terminology across the project.

## General ML Terms

**Autograd**: Automatic differentiation - computing gradients automatically by tracking
operations and applying the chain rule in reverse.

**Backpropagation**: Algorithm for computing gradients via the chain rule, propagating
error signals backward from output to input layers.

**Epoch**: One complete pass through the entire training dataset during model training.

**Gradient**: Derivative indicating the direction and rate of steepest increase for a
function, used to update model weights.

**Loss Function**: Measures the difference between model prediction and target value,
guiding the optimization process.

**Optimizer**: Algorithm for updating model weights based on gradients (e.g., SGD, Adam,
RMSprop).

**Tensor**: Multi-dimensional array of numbers, the fundamental data structure in deep
learning.

## Architecture Terms

**Activation Function**: Non-linear function applied to layer output to introduce
non-linearity (e.g., ReLU, Sigmoid, Tanh, Softmax).

**Batch Normalization**: Technique that normalizes layer inputs across a mini-batch,
improving training stability and speed.

**Convolution**: Sliding window operation that detects spatial patterns by applying learned
filters across input dimensions.

**Dropout**: Regularization technique that randomly zeros activations during training to
prevent overfitting.

**Fully Connected Layer**: Layer where each neuron connects to all neurons in the previous
layer, also called "dense" or "linear" layer.

**Pooling**: Downsampling operation that reduces spatial dimensions while preserving
important features (max pooling, average pooling).

## Training Terms

**Batch Size**: Number of training examples processed together before updating weights.

**Learning Rate**: Step size for gradient descent, controlling how much weights change per
update.

**Momentum**: Acceleration term for gradient descent that helps escape local minima and
smooths updates.

**Overfitting**: When a model memorizes training data patterns rather than learning
generalizable features, resulting in poor performance on new data.

**Regularization**: Techniques to prevent overfitting by constraining model complexity
(L1, L2 penalties, dropout, early stopping).

**Underfitting**: When a model is too simple to capture underlying patterns in the data.

**Validation Set**: Held-out data used to tune hyperparameters and monitor training
progress without contaminating test evaluation.

## Implementation Terms

**Broadcasting**: Automatically expanding tensor shapes to make element-wise operations
compatible between tensors of different dimensions.

**Contiguous**: Tensor data stored in row-major order in memory, enabling efficient
sequential access and SIMD operations.

**DType**: Data type specifying numeric precision and representation (float32, float64,
int32, bfloat16, etc.).

**In-place**: Operation that modifies a tensor's data directly without creating a copy,
saving memory but potentially breaking gradient computation.

**SIMD**: Single Instruction Multiple Data - parallel computation technique that processes
multiple data elements with a single instruction.

**Stride**: Step size between consecutive elements in each dimension, determining memory
access patterns for tensor operations.

**View**: Tensor that shares underlying data with another tensor but may have different
shape or strides, enabling zero-copy reshaping.

## Mojo-Specific Terms

**Borrowed**: Reference to a value without transferring ownership, enabling read access
while the original owner retains control.

**Owned**: Value with exclusive ownership that can be modified or transferred.

**Parameter Convention**: Keywords (`out`, `mut`, `read`) specifying how function
parameters handle ownership and mutability.

**Trait**: Interface defining required methods and behaviors that a struct must implement.

**Transfer Operator (^)**: The `^` operator transfers ownership of a value, consuming the
source and enabling the recipient to take ownership.

## Autograd Terms

**Backward Pass**: Phase of training where gradients are computed by propagating error
signals from the loss function back to input layers.

**Computational Graph**: Directed acyclic graph (DAG) representing the sequence of
operations performed during the forward pass.

**Forward Pass**: Phase where input data flows through the network to produce output
predictions.

**Gradient Tape**: Mechanism that records operations during forward pass to enable
automatic differentiation during backward pass.

**No-Grad Context**: Mode that disables gradient tracking, used during inference to save
memory and computation.

**Requires Grad**: Flag indicating that a tensor should track operations for gradient
computation.

## See Also

- [PyTorch Glossary](https://pytorch.org/docs/stable/glossary.html) - Similar concepts
  in PyTorch
