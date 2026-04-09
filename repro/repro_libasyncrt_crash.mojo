"""Mojo v0.26.1 runtime crash: heap corruption via cumulative conv2d operations.

Environment: Mojo 0.26.1 (156d3ac6), GLIBC 2.39, Linux 6.6.87 x86_64 (WSL2)

This is a variant of the heap corruption crash that demonstrates the bug
through cumulative tensor operations rather than bitcast writes.

In this variant, the VGG16-style forward pass (13 conv layers + 3 FC layers)
is called multiple times in separate functions. After ~3 function-scoped
forward passes, the allocator state becomes corrupted and the next forward
pass that also creates additional small tensors (like training targets)
triggers a crash.

This is the SAME underlying bug as repro_libkgen_crash.mojo — heap
corruption in the Mojo runtime allocator — but exposed through a more
realistic deep learning workflow: repeated forward passes in a training loop.

Key observations:
- Forward pass alone can be called 20+ times in a loop without crashing
- Crash requires function-scoped calls (tensor destructor ordering matters)
- Crash requires creating a small tensor with bitcast write between forward passes
- Same stack trace as repro_libkgen_crash.mojo (same root cause)

Stack trace (constant across runs):
  #0 libKGENCompilerRTShared.so +0x3cb78b  (crash handler)
  #1 libKGENCompilerRTShared.so +0x3c93c6  (crash handler)
  #2 libKGENCompilerRTShared.so +0x3cc397  (crash handler)
  #3 libc.so.6                  +0x45330   (sigaction)
  #4 libAsyncRTRuntimeGlobals.so +0x416ba  (allocator — crash origin)
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.conv import conv2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.pooling import maxpool2d


def conv_block(
    input_tensor: AnyTensor,
    out_channels: Int,
    num_convs: Int,
) raises -> AnyTensor:
    """VGG-style conv block: N sequential conv+relu layers."""
    var in_channels = input_tensor.shape()[1]
    var result = input_tensor
    for _ in range(num_convs):
        var ks = List[Int]()
        ks.append(out_channels)
        ks.append(in_channels)
        ks.append(3)
        ks.append(3)
        var kernel = ones(ks, DType.float32)
        var bs = List[Int]()
        bs.append(out_channels)
        var bias = zeros(bs, DType.float32)
        result = conv2d(result, kernel, bias, 1, 1)
        result = relu(result)
        in_channels = out_channels
    return result


def vgg16_forward(input_tensor: AnyTensor) raises -> AnyTensor:
    """Full VGG-16 forward pass: 13 conv layers + 3 FC layers."""
    var x = input_tensor
    x = conv_block(x, 64, 2)
    x = maxpool2d(x, 2, 2)
    x = conv_block(x, 128, 2)
    x = maxpool2d(x, 2, 2)
    x = conv_block(x, 256, 3)
    x = maxpool2d(x, 2, 2)
    x = conv_block(x, 512, 3)
    x = maxpool2d(x, 2, 2)
    x = conv_block(x, 512, 3)
    x = maxpool2d(x, 2, 2)

    var batch_size = x.shape()[0]
    var fs = List[Int]()
    fs.append(batch_size)
    fs.append(512)
    var x_flat = x.reshape(fs)

    # FC layers
    var w1 = List[Int]()
    w1.append(256)
    w1.append(512)
    var b1 = List[Int]()
    b1.append(256)
    x = linear(x_flat, ones(w1, DType.float32), zeros(b1, DType.float32))
    x = relu(x)
    var w2 = List[Int]()
    w2.append(256)
    w2.append(256)
    x = linear(x, ones(w2, DType.float32), zeros(b1, DType.float32))
    x = relu(x)
    var w3 = List[Int]()
    w3.append(10)
    w3.append(256)
    var b3 = List[Int]()
    b3.append(10)
    x = linear(x, ones(w3, DType.float32), zeros(b3, DType.float32))
    return x


def make_input(batch_size: Int) raises -> AnyTensor:
    var s = List[Int]()
    s.append(batch_size)
    s.append(3)
    s.append(32)
    s.append(32)
    return ones(s, DType.float32)


def test_inference(batch_size: Int) raises:
    """Inference test: forward pass + shape check."""
    var output = vgg16_forward(make_input(batch_size))
    var s = output.shape()
    if s[0] != batch_size or s[1] != 10:
        raise "bad shape"


def test_training_step() raises:
    """Training step: create targets with bitcast, then forward pass.

    This function triggers the crash because it:
    1. Creates a small target tensor
    2. Writes to it via bitcast (corrupts heap metadata)
    3. Calls vgg16_forward which does a large allocation -> CRASH
    """
    var input = make_input(2)

    # Create training targets with bitcast write
    var ts = List[Int]()
    ts.append(2)
    var target = zeros(ts, DType.float32)
    var td = target._data.bitcast[Float32]()
    td[0] = 0.0  # Class 0
    td[1] = 1.0  # Class 1

    # Forward pass -> crashes here
    var logits = vgg16_forward(input)
    if logits.shape()[0] != 2 or logits.shape()[1] != 10:
        raise "bad shape"


def main() raises:
    # These inference tests pass fine
    print("Inference batch=4...", end="")
    test_inference(4)
    print(" OK")

    print("Inference batch=2...", end="")
    test_inference(2)
    print(" OK")

    # This training step crashes
    print("Training step...", end="")
    test_training_step()
    print(" OK — no crash (bug may be environment-dependent)")
