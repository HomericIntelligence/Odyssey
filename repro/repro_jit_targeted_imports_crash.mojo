# Reproducer for JIT crash with targeted submodule imports.
#
# Background: ADR-015 action #1 converted package-level imports to
# targeted submodule imports to reduce JIT compilation volume. However,
# crash pattern libKGENCompilerRTShared.so+0x6d4ab/+0x6a686/+0x6e157
# is still observed intermittently on CI even after the conversion.
#
# This reproducer attempts to trigger the crash using only targeted imports
# (no package-level from projectodyssey.core import / from projectodyssey import) to isolate
# whether the crash is import-volume-triggered or something else.
#
# Environment: Ubuntu 24.04, GLIBC 2.39, Mojo 0.26.3 (dev2026040705)
# Runner UID: 1001 (GitHub Actions standard runner)
#
# To run: mojo repro/repro_jit_targeted_imports_crash.mojo
# Expected (bug): error: execution crashed (before any output)
# Expected (fixed): prints test results normally

from std.testing import assert_true
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
from projectodyssey.core.layers.linear import Linear
from projectodyssey.core.layers.conv2d import Conv2dLayer
from projectodyssey.core.layers.dropout import DropoutLayer
from projectodyssey.core.layers.relu import ReLULayer
from projectodyssey.core.activation import relu, sigmoid
from projectodyssey.core.matrix import matmul
from projectodyssey.core.shape import as_contiguous, reshape


fn test_linear() raises:
    print("test_linear: start")
    var t = zeros([4, 8], DType.float32)
    assert_true(t.shape()[0] == 4, "shape[0] should be 4")
    print("test_linear: pass")


fn test_conv2d() raises:
    print("test_conv2d: start")
    var t = zeros([1, 3, 8, 8], DType.float32)
    assert_true(t.shape()[1] == 3, "channels should be 3")
    print("test_conv2d: pass")


fn test_dropout() raises:
    print("test_dropout: start")
    var t = ones([2, 4], DType.float32)
    assert_true(t.shape()[0] == 2, "shape[0] should be 2")
    print("test_dropout: pass")


fn test_relu_activation() raises:
    print("test_relu_activation: start")
    var t = zeros([3, 3], DType.float32)
    var out = relu(t)
    assert_true(out.shape()[0] == 3, "relu output shape[0] should be 3")
    print("test_relu_activation: pass")


fn test_reshape() raises:
    print("test_reshape: start")
    var t = zeros([2, 6], DType.float32)
    var r = reshape(t, [3, 4])
    assert_true(r.shape()[0] == 3, "reshaped shape[0] should be 3")
    print("test_reshape: pass")


fn main() raises:
    print("=== repro_jit_targeted_imports_crash: start ===")
    test_linear()
    test_conv2d()
    test_dropout()
    test_relu_activation()
    test_reshape()
    print("=== All reproducer tests passed ===")
