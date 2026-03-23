"""JIT crash reproduction: heavy import via shared.core (package-level).

This file intentionally uses `from shared.core import` which forces the JIT
to compile all 37K+ lines across 60+ modules via __init__.mojo.
Run 30+ times to observe intermittent 'execution crashed'.

See docs/dev/mojo-jit-crash-workaround.md for context.
"""
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core import relu, sigmoid, matmul, softmax
from testing import assert_true


fn test_basic_tensor_creation() raises:
    var t = zeros([2, 3], DType.float32)
    assert_true(t.shape()[0] == 2, "shape[0] should be 2")
    assert_true(t.shape()[1] == 3, "shape[1] should be 3")


fn main() raises:
    print("Running JIT crash heavy import test...")
    test_basic_tensor_creation()
    print("PASS")
