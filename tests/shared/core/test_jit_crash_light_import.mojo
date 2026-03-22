"""JIT crash reproduction: light import via targeted submodule.

This file uses targeted submodule imports which only compile ~500 lines.
Should never crash even after 100+ runs.

See docs/dev/mojo-jit-crash-workaround.md for context.
"""
from shared.core.any_tensor import AnyTensor, zeros
from testing import assert_true


fn test_basic_tensor_creation() raises:
    var t = zeros([2, 3], DType.float32)
    assert_true(t.shape()[0] == 2, "shape[0] should be 2")
    assert_true(t.shape()[1] == 3, "shape[1] should be 3")


fn main() raises:
    print("Running JIT crash light import test...")
    test_basic_tensor_creation()
    print("PASS")
