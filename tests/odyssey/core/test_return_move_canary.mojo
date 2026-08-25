"""Permanent canary for the FM-A return-move premature-`__deinit__` class (modular/modular#6939).

The historical failure: a struct with a custom `deinit move` constructor and
shared-refcount ownership, when **returned by value from an owned local**
(`return r^`) or constructed inside a return (`return Pair(a^, b^)`), has its
moved-from source's `__deinit__` run by the 1.0.0 compiler -> the shared
refcount is decremented twice for one transfer -> premature free ->
use-after-free reads (garbage / NaN) in the returned value.

The minimal raw-pointer repro (docs/dev/reproducers/repro_uaf_return_move.mojo)
still fails on 1.0.0 stable and only shows deterministically under ASAN; these
shapes use the real `AnyTensor` (refcount cell + List fields) so a double
decrement frees the buffer while the destination is still alive. Values are
asserted exactly; on a regressed toolchain the freed blocks get reused and the
assertion trips (the same mechanism behind the FM-A / FM-E CI flakes).

Mirrors the exact in-repo shapes:
- whole-local return: `batchnorm.mojo:152` (`return output^`), `ssm.mojo:238`,
  `mamba.mojo:347`, `jvp.mojo:141/162`
- construction-in-return: `return GradientPair(grad_a^, grad_b^)`
  (`arithmetic.mojo:447...`), `return Tuple(t1^, t2^, t3^)`
  (`normalization_simd.mojo:319/474/596`)
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full
from odyssey.core.gradient_types import GradientPair


def make_whole_local() raises -> AnyTensor:
    """Shape: `return output^` on an owned local (batchnorm/mamba/ssm)."""
    var output = full([64, 16], 1.5, DType.float32)
    return output^


def make_gradient_pair() raises -> GradientPair:
    """Shape: `return GradientPair(grad_a^, grad_b^)` (arithmetic/linear)."""
    var grad_a = full([8, 8], 2.0, DType.float32)
    var grad_b = full([8, 8], 3.0, DType.float32)
    return GradientPair(grad_a^, grad_b^)


def make_triple() raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Shape: `return Tuple[AnyTensor, AnyTensor, AnyTensor](t1^, t2^, t3^)`
    (normalization_simd fused-training returns)."""
    var out = full([2, 4], 0.5, DType.float32)
    var mean = full([4], 0.25, DType.float32)
    var running_var = full([4], 0.75, DType.float32)
    return Tuple[AnyTensor, AnyTensor, AnyTensor](out^, mean^, running_var^)


def check_tensor(t: AnyTensor, expected: Float32) -> Bool:
    """All elements must equal `expected` exactly (heap reuse shows garbage)."""
    var ok = True
    for i in range(t.numel()):
        var v = t.load[DType.float32](i)
        if v != expected:
            ok = False
    return ok


def test_return_move_shapes() raises:
    # multiple iterations: FM-A is heap-reuse dependent, so a regressed
    # compiler may only corrupt when the freed block is reallocated
    for i in range(25):
        var t = make_whole_local()
        if not check_tensor(t, 1.5):
            raise Error(
                "FM-A whole-local return corrupted: shared-refcount "
                + "double decrement freed the buffer (iteration #"
                + String(i)
                + ")"
            )
        var pr = make_gradient_pair()
        if not check_tensor(pr.grad_a, 2.0) or not check_tensor(pr.grad_b, 3.0):
            raise Error(
                "FM-A GradientPair return corrupted (#6939): "
                + "moved-from source deinit over-decremented the refcount"
            )
        var (out, mean, running_var) = make_triple()
        if (
            not check_tensor(out, 0.5)
            or not check_tensor(mean, 0.25)
            or not check_tensor(running_var, 0.75)
        ):
            raise Error(
                "FM-A tuple-at-return corrupted (#6939): "
                + "BN-shaped triple return hit a premature free"
            )


def main() raises:
    print("Running FM-A return-move canary (modular/modular#6939)...")
    test_return_move_shapes()
    print("PASS: FM-A return-move canary")
