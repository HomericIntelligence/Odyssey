"""Byte-identical equivalence tests for the 5 core OO optimizer delegators.

For each of `SGD`, `Adam`, `AdamW`, `AdaGrad`, `RMSprop` this test:

    1. Constructs identical `(param, grad, state)` inputs.
    2. Runs the canonical functional step in
       `odyssey.training.optimizers.<name>` — returns updated param.
    3. Runs the OO wrapper `opt.step(params, tape)` from
       `odyssey.autograd.optimizers_oo.<name>` on a fresh
       `Variable + GradientTape` setup with the *same* initial values.
    4. Asserts the resulting parameter tensors are byte-identical
       element-wise (zero tolerance).

This locks in the *no behavioral drift* claim of the single-source-of-truth
refactor — any future divergence between the canonical functional step and
its OO wrapper is caught here immediately.

Reference numbers were transcribed from the same fixed-input vectors used
by the per-optimizer parity tests
(`tests/odyssey/training/optimizers/test_{sgd,adam,adamw,adagrad,rmsprop}.mojo`)
so both test suites stay in lockstep on shared fixtures.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.autograd.variable import Variable
from odyssey.autograd.tape import GradientTape
from odyssey.autograd.optimizer_base import Optimizer
from odyssey.autograd.optimizers_oo.sgd import SGD
from odyssey.autograd.optimizers_oo.adam import Adam
from odyssey.autograd.optimizers_oo.adamw import AdamW
from odyssey.autograd.optimizers_oo.adagrad import AdaGrad
from odyssey.autograd.optimizers_oo.rmsprop import RMSprop
from odyssey.training.optimizers.sgd import sgd_step
from odyssey.training.optimizers.adam import adam_step
from odyssey.training.optimizers.adamw import adamw_step
from odyssey.training.optimizers.adagrad import adagrad_step
from odyssey.training.optimizers.rmsprop import rmsprop_step


# ============================================================================
# Local helpers
# ============================================================================


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _allocate_filled(
    n: Int, values: List[Float64], dtype: DType
) raises -> AnyTensor:
    """Allocate an AnyTensor of length `n` and fill it with `values`.

    The returned tensor is owned (no borrow).
    """
    var t = zeros([n], dtype)
    for i in range(n):
        t.store[DType.float64](i, values[i])
    return t^


def _assert_tensors_byte_equal(
    expected: AnyTensor,
    actual: AnyTensor,
    label: String,
) raises:
    """Element-wise byte-identical assertion (zero tolerance)."""
    if expected.numel() != actual.numel():
        raise Error(
            label
            + " numel mismatch: expected "
            + String(expected.numel())
            + " got "
            + String(actual.numel())
        )
    for i in range(expected.numel()):
        var e = expected.load[DType.float64](i)
        var a = actual.load[DType.float64](i)
        if _abs_diff(e, a) > 0.0:
            raise Error(
                label
                + " mismatch at index "
                + String(i)
                + ": expected "
                + String(e)
                + ", got "
                + String(a)
            )


def _run_oo_step[
    OOType: Optimizer
](mut opt: OOType, p_init: AnyTensor, g_init: AnyTensor,) raises -> AnyTensor:
    """Run `opt.step([variable])` on a fresh Variable+GradientTape.

    Returns the updated parameter data tensor.
    """
    var tape = GradientTape()
    tape.enable()
    var pv = Variable(p_init^, True, tape)
    var pid = pv.id
    var params: List[Variable] = []
    params.append(pv.copy())
    tape.registry.set_grad(pid, g_init^)
    opt.step(params, tape)
    return params[0].data


# ============================================================================
# Tests
# ============================================================================


def test_sgd_oo_matches_canonical_with_momentum() raises:
    """SGD with momentum: OO `SGD.step()` == canonical `sgd_step`."""
    print("Running test_sgd_oo_matches_canonical_with_momentum...")
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    p_vals.append(-0.40)
    p_vals.append(0.50)
    p_vals.append(-0.60)
    var g_vals = List[Float64]()
    g_vals.append(0.02)
    g_vals.append(-0.03)
    g_vals.append(0.015)
    g_vals.append(0.025)
    g_vals.append(-0.01)
    g_vals.append(0.04)
    var lr = 0.1
    var momentum = 0.9
    var wd = 0.0

    # Functional canonical
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var v_func = zeros([6], DType.float64)
    var out_func = sgd_step(p_func, g_func, v_func, lr, momentum, wd)
    var p_func_out = out_func[0]

    # OO delegator (fresh, owned inputs)
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = SGD(learning_rate=lr, momentum=momentum, weight_decay=wd)
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "SGD(m)")
    print("  ok SGD(m) byte-identical to canonical sgd_step")
    print("test_sgd_oo_matches_canonical_with_momentum PASSED")


def test_sgd_oo_matches_canonical_plain() raises:
    """SGD with momentum=0: OO `SGD.step()` == canonical `sgd_step`."""
    print("Running test_sgd_oo_matches_canonical_plain...")
    var p_vals = List[Float64]()
    p_vals.append(1.0)
    p_vals.append(1.0)
    p_vals.append(1.0)
    var g_vals = List[Float64]()
    g_vals.append(0.5)
    g_vals.append(0.5)
    g_vals.append(0.5)
    var lr = 0.1

    # Functional canonical
    var p_func = _allocate_filled(3, p_vals, DType.float64)
    var g_func = _allocate_filled(3, g_vals, DType.float64)
    var v_func = zeros([3], DType.float64)
    var out_func = sgd_step(p_func, g_func, v_func, lr)
    var p_func_out = out_func[0]

    # OO delegator
    var p_oo = _allocate_filled(3, p_vals, DType.float64)
    var g_oo = _allocate_filled(3, g_vals, DType.float64)
    var opt = SGD(learning_rate=lr)
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "SGD")
    print("  ok SGD plain byte-identical to canonical sgd_step")
    print("test_sgd_oo_matches_canonical_plain PASSED")


def test_adam_oo_matches_canonical() raises:
    """Adam: OO `Adam.step()` == canonical `adam_step`."""
    print("Running test_adam_oo_matches_canonical...")
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    p_vals.append(-0.40)
    p_vals.append(0.50)
    p_vals.append(-0.60)
    var g_vals = List[Float64]()
    g_vals.append(0.02)
    g_vals.append(-0.03)
    g_vals.append(0.015)
    g_vals.append(0.025)
    g_vals.append(-0.01)
    g_vals.append(0.04)
    var lr = 0.001
    var b1 = 0.9
    var b2 = 0.999
    var eps = 1e-8
    var wd = 0.0
    var t = 1  # OO Adam increments self.t to 1 on its first call

    # Functional canonical
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var m_func = zeros([6], DType.float64)
    var v_func = zeros([6], DType.float64)
    var out_func = adam_step(
        p_func, g_func, m_func, v_func, t, lr, b1, b2, eps, wd
    )
    var p_func_out = out_func[0]

    # OO delegator
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = Adam(
        learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps, weight_decay=wd
    )
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "Adam")
    print("  ok Adam byte-identical to canonical adam_step")
    print("test_adam_oo_matches_canonical PASSED")


def test_adamw_oo_matches_canonical() raises:
    """AdamW: OO `AdamW.step()` == canonical `adamw_step`."""
    print("Running test_adamw_oo_matches_canonical...")
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    p_vals.append(-0.40)
    p_vals.append(0.50)
    p_vals.append(-0.60)
    var g_vals = List[Float64]()
    g_vals.append(0.02)
    g_vals.append(-0.03)
    g_vals.append(0.015)
    g_vals.append(0.025)
    g_vals.append(-0.01)
    g_vals.append(0.04)
    var lr = 0.001
    var b1 = 0.9
    var b2 = 0.999
    var eps = 1e-8
    var wd = 0.01
    var t = 1

    # Functional canonical
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var m_func = zeros([6], DType.float64)
    var v_func = zeros([6], DType.float64)
    var out_func = adamw_step(
        p_func, g_func, m_func, v_func, t, lr, b1, b2, eps, wd
    )
    var p_func_out = out_func[0]

    # OO delegator
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = AdamW(
        learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps, weight_decay=wd
    )
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "AdamW")
    print("  ok AdamW byte-identical to canonical adamw_step")
    print("test_adamw_oo_matches_canonical PASSED")


def test_adagrad_oo_matches_canonical_no_wd() raises:
    """AdaGrad with wd=0: OO `AdaGrad.step()` == canonical `adagrad_step`."""
    print("Running test_adagrad_oo_matches_canonical_no_wd...")
    var p_vals = List[Float64]()
    p_vals.append(0.50)
    p_vals.append(-0.50)
    p_vals.append(0.25)
    var g_vals = List[Float64]()
    g_vals.append(0.10)
    g_vals.append(-0.20)
    g_vals.append(0.05)
    var lr = 0.01
    var eps = 1e-10
    var wd = 0.0

    # Functional canonical (accum starts empty per AdaGrad spec)
    var p_func = _allocate_filled(3, p_vals, DType.float64)
    var g_func = _allocate_filled(3, g_vals, DType.float64)
    var accum_func = zeros([3], DType.float64)
    var out_func = adagrad_step(p_func, g_func, accum_func, lr, eps, wd)
    var p_func_out = out_func[0]

    # OO delegator
    var p_oo = _allocate_filled(3, p_vals, DType.float64)
    var g_oo = _allocate_filled(3, g_vals, DType.float64)
    var opt = AdaGrad(learning_rate=lr, epsilon=eps, weight_decay=wd)
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "AdaGrad(wd=0)")
    print("  ok AdaGrad(wd=0) byte-identical to canonical adagrad_step")
    print("test_adagrad_oo_matches_canonical_no_wd PASSED")


def test_adagrad_oo_matches_canonical_with_wd() raises:
    """AdaGrad with wd>0 (legacy additive semantics): OO matches canonical.

    `adagrad_step` applies weight decay *outside* the adaptive scaling
    (matching the legacy autograd AdaGrad semantics exactly):
        accum_t = accum + g^2
        update  = lr*g/(sqrt(accum)+eps) + (wd*param) if wd>0
    Both paths must agree because `AdaGrad.step()` thin-delegates.
    """
    print("Running test_adagrad_oo_matches_canonical_with_wd...")
    var p_vals = List[Float64]()
    p_vals.append(0.50)
    p_vals.append(-0.50)
    p_vals.append(0.25)
    var g_vals = List[Float64]()
    g_vals.append(0.10)
    g_vals.append(-0.20)
    g_vals.append(0.05)
    var lr = 0.01
    var eps = 1e-10
    var wd = 1e-3

    # Functional canonical
    var p_func = _allocate_filled(3, p_vals, DType.float64)
    var g_func = _allocate_filled(3, g_vals, DType.float64)
    var accum_func = zeros([3], DType.float64)
    var out_func = adagrad_step(p_func, g_func, accum_func, lr, eps, wd)
    var p_func_out = out_func[0]

    # OO delegator
    var p_oo = _allocate_filled(3, p_vals, DType.float64)
    var g_oo = _allocate_filled(3, g_vals, DType.float64)
    var opt = AdaGrad(learning_rate=lr, epsilon=eps, weight_decay=wd)
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "AdaGrad(wd>0)")
    print("  ok AdaGrad(wd>0) byte-identical to canonical adagrad_step")
    print("test_adagrad_oo_matches_canonical_with_wd PASSED")


def test_rmsprop_oo_matches_canonical_plain() raises:
    """RMSprop (momentum=0): OO `RMSprop.step()` == canonical `rmsprop_step`."""
    print("Running test_rmsprop_oo_matches_canonical_plain...")
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    p_vals.append(-0.40)
    p_vals.append(0.50)
    p_vals.append(-0.60)
    var g_vals = List[Float64]()
    g_vals.append(0.02)
    g_vals.append(-0.03)
    g_vals.append(0.015)
    g_vals.append(0.025)
    g_vals.append(-0.01)
    g_vals.append(0.04)
    var lr = 0.01
    var alpha = 0.99
    var eps = 1e-8
    var wd = 0.0
    var momentum = 0.0  # buf=None in canonical; OO also buf=None

    # Functional canonical
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var square_func = zeros([6], DType.float64)
    var out_func = rmsprop_step(
        p_func, g_func, square_func, 1, lr, alpha, eps, wd, momentum, None
    )
    var p_func_out = out_func[0]

    # OO delegator
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = RMSprop(
        learning_rate=lr,
        alpha=alpha,
        epsilon=eps,
        weight_decay=wd,
        momentum=momentum,
    )
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "RMSprop")
    print("  ok RMSprop byte-identical to canonical rmsprop_step")
    print("test_rmsprop_oo_matches_canonical_plain PASSED")


def test_rmsprop_oo_matches_canonical_with_momentum() raises:
    """RMSprop (momentum>0): OO `RMSprop.step()` == canonical `rmsprop_step`.

    Both paths supply a zero-initialised momentum-buffer at the first
    step and compute the same update; the comparison must be byte-equal.
    """
    print("Running test_rmsprop_oo_matches_canonical_with_momentum...")
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    p_vals.append(-0.40)
    p_vals.append(0.50)
    p_vals.append(-0.60)
    var g_vals = List[Float64]()
    g_vals.append(0.02)
    g_vals.append(-0.03)
    g_vals.append(0.015)
    g_vals.append(0.025)
    g_vals.append(-0.01)
    g_vals.append(0.04)
    var lr = 0.01
    var alpha = 0.99
    var eps = 1e-8
    var wd = 0.0
    var momentum = 0.9

    # Functional canonical — pass zero-initised buf (the same shape as
    # param) to match what the OO's lazy `m_buffers[param_id] =
    # zeros_like(param_data)` produces on first sight.
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var square_func = zeros([6], DType.float64)
    var buf_func = zeros([6], DType.float64)
    var out_func = rmsprop_step(
        p_func, g_func, square_func, 1, lr, alpha, eps, wd, momentum, buf_func
    )
    var p_func_out = out_func[0]

    # OO delegator — lazily creates m_buffers[param_id] = zeros_like(p)
    # on the first call, so the canonical buf shape/dtype/value matches.
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = RMSprop(
        learning_rate=lr,
        alpha=alpha,
        epsilon=eps,
        weight_decay=wd,
        momentum=momentum,
    )
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "RMSprop(m)")
    print("  ok RMSprop(m) byte-identical to canonical rmsprop_step")
    print("test_rmsprop_oo_matches_canonical_with_momentum PASSED")


# ============================================================================
# Test main
# ============================================================================


def main() raises:
    test_sgd_oo_matches_canonical_with_momentum()
    test_sgd_oo_matches_canonical_plain()
    test_adam_oo_matches_canonical()
    test_adamw_oo_matches_canonical()
    test_adagrad_oo_matches_canonical_no_wd()
    test_adagrad_oo_matches_canonical_with_wd()
    test_rmsprop_oo_matches_canonical_plain()
    test_rmsprop_oo_matches_canonical_with_momentum()
    print("\nAll optimizer delegator-equivalence tests PASSED")
