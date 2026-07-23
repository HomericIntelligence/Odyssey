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
from odyssey.autograd.optimizers_oo.lion import Lion
from odyssey.autograd.optimizers_oo.lars import LARS
from odyssey.autograd.optimizers_oo.ftrl import FTRLProximal
from odyssey.training.optimizers.sgd import sgd_step
from odyssey.training.optimizers.adam import adam_step
from odyssey.training.optimizers.adamw import adamw_step
from odyssey.training.optimizers.adagrad import adagrad_step
from odyssey.training.optimizers.rmsprop import rmsprop_step
from odyssey.training.optimizers.lion import lion_step
from odyssey.training.optimizers.lars import lars_step
from odyssey.training.optimizers.ftrl import ftrl_step


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


def _run_oo_k3_steps[
    OOType: Optimizer
](
    mut opt: OOType,
    p_init: AnyTensor,
    mut grads: List[AnyTensor],
) raises -> AnyTensor:
    """Run `opt.step([Variable])` 3 successive times on a single param.

    Reuses the same `Variable + param_id` across all 3 iterations so the
    OO's per-parameter state buffers (`m_buffers` / `v_buffers` /
    `G_buffers` / RMSprop `m_buffers`) carry-over across calls — proving
    that the OO optimizers thread their lazy-initialised state via
    `parameters[i].id` exactly the way the caller-managed canonical path
    threads `m, v, accum, buf`.

    Args:
        opt: OO optimizer state (hyperparameters already set at construct).
        p_init: initial parameter tensor (owned; consumed).
        grads: exactly 3 gradient tensors (each consumed in turn).

    Returns the parameter tensor after the 3rd step (zero-tolerance byte
    equality expected against the canonical-K=3 result).
    """
    if len(grads) != 3:
        raise Error(
            "_run_oo_k3_steps requires exactly 3 grads (got "
            + String(len(grads))
            + ")"
        )
    var tape = GradientTape()
    tape.enable()
    var pv = Variable(p_init^, True, tape)
    var pid = pv.id
    var params: List[Variable] = []
    params.append(pv.copy())
    for k in range(3):
        tape.registry.set_grad(pid, grads[k]^)
        opt.step(params, tape)
        opt.zero_grad(tape)
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


def test_adam_oo_matches_canonical_k3() raises:
    """Adam K=3: across 3 successive steps OO.t / m_buffers / v_buffers
    threading via `param_id` matches canonical's caller-managed
    `m_c, v_c, t_c` (t advances 1→2→3 across calls).

    The K=3 byte-identical assertion is the regression barrier against
    state-threading drift — if the OO re-lazy-inits buffers on each
    call, or threads `t` incorrectly, the second + third steps diverge
    from the canonical.
    """
    print("Running test_adam_oo_matches_canonical_k3...")
    var n = 3
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    var g1_vals = List[Float64]()
    g1_vals.append(0.02)
    g1_vals.append(-0.03)
    g1_vals.append(0.015)
    var g2_vals = List[Float64]()
    g2_vals.append(-0.04)
    g2_vals.append(0.05)
    g2_vals.append(-0.01)
    var g3_vals = List[Float64]()
    g3_vals.append(0.03)
    g3_vals.append(-0.02)
    g3_vals.append(0.04)
    var lr = 0.001
    var b1 = 0.9
    var b2 = 0.999
    var eps = 1e-8
    var wd = 0.0

    # === Canonical (caller-managed m_c, v_c; t_c = k+1 each iter = 1, 2, 3) ===
    var p_c = _allocate_filled(n, p_vals, DType.float64)
    var m_c = zeros([n], DType.float64)
    var v_c = zeros([n], DType.float64)
    var grads_c = List[AnyTensor]()
    grads_c.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g3_vals, DType.float64))
    for k in range(3):
        var t_c = k + 1
        var out = adam_step(p_c, grads_c[k], m_c, v_c, t_c, lr, b1, b2, eps, wd)
        p_c = out[0]
        m_c = out[1]
        v_c = out[2]

    # === OO (state threaded internally by `param_id` from one Variable) ===
    var p_oo = _allocate_filled(n, p_vals, DType.float64)
    var grads_oo = List[AnyTensor]()
    grads_oo.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g3_vals, DType.float64))
    var opt = Adam(
        learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps, weight_decay=wd
    )
    var p_oo_out = _run_oo_k3_steps(opt, p_oo, grads_oo)

    _assert_tensors_byte_equal(p_c, p_oo_out, "Adam K=3")
    print("  ok Adam K=3 byte-identical to canonical across 3 steps")
    print("test_adam_oo_matches_canonical_k3 PASSED")


def test_rmsprop_with_momentum_oo_matches_canonical_k3() raises:
    """RMSprop-with-momentum K=3: OO `m_buffers[pid]` carry-over matches
    canonical's caller-managed `buf_c` thread.

    The OO lazy-inits `m_buffers[pid] = zeros_like(param)` on iter 1
    (when `momentum > 0`) and threads via `result[2]` on subsequent
    iters; canonical mirrors the exact same pattern.
    """
    print("Running test_rmsprop_with_momentum_oo_matches_canonical_k3...")
    var n = 6
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    p_vals.append(-0.40)
    p_vals.append(0.50)
    p_vals.append(-0.60)
    var g1_vals = List[Float64]()
    g1_vals.append(0.02)
    g1_vals.append(-0.03)
    g1_vals.append(0.015)
    g1_vals.append(0.025)
    g1_vals.append(-0.01)
    g1_vals.append(0.04)
    var g2_vals = List[Float64]()
    g2_vals.append(-0.015)
    g2_vals.append(0.025)
    g2_vals.append(-0.010)
    g2_vals.append(0.020)
    g2_vals.append(-0.005)
    g2_vals.append(0.030)
    var g3_vals = List[Float64]()
    g3_vals.append(0.010)
    g3_vals.append(-0.020)
    g3_vals.append(0.005)
    g3_vals.append(-0.015)
    g3_vals.append(0.020)
    g3_vals.append(-0.025)
    var lr = 0.01
    var alpha = 0.99
    var eps = 1e-8
    var wd = 0.0
    var momentum = 0.9

    # === Canonical (square_c + buf_c threaded across iters) ===
    var p_c = _allocate_filled(n, p_vals, DType.float64)
    var square_c = zeros([n], DType.float64)
    var buf_c = zeros([n], DType.float64)
    var grads_c = List[AnyTensor]()
    grads_c.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g3_vals, DType.float64))
    for k in range(3):
        var out = rmsprop_step(
            p_c, grads_c[k], square_c, 1, lr, alpha, eps, wd, momentum, buf_c
        )
        p_c = out[0]
        square_c = out[1]
        buf_c = out[2]

    # === OO ===
    var p_oo = _allocate_filled(n, p_vals, DType.float64)
    var grads_oo = List[AnyTensor]()
    grads_oo.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g3_vals, DType.float64))
    var opt = RMSprop(
        learning_rate=lr,
        alpha=alpha,
        epsilon=eps,
        weight_decay=wd,
        momentum=momentum,
    )
    var p_oo_out = _run_oo_k3_steps(opt, p_oo, grads_oo)

    _assert_tensors_byte_equal(p_c, p_oo_out, "RMSprop(m) K=3")
    print("  ok RMSprop(m) K=3 byte-identical to canonical across 3 steps")
    print("test_rmsprop_with_momentum_oo_matches_canonical_k3 PASSED")


def test_adagrad_with_wd_oo_matches_canonical_k3() raises:
    """AdaGrad-with-wd K=3: OO `G_buffers[pid] += grad²` carry-over
    matches canonical's caller-managed `accum` thread.

    The OO lazy-inits `G_buffers[pid] = zeros_like(param)` on iter 1 and
    threads via `result[1]` (the new accumulator) on subsequent iters;
    canonical mirrors. With `wd>0`, weight decay is applied OUTSIDE the
    adaptive scaling (legacy semantics) so this test additionally pins
    that quirky additive-WD contract for multi-step iterates.
    """
    print("Running test_adagrad_with_wd_oo_matches_canonical_k3...")
    var n = 3
    var p_vals = List[Float64]()
    p_vals.append(0.50)
    p_vals.append(-0.50)
    p_vals.append(0.25)
    var g1_vals = List[Float64]()
    g1_vals.append(0.10)
    g1_vals.append(-0.20)
    g1_vals.append(0.05)
    var g2_vals = List[Float64]()
    g2_vals.append(0.08)
    g2_vals.append(-0.15)
    g2_vals.append(0.07)
    var g3_vals = List[Float64]()
    g3_vals.append(-0.05)
    g3_vals.append(0.10)
    g3_vals.append(-0.03)
    var lr = 0.01
    var eps = 1e-10
    var wd = 1e-3

    # === Canonical (accum_c threaded across iters) ===
    var p_c = _allocate_filled(n, p_vals, DType.float64)
    var accum_c = zeros([n], DType.float64)
    var grads_c = List[AnyTensor]()
    grads_c.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g3_vals, DType.float64))
    for k in range(3):
        var out = adagrad_step(p_c, grads_c[k], accum_c, lr, eps, wd)
        p_c = out[0]
        accum_c = out[1]

    # === OO ===
    var p_oo = _allocate_filled(n, p_vals, DType.float64)
    var grads_oo = List[AnyTensor]()
    grads_oo.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g3_vals, DType.float64))
    var opt = AdaGrad(learning_rate=lr, epsilon=eps, weight_decay=wd)
    var p_oo_out = _run_oo_k3_steps(opt, p_oo, grads_oo)

    _assert_tensors_byte_equal(p_c, p_oo_out, "AdaGrad(wd>0) K=3")
    print("  ok AdaGrad(wd>0) K=3 byte-identical to canonical across 3 steps")
    print("test_adagrad_with_wd_oo_matches_canonical_k3 PASSED")


def test_lion_oo_matches_canonical() raises:
    """Lion: OO `Lion.step()` == canonical `lion_step`.

    Pins the K=1 byte-identical claim for the Lion OO wrapper: the OO
    lazy-inits `momenta[pid] = zeros_like(param)` on first sight
    (matching the zero-initialised `m_func` we pass to the canonical
    here), and forwards all hyperparameters unchanged into `lion_step`.
    """
    print("Running test_lion_oo_matches_canonical...")
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
    # Lion uses 3-10x smaller LR than AdamW; pick a representative midpoint.
    var lr = 0.0001
    var b1 = 0.9
    var b2 = 0.99
    var wd = 0.0

    # Functional canonical (zero momentum init).
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var m_func = zeros([6], DType.float64)
    var out_func = lion_step(p_func, g_func, m_func, lr, b1, b2, wd)
    var p_func_out = out_func[0]

    # OO delegator lazy-inits momenta[pid] = zeros_like(param) on first call.
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = Lion(learning_rate=lr, beta1=b1, beta2=b2, weight_decay=wd)
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "Lion")
    print("  ok Lion byte-identical to canonical lion_step")
    print("test_lion_oo_matches_canonical PASSED")


def test_lars_oo_matches_canonical() raises:
    """LARS: OO `LARS.step()` == canonical `lars_step`.

    Pins the K=1 byte-identical claim for the LARS OO wrapper: the OO
    lazy-inits `velocities[pid] = zeros_like(param)` on first sight
    (matching the zero-initialised `v_func`), and forwards all
    hyperparameters unchanged into `lars_step`. The trust-ratio scaling
    is computed inside `lars_step` itself.
    """
    print("Running test_lars_oo_matches_canonical...")
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
    var wd = 0.0001
    var trust = 0.001
    var eps = 1e-8

    # Functional canonical (zero velocity).
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var v_func = zeros([6], DType.float64)
    var out_func = lars_step(
        p_func, g_func, v_func, lr, momentum, wd, trust, eps
    )
    var p_func_out = out_func[0]

    # OO delegator lazy-inits velocities[pid] = zeros_like(param) on first call.
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = LARS(
        learning_rate=lr,
        momentum=momentum,
        weight_decay=wd,
        trust_coefficient=trust,
        epsilon=eps,
    )
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "LARS")
    print("  ok LARS byte-identical to canonical lars_step")
    print("test_lars_oo_matches_canonical PASSED")


def test_ftrl_oo_matches_canonical() raises:
    """FTRL-Proximal (lambda1=lambda2=0): OO == canonical.

    Pins the K=1 byte-identical claim for FTRL-Proximal OO wrapper:
    the OO lazy-inits BOTH `z_buffers[pid]` and `n_buffers[pid]` to
    `zeros_like(param)` on first sight (matching the two zero-initialised
    `z_func, n_func` passed to canonical). Picking `lambda1=lambda2=0`
    keeps the test dense so byte-equality is element-wise stable.
    """
    print("Running test_ftrl_oo_matches_canonical...")
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
    # lr=1.0 -> textbook FTRL (per-coord rate lives in `alpha`).
    var lr = 1.0
    var alpha = 0.1
    var beta = 1.0
    var lambda1 = 0.0
    var lambda2 = 0.0

    # Functional canonical (both z and n start zero on K=1).
    var p_func = _allocate_filled(6, p_vals, DType.float64)
    var g_func = _allocate_filled(6, g_vals, DType.float64)
    var z_func = zeros([6], DType.float64)
    var n_func = zeros([6], DType.float64)
    var out_func = ftrl_step(
        p_func, g_func, z_func, n_func, lr, alpha, beta, lambda1, lambda2
    )
    var p_func_out = out_func[0]

    # OO delegator lazy-inits z_buffers AND n_buffers on first call.
    var p_oo = _allocate_filled(6, p_vals, DType.float64)
    var g_oo = _allocate_filled(6, g_vals, DType.float64)
    var opt = FTRLProximal(
        learning_rate=lr,
        alpha=alpha,
        beta=beta,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    var p_oo_out = _run_oo_step(opt, p_oo, g_oo)

    _assert_tensors_byte_equal(p_func_out, p_oo_out, "FTRL")
    print("  ok FTRL byte-identical to canonical ftrl_step")
    print("test_ftrl_oo_matches_canonical PASSED")


def test_lion_oo_matches_canonical_k3() raises:
    """Lion K=3: OO `momenta[pid]` carry-over across 3 successive steps
    matches the canonical caller-managed `m_c` thread.

    Pins the multi-step byte-identical claim for the Lion OO wrapper:
    the OO lazy-inits `momenta[pid] = zeros_like(param)` on iter 1
    and threads via `result[1]` on subsequent iters; canonical mirrors
    that pattern with caller-managed `m_c`. A regression that re-lazy-
    inits `momenta` on each call diverges from canonical on step 2 + 3.
    """
    print("Running test_lion_oo_matches_canonical_k3...")
    var n = 3
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    var g1_vals = List[Float64]()
    g1_vals.append(0.02)
    g1_vals.append(-0.03)
    g1_vals.append(0.015)
    var g2_vals = List[Float64]()
    g2_vals.append(-0.015)
    g2_vals.append(0.025)
    g2_vals.append(-0.010)
    var g3_vals = List[Float64]()
    g3_vals.append(0.010)
    g3_vals.append(-0.020)
    g3_vals.append(0.005)
    var lr = 0.0001
    var b1 = 0.9
    var b2 = 0.99
    var wd = 0.0

    # === Canonical (m_c threaded across iters) ===
    var p_c = _allocate_filled(n, p_vals, DType.float64)
    var m_c = zeros([n], DType.float64)
    var grads_c = List[AnyTensor]()
    grads_c.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g3_vals, DType.float64))
    for k in range(3):
        var out_func = lion_step(p_c, grads_c[k], m_c, lr, b1, b2, wd)
        p_c = out_func[0]
        m_c = out_func[1]

    # === OO (momenta[pid] threads via param_id + result[1]) ===
    var p_oo = _allocate_filled(n, p_vals, DType.float64)
    var grads_oo = List[AnyTensor]()
    grads_oo.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g3_vals, DType.float64))
    var opt = Lion(learning_rate=lr, beta1=b1, beta2=b2, weight_decay=wd)
    var p_oo_out = _run_oo_k3_steps(opt, p_oo, grads_oo)

    _assert_tensors_byte_equal(p_c, p_oo_out, "Lion K=3")
    print("  ok Lion K=3 byte-identical to canonical across 3 steps")
    print("test_lion_oo_matches_canonical_k3 PASSED")


def test_lars_oo_matches_canonical_k3() raises:
    """LARS K=3: OO `velocities[pid]` carry-over across 3 successive steps
    matches the canonical caller-managed `v_c` thread.

    Pins the multi-step byte-identical claim for LARS OO wrapper:
    OO lazy-inits `velocities[pid] = zeros_like(param)` on iter 1 and
    threads via `result[1]` on subsequent iters; canonical mirrors.
    """
    print("Running test_lars_oo_matches_canonical_k3...")
    var n = 3
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    var g1_vals = List[Float64]()
    g1_vals.append(0.02)
    g1_vals.append(-0.03)
    g1_vals.append(0.015)
    var g2_vals = List[Float64]()
    g2_vals.append(-0.015)
    g2_vals.append(0.025)
    g2_vals.append(-0.010)
    var g3_vals = List[Float64]()
    g3_vals.append(0.010)
    g3_vals.append(-0.020)
    g3_vals.append(0.005)
    var lr = 0.1
    var momentum = 0.9
    var wd = 0.0001
    var trust = 0.001
    var eps = 1e-8

    # === Canonical (v_c threaded across iters) ===
    var p_c = _allocate_filled(n, p_vals, DType.float64)
    var v_c = zeros([n], DType.float64)
    var grads_c = List[AnyTensor]()
    grads_c.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g3_vals, DType.float64))
    for k in range(3):
        var out_func = lars_step(
            p_c, grads_c[k], v_c, lr, momentum, wd, trust, eps
        )
        p_c = out_func[0]
        v_c = out_func[1]

    # === OO (velocities[pid] threads via param_id + result[1]) ===
    var p_oo = _allocate_filled(n, p_vals, DType.float64)
    var grads_oo = List[AnyTensor]()
    grads_oo.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g3_vals, DType.float64))
    var opt = LARS(
        learning_rate=lr,
        momentum=momentum,
        weight_decay=wd,
        trust_coefficient=trust,
        epsilon=eps,
    )
    var p_oo_out = _run_oo_k3_steps(opt, p_oo, grads_oo)

    _assert_tensors_byte_equal(p_c, p_oo_out, "LARS K=3")
    print("  ok LARS K=3 byte-identical to canonical across 3 steps")
    print("test_lars_oo_matches_canonical_k3 PASSED")


def test_ftrl_oo_matches_canonical_k3() raises:
    """FTRL-Proximal K=3 (lambda1=lambda2=0): OO threads BOTH
    `z_buffers[pid]` AND `n_buffers[pid]` correctly across 3 steps.

    FTRL is the most state-bearing of the 8 wrappers (2 buffers per param).
    Its K>1 inversion `params = -z/n` matches canonical only when BOTH
    `z` and `n` are threaded across calls. A regression that re-lazy-inits
    either buffer per call diverges from canonical on step 2.
    """
    print("Running test_ftrl_oo_matches_canonical_k3...")
    var n = 3
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    var g1_vals = List[Float64]()
    g1_vals.append(0.02)
    g1_vals.append(-0.03)
    g1_vals.append(0.015)
    var g2_vals = List[Float64]()
    g2_vals.append(-0.015)
    g2_vals.append(0.025)
    g2_vals.append(-0.010)
    var g3_vals = List[Float64]()
    g3_vals.append(0.010)
    g3_vals.append(-0.020)
    g3_vals.append(0.005)
    var lr = 1.0
    var alpha = 0.1
    var beta = 1.0
    var lambda1 = 0.0
    var lambda2 = 0.0

    # === Canonical (z_c + n_c threaded across iters) ===
    var p_c = _allocate_filled(n, p_vals, DType.float64)
    var z_c = zeros([n], DType.float64)
    var n_c = zeros([n], DType.float64)
    var grads_c = List[AnyTensor]()
    grads_c.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g3_vals, DType.float64))
    for k in range(3):
        var out_func = ftrl_step(
            p_c, grads_c[k], z_c, n_c, lr, alpha, beta, lambda1, lambda2
        )
        p_c = out_func[0]
        z_c = out_func[1]
        n_c = out_func[2]

    # === OO (both z_buffers AND n_buffers thread via result[1], result[2]) ===
    var p_oo = _allocate_filled(n, p_vals, DType.float64)
    var grads_oo = List[AnyTensor]()
    grads_oo.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g3_vals, DType.float64))
    var opt = FTRLProximal(
        learning_rate=lr,
        alpha=alpha,
        beta=beta,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    var p_oo_out = _run_oo_k3_steps(opt, p_oo, grads_oo)

    _assert_tensors_byte_equal(p_c, p_oo_out, "FTRL K=3")
    print("  ok FTRL K=3 byte-identical to canonical across 3 steps")
    print("test_ftrl_oo_matches_canonical_k3 PASSED")


# ============================================================================
# Raise-contract equivalence tests
# ============================================================================
#
# These tests pin the *error-path* equivalence: the canonical functional
# step and the OO wrapper must reject (or accept) the same inputs, with the
# same exception type. Locking the raise contract is what closes the
# "no behavioral drift" claim: a future divergence that silently swallowed
# a dtype mismatch, or that raised a *different* error on shape mismatch,
# would otherwise be invisible to K=1/K=3 happy-path tests.


def test_adam_dtypes_mismatch_raises() raises:
    """Adam: both `adam_step` and `Adam.step()` raise on dtype mismatch.

    Both paths go through the same `adam_step` rejection
    (`params.dtype() != gradients.dtype()`), so the exception semantics
    match by construction if both raise.
    """
    print("Running test_adam_dtypes_mismatch_raises...")
    var p_vals = List[Float64]()
    p_vals.append(1.0)
    p_vals.append(1.0)
    p_vals.append(1.0)
    var g_vals = List[Float64]()
    g_vals.append(0.5)
    g_vals.append(0.5)
    g_vals.append(0.5)
    var p_can = _allocate_filled(3, p_vals, DType.float64)
    var g_can = _allocate_filled(3, g_vals, DType.float32)  # dtype mismatch
    var m_c = zeros([3], DType.float64)
    var v_c = zeros([3], DType.float64)

    var can_raised = False
    try:
        adam_step(p_can, g_can, m_c, v_c, 1, 0.001, 0.9, 0.999, 1e-8, 0.0)
    except e:
        can_raised = True
    if not can_raised:
        raise Error("canonical adam_step did not raise on dtype mismatch")

    var p_oo = _allocate_filled(3, p_vals, DType.float64)
    var g_oo = _allocate_filled(3, g_vals, DType.float32)
    var oo_raised = False
    try:
        var opt = Adam(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
        )
        var tape = GradientTape()
        tape.enable()
        var pv = Variable(p_oo^, True, tape)
        var pid = pv.id
        var params: List[Variable] = []
        params.append(pv.copy())
        tape.registry.set_grad(pid, g_oo^)
        opt.step(params, tape)
    except e:
        oo_raised = True
    if not oo_raised:
        raise Error("OO Adam.step() did not raise on dtype mismatch")
    if can_raised != oo_raised:
        raise Error(
            "raise contract differs (dtype): canonical="
            + String(can_raised)
            + " oo="
            + String(oo_raised)
        )
    print("  ok Adam dtype-mismatch raises on both paths")
    print("test_adam_dtypes_mismatch_raises PASSED")


def test_adam_shape_mismatch_raises() raises:
    """Adam: both paths raise on shape mismatch (param=[3], grad=[4]).

    Same structure as the dtype-mismatch test — both paths share
    the canonical `adam_step` rejection of `params.shape() !=
    gradients.shape()`, so the contract agrees if both raise.
    """
    print("Running test_adam_shape_mismatch_raises...")
    var p_vals = List[Float64]()
    p_vals.append(1.0)
    p_vals.append(1.0)
    p_vals.append(1.0)
    var g_vals_4 = List[Float64]()
    g_vals_4.append(0.5)
    g_vals_4.append(0.5)
    g_vals_4.append(0.5)
    g_vals_4.append(0.5)
    var p = _allocate_filled(3, p_vals, DType.float64)
    var g = _allocate_filled(4, g_vals_4, DType.float64)  # shape mismatch
    var m_c = zeros([3], DType.float64)
    var v_c = zeros([3], DType.float64)

    var can_raised = False
    try:
        adam_step(p, g, m_c, v_c, 1, 0.001, 0.9, 0.999, 1e-8, 0.0)
    except e:
        can_raised = True
    if not can_raised:
        raise Error("canonical adam_step did not raise on shape mismatch")

    var oo_raised = False
    try:
        var opt = Adam(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
        )
        var tape = GradientTape()
        tape.enable()
        var pv = Variable(p^, True, tape)
        var pid = pv.id
        var params: List[Variable] = []
        params.append(pv.copy())
        tape.registry.set_grad(pid, g^)
        opt.step(params, tape)
    except e:
        oo_raised = True
    if not oo_raised:
        raise Error("OO Adam.step() did not raise on shape mismatch")
    if can_raised != oo_raised:
        raise Error("raise contract differs (shape)")
    print("  ok Adam shape-mismatch raises on both paths")
    print("test_adam_shape_mismatch_raises PASSED")


def test_adam_t_nonpositive_raises() raises:
    """Adam: canonical `adam_step` rejects `t<=0`; OO `Adam.step()`
    internally manages `t` and never exposes an invalid t to the
    canonical implementation.

    Both paths produce well-defined K=3 updates — the contract agrees
    in the sense that the canonical rejects caller-supplied bad `t`,
    while the OO avoids the bad-t condition by construction
    (no public `t` parameter; `self.t` increments internally across
    the K=3 step sequence). This test logs that asymmetry and pins
    the OO K=3 path against the canonical's well-formed `t=1,2,3`.
    """
    print("Running test_adam_t_nonpositive_raises...")

    # 1) Canonical rejects t<=0 (both t=0 and a negative t).
    var p_vals = List[Float64]()
    p_vals.append(1.0)
    p_vals.append(1.0)
    p_vals.append(1.0)
    var g_vals = List[Float64]()
    g_vals.append(0.5)
    g_vals.append(0.5)
    g_vals.append(0.5)
    var p = _allocate_filled(3, p_vals, DType.float64)
    var g = _allocate_filled(3, g_vals, DType.float64)
    var m_c = zeros([3], DType.float64)
    var v_c = zeros([3], DType.float64)

    var can_raised_t0 = False
    try:
        adam_step(p, g, m_c, v_c, 0, 0.001, 0.9, 0.999, 1e-8, 0.0)
    except e:
        can_raised_t0 = True
    if not can_raised_t0:
        raise Error("canonical adam_step did not raise on t=0")

    var can_raised_neg = False
    try:
        adam_step(p, g, m_c, v_c, -1, 0.001, 0.9, 0.999, 1e-8, 0.0)
    except e:
        can_raised_neg = True
    if not can_raised_neg:
        raise Error("canonical adam_step did not raise on t=-1")

    # 2) OO never exposes bad t — running K=3 must succeed because the
    #    OO manages `t` internally (init=0, +1 per step).
    var p2 = _allocate_filled(3, p_vals, DType.float64)
    var grads = List[AnyTensor]()
    grads.append(_allocate_filled(3, g_vals, DType.float64))
    grads.append(_allocate_filled(3, g_vals, DType.float64))
    grads.append(_allocate_filled(3, g_vals, DType.float64))
    var opt = Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
    )
    var oo_out = _run_oo_k3_steps(opt, p2, grads)
    if oo_out.numel() != 3:
        raise Error("OO K=3 output has wrong numel after internal t threading")

    print(
        "  ok Adam t<=0 contract: canonical rejects, OO avoids by"
        " construction (managed internally)"
    )
    print("test_adam_t_nonpositive_raises PASSED")


def test_adamw_oo_matches_canonical_k3() raises:
    """AdamW K=3 (wd=0.01): decoupled-WD contract pinned at multi-step.

    Across 3 successive steps the OO's `self.t` / `m_buffers` /
    `v_buffers` threading plus the wd=0.01 weight-decay penalty must
    match the canonical `adamw_step`'s caller-managed `m_c, v_c, t_c`
    thread. Complements `test_adamw_oo_matches_canonical` (K=1) by
    locking in the multi-step behavior — including how the wd penalty
    decays `params` across iterates 1→2→3.
    """
    print("Running test_adamw_oo_matches_canonical_k3...")
    var n = 3
    var p_vals = List[Float64]()
    p_vals.append(0.10)
    p_vals.append(-0.20)
    p_vals.append(0.30)
    var g1_vals = List[Float64]()
    g1_vals.append(0.02)
    g1_vals.append(-0.03)
    g1_vals.append(0.015)
    var g2_vals = List[Float64]()
    g2_vals.append(-0.04)
    g2_vals.append(0.05)
    g2_vals.append(-0.01)
    var g3_vals = List[Float64]()
    g3_vals.append(0.03)
    g3_vals.append(-0.02)
    g3_vals.append(0.04)
    var lr = 0.001
    var b1 = 0.9
    var b2 = 0.999
    var eps = 1e-8
    var wd = 0.01

    # Canonical (caller-managed m_c, v_c; t_c = k+1 each iter = 1, 2, 3)
    var p_c = _allocate_filled(n, p_vals, DType.float64)
    var m_c = zeros([n], DType.float64)
    var v_c = zeros([n], DType.float64)
    var grads_c = List[AnyTensor]()
    grads_c.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_c.append(_allocate_filled(n, g3_vals, DType.float64))
    for k in range(3):
        var t_c = k + 1
        var out = adamw_step(
            p_c, grads_c[k], m_c, v_c, t_c, lr, b1, b2, eps, wd
        )
        p_c = out[0]
        m_c = out[1]
        v_c = out[2]

    # OO (state threaded internally by `param_id` from one Variable)
    var p_oo = _allocate_filled(n, p_vals, DType.float64)
    var grads_oo = List[AnyTensor]()
    grads_oo.append(_allocate_filled(n, g1_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g2_vals, DType.float64))
    grads_oo.append(_allocate_filled(n, g3_vals, DType.float64))
    var opt = AdamW(
        learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps, weight_decay=wd
    )
    var p_oo_out = _run_oo_k3_steps(opt, p_oo, grads_oo)

    _assert_tensors_byte_equal(p_c, p_oo_out, "AdamW K=3 (wd=0.01)")
    print(
        "  ok AdamW K=3 byte-identical to canonical across 3 steps"
        " (decoupled WD)"
    )
    print("test_adamw_oo_matches_canonical_k3 PASSED")


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
    test_adam_oo_matches_canonical_k3()
    test_rmsprop_with_momentum_oo_matches_canonical_k3()
    test_adagrad_with_wd_oo_matches_canonical_k3()
    test_adam_dtypes_mismatch_raises()
    test_adam_shape_mismatch_raises()
    test_adam_t_nonpositive_raises()
    test_adamw_oo_matches_canonical_k3()
    test_lion_oo_matches_canonical()
    test_lars_oo_matches_canonical()
    test_ftrl_oo_matches_canonical()
    test_lion_oo_matches_canonical_k3()
    test_lars_oo_matches_canonical_k3()
    test_ftrl_oo_matches_canonical_k3()
    print(
        "\nAll optimizer delegator-equivalence tests PASSED"
        " (K=1 + K=3 + raise-contract + AdamW K=3 + Lion/LARS/FTRL)"
    )
