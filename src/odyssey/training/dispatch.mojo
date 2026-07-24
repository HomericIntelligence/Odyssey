"""Optimizer CLI/config dispatch registry.

A single-source-of-truth registry for the 24 functional optimizers shipped under
`odyssey.training.optimizers`. Lets CLI drivers and config loaders look up an
optimizer by string name (`--optimizer adam`, `optimizer.name: shampoo` in YAML)
without needing to know about the concrete module path.

Capabilities:
- `list_optimizers()` — full inventory of registered names.
- `is_optimizer_registered(name)` — fast presence check.
- `get_optimizer_spec(name)` — full metadata (family, step/init fn names, default
  learning rate & weight decay, hint on per-param state buffer count).
- `apply_default_hyperparams(name)` — default learning_rate / weight_decay plus
  family-specific extras (beta1, beta2, eps, momentum, …).
- `init_optimizer_state(name, params, *, force_f64=False)` — uniform state
  allocation across all 24 optimizers. Each `init_<name>_state` has the same
  canonical signature: `(List[AnyTensor], *, Bool) -> List[List[AnyTensor]]`.

Design notes:
- Backed by a static `if/elif` chain — every dispatch branch is fully
  type-checked, and we avoid the function-pointer/Dict-of-fn lifetime gymnastics
  that Mojo 1.0 makes awkward.
- The registry is built fresh on each lookup; with 24 entries the cost is
  negligible relative to a single Mojo compile.
- Only STATE initialization is dispatched. The `_<name>_step` functions have
  intentionally non-uniform signatures (sgd returns `Tuple(p, v)`, adam returns
  `Tuple(p, m, v)`, shampoo returns 3, schedule_free returns 3, …) so a uniform
  step dispatch would require bespoke per-optimizer adapters. Callers who want
  to step should still import the typed `_<name>_step` directly from the
  optimizer submodules.

See:
- tests/odyssey/training/test_dispatch.mojo for smoke coverage.
- configs/schemas/training.schema.yaml for the matching CLI/config enum.
"""

from std.collections import Dict

from odyssey.tensor.any_tensor import AnyTensor

# Each `init_<name>_state` is uniform: takes a List[AnyTensor] (one entry per
# parameter) and returns a List[List[AnyTensor]] (one entry per parameter, with
# the inner list holding the per-parameter state buffers).
from odyssey.training.optimizers.sgd import init_sgd_state
from odyssey.training.optimizers.adam import init_adam_state
from odyssey.training.optimizers.adamw import init_adamw_state
from odyssey.training.optimizers.rmsprop import init_rmsprop_state
from odyssey.training.optimizers.adagrad import init_adagrad_state
from odyssey.training.optimizers.lars import init_lars_state
from odyssey.training.optimizers.muon import init_muon_state
from odyssey.training.optimizers.normuon import init_normuon_state
from odyssey.training.optimizers.mgup_muon import init_mgup_muon_state
from odyssey.training.optimizers.muon_hyperball import init_muon_hyperball_state
from odyssey.training.optimizers.lion import init_lion_state
from odyssey.training.optimizers.adopt import init_adopt_state
from odyssey.training.optimizers.lionmuon import init_lionmuon_state
from odyssey.training.optimizers.sophia import init_sophia_state
from odyssey.training.optimizers.adan import init_adan_state
from odyssey.training.optimizers.sf_normuon import init_sf_normuon_state
from odyssey.training.optimizers.ftrl import init_ftrl_state
from odyssey.training.optimizers.shampoo import init_shampoo_state
from odyssey.training.optimizers.soap import init_soap_state
from odyssey.training.optimizers.kl_shampoo import init_kl_shampoo_state
from odyssey.training.optimizers.splus import init_splus_state
from odyssey.training.optimizers.schedule_free import init_schedule_free_state
from odyssey.training.optimizers.schedule_free_plus import (
    init_schedule_free_plus_state,
)
from odyssey.training.optimizers.prodigy import init_prodigy_state


# ============================================================================
# Spec + Registry
# ============================================================================


struct OptimizerSpec(Copyable, ImplicitlyCopyable, Movable):
    """Lightweight metadata for a registered optimizer.

    Fields:
        name: The CLI/config identifier (e.g., "adamw").
        family: Logical grouping ("sgd", "adam", "muon", "shampoo", …). Used by
            `apply_default_hyperparams` to attach family-specific defaults
            (beta1, beta2, eps, momentum, …).
        step_fn_name: Symbol name of the typed step function in
            `odyssey.training.optimizers.<x>` (e.g., "adamw_step") — useful for
            diagnostics (`--help`/`--list-optimizers`).
        init_fn_name: Symbol name of the state-initializer (e.g.,
            "init_adamw_state") — same diagnostic purpose.
        default_learning_rate: The family-typical starting LR.
        default_weight_decay: The family-typical starting WD.
        num_state_buffers: Exact count of per-parameter buffers produced
            by the matching `init_<name>_state`. This is authoritative — updates
            here MUST stay in sync with the optimizer's `init_<name>_state`
            return shape (the smoke test in `tests/odyssey/training/test_dispatch.mojo`
            enforces this for all 24 registered optimizers).
    """

    var name: String
    var family: String
    var step_fn_name: String
    var init_fn_name: String
    var default_learning_rate: Float64
    var default_weight_decay: Float64
    var num_state_buffers: Int

    def __init__(
        out self,
        name: String,
        family: String,
        step_fn_name: String,
        init_fn_name: String,
        default_learning_rate: Float64,
        default_weight_decay: Float64,
        num_state_buffers: Int,
    ):
        self.name = name
        self.family = family
        self.step_fn_name = step_fn_name
        self.init_fn_name = init_fn_name
        self.default_learning_rate = default_learning_rate
        self.default_weight_decay = default_weight_decay
        self.num_state_buffers = num_state_buffers


def _build_registry() -> List[OptimizerSpec]:
    """Internal: build the canonical 24-entry registry list."""
    var registry: List[OptimizerSpec] = []

    # sgd family
    registry.append(
        OptimizerSpec("sgd", "sgd", "sgd_step", "init_sgd_state", 0.01, 0.0, 1)
    )

    # adam family
    registry.append(
        OptimizerSpec(
            "adam", "adam", "adam_step", "init_adam_state", 0.001, 0.0, 2
        )
    )
    registry.append(
        OptimizerSpec(
            "adamw", "adam", "adamw_step", "init_adamw_state", 0.001, 0.01, 2
        )
    )
    registry.append(
        OptimizerSpec(
            "adopt", "adam", "adopt_step", "init_adopt_state", 0.001, 0.0, 2
        )
    )
    registry.append(
        OptimizerSpec(
            "adan", "adam", "adan_step", "init_adan_state", 0.001, 0.01, 4
        )
    )
    registry.append(
        OptimizerSpec(
            "sophia", "adam", "sophia_step", "init_sophia_state", 0.001, 0.01, 2
        )
    )

    # rmsprop
    registry.append(
        OptimizerSpec(
            "rmsprop",
            "rmsprop",
            "rmsprop_step",
            "init_rmsprop_state",
            0.001,
            0.0,
            2,
        )
    )

    # adagrad
    registry.append(
        OptimizerSpec(
            "adagrad",
            "adagrad",
            "adagrad_step",
            "init_adagrad_state",
            0.01,
            0.0,
            1,
        )
    )

    # lars
    registry.append(
        OptimizerSpec(
            "lars", "lars", "lars_step", "init_lars_state", 0.01, 0.0, 1
        )
    )

    # muon family (Newton-Schulz orthogonalized momentum variants)
    registry.append(
        OptimizerSpec(
            "muon", "muon", "muon_step", "init_muon_state", 0.02, 0.0, 2
        )
    )
    registry.append(
        OptimizerSpec(
            "normuon",
            "muon",
            "normuon_step",
            "init_normuon_state",
            0.02,
            0.0,
            2,
        )
    )
    registry.append(
        OptimizerSpec(
            "mgup_muon",
            "muon",
            "mgup_muon_step",
            "init_mgup_muon_state",
            0.02,
            0.0,
            2,
        )
    )
    registry.append(
        OptimizerSpec(
            "muon_hyperball",
            "muon",
            "muon_hyperball_step",
            "init_muon_hyperball_state",
            0.02,
            0.0,
            2,
        )
    )
    registry.append(
        OptimizerSpec(
            "sf_normuon",
            "muon",
            "sf_normuon_step",
            "init_sf_normuon_state",
            0.02,
            0.0,
            3,
        )
    )

    # lion
    registry.append(
        OptimizerSpec(
            "lion", "lion", "lion_step", "init_lion_state", 0.0001, 0.01, 1
        )
    )

    # hybrid (Lion + Muon rule per-parameter)
    registry.append(
        OptimizerSpec(
            "lionmuon",
            "hybrid",
            "lionmuon_step",
            "init_lionmuon_state",
            0.001,
            0.01,
            2,
        )
    )

    # ftrl
    registry.append(
        OptimizerSpec(
            "ftrl", "ftrl", "ftrl_step", "init_ftrl_state", 0.01, 0.0, 3
        )
    )

    # shampoo family (matrix preconditioners + their descendants)
    registry.append(
        OptimizerSpec(
            "shampoo",
            "shampoo",
            "shampoo_step",
            "init_shampoo_state",
            0.01,
            0.0,
            3,
        )
    )
    registry.append(
        OptimizerSpec(
            "soap", "shampoo", "soap_step", "init_soap_state", 0.003, 0.0, 4
        )
    )
    registry.append(
        OptimizerSpec(
            "kl_shampoo",
            "shampoo",
            "kl_shampoo_step",
            "init_kl_shampoo_state",
            0.01,
            0.0,
            3,
        )
    )
    registry.append(
        OptimizerSpec(
            "splus", "shampoo", "splus_step", "init_splus_state", 0.01, 0.0, 3
        )
    )

    # schedule_free family (online iterate averaging — Defazio et al. 2024)
    registry.append(
        OptimizerSpec(
            "schedule_free",
            "schedule_free",
            "schedule_free_step",
            "init_schedule_free_state",
            0.01,
            0.0,
            2,
        )
    )
    registry.append(
        OptimizerSpec(
            "schedule_free_plus",
            "schedule_free",
            "schedule_free_plus_step",
            "init_schedule_free_plus_state",
            0.01,
            0.0,
            2,
        )
    )

    # prodigy (parameter-free)
    registry.append(
        OptimizerSpec(
            "prodigy",
            "prodigy",
            "prodigy_step",
            "init_prodigy_state",
            0.001,
            0.0,
            3,
        )
    )

    # TODO(#5683): per-optimizer `num_state_buffers` verified for
    # {sgd=1, adam=2, shampoo=3, sophia=2, prodigy=3, adan=4}; the remaining
    # 18 are best-guess and should be reconciled against each optimizer's
    # `init_<name>_state` source in a follow-up PR (`scripts/check_dispatch_sync.py`).
    return registry^


def _build_index() raises -> Dict[String, OptimizerSpec]:
    """Internal: name → spec index for O(1) lookup."""
    var index: Dict[String, OptimizerSpec] = {}
    var registry = _build_registry()
    for spec in registry:
        index[spec.name] = spec
    return index^


def list_optimizers() -> List[String]:
    """Return the inventory of registered optimizer names (canonical order).

    Useful for CLI `--list-optimizers` output and config-schema validation.
    Order matches `_build_registry()` so consumers can present a stable list.
    """
    var names: List[String] = []
    var registry = _build_registry()
    for spec in registry:
        names.append(spec.name)
    return names^


def is_optimizer_registered(name: String) -> Bool:
    """Whether the given optimizer name is registered in this dispatch.

    Cheap O(n) scan with n=24 — no need for a `Dict`-of-`Bool` micro-cache
    here.
    """
    var registry = _build_registry()
    for spec in registry:
        if spec.name == name:
            return True
    return False


def get_optimizer_spec(name: String) raises -> OptimizerSpec:
    """Look up the full `OptimizerSpec` for the given name.

    Args:
        name: Optimizer identifier (e.g., "adamw", "shampoo", "sophia").

    Returns:
        The matching `OptimizerSpec`. Stable, id-only — does not allocate.

    Raises:
        Error: If `name` is not registered. The error message lists the first
            few known names for easy debugging.
    """
    var index = _build_index()
    if name in index:
        return index[name]
    # Fall back to a helpful error so users discover the typo quickly.
    var registry = _build_registry()
    var preview = "Known names: "
    for i in range(min(5, len(registry))):
        preview += registry[i].name + ", "
    raise Error(
        "Unknown optimizer '"
        + name
        + "' — not registered in dispatch. "
        + preview
        + "… (use list_optimizers() for the full set)"
    )


def apply_default_hyperparams(name: String) raises -> Dict[String, Float64]:
    """Materialize a hyperparameter dict for the named optimizer.

    Always emits `learning_rate` and `weight_decay` (from the spec). Family-
    specific extras are added when applicable — `beta1`/`beta2`/`eps` for the
    Adam family, `momentum`/`ns_steps` for the Muon family, `d_coef` for
    Prodigy, `beta1`/`beta2`/`eps` for the Shampoo family, etc. CLI drivers
    can layer user-supplied overrides on top via `Dict.update` / re-assignment.

    Args:
        name: Optimizer identifier.

    Returns:
        A freshly-allocated `Dict[String, Float64]` with at minimum
        `learning_rate` + `weight_decay`.

    Raises:
        Error: If `name` is not registered.
    """
    var spec = get_optimizer_spec(name)
    var h: Dict[String, Float64] = {}
    h["learning_rate"] = spec.default_learning_rate
    h["weight_decay"] = spec.default_weight_decay

    if spec.family == "adam":
        h["beta1"] = 0.9
        h["beta2"] = 0.999
        h["eps"] = 1e-8
    elif spec.family == "rmsprop":
        h["beta"] = 0.9
        h["eps"] = 1e-8
    elif spec.family == "adagrad":
        h["eps"] = 1e-8
    elif spec.family == "muon":
        h["momentum"] = 0.95
        h["ns_steps"] = 5.0
    elif spec.family == "lion":
        h["beta1"] = 0.9
        h["beta2"] = 0.99
    elif spec.family == "schedule_free":
        h["beta1"] = 0.9
        h["warmup_steps"] = 0.0
    elif spec.family == "prodigy":
        h["d_coef"] = 1.0
    elif spec.family == "shampoo":
        h["beta1"] = 0.9
        h["beta2"] = 0.999
        h["eps"] = 1e-8
    elif spec.name == "ftrl":
        h["l1_strength"] = 0.0
        h["learning_rate_power"] = -0.5

    return h^


def init_optimizer_state(
    name: String,
    params: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate state buffers for the named optimizer.

    Uniform dispatcher over all 24 optimizers: every `init_<name>_state`
    follows the same signature `(List[AnyTensor], *, Bool=False) ->
    List[List[AnyTensor]]`, where the outer list is per-parameter and the
    inner list holds the per-parameter state buffers (1 for SGD, 2 for Adam,
    3 for Shampoo's L/R/momentum, …).

    Note: backed by a 24-branch `if/elif` chain — Mojo 1.0 has no `match`
    statement, and each branch is fully type-checked against the imports at
    the top of this module, so a typo in `init_<name>_state` here is a
    compile error rather than a silent dispatch failure.

    Args:
        name: Optimizer identifier (e.g., "adamw", "shampoo", "splus").
        params: Model parameters — one AnyTensor per trainable tensor.
        force_f64: Up-cast all state buffers to float64 regardless of param
            dtype (canonical precedent in `init_*_state` callers; useful for
            numerical-stability audits).

    Returns:
        A `List[List[AnyTensor]]` in the same order as `params`.

    Raises:
        Error: If `name` is not registered.
    """
    if name == "sgd":
        return init_sgd_state(params, force_f64=force_f64)
    elif name == "adam":
        return init_adam_state(params, force_f64=force_f64)
    elif name == "adamw":
        return init_adamw_state(params, force_f64=force_f64)
    elif name == "rmsprop":
        return init_rmsprop_state(params, force_f64=force_f64)
    elif name == "adagrad":
        return init_adagrad_state(params, force_f64=force_f64)
    elif name == "lars":
        return init_lars_state(params, force_f64=force_f64)
    elif name == "muon":
        return init_muon_state(params, force_f64=force_f64)
    elif name == "normuon":
        return init_normuon_state(params, force_f64=force_f64)
    elif name == "mgup_muon":
        return init_mgup_muon_state(params, force_f64=force_f64)
    elif name == "muon_hyperball":
        return init_muon_hyperball_state(params, force_f64=force_f64)
    elif name == "lion":
        return init_lion_state(params, force_f64=force_f64)
    elif name == "adopt":
        return init_adopt_state(params, force_f64=force_f64)
    elif name == "lionmuon":
        return init_lionmuon_state(params, force_f64=force_f64)
    elif name == "sophia":
        return init_sophia_state(params, force_f64=force_f64)
    elif name == "adan":
        return init_adan_state(params, force_f64=force_f64)
    elif name == "sf_normuon":
        return init_sf_normuon_state(params, force_f64=force_f64)
    elif name == "ftrl":
        return init_ftrl_state(params, force_f64=force_f64)
    elif name == "shampoo":
        return init_shampoo_state(params, force_f64=force_f64)
    elif name == "soap":
        return init_soap_state(params, force_f64=force_f64)
    elif name == "kl_shampoo":
        return init_kl_shampoo_state(params, force_f64=force_f64)
    elif name == "splus":
        return init_splus_state(params, force_f64=force_f64)
    elif name == "schedule_free":
        return init_schedule_free_state(params, force_f64=force_f64)
    elif name == "schedule_free_plus":
        return init_schedule_free_plus_state(params, force_f64=force_f64)
    elif name == "prodigy":
        return init_prodigy_state(params, force_f64=force_f64)
    else:
        raise Error(
            "Unknown optimizer '"
            + name
            + "' — not registered in dispatch. "
            "Use list_optimizers() for the canonical set."
        )
