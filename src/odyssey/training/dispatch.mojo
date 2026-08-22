"""Optimizer-name dispatch yard.

Maps an optimizer name (e.g. ``"adan"``, ``"adam"``, ``"muon"``) to a
``Dict[String, Float64]`` of paper-accurate default Float64 hyperparameters.

Public API:

- ``get_default_hyperparams(name) raises -> Dict[String, Float64]``
- ``all_supported_optimizers() -> List[String]`` (24 entries, sorted)
- ``init_optimizer_state(name, params, *, force_f64=False) raises -> List[List[AnyTensor]]``
  — uniform state allocation across all 24 optimizers

This module closes the loop documented in PR #5707's "What this PR does NOT
fix" caveat: the previous implicit family fallback silently stripped Adan's
``beta3`` (Xie et al. 2022 -- arXiv:2208.06677) when callers requested
``"adan"`` by name. Routing is now strict.

Note:
    The dispatcher imports ``adan_default_hyperparams`` directly from
    ``odyssey.training.optimizers.adan`` to preserve a single source of
    truth for Adan paper defaults. The other 23 optimizers have inline
    dicts here; future PRs may migrate them to sibling helpers
    (e.g. ``adam_default_hyperparams``) that mirror the adan pattern.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.training.optimizers.adan import adan_default_hyperparams

# Uniform state-allocation dispatch: each ``init_<name>_state`` has the same
# canonical signature ``(List[AnyTensor], *, Bool) -> List[List[AnyTensor]]``.
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


def all_supported_optimizers() -> List[String]:
    """Return the canonical roster of optimizer names (exactly 24 entries).

    The list is sorted alphabetically for stable iteration order across YAML
    schemas, CLI completions, and Python generators. **Each entry in this
    list must have a corresponding branch in** ``get_default_hyperparams``;
    adding a new optimizer requires both edits.

    Returns:
        List[String] of 24 optimizer names.
    """
    return [
        String("adagrad"),
        String("adam"),
        String("adamw"),
        String("adan"),
        String("adopt"),
        String("ftrl"),
        String("kl_shampoo"),
        String("lars"),
        String("lion"),
        String("lionmuon"),
        String("mgup_muon"),
        String("muon"),
        String("muon_hyperball"),
        String("normuon"),
        String("prodigy"),
        String("rmsprop"),
        String("schedule_free"),
        String("schedule_free_plus"),
        String("sf_normuon"),
        String("sgd"),
        String("shampoo"),
        String("soap"),
        String("sophia"),
        String("splus"),
    ]


def get_default_hyperparams(name: String) raises -> Dict[String, Float64]:
    """Return paper-accurate default Float64 hyperparameters for ``name``.

    Args:
        name: Optimizer name as a string (e.g. ``"adan"``, ``"adam"``, ``"muon"``).
            Must be one of ``all_supported_optimizers()``.

    Returns:
        ``Dict[String, Float64]`` mapping hyperparameter name (``"beta1"``,
        ``"epsilon"``, etc.) to its default value. The dict excludes non-Float64
        hyperparameters -- Int counts like ``ns_steps``, Bool flags like
        ``nesterov`` -- those remain at the underlying ``<name>_step`` function's
        positional defaults.

    Raises:
        Raises ``Error`` if ``name`` is not in ``all_supported_optimizers()``.
        Strict routing is intentional: the silent family fallback enabled the
        Adan beta3 regression (Xie et al. 2022). Callers wanting a fallback must
        implement it around this dispatcher, not via here.
    """
    var defaults = Dict[String, Float64]()
    if name == "adan":
        return adan_default_hyperparams()
    elif name == "adagrad":
        defaults["lr_decay"] = 0.0
        defaults["weight_decay"] = 0.0
    elif name == "adam":
        defaults["beta1"] = 0.9
        defaults["beta2"] = 0.999
        defaults["epsilon"] = 1e-8
    elif name == "adamw":
        defaults["beta1"] = 0.9
        defaults["beta2"] = 0.999
        defaults["epsilon"] = 1e-8
        defaults["weight_decay"] = 0.01
    elif name == "adopt":
        defaults["beta1"] = 0.9
        defaults["beta2"] = 0.9999
        defaults["epsilon"] = 1e-8
        defaults["weight_decay"] = 0.0
    elif name == "ftrl":
        defaults["alpha"] = 0.1
        defaults["beta"] = 1.0
        defaults["lambda1"] = 0.0
        defaults["lambda2"] = 0.0
    elif name == "kl_shampoo":
        defaults["beta"] = 0.95
        defaults["weight_decay"] = 0.0
        defaults["ridge"] = 1e-8
    elif name == "lars":
        defaults["trust_coefficient"] = 0.001
    elif name == "lion":
        defaults["beta1"] = 0.9
        defaults["beta2"] = 0.99
        defaults["weight_decay"] = 0.0
    elif name == "lionmuon":
        defaults["beta1"] = 0.9
        defaults["beta2"] = 0.99
    elif name == "mgup_muon":
        defaults["momentum"] = 0.95
    elif name == "muon":
        defaults["momentum"] = 0.95
    elif name == "muon_hyperball":
        defaults["momentum"] = 0.95
    elif name == "normuon":
        defaults["momentum"] = 0.95
    elif name == "prodigy":
        defaults["gamma"] = 1.0
        defaults["beta1"] = 0.9
        defaults["beta2"] = 0.999
        defaults["epsilon"] = 1e-8
        defaults["growth_rate"] = 1e30
    elif name == "rmsprop":
        defaults["alpha"] = 0.99
        defaults["epsilon"] = 1e-8
        defaults["momentum"] = 0.0
    elif name == "schedule_free":
        defaults["beta"] = 0.9
        defaults["weight_power"] = 0.0
    elif name == "schedule_free_plus":
        # Verified against schedule_free_plus_step defaults (Defazio 2026).
        # horizon is Int in the step function, excluded per the Float64-only
        # dict contract; callers pass it positionally.
        defaults["mu"] = 0.9
        defaults["beta_sf"] = 0.9
        defaults["beta_max"] = 0.98
        defaults["rho"] = 0.9
        defaults["epsilon"] = 1e-8
    elif name == "sf_normuon":
        defaults["beta"] = 0.9
        defaults["mu"] = 0.95
        defaults["weight_decay"] = 0.0
        defaults["weight_power"] = 0.0
        defaults["eps"] = 1e-8
    elif name == "shampoo":
        defaults["beta_precond"] = 0.95
        defaults["beta_momentum"] = 0.95
        defaults["weight_decay"] = 0.0
        defaults["eps"] = 1e-10
        defaults["max_precond_norm"] = 1e6
    elif name == "sgd":
        defaults["momentum"] = 0.0
        defaults["dampening"] = 0.0
        defaults["weight_decay"] = 0.0
    elif name == "soap":
        defaults["beta1"] = 0.95
        defaults["beta2"] = 0.95
        defaults["shampoo_beta"] = 0.95
        defaults["weight_decay"] = 0.01
        defaults["epsilon"] = 1e-8
    elif name == "sophia":
        defaults["rho"] = 0.04
    elif name == "splus":
        defaults["beta1"] = 0.9
        defaults["beta2"] = 0.999
        defaults["ema_rate"] = 0.999
        defaults["weight_decay"] = 0.0
        defaults["sign_eps"] = 1e-12
        defaults["eig_eps"] = 1e-30
    else:
        raise Error(
            String("get_default_hyperparams: unknown optimizer name: ")
            + name
            + String(
                " (call all_supported_optimizers() for the canonical roster)"
            )
        )
    return defaults^


def init_optimizer_state(
    name: String,
    params: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate state buffers for the named optimizer.

    Uniform dispatcher over all 24 optimizers: every ``init_<name>_state``
    follows the same signature ``(List[AnyTensor], *, Bool=False) ->
    List[List[AnyTensor]]``, where the outer list is per-parameter and the
    inner list holds the per-parameter state buffers (1 for SGD, 2 for Adam,
    3 for Shampoo's L/R/momentum, …).

    Backed by a 24-branch ``if/elif`` chain — Mojo 1.0 has no ``match``
    statement, and each branch is fully type-checked against the imports at
    the top of this module, so a typo in ``init_<name>_state`` here is a
    compile error rather than a silent dispatch failure.

    Args:
        name: Optimizer identifier (e.g., "adamw", "shampoo", "splus").
        params: Model parameters — one AnyTensor per trainable tensor.
        force_f64: Up-cast all state buffers to float64 regardless of param
            dtype.

    Returns:
        A ``List[List[AnyTensor]]`` in the same order as ``params``.

    Raises:
        Error: If ``name`` is not in ``all_supported_optimizers()``.
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
            String("init_optimizer_state: unknown optimizer name: ")
            + name
            + String(
                " (call all_supported_optimizers() for the canonical roster)"
            )
        )
