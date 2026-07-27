"""Optimizer-name dispatch yard.

Maps an optimizer name (e.g. ``"adan"``, ``"adam"``, ``"muon"``) to a
``Dict[String, Float64]`` of paper-accurate default Float64 hyperparameters.

Public API:

- ``get_default_hyperparams(name) raises -> Dict[String, Float64]``
- ``all_supported_optimizers() -> List[String]`` (24 entries, sorted)

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

from odyssey.training.optimizers.adan import adan_default_hyperparams


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
        # TODO: verify against `schedule_free_plus_step` concrete kwargs
        # (Defazio 2026 reference values; may not match implementation).
        defaults["mu"] = 0.9
        defaults["beta_sf"] = 0.9
        defaults["beta_max"] = 0.98
        defaults["rho"] = 0.9
        defaults["epsilon"] = 1e-8
        defaults["horizon"] = 1000.0
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
