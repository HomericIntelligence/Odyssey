"""Tests for ``src/odyssey/training/dispatch.mojo``.

Contract under test:

- ``all_supported_optimizers()`` returns exactly 24 names, alphabetically sorted.
- ``get_default_hyperparams("adan")`` routes through
  ``adan_default_hyperparams()`` and exposes Adan's paper-accurate ``beta3``.
- ``get_default_hyperparams("adam")`` exposes Kingma & Ba 2014 defaults.
- ``get_default_hyperparams(<unknown>)`` raises.
- ``get_default_hyperparams("muon")`` returns a non-empty dict with momentum.

The ``adan`` test is the regression-locking test for the family-fallback bug:
any future refactor that re-implements silent family-fallback defaults will
fail CI at this point.
"""

from odyssey.training import (
    all_supported_optimizers,
    get_default_hyperparams,
)


def test_supported_count_is_24() raises:
    """The roster must contain exactly 24 entries -- one per optimizer file."""
    var names = all_supported_optimizers()
    if len(names) != 24:
        raise Error(
            String("all_supported_optimizers() should return 24 names, got ")
            + String(len(names))
        )
    print("  ok all_supported_optimizers() returns 24 names")


def test_supported_names_are_alphabetical() raises:
    """Alphabetical strict-sorted -- stable iteration order for YAMLs/CLIs."""
    var names = all_supported_optimizers()
    for i in range(len(names) - 1):
        if names[i] >= names[i + 1]:
            raise Error(
                String("names not in sorted order at index ")
                + String(i)
                + String(": ")
                + names[i]
                + String(" >= ")
                + names[i + 1]
            )
    print("  ok all_supported_optimizers() is strictly sorted")


def test_adan_paper_defaults() raises:
    """The regression-locking test.

    Per the sail-sg/Adan reference and Xie et al. 2022 (arXiv:2208.06677),
    Adan's defaults are ``beta1=0.98, beta2=0.92, beta3=0.99, epsilon=1e-8``.
    An earlier dispatch yard fell back to ``family="adam"`` and silently
    dropped ``beta3`` -- this test catches that drift.
    """
    var defaults = get_default_hyperparams(String("adan"))
    if defaults["beta1"] != 0.98:
        raise Error(String("adan: beta1 expected 0.98"))
    if defaults["beta2"] != 0.92:
        raise Error(String("adan: beta2 expected 0.92"))
    if defaults["beta3"] != 0.99:
        raise Error(
            String("adan: beta3 expected 0.99 -- regression of ")
            + String("family=adam fallback bug")
        )
    if defaults["epsilon"] != 1e-8:
        raise Error(String("adan: epsilon expected 1e-8"))
    print(
        "  ok adan paper defaults "
        + String("(beta1=0.98, beta2=0.92, beta3=0.99, epsilon=1e-8)")
    )


def test_adam_paper_defaults() raises:
    """Verify a routine adam-family branch -- baseline for non-adan cases."""
    var defaults = get_default_hyperparams(String("adam"))
    if defaults["beta1"] != 0.9:
        raise Error(String("adam: beta1 expected 0.9"))
    if defaults["beta2"] != 0.999:
        raise Error(String("adam: beta2 expected 0.999"))
    if defaults["epsilon"] != 1e-8:
        raise Error(String("adam: epsilon expected 1e-8"))
    print(
        "  ok adam paper defaults "
        + String("(beta1=0.9, beta2=0.999, epsilon=1e-8)")
    )


def test_unknown_name_raises() raises:
    """Strict routing: an unsupported name must raise, not silently fall back.

    This is the contract that prevents future family-fallback bugs from
    regressing. Silent fallback was the root cause of the Adan beta3 bug.
    """
    var raised = False
    try:
        _ = get_default_hyperparams(String("definitely_not_a_real_optimizer"))
    except _:
        raised = True
    if not raised:
        raise Error(
            String("get_default_hyperparams should have raised for an ")
            + String("unknown name, but no exception was raised")
        )
    print("  ok unknown name raises Error")


def test_muon_returns_non_empty_defaults() raises:
    """Spot-check: a non-adam-family optimizer returns a non-empty dict."""
    var defaults = get_default_hyperparams(String("muon"))
    if len(defaults) == 0:
        raise Error(String("muon defaults should be non-empty"))
    if "momentum" not in defaults:
        raise Error(String("muon defaults should expose a 'momentum' key"))
    print("  ok muon returns non-empty defaults including 'momentum'")


def test_loop_all_24_known() raises:
    """Loop over the 24-name roster: every entry must yield non-empty defaults.

    This locks the registry-vs-defaults contract. Any future PR that adds
    a name to ``all_supported_optimizers()`` without adding a corresponding
    elif branch in ``get_default_hyperparams()`` fails this test.
    """
    var names = all_supported_optimizers()
    for i in range(len(names)):
        var defaults = get_default_hyperparams(names[i])
        if len(defaults) == 0:
            raise Error(
                String("get_default_hyperparams(")
                + names[i]
                + String(") returned empty dict -- registry-vs-defaults drift")
            )
    print("  ok all 24 names return non-empty defaults")


def main() raises:
    test_supported_count_is_24()
    test_supported_names_are_alphabetical()
    test_loop_all_24_known()
    test_adan_paper_defaults()
    test_adam_paper_defaults()
    test_unknown_name_raises()
    test_muon_returns_non_empty_defaults()
    print("All dispatch tests PASSED")
