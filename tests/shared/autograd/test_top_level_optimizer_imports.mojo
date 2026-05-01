"""Tests for top-level optimizer imports from shared package.

Verifies that AdaGrad, RMSprop, SGD, Adam, and AdamW are all importable
directly from the top-level `shared` package (Issue #3745).

Run with: mojo test tests/shared/autograd/test_top_level_optimizer_imports.mojo

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""

from std.testing import assert_true
from tests.shared.conftest import assert_almost_equal

# ADR-015: This file intentionally validates top-level shared package exports.
# Do not convert to targeted imports — that would defeat its purpose.
from shared import AdaGrad, RMSprop, SGD, Adam, AdamW


# ============================================================================
# Top-Level Import Tests (Issue #3745)
# ============================================================================


def test_adagrad_top_level_import() raises:
    """Test AdaGrad is importable from top-level shared package."""
    var opt = AdaGrad(learning_rate=0.01)
    assert_almost_equal(opt.get_lr(), 0.01, tolerance=1e-10)

    opt.set_lr(0.005)
    assert_almost_equal(opt.get_lr(), 0.005, tolerance=1e-10)

    print("✓ AdaGrad top-level import test passed")


def test_rmsprop_top_level_import() raises:
    """Test RMSprop is importable from top-level shared package."""
    var opt = RMSprop(learning_rate=0.001)
    assert_almost_equal(opt.get_lr(), 0.001, tolerance=1e-10)

    opt.set_lr(0.01)
    assert_almost_equal(opt.get_lr(), 0.01, tolerance=1e-10)

    print("✓ RMSprop top-level import test passed")


def test_sgd_top_level_import() raises:
    """Test SGD is importable from top-level shared package (regression)."""
    var opt = SGD(learning_rate=0.01)
    assert_almost_equal(opt.get_lr(), 0.01, tolerance=1e-10)

    print("✓ SGD top-level import test passed")


def test_adam_top_level_import() raises:
    """Test Adam is importable from top-level shared package (regression)."""
    var opt = Adam(learning_rate=0.001)
    assert_almost_equal(opt.get_lr(), 0.001, tolerance=1e-10)

    print("✓ Adam top-level import test passed")


def test_adamw_top_level_import() raises:
    """Test AdamW is importable from top-level shared package (regression)."""
    var opt = AdamW(learning_rate=0.001)
    assert_almost_equal(opt.get_lr(), 0.001, tolerance=1e-10)

    print("✓ AdamW top-level import test passed")


# ============================================================================
# Test Main
# ============================================================================


def main() raises:
    """Run top-level optimizer import tests."""
    print("Running top-level optimizer import tests (Issue #3745)...")

    test_adagrad_top_level_import()
    test_rmsprop_top_level_import()
    test_sgd_top_level_import()
    test_adam_top_level_import()
    test_adamw_top_level_import()

    print("\nAll top-level optimizer import tests passed! ✓")
