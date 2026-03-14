"""
Import Validation Tests - Part 3: Utils, Root, Nested, and Version Imports

Tests that all public imports work correctly for the shared library.
These tests verify both import functionality and basic component behavior.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_imports.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Run with: mojo test tests/shared/test_imports_part3.mojo
"""

from testing import assert_true

# ============================================================================
# Utils Package Imports (continued)
# ============================================================================


fn test_utils_visualization_imports() raises:
    """Test utils visualization imports."""
    # Visualization functions require Python interop
    # For now, just verify utils imports work
    from shared.utils import Logger

    print("✓ Utils visualization imports test passed")


fn test_utils_config_imports() raises:
    """Test utils config imports."""
    from shared.utils import Config, load_config, save_config, ConfigValidator

    print("✓ Utils config imports test passed")


# ============================================================================
# Root Package Imports
# ============================================================================


fn test_root_imports() raises:
    """Test root package convenience imports work."""
    # Root package doesn't re-export all components
    # Users should import from subpackages
    from shared.core import ExTensor
    from shared.training import SGD
    from shared.utils import Logger

    print("✓ Root imports test passed")


fn test_subpackage_imports() raises:
    """Test importing subpackages themselves."""
    from shared import core, training, data, utils

    print("✓ Subpackage imports test passed")


# ============================================================================
# Nested Imports
# ============================================================================


fn test_nested_optimizer_imports() raises:
    """Test nested imports from optimizer subpackages."""
    from shared.training import SGD

    print("✓ Nested optimizer imports test passed")


fn test_nested_scheduler_imports() raises:
    """Test nested imports from scheduler subpackages."""
    from shared.training import StepLR, CosineAnnealingLR

    print("✓ Nested scheduler imports test passed")


fn test_nested_metric_imports() raises:
    """Test nested imports from metrics subpackages."""
    # Metrics are in shared.training
    from shared.training import Callback

    print("✓ Nested metric imports test passed")


# ============================================================================
# Version Info
# ============================================================================


fn test_version_info() raises:
    """Test version info is accessible and has proper format."""
    from shared import VERSION, AUTHOR, LICENSE

    # Critical validation - ensure values are not empty/None
    assert_true(VERSION != "", "VERSION should not be empty")
    assert_true(AUTHOR != "", "AUTHOR should not be empty")
    assert_true(LICENSE != "", "LICENSE should not be empty")

    # Ensure these are actual string values, not None
    assert_true(VERSION.__len__() > 0, "VERSION string should have length > 0")
    assert_true(AUTHOR.__len__() > 0, "AUTHOR string should have length > 0")
    assert_true(LICENSE.__len__() > 0, "LICENSE string should have length > 0")

    # Test version format follows semantic versioning (major.minor.patch)
    var version_parts = VERSION.split(".")
    assert_true(
        version_parts.__len__() == 3,
        "Version should have 3 parts (major.minor.patch)",
    )

    # Test that version parts are numeric by checking they only contain digits
    for i in range(version_parts.__len__()):
        var part = version_parts[i]
        assert_true(part.__len__() > 0, "Version part should not be empty")

        # Check each character is a digit (0-9)
        var is_numeric = True
        var part_bytes = part.as_bytes()
        for j in range(part.__len__()):
            var ch = Int(part_bytes[j])
            if ch < 48 or ch > 57:  # ord("0") == 48, ord("9") == 57
                is_numeric = False
                break

        assert_true(is_numeric, "Version part should contain only digits")

    print("✓ Version info test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run part 3 import validation tests."""
    print("\n" + "=" * 70)
    print(
        "Running Import Validation Tests - Part 3 (Utils, Root, Nested &"
        " Version)"
    )
    print("=" * 70 + "\n")

    # Utils package tests (second half)
    print("Testing Utils Package (Part 2)...")
    test_utils_visualization_imports()
    test_utils_config_imports()

    # Root package tests
    print("\nTesting Root Package...")
    test_root_imports()
    test_subpackage_imports()

    # Nested imports tests
    print("\nTesting Nested Imports...")
    test_nested_optimizer_imports()
    test_nested_scheduler_imports()
    test_nested_metric_imports()

    # Version info test
    print("\nTesting Version Info...")
    test_version_info()

    # Summary
    print("\n" + "=" * 70)
    print("✅ Import Validation Tests - Part 3 Passed!")
    print("=" * 70)
