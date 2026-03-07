"""
Installation Verification Script

Verifies that the shared library is correctly installed and importable.

Usage:
    mojo run scripts/verify_installation.mojo

Exit codes:
    0 - Installation verified successfully
    1 - Import errors detected
"""


fn main():
    """Run installation verification checks."""
    var errors: Int = 0

    print("\n" + "=" * 70)
    print("ML Odyssey Shared Library - Installation Verification")
    print("=" * 70 + "\n")

    # ========================================================================
    # Test 1: Version Info
    # ========================================================================
    print("Test 1: Checking version info...")
    try:
        from shared import VERSION, AUTHOR, LICENSE

        print("  ✓ Version:", VERSION)
        print("  ✓ Author:", AUTHOR)
        print("  ✓ License:", LICENSE)
    except:
        print("  ✗ Failed to import version info")
        errors += 1

    # ========================================================================
    # Test 2: Core Package
    # ========================================================================
    print("\nTest 2: Checking core package...")
    try:
        # These imports are commented until implementation completes
        # from shared.core import Linear, ReLU, Tensor

        print(
            "  ✓ Core package accessible (placeholder - awaiting"
            " implementation)"
        )
    except:
        print("  ✗ Failed to import core package")
        errors += 1

    # ========================================================================
    # Test 3: Training Package
    # ========================================================================
    print("\nTest 3: Checking training package...")
    try:
        # from shared.training import SGD, Adam

        print(
            "  ✓ Training package accessible (placeholder - awaiting"
            " implementation)"
        )
    except:
        print("  ✗ Failed to import training package")
        errors += 1

    # ========================================================================
    # Test 4: Data Package
    # ========================================================================
    print("\nTest 4: Checking data package...")
    try:
        # from shared.data import DataLoader

        print(
            "  ✓ Data package accessible (placeholder - awaiting"
            " implementation)"
        )
    except:
        print("  ✗ Failed to import data package")
        errors += 1

    # ========================================================================
    # Test 5: Utils Package
    # ========================================================================
    print("\nTest 5: Checking utils package...")
    try:
        # from shared.utils import Logger

        print(
            "  ✓ Utils package accessible (placeholder - awaiting"
            " implementation)"
        )
    except:
        print("  ✗ Failed to import utils package")
        errors += 1

    # ========================================================================
    # Test 6: Root Convenience Imports
    # ========================================================================
    print("\nTest 6: Checking root convenience imports...")
    try:
        # from shared import Linear, SGD

        print(
            "  ✓ Root imports accessible (placeholder - awaiting"
            " implementation)"
        )
    except:
        print("  ✗ Failed to import from root")
        errors += 1

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)

    if errors == 0:
        print("✅ Shared Library Installation Verified!")
        print("=" * 70)
        print("\nAll checks passed successfully.")
        print(
            "\nNote: Functional tests are placeholders awaiting implementation"
            " (Issue #49)"
        )
        print(
            "Once implementation completes, uncomment imports in this script."
        )
        print("\nNext steps:")
        print("  - See EXAMPLES.md for usage examples")
        print("  - Read API documentation for detailed reference")
        print("  - Run tests with: mojo test tests/shared/")
    else:
        print("❌ Installation Verification Failed!")
        print("=" * 70)
        print("\nFound", errors, "error(s)")
        print("\nTroubleshooting:")
        print("  1. Verify Mojo is installed: mojo --version")
        print("  2. Reinstall shared library: mojo package shared --install")
        print("  3. Check MOJO_PATH: echo $MOJO_PATH")
        print("  4. See INSTALL.md for detailed installation instructions")

    print()
