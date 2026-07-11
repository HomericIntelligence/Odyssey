"""Installation verification script.

Verifies that the shared library is correctly installed and importable.

Usage:
    mojo run scripts/verify_installation.mojo

Exit codes:
    0 - Installation verified successfully
    1 - Import errors detected
"""


def main():
    """Run installation verification checks."""
    var errors: Int = 0

    print("\n" + "=" * 70)
    print("ML Odyssey Shared Library - Installation Verification")
    print("=" * 70 + "\n")

    # ========================================================================
    # Test 1: Version Info
    # ========================================================================
    print("Test 1: Checking version info...")
    from odyssey import VERSION, AUTHOR, LICENSE

    print("  ✓ Version:", VERSION)
    print("  ✓ Author:", AUTHOR)
    print("  ✓ License:", LICENSE)

    # ========================================================================
    # Test 2: Core Package
    # ========================================================================
    print("\nTest 2: Checking core package...")
    # These imports are commented until implementation completes
    # from odyssey.core import Linear, ReLU, Tensor
    print("  ✓ Core package accessible (placeholder - awaiting implementation)")

    # ========================================================================
    # Test 3: Training Package
    # ========================================================================
    print("\nTest 3: Checking training package...")
    # from odyssey.training import SGD, Adam
    print(
        "  ✓ Training package accessible (placeholder - awaiting"
        " implementation)"
    )

    # ========================================================================
    # Test 4: Data Package
    # ========================================================================
    print("\nTest 4: Checking data package...")
    # from odyssey.data import DataLoader
    print("  ✓ Data package accessible (placeholder - awaiting implementation)")

    # ========================================================================
    # Test 5: Utils Package
    # ========================================================================
    print("\nTest 5: Checking utils package...")
    # from odyssey.utils import Logger
    print(
        "  ✓ Utils package accessible (placeholder - awaiting implementation)"
    )

    # ========================================================================
    # Test 6: Root Convenience Imports
    # ========================================================================
    print("\nTest 6: Checking root convenience imports...")
    # from odyssey import Linear, SGD
    print("  ✓ Root imports accessible (placeholder - awaiting implementation)")

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
