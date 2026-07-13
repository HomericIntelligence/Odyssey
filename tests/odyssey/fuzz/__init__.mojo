"""Fuzz testing package for ML Odyssey.

This package contains property-based tests that use fuzzing to discover
edge cases and unexpected behaviors in tensor operations.

Modules:
    test_tensor_fuzz: Fuzz tests for tensor creation and arithmetic operations

Usage:
    # Run all fuzz tests
    mojo test tests/shared/fuzz/

    # Run specific fuzz test file
    mojo test tests/shared/fuzz/test_tensor_fuzz.mojo
"""
