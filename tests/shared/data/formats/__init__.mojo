"""Tests for data format loaders.

This package contains tests for various data format loaders used in ML Odyssey.

Modules:
    test_cifar_loader_part1: Tests for CIFAR-10 and CIFAR-100 binary format loader (init, labels, shapes)
    test_cifar_loader_part2: Tests for CIFAR-10 and CIFAR-100 binary format loader (validation, counts)

Note: test_cifar_loader.mojo was split per ADR-009 to stay within the ≤10 fn test_ limit,
avoiding Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so).
"""
