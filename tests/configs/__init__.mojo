"""Configuration tests package.

This package contains comprehensive tests for the configuration management system.
Tests follow TDD principles and will be validated by Issue #74 implementation.

Test Files:
- test_loading.mojo: Configuration loading from YAML/JSON
- test_merging.mojo: Configuration merging (defaults → paper → experiment)
- test_validation_part1.mojo: Required key and type validation (ADR-009 split)
- test_validation_part2.mojo: Range and enum validation (ADR-009 split)
- test_validation_part3.mojo: Exclusive, complex, and validator builder tests (ADR-009 split)
- test_env_vars.mojo: Environment variable substitution
- test_schema.py: JSON schema validation (Python/pytest)
- test_integration.mojo: End-to-end integration tests

Fixtures:
- fixtures/valid_training.yaml: Valid training configuration
- fixtures/invalid_training.yaml: Invalid configuration for error testing
- fixtures/minimal.yaml: Minimal valid configuration
- fixtures/complex.yaml: Complex nested configuration
- fixtures/env_vars.yaml: Configuration with environment variables

Run Tests:
    mojo test tests/configs/test_loading.mojo
    mojo test tests/configs/test_merging.mojo
    mojo test tests/configs/test_validation_part1.mojo
    mojo test tests/configs/test_validation_part2.mojo
    mojo test tests/configs/test_validation_part3.mojo
    mojo test tests/configs/test_env_vars.mojo
    mojo test tests/configs/test_integration.mojo
    pytest tests/configs/test_schema.py
"""
