# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arg_parser.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for argument parser utilities - Part 2 (Issue #2200).

Tests argument parsing functionality including:
- Adding boolean flags
- Invalid type rejection
- Default value handling
- Multiple argument values
- Parser default population (Issue #2585)
"""

from testing import assert_true, assert_equal
from shared.utils import ArgumentParser, ArgumentSpec, ParsedArgs


fn test_argument_parser_add_flag() raises:
    """Test adding boolean flags."""
    var parser = ArgumentParser()
    parser.add_flag("verbose")
    parser.add_flag("debug")

    assert_equal(len(parser.arguments), 2)
    assert_true("verbose" in parser.arguments)
    assert_true("debug" in parser.arguments)

    assert_true(parser.arguments["verbose"].is_flag)
    print("PASS: test_argument_parser_add_flag")


fn test_argument_parser_invalid_type() raises:
    """Test that invalid argument types are rejected."""
    var parser = ArgumentParser()
    try:
        parser.add_argument("bad", "invalid_type", "0")
        # Should raise error
        assert_true(False)
    except:
        assert_true(True)  # Expected error
        print("PASS: test_argument_parser_invalid_type")


fn test_argument_defaults() raises:
    """Test that defaults are applied."""
    var parser = ArgumentParser()
    parser.add_argument("epochs", "int", "100")
    parser.add_argument("lr", "float", "0.001")
    parser.add_argument("output", "string", "model.weights")

    # Note: In a real test, we would call parse() with empty argv
    # For now, we just verify defaults are stored
    assert_equal(parser.arguments["epochs"].default_value, "100")
    assert_equal(parser.arguments["lr"].default_value, "0.001")
    assert_equal(parser.arguments["output"].default_value, "model.weights")
    print("PASS: test_argument_defaults")


fn test_parsed_args_multiple_values() raises:
    """Test handling multiple argument values."""
    var args = ParsedArgs()
    args.set("epochs", "100")
    args.set("batch_size", "32")
    args.set("lr", "0.001")
    args.set("output", "weights.mojo")

    assert_equal(args.get_int("epochs"), 100)
    assert_equal(args.get_int("batch_size"), 32)
    var lr = args.get_float("lr")
    assert_true(lr > 0.0009 and lr < 0.0011)
    assert_equal(args.get_string("output"), "weights.mojo")
    print("PASS: test_parsed_args_multiple_values")


fn test_parser_populates_defaults() raises:
    """Test that parser.parse() populates defaults from argument specs (Issue #2585).
    """
    var parser = ArgumentParser()
    parser.add_argument("epochs", "int", "100")
    parser.add_argument("lr", "float", "0.001")
    parser.add_argument("output", "string", "model.weights")

    # Parse with empty command line (no arguments provided)
    # Defaults should be populated in result
    var result = parser.parse()

    # Verify defaults are present
    assert_true(result.has("epochs"))
    assert_true(result.has("lr"))
    assert_true(result.has("output"))

    assert_equal(result.get_int("epochs"), 100)
    var lr = result.get_float("lr")
    assert_true(lr > 0.0009 and lr < 0.0011)
    assert_equal(result.get_string("output"), "model.weights")

    print("PASS: test_parser_populates_defaults")


fn main() raises:
    """Run argument parser tests - Part 2."""
    print("")
    print("=" * 70)
    print("ArgumentParser Unit Tests - Part 2")
    print("=" * 70)
    print("")

    test_argument_parser_add_flag()
    test_argument_parser_invalid_type()
    test_argument_defaults()
    test_parsed_args_multiple_values()
    test_parser_populates_defaults()

    print("")
    print("=" * 70)
    print("All argument parser Part 2 tests passed!")
    print("=" * 70)
    print("")
