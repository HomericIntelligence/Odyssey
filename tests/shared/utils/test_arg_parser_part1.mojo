# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arg_parser.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for argument parser utilities - Part 1 (Issue #2200).

Tests basic argument parsing functionality including:
- ArgumentSpec creation and fields
- ParsedArgs string, int, float, bool, and has() getters
- ArgumentParser creation and add_argument()
"""

from testing import assert_true, assert_equal
from shared.utils import ArgumentParser, ArgumentSpec, ParsedArgs


fn test_argument_spec_creation() raises:
    """Test creating argument specifications."""
    var spec = ArgumentSpec(
        name="epochs", arg_type="int", default_value="100", is_flag=False
    )
    assert_equal(spec.name, "epochs")
    assert_equal(spec.arg_type, "int")
    assert_equal(spec.default_value, "100")
    assert_true(not spec.is_flag)
    print("PASS: test_argument_spec_creation")


fn test_parsed_args_string() raises:
    """Test ParsedArgs string getter."""
    var args = ParsedArgs()
    args.set("output", "model.weights")
    assert_equal(args.get_string("output"), "model.weights")
    assert_equal(args.get_string("missing", "default"), "default")
    print("PASS: test_parsed_args_string")


fn test_parsed_args_int() raises:
    """Test ParsedArgs integer getter."""
    var args = ParsedArgs()
    args.set("epochs", "100")
    assert_equal(args.get_int("epochs"), 100)
    assert_equal(args.get_int("missing", 42), 42)
    print("PASS: test_parsed_args_int")


fn test_parsed_args_float() raises:
    """Test ParsedArgs float getter."""
    var args = ParsedArgs()
    args.set("lr", "0.001")
    var lr = args.get_float("lr")
    # Float comparison with tolerance
    assert_true(lr > 0.0009 and lr < 0.0011)
    assert_equal(args.get_float("missing", 0.1), 0.1)
    print("PASS: test_parsed_args_float")


fn test_parsed_args_bool() raises:
    """Test ParsedArgs boolean flag getter."""
    var args = ParsedArgs()
    args.set("verbose", "true")
    assert_true(args.get_bool("verbose"))
    assert_true(not args.get_bool("missing"))
    print("PASS: test_parsed_args_bool")


fn test_parsed_args_has() raises:
    """Test ParsedArgs has() method."""
    var args = ParsedArgs()
    args.set("epochs", "100")
    assert_true(args.has("epochs"))
    assert_true(not args.has("missing"))
    print("PASS: test_parsed_args_has")


fn test_argument_parser_creation() raises:
    """Test creating an argument parser."""
    var parser = ArgumentParser()
    parser.add_argument("epochs", "int", "100")
    assert_equal(len(parser.arguments), 1)
    print("PASS: test_argument_parser_creation")


fn test_argument_parser_add_arguments() raises:
    """Test adding typed arguments."""
    var parser = ArgumentParser()
    parser.add_argument("epochs", "int", "100")
    parser.add_argument("batch-size", "int", "32")
    parser.add_argument("lr", "float", "0.001")
    parser.add_argument("output", "string", "model.weights")

    assert_equal(len(parser.arguments), 4)
    assert_true("epochs" in parser.arguments)
    assert_true("batch-size" in parser.arguments)
    assert_true("lr" in parser.arguments)
    assert_true("output" in parser.arguments)
    print("PASS: test_argument_parser_add_arguments")


fn main() raises:
    """Run argument parser tests - Part 1."""
    print("")
    print("=" * 70)
    print("ArgumentParser Unit Tests - Part 1")
    print("=" * 70)
    print("")

    test_argument_spec_creation()
    test_parsed_args_string()
    test_parsed_args_int()
    test_parsed_args_float()
    test_parsed_args_bool()
    test_parsed_args_has()
    test_argument_parser_creation()
    test_argument_parser_add_arguments()

    print("")
    print("=" * 70)
    print("All argument parser Part 1 tests passed!")
    print("=" * 70)
    print("")
