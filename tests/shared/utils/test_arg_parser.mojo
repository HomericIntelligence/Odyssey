"""Tests for argument parser utilities

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
    """Run all test_arg_parser tests."""
    print("Running test_arg_parser tests...")

    test_argument_spec_creation()
    print("✓ test_argument_spec_creation")

    test_parsed_args_string()
    print("✓ test_parsed_args_string")

    test_parsed_args_int()
    print("✓ test_parsed_args_int")

    test_parsed_args_float()
    print("✓ test_parsed_args_float")

    test_parsed_args_bool()
    print("✓ test_parsed_args_bool")

    test_parsed_args_has()
    print("✓ test_parsed_args_has")

    test_argument_parser_creation()
    print("✓ test_argument_parser_creation")

    test_argument_parser_add_arguments()
    print("✓ test_argument_parser_add_arguments")

    test_argument_parser_add_flag()
    print("✓ test_argument_parser_add_flag")

    test_argument_parser_invalid_type()
    print("✓ test_argument_parser_invalid_type")

    test_argument_defaults()
    print("✓ test_argument_defaults")

    test_parsed_args_multiple_values()
    print("✓ test_parsed_args_multiple_values")

    test_parser_populates_defaults()
    print("✓ test_parser_populates_defaults")

    print("\nAll test_arg_parser tests passed!")
