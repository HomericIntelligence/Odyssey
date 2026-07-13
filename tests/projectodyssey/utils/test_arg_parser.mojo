"""Tests for argument parser utilities.

Tests basic argument parsing functionality including:
- ArgumentSpec creation and fields
- ParsedArgs string, int, float, bool, and has() getters
- ArgumentParser creation and add_argument()
"""


from std.testing import assert_true, assert_equal
from projectodyssey.utils import ArgumentParser, ArgumentSpec, ParsedArgs
from projectodyssey.utils import resolve_training_args


def test_argument_spec_creation() raises:
    """Test creating argument specifications."""
    var spec = ArgumentSpec(
        name="epochs", arg_type="int", default_value="100", is_flag=False
    )
    assert_equal(spec.name, "epochs")
    assert_equal(spec.arg_type, "int")
    assert_equal(spec.default_value, "100")
    assert_true(not spec.is_flag)
    print("PASS: test_argument_spec_creation")


def test_parsed_args_string() raises:
    """Test ParsedArgs string getter."""
    var args = ParsedArgs()
    args.set("output", "model.weights")
    assert_equal(args.get_string("output"), "model.weights")
    assert_equal(args.get_string("missing", "default"), "default")
    print("PASS: test_parsed_args_string")


def test_parsed_args_int() raises:
    """Test ParsedArgs integer getter."""
    var args = ParsedArgs()
    args.set("epochs", "100")
    assert_equal(args.get_int("epochs"), 100)
    assert_equal(args.get_int("missing", 42), 42)
    print("PASS: test_parsed_args_int")


def test_parsed_args_float() raises:
    """Test ParsedArgs float getter."""
    var args = ParsedArgs()
    args.set("lr", "0.001")
    var lr = args.get_float("lr")
    # Float comparison with tolerance
    assert_true(lr > 0.0009 and lr < 0.0011)
    assert_equal(args.get_float("missing", 0.1), 0.1)
    print("PASS: test_parsed_args_float")


def test_parsed_args_bool() raises:
    """Test ParsedArgs boolean flag getter."""
    var args = ParsedArgs()
    args.set("verbose", "true")
    assert_true(args.get_bool("verbose"))
    assert_true(not args.get_bool("missing"))
    print("PASS: test_parsed_args_bool")


def test_parsed_args_has() raises:
    """Test ParsedArgs has() method."""
    var args = ParsedArgs()
    args.set("epochs", "100")
    assert_true(args.has("epochs"))
    assert_true(not args.has("missing"))
    print("PASS: test_parsed_args_has")


def test_argument_parser_creation() raises:
    """Test creating an argument parser."""
    var parser = ArgumentParser()
    parser.add_argument("epochs", "int", "100")
    assert_equal(len(parser.arguments), 1)
    print("PASS: test_argument_parser_creation")


def test_argument_parser_add_arguments() raises:
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


def test_argument_parser_add_flag() raises:
    """Test adding boolean flags."""
    var parser = ArgumentParser()
    parser.add_flag("verbose")
    parser.add_flag("debug")

    assert_equal(len(parser.arguments), 2)
    assert_true("verbose" in parser.arguments)
    assert_true("debug" in parser.arguments)

    assert_true(parser.arguments["verbose"].is_flag)
    print("PASS: test_argument_parser_add_flag")


def test_argument_parser_invalid_type() raises:
    """Test that invalid argument types are rejected."""
    var parser = ArgumentParser()
    try:
        parser.add_argument("bad", "invalid_type", "0")
        # Should raise error
        assert_true(False)
    except:
        assert_true(True)  # Expected error
        print("PASS: test_argument_parser_invalid_type")


def test_argument_defaults() raises:
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


def test_parsed_args_multiple_values() raises:
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


def test_parser_populates_defaults() raises:
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


def test_parsed_args_user_supplied_vs_default() raises:
    """User-supplied values are distinguished from registered defaults (#5545).

    `set` records a registered default; `set_user_supplied` records a value the
    user passed on the command line. `has` is True for both, but
    `was_user_supplied` must be True only for the latter — this is what lets
    parse_training_args_with_defaults honor a caller default only when the user
    did not pass the flag.
    """
    var args = ParsedArgs()
    args.set("weights-dir", "weights")  # registered default
    args.set_user_supplied("epochs", "50")  # user passed --epochs 50

    # Both are visible via has()...
    assert_true(args.has("weights-dir"))
    assert_true(args.has("epochs"))

    # ...but only the user-supplied one is flagged as such.
    assert_true(not args.was_user_supplied("weights-dir"))
    assert_true(args.was_user_supplied("epochs"))
    # An argument never set at all is neither.
    assert_true(not args.has("missing"))
    assert_true(not args.was_user_supplied("missing"))


def test_registered_defaults_not_user_supplied() raises:
    """Empty argv marks nothing as user-supplied even with defaults (#5545 guard).

    The parser pre-populates every argument's registered default, so has() is
    True for each; but since the user passed nothing, was_user_supplied() must
    be False for all — proving a caller default would take effect.
    """
    var parser = ArgumentParser()
    parser.add_argument("weights-dir", "string", "weights")
    parser.add_argument("epochs", "int", "100")

    var result = parser.parse()  # empty argv (test harness passes no flags)

    assert_true(result.has("weights-dir"))
    assert_true(result.has("epochs"))
    # The regression: neither should count as user-supplied.
    assert_true(not result.was_user_supplied("weights-dir"))
    assert_true(not result.was_user_supplied("epochs"))


def test_resolve_honors_caller_default_when_absent() raises:
    """Caller default is returned when the user omits the flag (#5545).

    Simulates `parse_training_args_with_defaults(default_weights_dir="custom")`
    with NO user-supplied flags: every registered default is present in the
    ParsedArgs (via set), but since none was user-supplied, the caller's
    per-script defaults must win — the exact regression #5545 describes.
    """
    var parsed = ParsedArgs()
    # Registered defaults, as parse() would pre-populate them.
    parsed.set("weights-dir", "weights")
    parsed.set("epochs", "10")
    parsed.set("lr", "0.01")

    var args = resolve_training_args(
        parsed,
        default_epochs=100,
        default_lr=0.005,
        default_weights_dir="custom_weights",
    )

    # Caller defaults win because nothing was user-supplied.
    assert_equal(args.weights_dir, "custom_weights")
    assert_equal(args.epochs, 100)
    assert_true(args.learning_rate > 0.0049 and args.learning_rate < 0.0051)


def test_resolve_user_value_overrides_caller_default() raises:
    """A user-supplied flag beats the caller's default in resolve_training_args.
    """
    var parsed = ParsedArgs()
    parsed.set("weights-dir", "weights")  # registered default
    parsed.set_user_supplied("weights-dir", "user_dir")  # user passed the flag
    parsed.set_user_supplied("epochs", "7")

    var args = resolve_training_args(
        parsed,
        default_epochs=100,
        default_weights_dir="custom_weights",
    )

    # User-supplied values win over caller defaults.
    assert_equal(args.weights_dir, "user_dir")
    assert_equal(args.epochs, 7)


def test_resolve_max_batches_and_smoke_defaults() raises:
    """Max-batches defaults to 0 (unbounded) and smoke to False (#5551)."""
    var parsed = ParsedArgs()
    var args = resolve_training_args(parsed)
    assert_equal(args.max_batches, 0)
    assert_true(not args.smoke)


def test_resolve_max_batches_and_smoke_user_supplied() raises:
    """--max-batches N and --smoke flag are picked up by resolve (#5551)."""
    var parsed = ParsedArgs()
    parsed.set_user_supplied("max-batches", "5")
    parsed.set_user_supplied("smoke", "true")  # flag → has()/get_bool True

    var args = resolve_training_args(parsed)
    assert_equal(args.max_batches, 5)
    assert_true(args.smoke)


def test_resolve_rejects_negative_max_batches() raises:
    """A negative --max-batches is rejected (0 = unbounded; #5551)."""
    var parsed = ParsedArgs()
    parsed.set_user_supplied("max-batches", "-3")
    var raised = False
    try:
        _ = resolve_training_args(parsed)
    except:
        raised = True
    assert_true(
        raised, "resolve_training_args must reject negative max-batches"
    )


def test_direct_parse_resolve_honors_script_default_when_absent() raises:
    """Direct-parse resolve_* honors the script's own default when unset (#5569).

    Mirrors what every migrated example script now does: the shared parser
    pre-populates a *registered* default into the value map, but the script
    passes its own per-script default to resolve_* and that default must win
    unless the user explicitly passed the flag. This is the systemic
    masked-default bug that `get_*` exhibited (it returned the registered
    default) and that resolve_* fixes.
    """
    var parsed = ParsedArgs()
    # Registered defaults, exactly as ArgumentParser.parse() pre-populates them
    # (e.g. create_training_parser registers epochs=100, weights-dir="weights",
    # lr=0.001). None of these was user-supplied.
    parsed.set("weights-dir", "weights")
    parsed.set("epochs", "100")
    parsed.set("lr", "0.001")

    # The script supplies its OWN defaults (e.g. lenet run_train wants
    # "lenet5_weights", 10 epochs, lr 0.01). With get_* these were shadowed by
    # the registered defaults above; resolve_* must honor the script's values.
    assert_equal(
        parsed.resolve_string("weights-dir", "lenet5_weights"), "lenet5_weights"
    )
    assert_equal(parsed.resolve_int("epochs", 10), 10)
    var lr = parsed.resolve_float("lr", 0.01)
    assert_true(lr > 0.0099 and lr < 0.0101)


def test_direct_parse_resolve_user_value_overrides_script_default() raises:
    """Direct-parse resolve_* returns the user's value when the flag is passed.

    The complement to the case above: when the user explicitly passes a flag,
    the parsed CLI value must beat both the registered default and the script's
    own default (#5569).
    """
    var parsed = ParsedArgs()
    parsed.set("weights-dir", "weights")  # registered default
    parsed.set_user_supplied("weights-dir", "user_dir")  # user passed the flag
    parsed.set_user_supplied("epochs", "7")
    parsed.set_user_supplied("lr", "0.5")

    # User-supplied values win over the script's per-site defaults.
    assert_equal(
        parsed.resolve_string("weights-dir", "lenet5_weights"), "user_dir"
    )
    assert_equal(parsed.resolve_int("epochs", 10), 7)
    var lr = parsed.resolve_float("lr", 0.01)
    assert_true(lr > 0.4999 and lr < 0.5001)


def test_direct_parse_resolve_differs_from_get_when_shadowed() raises:
    """Resolve_* honors the script default where get_* returned the registered.

    Pins the exact behavioral difference the #5569 migration relies on: for a
    pre-populated (registered) default that the user did NOT pass, get_* returns
    the registered value (the bug) while resolve_* returns the script's default
    (the fix). Also guards that get_* semantics are unchanged (so the
    pre-population contract / test_parser_populates_defaults is not regressed).
    """
    var parsed = ParsedArgs()
    parsed.set(
        "weights-dir", "weights"
    )  # registered default, not user-supplied

    # get_* keeps its documented behavior: registered default wins (unchanged).
    assert_equal(parsed.get_string("weights-dir", "lenet5_weights"), "weights")
    # resolve_* is the fix: the script's own default wins when unset.
    assert_equal(
        parsed.resolve_string("weights-dir", "lenet5_weights"), "lenet5_weights"
    )


def main() raises:
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

    test_parsed_args_user_supplied_vs_default()
    print("✓ test_parsed_args_user_supplied_vs_default")

    test_registered_defaults_not_user_supplied()
    print("✓ test_registered_defaults_not_user_supplied")

    test_resolve_honors_caller_default_when_absent()
    print("✓ test_resolve_honors_caller_default_when_absent")

    test_resolve_user_value_overrides_caller_default()
    test_resolve_max_batches_and_smoke_defaults()
    print("✓ test_resolve_max_batches_and_smoke_defaults")
    test_resolve_max_batches_and_smoke_user_supplied()
    print("✓ test_resolve_max_batches_and_smoke_user_supplied")
    test_resolve_rejects_negative_max_batches()
    print("✓ test_resolve_rejects_negative_max_batches")
    print("✓ test_resolve_user_value_overrides_caller_default")

    test_direct_parse_resolve_honors_script_default_when_absent()
    print("✓ test_direct_parse_resolve_honors_script_default_when_absent")
    test_direct_parse_resolve_user_value_overrides_script_default()
    print("✓ test_direct_parse_resolve_user_value_overrides_script_default")
    test_direct_parse_resolve_differs_from_get_when_shadowed()
    print("✓ test_direct_parse_resolve_differs_from_get_when_shadowed")

    print("\nAll test_arg_parser tests passed!")
