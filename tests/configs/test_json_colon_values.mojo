"""Tests for JSON parsing with values containing colons.

Tests for issue #2128: Handle values with colons (URLs, timestamps, etc.)

Run with: mojo test tests/configs/test_json_colon_values.mojo
"""

from std.testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config


# ============================================================================
# JSON Parsing with Colon-Containing Values
# ============================================================================


def test_json_with_url_values() raises:
    """Test loading JSON with URL values containing colons.

    Verifies that URLs like "http://example.com" are parsed correctly
    without being split on the colon.
    """
    var config = load_config("tests/configs/fixtures/urls.json")

    # Test HTTP URL
    var api_url = config.get_string("api_url")
    assert_equal(
        api_url, "http://localhost:8080", "Should preserve HTTP URL with port"
    )

    # Test HTTPS URL
    var model_url = config.get_string("model_url")
    assert_equal(
        model_url,
        "https://models.example.com/lenet5.bin",
        "Should preserve HTTPS URL",
    )

    print("✓ test_json_with_url_values passed")


def test_json_with_database_url() raises:
    """Test loading JSON with database connection URL.

    Verifies that PostgreSQL URLs with multiple colons are parsed correctly.
    """
    var config = load_config("tests/configs/fixtures/urls.json")

    var db_url = config.get_string("database_url")
    assert_equal(
        db_url,
        "postgresql://user:pass@localhost:5432/db",
        "Should preserve database URL with credentials and port",
    )

    print("✓ test_json_with_database_url passed")


def test_json_with_time_values() raises:
    """Test loading JSON with time values containing colons.

    Verifies that time strings like "12:30:45" are preserved.
    """
    var config = load_config("tests/configs/fixtures/urls.json")

    var time_start = config.get_string("time_start")
    assert_equal(time_start, "12:30:45", "Should preserve time format HH:MM:SS")

    print("✓ test_json_with_time_values passed")


def test_json_numeric_values_still_work() raises:
    """Test that numeric values are still correctly parsed.

    Verifies that the fix doesn't break number parsing.
    """
    var config = load_config("tests/configs/fixtures/urls.json")

    var port = config.get_int("port")
    assert_equal(port, 8080, "Should parse integer port correctly")

    var lr = config.get_float("learning_rate")
    assert_equal(lr, 0.001, "Should parse float learning rate correctly")

    print("✓ test_json_numeric_values_still_work passed")


# ============================================================================
# Main Test Runner
# ============================================================================


def main() raises:
    """Run all JSON colon-value tests."""
    print("\n" + "=" * 70)
    print("Running JSON Colon-Value Tests (Issue #2128)")
    print("=" * 70 + "\n")

    print("Testing URL and Colon-Containing Values...")
    test_json_with_url_values()
    test_json_with_database_url()
    test_json_with_time_values()
    test_json_numeric_values_still_work()

    print("\n" + "=" * 70)
    print("✅ All JSON Colon-Value Tests Passed!")
    print("=" * 70)
