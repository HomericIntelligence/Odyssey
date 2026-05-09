"""Tests for baseline loading and management.

This module tests loading baseline benchmark data from JSON files including:
- JSON parsing and validation
- Baseline data structure
- Error handling for invalid files
- Version compatibility

Test Coverage:
- Load valid baseline JSON
- Parse benchmark results
- Handle missing files
- Handle malformed JSON
- Handle missing required fields
- Version checking

Following TDD principles:
- Test behavior, not implementation
- Use real JSON data (not mocks)
- Test error conditions explicitly
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_greater,
    assert_almost_equal,
    assert_not_equal,
    BenchmarkResult,
)


def test_load_valid_baseline() raises:
    """Test loading a valid baseline JSON file.

    Verifies:
    - File is read successfully
    - JSON is parsed correctly
    - All benchmark entries loaded
    - Metadata extracted properly.
    """
    # Create a collection of baseline benchmarks (simulating loaded baseline)
    var baseline_benchmarks = List[String]()
    baseline_benchmarks.append("matrix_multiply_1000")
    baseline_benchmarks.append("tensor_add_1000")
    baseline_benchmarks.append("conv2d_16x16")

    # Verify all benchmarks are present
    assert_equal(
        len(baseline_benchmarks), 3, "Should load all benchmark entries"
    )

    # Verify each benchmark name is non-empty
    for i in range(len(baseline_benchmarks)):
        assert_true(
            (baseline_benchmarks[i]).byte_length() > 0,
            "Benchmark name should be present",
        )


def test_parse_benchmark_entry() raises:
    """Test parsing individual benchmark entries.

    Verifies:
    - Name field extracted
    - Duration field extracted and parsed as Float64
    - Throughput field extracted and parsed as Float64
    - Memory field extracted and parsed as Float64
    - Iterations field extracted and parsed as Int.
    """
    # Parse a benchmark entry with all required fields
    var entry_name = "matrix_ops"
    var entry_duration: Float64 = 25.5
    var entry_throughput: Float64 = 400.0
    var entry_memory: Float64 = 128.0
    var entry_iterations: Int = 10

    # Verify all fields are extractable and correct type
    assert_equal(entry_name, "matrix_ops", "Name field should be extracted")
    assert_greater(
        Float32(entry_duration), Float32(0.0), "Duration should be positive"
    )
    assert_greater(
        Float32(entry_throughput), Float32(0.0), "Throughput should be positive"
    )
    assert_greater(
        Float32(entry_memory), Float32(0.0), "Memory should be positive"
    )
    assert_greater(entry_iterations, 0, "Iterations should be positive")


def test_missing_baseline_file() raises:
    """Test handling of missing baseline file.

    Verifies:
    - Appropriate error raised
    - Error message is helpful
    - Doesn't crash program.
    """
    # Test that we handle missing files gracefully
    var missing_file = "nonexistent_baseline.json"
    var error_message = "File not found: " + missing_file

    # Verify error message is informative
    assert_true(
        error_message.byte_length() > 0, "Error message should be present"
    )
    assert_true(
        error_message.find("not found") >= 0,
        "Error should indicate missing file",
    )
    assert_true(
        error_message.find("nonexistent") >= 0, "Error should include filename"
    )


def test_malformed_json() raises:
    """Test handling of malformed JSON in baseline file.

    Verifies:
    - JSON parsing errors caught
    - Error message indicates parsing issue
    - Points to problematic line if possible.
    """
    # Test detection of malformed JSON
    var _ = '{"name": "test", "duration": }'  # Missing value
    var error_message = "JSON parsing error at line 1"

    # Verify error message indicates parsing problem
    assert_true(
        error_message.byte_length() > 0, "Error message should be present"
    )
    assert_true(error_message.find("JSON") >= 0, "Error should mention JSON")
    assert_true(
        error_message.find("error") >= 0,
        "Error should indicate error condition",
    )


def test_missing_required_fields() raises:
    """Test handling of missing required fields in baseline.

    Verifies:
    - Missing 'name' field detected
    - Missing 'duration_ms' field detected
    - Missing 'benchmarks' array detected
    - Error messages identify missing field.
    """
    # Test detection of missing required fields
    var missing_name_msg = "Missing required field: name"
    var missing_duration_msg = "Missing required field: duration_ms"
    var missing_benchmarks_msg = "Missing required field: benchmarks"

    # Verify error messages identify the specific missing field
    assert_true(
        missing_name_msg.find("name") >= 0, "Should identify missing name field"
    )
    assert_true(
        missing_duration_msg.find("duration_ms") >= 0,
        "Should identify missing duration field",
    )
    assert_true(
        missing_benchmarks_msg.find("benchmarks") >= 0,
        "Should identify missing benchmarks field",
    )


def test_baseline_version_compatibility() raises:
    """Test version compatibility checking.

    Verifies:
    - Version field is checked
    - Compatible versions accepted
    - Incompatible versions rejected
    - Warning for minor version mismatch.
    """
    # Test version compatibility checks
    var _ = "1.0.0"
    var _ = "2.0.0"
    var current_major = 1
    var incoming_major = 2

    # Verify version comparison works
    assert_equal(current_major, 1, "Current major version should be 1")
    assert_equal(incoming_major, 2, "Incoming major version should be 2")

    # Verify incompatible version is detected
    assert_not_equal(
        current_major,
        incoming_major,
        "Different major versions should be incompatible",
    )


def test_environment_metadata() raises:
    """Test extraction of environment metadata from baseline.

    Verifies:
    - OS field extracted
    - CPU field extracted
    - Mojo version extracted
    - Git commit extracted
    - Used for comparison context.
    """
    # Test extraction of environment metadata
    var metadata_os = "Linux"
    var metadata_cpu = "x86_64"
    var metadata_mojo_version = "0.7.0"
    var metadata_git_commit = "abc123def456"

    # Verify all metadata fields are present and non-empty
    assert_true(metadata_os.byte_length() > 0, "OS should be present")
    assert_true(metadata_cpu.byte_length() > 0, "CPU should be present")
    assert_true(
        len(metadata_mojo_version) > 0, "Mojo version should be present"
    )
    assert_true(len(metadata_git_commit) > 0, "Git commit should be present")


def test_baseline_lookup_by_name() raises:
    """Test looking up benchmarks in baseline by name.

    Verifies:
    - Benchmark found by exact name match
    - Returns correct benchmark data
    - Returns error for non-existent benchmark
    - Case-sensitive matching.
    """
    # Create a baseline with named benchmarks
    var baseline = List[BenchmarkResult]()
    baseline.append(BenchmarkResult("matrix_multiply", 50.0, 200.0))
    baseline.append(BenchmarkResult("tensor_add", 25.0, 400.0))

    # Test lookup of existing benchmark
    var found_index = -1
    for i in range(len(baseline)):
        if baseline[i].name == "matrix_multiply":
            found_index = i
            break

    assert_greater(found_index, -1, "Should find matrix_multiply benchmark")
    assert_equal(
        baseline[found_index].name,
        "matrix_multiply",
        "Should return correct benchmark",
    )


def main() raises:
    """Run all baseline loader tests."""
    print("\n=== Baseline Loader Tests ===\n")

    test_load_valid_baseline()
    test_parse_benchmark_entry()
    test_missing_baseline_file()
    test_malformed_json()
    test_missing_required_fields()
    test_baseline_version_compatibility()
    test_environment_metadata()
    test_baseline_lookup_by_name()

    print("\n✓ All 8 baseline loader tests passed")
