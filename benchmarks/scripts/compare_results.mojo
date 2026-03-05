"""Benchmark comparison script.

This script compares current benchmark results against a baseline to detect
performance regressions.

Usage:
    mojo compare_results.mojo \
      --baseline baselines/baseline_results.json \
      --current results/{timestamp}_results.json

Exit Codes:
    0: No regressions detected
    1: Regressions detected (>10% slowdown)

Output:
    - Comparison summary (console)
    - Regression report if regressions found
"""

from sys import argv
from python import Python


# ============================================================================
# Data Structures
# ============================================================================


struct BenchmarkData(Copyable, Movable):
    """Data from a single benchmark result."""

    var name: String
    var duration_ms: Float64
    var throughput: Float64
    var memory_mb: Float64
    var iterations: Int

    fn __init__(
        out self,
        name: String,
        duration_ms: Float64,
        throughput: Float64,
        memory_mb: Float64,
        iterations: Int,
    ):
        self.name = name
        self.duration_ms = duration_ms
        self.throughput = throughput
        self.memory_mb = memory_mb
        self.iterations = iterations


struct ComparisonResult(Copyable, Movable):
    """Result of comparing baseline and current benchmark."""

    var name: String
    var baseline_duration: Float64
    var current_duration: Float64
    var percentage_change: Float64
    var is_regression: Bool
    var severity: String  # "none", "minor", "moderate", "severe"

    fn __init__(
        out self,
        name: String,
        baseline_duration: Float64,
        current_duration: Float64,
        percentage_change: Float64,
        is_regression: Bool,
        severity: String,
    ):
        self.name = name
        self.baseline_duration = baseline_duration
        self.current_duration = current_duration
        self.percentage_change = percentage_change
        self.is_regression = is_regression
        self.severity = severity


# ============================================================================
# JSON Parsing
# ============================================================================


fn parse_benchmark_name(line: String) raises -> String:
    """Extract benchmark name from JSON line.

    Args:
        line: JSON line containing "name": "value".

    Returns:
        Extracted name value.
    """
    var start = line.find('"name": "')
    if start == -1:
        raise Error("Could not find name in benchmark entry")

    start = start + 9  # Length of '"name": "'
    var end = line.find('"', start)
    if end == -1:
        raise Error("Could not find closing quote for name")

    return String(line[start:end])


fn atof(s: String) -> Float64:
    """Convert string to float64 with scientific notation support.

    Supports formats:
    - Regular: "123.456", "-78.9"
    - Scientific: "1.23e-4", "5.67E+10", "-3.14e2"

    Args:
        s: String to convert.

    Returns:
        Float64 value.
    """
    var result: Float64 = 0.0
    var sign: Float64 = 1.0
    var i = 0

    # Handle negative numbers
    if len(s) > 0 and s[0] == "-":
        sign = -1.0
        i = 1

    # Parse integer part
    while (
        i < len(s)
        and String(s[i : i + 1]) != "."
        and String(s[i : i + 1]) != "e"
        and String(s[i : i + 1]) != "E"
    ):
        result = result * 10.0 + Float64(ord(s[i]) - ord("0"))
        i += 1

    # Parse decimal part
    if i < len(s) and String(s[i : i + 1]) == ".":
        i += 1
        var decimal_place: Float64 = 0.1
        while (
            i < len(s)
            and String(s[i : i + 1]) != "e"
            and String(s[i : i + 1]) != "E"
        ):
            result = result + Float64(ord(s[i]) - ord("0")) * decimal_place
            decimal_place = decimal_place * 0.1
            i += 1

    # Parse exponent (scientific notation)
    if i < len(s) and (
        String(s[i : i + 1]) == "e" or String(s[i : i + 1]) == "E"
    ):
        i += 1
        var exp_sign = 1
        if i < len(s) and String(s[i : i + 1]) == "-":
            exp_sign = -1
            i += 1
        elif i < len(s) and String(s[i : i + 1]) == "+":
            i += 1

        var exponent = 0
        while i < len(s):
            exponent = exponent * 10 + (ord(s[i]) - ord("0"))
            i += 1

        # Apply exponent: result * 10^exponent
        var exp_value = Float64(exp_sign * exponent)
        result = result * (10.0**exp_value)

    return result * sign


fn parse_float_value(line: String, field_name: String) raises -> Float64:
    """Extract float value from JSON line.

    Args:
        line: JSON line containing field.
        field_name: Name of field to extract.

    Returns:
        Extracted float value.
    """
    var search_str = field_name + '": '
    var start = line.find(search_str)
    if start == -1:
        raise Error("Could not find " + field_name + " in line")

    start = start + len(search_str)

    # Find end of number (comma, newline, or closing brace)
    var end = start
    while (
        end < len(line)
        and String(line[end : end + 1]) != ","
        and String(line[end : end + 1]) != "}"
        and String(line[end : end + 1]) != "\n"
    ):
        end += 1

    var value_str = String(line[start:end].strip())
    return atof(value_str)


fn parse_int_value(line: String, field_name: String) raises -> Int:
    """Extract int value from JSON line.

    Args:
        line: JSON line containing field.
        field_name: Name of field to extract.

    Returns:
        Extracted int value.
    """
    var search_str = field_name + '": '
    var start = line.find(search_str)
    if start == -1:
        raise Error("Could not find " + field_name + " in line")

    start = start + len(search_str)

    # Find end of number (comma, newline, or closing brace)
    var end = start
    while (
        end < len(line)
        and String(line[end : end + 1]) != ","
        and String(line[end : end + 1]) != "}"
        and String(line[end : end + 1]) != "\n"
    ):
        end += 1

    var value_str = String(line[start:end].strip())

    # Convert string to int
    var result = 0
    for i in range(len(value_str)):
        result = result * 10 + (ord(value_str[i]) - ord("0"))

    return result


fn load_benchmark_results(filepath: String) raises -> List[BenchmarkData]:
    """Load benchmark results from JSON file.

    Args:
        filepath: Path to JSON results file.

    Returns:
        List of parsed benchmark data.

    Raises:
        Error if file cannot be parsed.
    """
    var results = List[BenchmarkData](capacity=10)

    try:
        # Check file exists using Python (no Mojo os.path alternative yet)
        var os_path = Python.import_module("os.path")

        if not os_path.exists(filepath):
            raise Error("Benchmark file not found: " + filepath)

        # Read file using Mojo native file I/O
        var content: String
        with open(filepath, "r") as f:
            content = f.read()

        # Parse JSON content for benchmarks array
        var benchmarks_start = content.find('"benchmarks": [')
        if benchmarks_start == -1:
            raise Error("Could not find benchmarks array in JSON")

        # Find benchmark entries within the array
        var current_pos = benchmarks_start
        while True:
            var entry_start = content.find("{", current_pos)
            if entry_start == -1:
                break

            # Check if we've gone past the benchmarks array
            var array_end = content.find("]", benchmarks_start)
            if entry_start > array_end:
                break

            var entry_end = content.find("}", entry_start)
            if entry_end == -1:
                break

            var entry_slice = content[entry_start : entry_end + 1]
            var entry = String(entry_slice)

            # Parse benchmark entry using existing parsing functions
            try:
                var name = parse_benchmark_name(entry)
                var duration_ms = parse_float_value(entry, '"duration_ms')
                var throughput = parse_float_value(entry, '"throughput')
                var memory_mb = parse_float_value(entry, '"memory_mb')
                var iterations = parse_int_value(entry, '"iterations')

                results.append(
                    BenchmarkData(
                        name, duration_ms, throughput, memory_mb, iterations
                    )
                )
            except:
                # Skip entries that can't be parsed
                pass

            # Move to next entry
            current_pos = entry_end + 1

    except e:
        raise Error("Failed to load benchmark results: " + String(e))

    return results^


fn atoi(s: String) -> Int:
    """Convert string to integer.

    Args:
        s: String to convert.

    Returns:
        Integer value.
    """
    var result: Int = 0
    var sign: Int = 1
    var i = 0

    # Handle negative numbers
    if len(s) > 0 and s[0] == "-":
        sign = -1
        i = 1

    # Parse digits
    while i < len(s):
        if s[i] >= "0" and s[i] <= "9":
            result = result * 10 + (ord(s[i]) - ord("0"))
        i += 1

    return result * sign


# ============================================================================
# Comparison Logic
# ============================================================================


fn calculate_percentage_change(
    baseline: Float64, current: Float64
) raises -> Float64:
    """Calculate percentage change from baseline to current.

    Formula: ((current - baseline) / baseline) * 100

    Positive = slower (worse)
    Negative = faster (better)

    Args:
        baseline: Baseline value.
        current: Current value.

    Returns:
        Percentage change.
    """
    if baseline == 0.0:
        raise Error("Baseline value cannot be zero")

    return ((current - baseline) / baseline) * 100.0


fn determine_severity(percentage_change: Float64) -> String:
    """Determine severity level of regression.

    Args:
        percentage_change: Percentage change (positive for slower).

    Returns:
        Severity level: "none", "minor", "moderate", "severe".
    """
    if percentage_change <= 10.0:
        return "none"
    elif percentage_change <= 20.0:
        return "minor"
    elif percentage_change <= 50.0:
        return "moderate"
    else:
        return "severe"


fn is_regression(percentage_change: Float64) -> Bool:
    """Check if change is a regression.

    Regression threshold: >10% slowdown (exclusive)
    10% exactly = no regression
    10.1% = regression

    Args:
        percentage_change: Percentage change (positive for slower).

    Returns:
        True if regression detected.
    """
    return percentage_change > 10.0


fn compare_benchmarks(
    baseline_results: List[BenchmarkData], current_results: List[BenchmarkData]
) raises -> List[ComparisonResult]:
    """Compare baseline and current results.

    Optimized with O(1) lookup using dictionary instead of O(n) linear search.

    Args:
        baseline_results: List of baseline benchmark data.
        current_results: List of current benchmark data.

    Returns:
        List of comparison results.
    """
    var comparisons = List[ComparisonResult](capacity=len(baseline_results))

    # Create dictionary (hash map) of current results by name for O(1) lookup
    # Use Python dict since Mojo v0.26.1 doesn't have native Dict with custom types
    var builtins = Python.import_module("builtins")
    var current_dict = builtins.dict()

    # Build dictionary mapping name -> index in current_results
    for i in range(len(current_results)):
        var name_py = Python.import_module("builtins").str(
            current_results[i].name
        )
        current_dict[name_py] = i

    # Compare each baseline to current (now O(n) instead of O(n²))
    for i in range(len(baseline_results)):
        var baseline = baseline_results[i].copy()

        # O(1) lookup in dictionary
        var baseline_name_py = Python.import_module("builtins").str(
            baseline.name
        )
        if baseline_name_py not in current_dict:
            print(
                "Warning: Benchmark '"
                + baseline.name
                + "' not found in current results"
            )
            continue

        # Get current result via index from dictionary
        var current_idx = Int(current_dict[baseline_name_py])
        var current = current_results[current_idx].copy()
        var current_duration = current.duration_ms

        # Calculate percentage change
        var pct_change = calculate_percentage_change(
            baseline.duration_ms, current_duration
        )

        # Determine if regression
        var is_regr = is_regression(pct_change)
        var severity = determine_severity(pct_change)

        comparisons.append(
            ComparisonResult(
                baseline.name,
                baseline.duration_ms,
                current_duration,
                pct_change,
                is_regr,
                severity,
            )
        )

    return comparisons^


# ============================================================================
# Reporting
# ============================================================================


fn generate_comparison_report(
    comparisons: List[ComparisonResult],
) raises -> String:
    """Generate comparison report.

    Args:
        comparisons: List of comparison results.

    Returns:
        Formatted comparison report.
    """
    var report = String("=== Benchmark Comparison Report ===\n\n")

    var regression_count = 0
    var improvement_count = 0
    var within_tolerance = 0

    # Count results
    for i in range(len(comparisons)):
        var comp = comparisons[i].copy()
        if comp.is_regression:
            regression_count += 1
        elif comp.percentage_change < -5.0:
            improvement_count += 1
        else:
            within_tolerance += 1

    # Summary
    report += "Summary:\n"
    report += "  Total benchmarks: " + String(len(comparisons)) + "\n"
    report += "  Regressions (>10%): " + String(regression_count) + "\n"
    report += "  Improvements: " + String(improvement_count) + "\n"
    report += "  Within tolerance: " + String(within_tolerance) + "\n"
    report += "\n"

    # Detailed results
    report += "Detailed Results:\n"
    for i in range(len(comparisons)):
        var comp = comparisons[i].copy()
        report += "\n" + comp.name + ":\n"
        report += "  Baseline: " + String(comp.baseline_duration) + " ms\n"
        report += "  Current:  " + String(comp.current_duration) + " ms\n"
        report += "  Change:   " + String(comp.percentage_change) + "%\n"

        if comp.is_regression:
            report += "  Status:   REGRESSION (" + comp.severity + ")\n"
        elif comp.percentage_change < -5.0:
            report += "  Status:   IMPROVEMENT\n"
        else:
            report += "  Status:   OK\n"

    return report


fn generate_regression_report(
    comparisons: List[ComparisonResult],
) raises -> String:
    """Generate report focused on regressions.

    Args:
        comparisons: List of comparison results.

    Returns:
        Regression-focused report.
    """
    var report = String("\n=== REGRESSION DETECTED ===\n\n")

    var regressions = List[ComparisonResult](capacity=10)
    for i in range(len(comparisons)):
        if comparisons[i].is_regression:
            regressions.append(comparisons[i].copy())

    report += "Found " + String(len(regressions)) + " regressions:\n\n"

    for i in range(len(regressions)):
        var reg = regressions[i].copy()
        report += reg.name + " (" + reg.severity + " regression):\n"
        report += "  Baseline: " + String(reg.baseline_duration) + " ms\n"
        report += "  Current:  " + String(reg.current_duration) + " ms\n"
        report += "  Slowdown: " + String(reg.percentage_change) + "%\n\n"

    return report


# ============================================================================
# Main
# ============================================================================


fn main() raises:
    """Compare benchmark results and detect regressions.

    The implementation:
    1. Parse command line arguments for baseline and current files
    2. Load baseline results from JSON
    3. Load current results from JSON
    4. Compare each benchmark:
       - Calculate percentage change
       - Apply tolerance (~5% for normal variance)
       - Flag regressions (>10% slowdown)
    5. Generate comparison report
    6. Exit with appropriate code (0 = success, 1 = regressions).
    """
    print("Benchmark Comparison Tool\n")

    var baseline_file = "benchmarks/baselines/baseline_results.json"
    var current_file = "benchmarks/results/benchmark_results.json"

    # Parse command line arguments
    # NOTE (Mojo v0.26.1): Hardcoded paths are intentional. While Mojo's sys.argv provides
    # basic argv access, comprehensive argument parsing (flags, options,
    # help text) would require a custom parser. For internal benchmarking
    # tooling, hardcoded defaults are sufficient and simpler to maintain.
    print(
        "Note: Using default file paths (argument parsing limited in current"
        " Mojo version)"
    )
    print("  Baseline: " + baseline_file)
    print("  Current:  " + current_file)

    print("Loading baseline from: " + baseline_file)
    var baseline_results = load_benchmark_results(baseline_file)
    print("  Loaded " + String(len(baseline_results)) + " benchmarks\n")

    print("Loading current results from: " + current_file)
    var current_results = load_benchmark_results(current_file)
    print("  Loaded " + String(len(current_results)) + " benchmarks\n")

    # Perform comparison
    print("Comparing results...\n")
    var comparisons = compare_benchmarks(baseline_results, current_results)

    # Generate and print report
    var report = generate_comparison_report(comparisons)
    print(report)

    # Check for regressions
    var has_regressions = False
    for i in range(len(comparisons)):
        if comparisons[i].is_regression:
            has_regressions = True
            break

    # Print regression report if needed
    if has_regressions:
        var regression_report = generate_regression_report(comparisons)
        print(regression_report)
        print("Status: FAILED - Performance regressions detected")
        # Exit with code 1 to signal failure to CI/CD
        var sys = Python.import_module("sys")
        sys.exit(1)
    else:
        print("Status: PASSED - No performance regressions detected")
        # Exit with code 0 to signal success
        var sys = Python.import_module("sys")
        sys.exit(0)
