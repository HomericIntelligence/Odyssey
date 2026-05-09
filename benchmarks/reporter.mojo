"""Benchmark results formatting and export.

Provides utilities for printing benchmark results as formatted tables and
exporting to JSON format for CI/CD tracking and historical analysis.
"""

from shared.benchmarking import LegacyBenchmarkResult


def _str_trunc(s: String, n: Int) -> String:
    """Truncate string to n bytes (ASCII-safe, for display formatting).

    mojo-1.0: String slicing with [:n] and .substr() removed;
    use byte-slice + String(unsafe_from_utf8=) instead.
    """
    var b = s.as_bytes()
    if len(b) <= n:
        return s
    return String(unsafe_from_utf8=b[:n])


def format_throughput(value: Float64) -> String:
    """Format throughput value with appropriate units.

    Args:
        value: Throughput in operations per second.

    Returns:
        Formatted string with units (ops/s, kops/s, Mops/s, Gops/s).
    """
    if value < 1000.0:
        return String(Int(value)) + " ops/s"
    elif value < 1_000_000.0:
        return String(value / 1000.0) + " kops/s"
    elif value < 1_000_000_000.0:
        return String(value / 1_000_000.0) + " Mops/s"
    else:
        return String(value / 1_000_000_000.0) + " Gops/s"


def format_time(value_us: Float64) -> String:
    """Format time value with appropriate units.

    Args:
        value_us: Time in microseconds.

    Returns:
        Formatted string with units (us, ms, s).
    """
    if value_us < 1000.0:
        return String(value_us) + " us"
    elif value_us < 1_000_000.0:
        return String(value_us / 1000.0) + " ms"
    else:
        return String(value_us / 1_000_000.0) + " s"


def print_table(results: List[LegacyBenchmarkResult]):
    """Print benchmark results as formatted table.

    Args:
        results: List of benchmark results to display.
    """
    print(
        "╔════════════════════════════════════════════════════════════════════════════════════╗"
    )
    print(
        "║ Operation                       Mean (us)  Std Dev    Min       Max "
        "     Throughput ║"
    )
    print(
        "╠════════════════════════════════════════════════════════════════════════════════════╣"
    )

    for i in range(len(results)):
        ref r = results[i]

        # Format operation name (max 30 chars)
        # mojo-1.0: String[:n] removed; use _str_trunc() helper
        var name_display = r.name
        if len(r.name) > 30:
            name_display = _str_trunc(r.name, 27) + "..."

        # Print row with aligned columns
        var mean_str = String(Int(r.mean_time_us))
        var std_str = String(Int(r.std_dev_us))
        var min_str = String(Int(r.min_time_us))
        var through_str = format_throughput(r.throughput_ops_per_sec)

        print(
            "║ "
            + name_display
            + " " * (30 - len(name_display))
            + " │ "
            + mean_str
            + " " * (8 - len(mean_str))
            + " │ "
            + std_str
            + " " * (8 - len(std_str))
            + " │ "
            + min_str
            + " " * (8 - len(min_str))
            + " │ "
            + through_str
            + " ║"
        )

    print(
        "╚════════════════════════════════════════════════════════════════════════════════════╝"
    )


def export_json_simple(
    results: List[LegacyBenchmarkResult], filename: String
) raises:
    """Export results to JSON file.

    Uses Python interop for file I/O (Mojo v0.26.1 limitation).

    Args:
        results: List of benchmark results.
        filename: Output JSON file path.

    Raises:
        Error: If file cannot be written.
    """
    print("Exporting results to JSON:", filename)

    try:
        # Build JSON manually to avoid complex serialization
        var json = String("{\n")
        json += '  "version": "1.0.0",\n'
        json += '  "benchmarks": [\n'

        for i in range(len(results)):
            ref r = results[i]
            json += "    {\n"
            json += '      "name": "' + r.name + '",\n'
            json += '      "mean_time_us": ' + String(r.mean_time_us) + ",\n"
            json += '      "std_dev_us": ' + String(r.std_dev_us) + ",\n"
            json += '      "min_time_us": ' + String(r.min_time_us) + ",\n"
            json += '      "max_time_us": ' + String(r.max_time_us) + ",\n"
            json += '      "p50_us": ' + String(r.p50_us) + ",\n"
            json += '      "p95_us": ' + String(r.p95_us) + ",\n"
            json += '      "p99_us": ' + String(r.p99_us) + ",\n"
            json += (
                '      "throughput_ops_per_sec": '
                + String(r.throughput_ops_per_sec)
                + ",\n"
            )
            json += '      "memory_mb": ' + String(r.memory_mb) + "\n"
            json += "    }"

            if i < len(results) - 1:
                json += ","
            json += "\n"

        json += "  ]\n"
        json += "}\n"

        # Write file using Mojo native file I/O
        with open(filename, "w") as f:
            _ = f.write(json)

        print("✓ Results exported to:", filename)

    except e:
        print("✗ Failed to export results:", e)
        raise Error("Export failed")


def print_summary(results: List[LegacyBenchmarkResult]):
    """Print summary statistics.

    Args:
        results: List of benchmark results.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print("Total benchmarks:", len(results))
    print()

    # Find slowest and fastest operations
    if len(results) > 0:
        var slowest_idx = 0
        var fastest_idx = 0
        var slowest_time = results[0].mean_time_us
        var fastest_time = results[0].mean_time_us

        for i in range(1, len(results)):
            if results[i].mean_time_us > slowest_time:
                slowest_idx = i
                slowest_time = results[i].mean_time_us
            if results[i].mean_time_us < fastest_time:
                fastest_idx = i
                fastest_time = results[i].mean_time_us

        print(
            "Slowest:  ",
            results[slowest_idx].name,
            " (",
            format_time(slowest_time),
            ")",
        )
        print(
            "Fastest:  ",
            results[fastest_idx].name,
            " (",
            format_time(fastest_time),
            ")",
        )
        print()
