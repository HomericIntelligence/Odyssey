"""
ML Odyssey Version Module.

Provides centralized version information for all Mojo modules

Usage:
    from shared.version import VERSION, get_version

    def main():
        print("ML Odyssey version:", VERSION)
        var v = get_version()
        print("Version string:", v)
"""

# Version constants (updated by scripts/update_version.py)
comptime VERSION = "0.1.0"
comptime VERSION_MAJOR = 0
comptime VERSION_MINOR = 1
comptime VERSION_PATCH = 0
comptime AUTHOR = "ML Odyssey Team"
comptime LICENSE = "BSD"


def get_version() -> String:
    """
        Get the version string.

    Returns:
            Version string in format "MAJOR.MINOR.PATCH".
    """
    return VERSION


def get_version_tuple() -> Tuple[Int, Int, Int]:
    """
        Get the version as a tuple of integers.

    Returns:
            Tuple of (major, minor, patch).
    """
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)


def version_info() -> String:
    """
        Get detailed version information.

    Returns:
            Formatted string with version details.
    """
    return "ML Odyssey v" + VERSION + " (Mojo-based AI Research Platform)"
