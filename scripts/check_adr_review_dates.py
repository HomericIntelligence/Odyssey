#!/usr/bin/env python3
"""Fail if any ADR has a `**Next Review**` date in the past.

Used by the weekly mojo-version-check workflow to enforce ADR-008's
quarterly review cadence (Issue #5040).
"""

import re
import sys
import datetime
from pathlib import Path

PATTERN = re.compile(r"^\*\*Next Review\*\*:\s*(\d{4}-\d{2}-\d{2})", re.M)


def main() -> int:
    """Check that all ADRs have future review dates."""
    today = datetime.date.today()
    overdue = []
    for adr in sorted(Path("docs/adr").glob("ADR-*.md")):
        for m in PATTERN.finditer(adr.read_text()):
            due = datetime.date.fromisoformat(m.group(1))
            if due < today:
                overdue.append((adr.name, due))
    for name, due in overdue:
        print(f"::warning::ADR review overdue: {name} (due {due})")
    if overdue:
        print(f"\n❌ {len(overdue)} ADR(s) overdue for review.")
        return 1
    print(f"✅ All ADRs are within their review window (checked: {today}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
