#!/usr/bin/env python3
"""
Hook script triggered on SessionEnd to prompt for /learn.

Receives JSON input with:
- session_id: Session identifier
- transcript_path: Path to session transcript (.jsonl)
- reason: "exit" | "clear" | "logout" | "prompt_input_exit" | "other"
- cwd: Current working directory

Outputs JSON to stdout for Claude to process.
"""

import json
import sys
from pathlib import Path


def main():
    """Main entry point for the hook."""
    try:
        # Read hook input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Invalid input, exit silently
        sys.exit(0)

    reason = input_data.get("reason", "other")
    transcript_path = input_data.get("transcript_path", "")

    # Only trigger on explicit session end (not other reasons)
    if reason not in ("exit", "clear"):
        sys.exit(0)

    # Check if transcript has meaningful content (> 10 messages)
    try:
        transcript = Path(transcript_path)
        if transcript.exists():
            line_count = sum(1 for _ in transcript.open())
            if line_count < 10:
                # Session too short for retrospective
                sys.exit(0)
        else:
            # No transcript file
            sys.exit(0)
    except Exception:
        # Error reading transcript, exit silently
        sys.exit(0)

    # Output message prompting /learn
    output = {
        "systemMessage": (
            "Session ending. Consider running /learn to capture learnings "
            "from this session. Would you like to save your learnings to the "
            "skills marketplace before ending?"
        )
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
