#!/usr/bin/env python3
"""
Tool: paper-scaffold/prompts.py
Purpose: Interactive user prompts for collecting paper metadata

Implements Issue #771 (User Prompts - Implementation):
- Interactive prompts for paper metadata when CLI arguments aren't provided
- Real-time input validation
- Default values where appropriate
- Helpful error messages

Language: Python
Justification: User input/output, string validation, no ML/AI computation
Reference: ADR-001
"""

import datetime
import re
from typing import Callable, Dict, Optional


class InteractivePrompter:
    """
    Interactive prompter for collecting paper metadata.

    Provides conversational prompts with validation, defaults, and helpful
    error messages to guide users through paper creation.
    """

    def __init__(self, quiet: bool = False):
        """
        Initialize interactive prompter.

        Args:
            quiet: If True, suppress informational messages
        """
        self.quiet = quiet

    def _prompt(
        self,
        message: str,
        default: Optional[str] = None,
        required: bool = True,
        validator: Optional[Callable[[str], tuple[bool, str]]] = None,
        example: Optional[str] = None,
    ) -> str:
        """
        Prompt user for input with validation.

        Args:
            message: Prompt message to display
            default: Default value if user presses enter
            required: Whether field is required
            validator: Optional validation function (str) -> (bool, str)
            example: Example value to show in prompt

        Returns:
            User input (validated).
        """
        # Build prompt message
        prompt_parts = [message]

        if example:
            prompt_parts.append(f"\n  Example: {example}")

        if default:
            prompt_parts.append(f" [{default}]")

        prompt_parts.append(": ")
        full_prompt = "".join(prompt_parts)

        while True:
            user_input = input(full_prompt).strip()

            # Use default if provided and input is empty
            if not user_input and default:
                return default

            # Check required field
            if required and not user_input:
                print("  ⚠ This field is required and cannot be empty. Please try again.")
                continue

            # If optional and empty, return empty
            if not required and not user_input:
                return ""

            # Run validator if provided
            if validator:
                is_valid, error_message = validator(user_input)
                if not is_valid:
                    print(f"  ⚠ {error_message}")
                    continue

            return user_input

    def prompt_for_paper_name(self) -> str:
        """
        Prompt for paper name.

        Returns:
            Paper name (will be normalized to directory name).
        """
        return self._prompt(
            message="Paper name (short identifier)",
            example="LeNet-5, BERT, GPT-2",
            required=True,
        )

    def prompt_for_title(self) -> str:
        """
        Prompt for full paper title.

        Returns:
            Full paper title
        """
        return self._prompt(
            message="Full paper title",
            example="LeNet-5: Gradient-Based Learning Applied to Document Recognition",
            required=True,
        )

    def prompt_for_authors(self) -> str:
        """
        Prompt for paper authors.

        Returns:
            Paper authors
        """
        return self._prompt(message="Authors", example="LeCun et al., Vaswani et al.", required=True)

    def _validate_year(self, year_str: str) -> tuple[bool, str]:
        """
        Validate year is a reasonable 4-digit number.

        Args:
            year_str: Year string to validate

        Returns:
            (is_valid, error_message) tuple
        """
        # Check it's a number
        if not year_str.isdigit():
            return False, "Year must be a number (e.g., 1998)"

        year = int(year_str)
        current_year = datetime.datetime.now().year

        # Check reasonable range (1950 to current year)
        if year < 1950:
            return False, f"Year must be 1950 or later (got {year})"

        if year > current_year:
            return (
                False,
                f"Year cannot be in the future (current year is {current_year})",
            )

        return True, ""

    def prompt_for_year(self) -> str:
        """
        Prompt for publication year.

        Returns:
            Publication year (defaults to current year).
        """
        current_year = str(datetime.datetime.now().year)

        return self._prompt(
            message="Publication year",
            default=current_year,
            required=False,
            validator=self._validate_year,
            example="1998, 2017",
        )

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """
        Validate URL format.

        Args:
            url: URL string to validate

        Returns:
            (is_valid, error_message) tuple
        """
        # If empty, it's valid (optional field)
        if not url:
            return True, ""

        # Simple URL validation (starts with http:// or https://)
        url_pattern = re.compile(r"^https?://")
        if not url_pattern.match(url):
            return False, "URL must start with http:// or https://"

        return True, ""

    def prompt_for_url(self) -> str:
        """
        Prompt for paper URL.

        Returns:
            Paper URL (optional).
        """
        result = self._prompt(
            message="Paper URL (optional)",
            default="",
            required=False,
            validator=self._validate_url,
            example="https://arxiv.org/abs/1234.5678",
        )

        # Return placeholder if empty
        return result if result else "TODO: Add paper URL"

    def prompt_for_description(self) -> str:
        """
        Prompt for paper description.

        Returns:
            Brief paper description (optional).
        """
        result = self._prompt(
            message="Brief description (optional)",
            default="",
            required=False,
            example="Convolutional neural network for handwritten digit recognition",
        )

        # Return placeholder if empty
        return result if result else "TODO: Add description"

    def collect_metadata(self, existing: Optional[Dict] = None) -> Dict[str, str]:
        """
        Collect complete paper metadata interactively.

        Only prompts for fields that are missing in existing arguments.

        Args:
            existing: Dictionary of existing CLI arguments (optional)

        Returns:
            Dictionary of complete metadata

        Example:
            ```mojo
            >> prompter = InteractivePrompter()
            >>> metadata = prompter.collect_metadata()
            >>> # User is prompted for all fields
            >>> metadata = prompter.collect_metadata(existing={'paper': 'LeNet-5'})
            >>> # User is only prompted for missing fields
        ```
        """
        if not self.quiet:
            print("\n" + "=" * 60)
            print("Paper Metadata Collection")
            print("=" * 60)
            print("Please provide information about the paper.")
            print("Press Enter to accept default values shown in [brackets].\n")

        metadata = {}

        # Helper to check if field needs prompting
        def needs_prompt(field_name: str) -> bool:
            if not existing:
                return True
            value = existing.get(field_name)
            return value is None or value == "" or (isinstance(value, str) and value.startswith("TODO"))

        # Paper name (required)
        if needs_prompt("paper"):
            metadata["PAPER_NAME"] = self.prompt_for_paper_name()
        else:
            metadata["PAPER_NAME"] = existing["paper"]
            if not self.quiet:
                print(f"Paper name: {metadata['PAPER_NAME']} (from arguments)")

        # Title (required)
        if needs_prompt("title"):
            metadata["PAPER_TITLE"] = self.prompt_for_title()
        else:
            metadata["PAPER_TITLE"] = existing["title"]
            if not self.quiet:
                print(f"Title: {metadata['PAPER_TITLE']} (from arguments)")

        # Authors (required)
        if needs_prompt("authors"):
            metadata["AUTHORS"] = self.prompt_for_authors()
        else:
            metadata["AUTHORS"] = existing["authors"]
            if not self.quiet:
                print(f"Authors: {metadata['AUTHORS']} (from arguments)")

        # Year (optional, defaults to current year)
        if needs_prompt("year"):
            metadata["YEAR"] = self.prompt_for_year()
        else:
            metadata["YEAR"] = existing["year"]
            if not self.quiet:
                print(f"Year: {metadata['YEAR']} (from arguments)")

        # URL (optional)
        if needs_prompt("url"):
            metadata["PAPER_URL"] = self.prompt_for_url()
        else:
            metadata["PAPER_URL"] = existing["url"]
            if not self.quiet:
                print(f"URL: {metadata['PAPER_URL']} (from arguments)")

        # Description (optional)
        if needs_prompt("description"):
            metadata["DESCRIPTION"] = self.prompt_for_description()
        else:
            metadata["DESCRIPTION"] = existing["description"]
            if not self.quiet:
                print(f"Description: {metadata['DESCRIPTION']} (from arguments)")

        if not self.quiet:
            print("\n" + "=" * 60)
            print("Metadata collection complete!")
            print("=" * 60 + "\n")

        return metadata


def main():
    """Example usage of interactive prompter."""
    print("Interactive Paper Metadata Prompter - Example")
    print("=" * 60)

    prompter = InteractivePrompter(quiet=False)
    metadata = prompter.collect_metadata()

    print("\nCollected metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
