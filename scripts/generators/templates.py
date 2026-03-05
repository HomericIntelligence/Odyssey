#!/usr/bin/env python3
"""
Template handling utilities for code generation.

This module provides:
- Template loading and caching
- Variable substitution
- Template validation
"""

import re
from pathlib import Path
from typing import Any


TEMPLATE_DIR = Path(__file__).parent.parent.parent / ".templates"


def load_template(template_name: str) -> str:
    """Load a template file by name.

    Args:
        template_name: Name of template file (without .mojo extension)

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template doesn't exist
    """
    template_path = TEMPLATE_DIR / f"{template_name}.mojo"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text()


def substitute_variables(template: str, variables: dict[str, Any]) -> str:
    """Substitute variables in a template.

    Variables are marked as {{variable_name}} in templates.

    Args:
        template: Template string with {{variable}} placeholders
        variables: Dictionary of variable names to values

    Returns:
        Template with variables substituted
    """
    result = template
    for name, value in variables.items():
        pattern = r"\{\{\s*" + re.escape(name) + r"\s*\}\}"
        result = re.sub(pattern, str(value), result)
    return result


def validate_template(template: str, required_vars: list[str]) -> list[str]:
    """Validate that a template contains required variables.

    Args:
        template: Template string to validate
        required_vars: List of required variable names

    Returns:
        List of missing variable names (empty if all present)
    """
    missing: list[str] = []
    for var in required_vars:
        pattern = r"\{\{\s*" + re.escape(var) + r"\s*\}\}"
        if not re.search(pattern, template):
            missing.append(var)
    return missing


def to_snake_case(name: str) -> str:
    """Convert PascalCase/camelCase to snake_case.

    Args:
        name: Name in PascalCase or camelCase

    Returns:
        Name in snake_case
    """
    # Insert underscore before uppercase letters
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters followed by lowercase
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase.

    Args:
        name: Name in snake_case

    Returns:
        Name in PascalCase
    """
    components = name.split("_")
    return "".join(x.title() for x in components)


def generate_imports(layer_types: list[str]) -> str:
    """Generate import statements based on layer types used.

    Args:
        layer_types: List of layer type names (e.g., ['Conv2d', 'Linear'])

    Returns:
        Import statement string
    """
    nn_layers = []
    core_imports = ["ExTensor"]

    for layer in layer_types:
        if layer in (
            "Conv2d",
            "Linear",
            "BatchNorm2d",
            "Dropout",
            "MaxPool2d",
            "AvgPool2d",
            "Flatten",
            "LayerNorm",
            "GroupNorm",
        ):
            nn_layers.append(layer)
        elif layer in ("ReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU"):
            nn_layers.append(layer)

    imports = []
    if nn_layers:
        imports.append(f"from shared.nn import Module, {', '.join(sorted(set(nn_layers)))}")
    else:
        imports.append("from shared.nn import Module")
    imports.append(f"from shared.core import {', '.join(sorted(set(core_imports)))}")

    return "\n".join(imports)


def indent_code(code: str, spaces: int = 4) -> str:
    """Indent each line of code by the specified number of spaces.

    Args:
        code: Code to indent
        spaces: Number of spaces to indent

    Returns:
        Indented code
    """
    prefix = " " * spaces
    lines = code.split("\n")
    return "\n".join(prefix + line if line.strip() else line for line in lines)
