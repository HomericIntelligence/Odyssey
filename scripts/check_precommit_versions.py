#!/usr/bin/env python3
"""Check pre-commit version consistency against pixi.toml."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from hephaestus.ci.precommit import (
    DEFAULT_HOOK_TO_PIXI_MAP,
    check_version_drift,
    extract_external_hooks,
    load_precommit_config,
    parse_pixi_constraint,
)
from hephaestus.utils.helpers import get_repo_root


def load_all_pixi_versions(pixi_path: Path) -> dict[str, str]:
    """Load pixi.toml and return package→lower-bound from ALL dependency sections.

    hephaestus.ci.precommit.load_pixi_versions only reads [dependencies].
    This wrapper also reads [feature.*.dependencies] so dev-only packages
    (mypy, nbstripout, pre-commit-hooks) are found.
    """
    with pixi_path.open("rb") as fh:
        data = tomllib.load(fh)

    versions: dict[str, str] = {}

    def _ingest(deps: dict) -> None:
        for pkg, constraint in deps.items():
            v = parse_pixi_constraint(str(constraint))
            if v:
                versions[pkg.lower()] = v

    _ingest(data.get("dependencies", {}))
    for feat in data.get("feature", {}).values():
        _ingest(feat.get("dependencies", {}))

    return versions


def main() -> int:
    root = get_repo_root()
    precommit_path = root / ".pre-commit-config.yaml"
    pixi_path = root / "pixi.toml"

    repos = load_precommit_config(precommit_path)
    external_hooks = extract_external_hooks(repos)
    pixi_versions = load_all_pixi_versions(pixi_path)
    issues = check_version_drift(external_hooks, pixi_versions, DEFAULT_HOOK_TO_PIXI_MAP)

    if issues:
        print("Pre-commit version drift detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix: update the rev in .pre-commit-config.yaml or the version constraint in pixi.toml so they match.")
        return 1

    print("OK: all pre-commit hook versions are consistent with pixi.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
