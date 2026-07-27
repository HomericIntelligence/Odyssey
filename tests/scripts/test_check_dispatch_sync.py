#!/usr/bin/env python3
"""Tests for `scripts/check_dispatch_sync.py` — three-way dispatch sync check.

Issue: #5682. Validates:

1. `parse_yaml_enum` returns the canonical names from the schema.
2. `parse_dispatch_registry` extracts every `String("name")` entry from
   `all_supported_optimizers()` (scoped to function body only).
3. `has_init_state` returns True if `def init_<name>_state(` is defined.
4. `check_dispatch_sync` is internally consistent for a synthetic fixture.
5. The full script runs against the live repo without errors.

Usage:
    python tests/scripts/test_check_dispatch_sync.py
    pytest tests/scripts/test_check_dispatch_sync.py -v
"""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load `check_dispatch_sync` from scripts/ (which has no __init__.py)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_dispatch_sync",
    _PROJECT_ROOT / "scripts" / "check_dispatch_sync.py",
)
_mod = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(_mod)  # type: ignore[union-attr]

parse_yaml_enum = _mod.parse_yaml_enum
parse_dispatch_registry = _mod.parse_dispatch_registry
has_init_state = _mod.has_init_state
check_dispatch_sync = _mod.check_dispatch_sync


# ============================================================================
# Layer 1 — YAML parser
# ============================================================================


class TestParseYamlEnum:
    def test_extracts_all_names(self, tmp_path: Path) -> None:
        schema = tmp_path / "training.schema.yaml"
        schema.write_text(
            textwrap.dedent(
                """\
                $schema: "http://json-schema.org/draft-07/schema#"
                title: Training
                type: object
                properties:
                  optimizer:
                    type: object
                    properties:
                      name:
                        type: string
                        enum:
                          - sgd
                          - adam
                          - shampoo
                          - prodigy
                """
            )
        )
        assert parse_yaml_enum(schema) == ["adam", "prodigy", "sgd", "shampoo"]

    def test_returns_sorted(self, tmp_path: Path) -> None:
        schema = tmp_path / "s.yaml"
        schema.write_text(
            textwrap.dedent(
                """\
                properties:
                  optimizer:
                    properties:
                      name:
                        enum:
                          - zebra
                          - apple
                          - mango
                """
            )
        )
        assert parse_yaml_enum(schema) == ["apple", "mango", "zebra"]


# ============================================================================
# Layer 2 — dispatch.mojo parser (all_supported_optimizers)
# ============================================================================


class TestParseDispatchRegistry:
    def test_single_line_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "dispatch.mojo"
        f.write_text(
            textwrap.dedent(
                """\
                def all_supported_optimizers() -> List[String]:
                    var names: List[String] = []
                    names.append(String("sgd"))
                    names.append(String("adam"))
                    return names^
                """
            )
        )
        reg = parse_dispatch_registry(f)
        assert reg == {"sgd", "adam"}

    def test_multi_line_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "dispatch.mojo"
        f.write_text(
            textwrap.dedent(
                """\
                def all_supported_optimizers() -> List[String]:
                    var names: List[String] = []
                    names.append(
                        String("shampoo")
                    )
                    names.append(
                        String("prodigy")
                    )
                    return names^
                """
            )
        )
        reg = parse_dispatch_registry(f)
        assert reg == {"shampoo", "prodigy"}

    def test_no_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "dispatch.mojo"
        f.write_text("# no specs here\n")
        assert parse_dispatch_registry(f) == set()

    def test_duplicate_entries_collapsed(self, tmp_path: Path) -> None:
        """Duplicate `String("name")` entries collapse into one set member."""
        f = tmp_path / "dispatch.mojo"
        f.write_text(
            "def all_supported_optimizers() -> List[String]:\n"
            "    var names: List[String] = []\n"
            '    names.append(String("sgd"))\n'
            '    names.append(String("sgd"))\n'
            "    return names^\n"
        )
        reg = parse_dispatch_registry(f)
        assert reg == {"sgd"}


# ============================================================================
# Layer 3 — per-source existence check
# ============================================================================


_INIT_STATE_FIXTURE = textwrap.dedent(
    """\
    def init_{name}_state(
        params_list: List[AnyTensor],
        *,
        force_f64: Bool = False,
    ) raises -> List[List[AnyTensor]]:
        from odyssey.tensor.tensor_creation import zeros
        return [list[zeros(p.shape(), p.dtype())] for p in params_list]
    """
)


_NO_INIT_SOURCE = textwrap.dedent(
    """\
    # No init function defined here.
    def another_function(p, g):
        return p
    """
)


class TestHasInitState:
    def test_init_state_present(self, tmp_path: Path) -> None:
        """Source with matching `init_<name>_state` header returns True."""
        f = tmp_path / "sgd.mojo"
        f.write_text(_INIT_STATE_FIXTURE.replace("{name}", "sgd"))
        assert has_init_state(f, "sgd") is True

    def test_missing_function_returns_false(self, tmp_path: Path) -> None:
        """Source without any `init_*_state` returns False."""
        f = tmp_path / "noop.mojo"
        f.write_text(_NO_INIT_SOURCE)
        assert has_init_state(f, "noop") is False

    def test_wrong_name_returns_false(self, tmp_path: Path) -> None:
        """Source contains `init_sgd_state` but caller asks for 'adam'."""
        f = tmp_path / "sgd.mojo"
        f.write_text(_INIT_STATE_FIXTURE.replace("{name}", "sgd"))
        assert has_init_state(f, "adam") is False


# ============================================================================
# Layer 4 — end-to-end on a synthetic repo
# ============================================================================


def _make_synthetic_repo(tmp_path: Path, *, mismatch: str | None = None) -> Path:
    """Build a 3-optimizer synthetic repo for end-to-end testing.

    Args:
        tmp_path: Test temp dir.
        mismatch: One of None, "yaml_only", "reg_only", "source_missing".
            Each introduces a different kind of consistency error.
    """
    root = tmp_path

    # YAML schema (3 optimizers)
    schema_path = root / "configs" / "schemas" / "training.schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_names = ["sgd", "adam", "shampoo"]
    if mismatch == "yaml_only":
        yaml_names.append("phantom_optimizer")
    yaml_text = (
        '$schema: "http://json-schema.org/draft-07/schema#"\n'
        "title: Training\n"
        "type: object\n"
        "properties:\n"
        "  optimizer:\n"
        "    properties:\n"
        "      name:\n"
        "        type: string\n"
        "        enum:\n"
    )
    for n in yaml_names:
        yaml_text += f"          - {n}\n"
    schema_path.write_text(yaml_text)

    # dispatch.mojo (all_supported_optimizers with String("name") entries)
    dispatch_path = root / "src" / "odyssey" / "training" / "dispatch.mojo"
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    reg_names = ["sgd", "adam", "shampoo"]
    if mismatch == "reg_only":
        reg_names.append("phantom_optimizer")
    dispatch_text = "def all_supported_optimizers() -> List[String]:\n"
    dispatch_text += "    var names: List[String] = []\n"
    for n in reg_names:
        dispatch_text += f'    names.append(String("{n}"))\n'
    dispatch_text += "    return names^\n"
    dispatch_path.write_text(dispatch_text)

    # optimizer sources — one per NAME in the canonical set (sgd, adam, shampoo).
    # Phantom optimizers added by mismatch cases intentionally have NO source
    # file generated, so the SOURCE MISSING code path is exercised alongside
    # the NAME MISMATCH detection.
    opt_dir = root / "src" / "odyssey" / "training" / "optimizers"
    opt_dir.mkdir(parents=True, exist_ok=True)
    canonical_names = ["sgd", "adam", "shampoo"]
    for n in canonical_names:
        (opt_dir / f"{n}.mojo").write_text(_INIT_STATE_FIXTURE.format(name=n))

    return root


class TestCheckDispatchSync:
    def test_clean_synthetic_repo_passes(self, tmp_path: Path) -> None:
        root = _make_synthetic_repo(tmp_path)
        assert check_dispatch_sync(root, quiet=True) == 0

    def test_yaml_only_optimizer_fails(self, tmp_path: Path) -> None:
        root = _make_synthetic_repo(tmp_path, mismatch="yaml_only")
        rc = check_dispatch_sync(root, quiet=True)
        assert rc == 1

    def test_reg_only_optimizer_fails(self, tmp_path: Path) -> None:
        root = _make_synthetic_repo(tmp_path, mismatch="reg_only")
        rc = check_dispatch_sync(root, quiet=True)
        assert rc == 1


# ============================================================================
# Layer 5 — live-repo smoke test
# ============================================================================


def test_live_repo_dispatch_is_consistent(capsys: pytest.CaptureFixture[str]) -> None:
    """STRICT live-repo check: any inconsistency fails this test.

    This test runs `check_dispatch_sync` against the live repository and
    asserts `rc == 0`. There is **no EXPECTED_DRIFT safety net** — any
    mismatch between the YAML enum, `dispatch.mojo`'s
    `all_supported_optimizers()`, and per-source `init_<name>_state`
    functions will fail this test.

    If the test fails:
    - A new optimizer was added without updating all three sources.
    - An optimizer was renamed/removed without updating all three sources.
    - The YAML enum, dispatch roster, or source files have diverged.

    The fix is to reconcile the three sources, NOT to add the divergence
    to an allowed-drift set (no such safety net exists in this codebase).

    Skips if `dispatch.mojo` doesn't exist (e.g., partial checkout).
    """
    dispatch_path = _PROJECT_ROOT / "src" / "odyssey" / "training" / "dispatch.mojo"
    if not dispatch_path.exists():
        pytest.skip("dispatch.mojo not present in this checkout")

    rc = check_dispatch_sync(_PROJECT_ROOT, quiet=False)
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert rc == 0, (
        f"check_dispatch_sync returned {rc}, expected 0 (all consistent). "
        f"This is a STRICT check — reconcile the YAML enum, "
        f"`all_supported_optimizers()`, and per-source `init_<name>_state` "
        f"function names. Output:\n{combined}"
    )


def test_live_repo_cli_invocation() -> None:
    """`python scripts/check_dispatch_sync.py --help` exits 0 and shows expected flags."""
    script = _PROJECT_ROOT / "scripts" / "check_dispatch_sync.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--root" in proc.stdout
    assert "--quiet" in proc.stdout
