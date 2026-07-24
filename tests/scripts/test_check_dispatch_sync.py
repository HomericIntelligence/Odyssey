#!/usr/bin/env python3
"""Tests for `scripts/check_dispatch_sync.py` — three-way dispatch sync check.

Issue: #5682. Validates:

1. `parse_yaml_enum` returns the canonical names from the schema.
2. `parse_dispatch_registry` extracts every `String("name")` entry from
   `all_supported_optimizers()`.
3. `extract_init_state_buffer_count` returns:
   - The integer N for loop-based `init_<name>_state` (sgd=1, adam=2).
   - The count of `per.append(...)` calls for append-based (shampoo=3).
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
extract_init_state_buffer_count = _mod.extract_init_state_buffer_count
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
# Layer 3 — per-source buffer-count extractor
# ============================================================================


_LOOP_BASED_OPTIMIZER = textwrap.dedent(
    """\
    def init_sgd_state(
        params_list: List[AnyTensor],
        *,
        force_f64: Bool = False,
    ) raises -> List[List[AnyTensor]]:
        from odyssey.tensor.tensor_creation import zeros

        var all_states: List[List[AnyTensor]] = []
        for i in range(len(params_list)):
            var p = params_list[i]
            var d = p.dtype() if not force_f64 else DType.float64
            var per: List[AnyTensor] = []
            for _ in range(1):
                per.append(zeros(p.shape(), d))
            all_states.append(per^)
        return all_states^

    def other_function():
        # This should NOT be picked up by the regex.
        for _ in range(99):
            pass
    """
)


_APPEND_BASED_OPTIMIZER = textwrap.dedent(
    """\
    def is_shampoo_eligible(p) -> Bool:
        return p.ndim() == 2 and p.shape()[0] >= 2 and p.shape()[1] >= 2


    def init_shampoo_state(
        params_list: List[AnyTensor],
        *,
        force_f64: Bool = False,
    ) raises -> List[List[AnyTensor]]:
        from odyssey.tensor.tensor_creation import eye, zeros_like

        var all_states: List[List[AnyTensor]] = []
        for i in range(len(params_list)):
            var p = params_list[i]
            var per: List[AnyTensor] = []
            if is_shampoo_eligible(p):
                var dtype = p.dtype() if not force_f64 else DType.float64
                var sh = p.shape()
                var m = sh[0]
                var n = sh[1]
                per.append(eye(m, m, 0, dtype))
                per.append(eye(n, n, 0, dtype))
                per.append(zeros_like(p))
            all_states.append(per^)
        return all_states^
    """
)


_NO_INIT_OPTIMIZER = textwrap.dedent(
    """\
    # No init function defined here.
    def step_function(p, g):
        return p
    """
)


class TestExtractInitStateBufferCount:
    def test_loop_based_returns_range_int(self, tmp_path: Path) -> None:
        f = tmp_path / "sgd.mojo"
        f.write_text(_LOOP_BASED_OPTIMIZER)
        assert extract_init_state_buffer_count(f, "sgd") == 1

    def test_loop_based_ignores_other_functions(self, tmp_path: Path) -> None:
        """The other_function's `for _ in range(99)` must not be picked up."""
        f = tmp_path / "sgd.mojo"
        f.write_text(_LOOP_BASED_OPTIMIZER)
        assert extract_init_state_buffer_count(f, "sgd") == 1

    def test_append_based_returns_per_append_count(self, tmp_path: Path) -> None:
        f = tmp_path / "shampoo.mojo"
        f.write_text(_APPEND_BASED_OPTIMIZER)
        assert extract_init_state_buffer_count(f, "shampoo") == 3

    def test_missing_init_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "noop.mojo"
        f.write_text(_NO_INIT_OPTIMIZER)
        assert extract_init_state_buffer_count(f, "noop") is None

    def test_name_mismatch_returns_none(self, tmp_path: Path) -> None:
        """Source contains init_sgd_state but caller asks for 'adam'."""
        f = tmp_path / "sgd.mojo"
        f.write_text(_LOOP_BASED_OPTIMIZER)
        assert extract_init_state_buffer_count(f, "adam") is None


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

    # optimizer sources
    opt_dir = root / "src" / "odyssey" / "training" / "optimizers"
    opt_dir.mkdir(parents=True, exist_ok=True)
    # sgd — loop-based, 1 buffer
    (opt_dir / "sgd.mojo").write_text(_LOOP_BASED_OPTIMIZER)
    # adam — loop-based with N=2 (rewrite the loop to range(2))
    adam_src = _LOOP_BASED_OPTIMIZER.replace("for _ in range(1):", "for _ in range(2):")
    adam_src = adam_src.replace("def init_sgd_state", "def init_adam_state")
    (opt_dir / "adam.mojo").write_text(adam_src)
    # shampoo — append-based, 3 buffers
    (opt_dir / "shampoo.mojo").write_text(_APPEND_BASED_OPTIMIZER)
    if mismatch == "reg_only":
        # Source for phantom optimizer is missing — registry should error.
        pass

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
    """Run check_dispatch_sync on the live repo.

    The live repo should have a consistent dispatch registry: the YAML enum,
    dispatch.mojo's `all_supported_optimizers()`, and each optimizer's
    `init_<name>_state` source function should all agree on the optimizer
    name set.

    Skips if `dispatch.mojo` doesn't exist (e.g., partial checkout).
    """
    dispatch_path = _PROJECT_ROOT / "src" / "odyssey" / "training" / "dispatch.mojo"
    if not dispatch_path.exists():
        pytest.skip("dispatch.mojo not present in this checkout")

    rc = check_dispatch_sync(_PROJECT_ROOT, quiet=False)
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert rc == 0, f"check_dispatch_sync returned {rc}, expected 0 (all consistent). Output:\n{combined}"


def test_live_repo_cli_invocation() -> None:
    """`python scripts/check_dispatch_sync.py --help` exits 0."""
    script = _PROJECT_ROOT / "scripts" / "check_dispatch_sync.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--root" in proc.stdout
    assert "--strict" in proc.stdout
