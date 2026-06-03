"""Tests for scripts/migrate_any_tensor_imports.py (issue #5159)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import migrate_any_tensor_imports as m


# --- partition-table validation -------------------------------------------------

def _write_any_tensor_mojo(tmp_path: Path, creation: list[str], utils: list[str]) -> Path:
    p = tmp_path / "any_tensor.mojo"
    body = (
        "from .tensor_creation import (\n"
        + "".join(f"    {s},\n" for s in creation)
        + ")\n\n"
        + "from .tensor_utils import (\n"
        + "".join(f"    {s},\n" for s in utils)
        + ")\n"
    )
    p.write_text(body)
    return p

def test_partition_table_validates_exact_match(tmp_path):
    p = _write_any_tensor_mojo(tmp_path, sorted(m.CREATION), sorted(m.UTILS))
    m._validate_partition_table(p)  # must not raise

def test_partition_table_aborts_on_missing_symbol(tmp_path):
    bad = sorted(m.CREATION - {"zeros"})
    p = _write_any_tensor_mojo(tmp_path, bad, sorted(m.UTILS))
    with pytest.raises(SystemExit, match="partition-table drift"):
        m._validate_partition_table(p)

def test_partition_table_aborts_on_extra_symbol(tmp_path):
    bad = sorted(m.CREATION | {"mystery_func"})
    p = _write_any_tensor_mojo(tmp_path, bad, sorted(m.UTILS))
    with pytest.raises(SystemExit, match="partition-table drift"):
        m._validate_partition_table(p)


# --- fold/rewrite correctness ---------------------------------------------------

@pytest.mark.parametrize("body,expected", [
    # single-line, mixed
    ("from projectodyssey.tensor.any_tensor import AnyTensor, zeros\n",
     "from projectodyssey.tensor.any_tensor import AnyTensor\n"
     "from projectodyssey.tensor.tensor_creation import zeros\n"),
    # multi-line paren
    ("from projectodyssey.tensor.any_tensor import (\n"
     "    AnyTensor,\n"
     "    zeros,\n"
     "    ones,\n"
     ")\n",
     "from projectodyssey.tensor.any_tensor import AnyTensor\n"
     "from projectodyssey.tensor.tensor_creation import zeros, ones\n"),
    # paren + comment containing parens (the canary R1 raised)
    ("from projectodyssey.tensor.any_tensor import (\n"
     "    zeros,  # zeros_like creates (shape-matching) tensor\n"
     "    ones,\n"
     ")\n",
     "from projectodyssey.tensor.tensor_creation import zeros, ones\n"),
    # alias preservation
    ("from projectodyssey.tensor.any_tensor import nan_tensor as shared_nan_tensor\n",
     "from projectodyssey.tensor.tensor_creation import nan_tensor as shared_nan_tensor\n"),
    # util-only file
    ("from projectodyssey.tensor.any_tensor import zeros, item\n",
     "from projectodyssey.tensor.tensor_creation import zeros\n"
     "from projectodyssey.tensor.tensor_utils import item\n"),
    # relative form
    ("from .any_tensor import AnyTensor, zeros\n",
     "from .any_tensor import AnyTensor\n"
     "from .tensor_creation import zeros\n"),
    # indented import
    ("    from projectodyssey.tensor.any_tensor import zeros\n",
     "    from projectodyssey.tensor.tensor_creation import zeros\n"),
])
def test_rewrite_forms(tmp_path, body, expected):
    f = tmp_path / "f.mojo"
    f.write_text(body)
    assert m._process(f) is True
    assert f.read_text() == expected


def test_rewrite_idempotent(tmp_path):
    f = tmp_path / "f.mojo"
    f.write_text("from projectodyssey.tensor.any_tensor import AnyTensor, zeros\n")
    assert m._process(f) is True
    first = f.read_text()
    assert m._process(f) is False           # no further change
    assert f.read_text() == first


# --- preflight ------------------------------------------------------------------

def test_preflight_passes_on_unrelated_comments(tmp_path):
    f = tmp_path / "f.mojo"
    f.write_text(
        "# TODO: drop `from tensor_creation import zeros` legacy line\n"
        "from projectodyssey.tensor.any_tensor import zeros\n"
    )
    m._preflight_partial_state([f], tmp_path)  # must not raise

def test_preflight_aborts_on_real_clash(tmp_path):
    f = tmp_path / "f.mojo"
    f.write_text(
        "from projectodyssey.tensor.any_tensor import zeros\n"
        "from projectodyssey.tensor.tensor_creation import zeros\n"
    )
    with pytest.raises(SystemExit, match="partial-migration state"):
        m._preflight_partial_state([f], tmp_path)


# --- end-to-end against sandboxed --root ---------------------------------------

def test_end_to_end_sandbox(tmp_path):
    root = tmp_path / "repo"
    (root / "src/projectodyssey/tensor").mkdir(parents=True)
    (root / "examples").mkdir()
    # Minimal any_tensor.mojo with the real partition table.
    _write_any_tensor_mojo(
        root / "src/projectodyssey/tensor", sorted(m.CREATION), sorted(m.UTILS)
    ).rename(root / "src/projectodyssey/tensor/any_tensor.mojo")
    consumer = root / "examples/c.mojo"
    consumer.write_text(
        "from projectodyssey.tensor.any_tensor import AnyTensor, zeros, item\n"
    )
    rc = m.main(["--root", str(root)])
    assert rc == 0
    assert consumer.read_text() == (
        "from projectodyssey.tensor.any_tensor import AnyTensor\n"
        "from projectodyssey.tensor.tensor_creation import zeros\n"
        "from projectodyssey.tensor.tensor_utils import item\n"
    )
    # Idempotency.
    rc2 = m.main(["--root", str(root)])
    assert rc2 == 0
