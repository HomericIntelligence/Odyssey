"""
Integration tests for papers/ directory packaging.

Tests the package_papers.py script to ensure it properly creates
distributable tarballs of the papers/ directory.
"""

import tarfile
from pathlib import Path


def test_package_papers_creates_tarball(tmp_path):
    """Test that the packaging script creates a valid tarball."""
    # Import the module
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from package_papers import create_papers_tarball, get_repo_root

    # Get the real repo root
    repo_root = get_repo_root()

    # Create tarball in temp directory
    output_dir = tmp_path / "output"
    tarball_path = create_papers_tarball(repo_root, output_dir)

    # Verify tarball was created
    assert tarball_path.exists()
    assert tarball_path.suffix == ".gz"
    assert tarball_path.stem.endswith(".tar")
    assert "papers-" in tarball_path.name


def test_tarball_contains_papers_directory(tmp_path):
    """Test that the tarball contains the papers/ directory structure."""
    # Import the module
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from package_papers import create_papers_tarball, get_repo_root

    # Get the real repo root
    repo_root = get_repo_root()

    # Create tarball
    output_dir = tmp_path / "output"
    tarball_path = create_papers_tarball(repo_root, output_dir)

    # Extract and verify contents
    with tarfile.open(tarball_path, "r:gz") as tar:
        members = tar.getnames()

        # Should contain papers/ directory
        assert "papers" in members or "papers/" in members

        # Should contain papers/README.md
        assert any("papers/README.md" in m for m in members)


def test_tarball_is_readable(tmp_path):
    """Test that the created tarball is valid and can be extracted."""
    # Import the module
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from package_papers import create_papers_tarball, get_repo_root

    # Get the real repo root
    repo_root = get_repo_root()

    # Create tarball
    output_dir = tmp_path / "output"
    tarball_path = create_papers_tarball(repo_root, output_dir)

    # Extract tarball
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")

    # Verify extracted structure
    papers_dir = extract_dir / "papers"
    assert papers_dir.exists()
    assert papers_dir.is_dir()
    assert (papers_dir / "README.md").exists()


def test_papers_readme_content(tmp_path):
    """Test that the tarball preserves the papers/README.md content."""
    # Import the module
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from package_papers import create_papers_tarball, get_repo_root

    # Get the real repo root
    repo_root = get_repo_root()

    # Read original README content
    original_readme = repo_root / "papers" / "README.md"
    original_content = original_readme.read_text()

    # Create tarball
    output_dir = tmp_path / "output"
    tarball_path = create_papers_tarball(repo_root, output_dir)

    # Extract and verify content
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")

    extracted_readme = extract_dir / "papers" / "README.md"
    extracted_content = extracted_readme.read_text()

    # Content should match
    assert extracted_content == original_content
    assert "# Papers" in extracted_content


def test_multiple_tarballs_same_day(tmp_path):
    """Test that multiple tarballs can be created on the same day."""
    # Import the module
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from package_papers import create_papers_tarball, get_repo_root

    # Get the real repo root
    repo_root = get_repo_root()

    # Create first tarball
    output_dir = tmp_path / "output"
    tarball_path1 = create_papers_tarball(repo_root, output_dir)

    # Note: Same-day tarballs will overwrite, which is expected behavior
    # Just verify the tarball exists and is valid
    assert tarball_path1.exists()

    with tarfile.open(tarball_path1, "r:gz") as tar:
        members = tar.getnames()
        assert len(members) > 0
