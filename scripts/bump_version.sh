#!/bin/bash
# Bump version and create release tag
#
# Usage:
#   ./scripts/bump_version.sh [major|minor|patch]
#   ./scripts/bump_version.sh patch        # 0.1.0 -> 0.1.1
#   ./scripts/bump_version.sh minor        # 0.1.0 -> 0.2.0
#   ./scripts/bump_version.sh major        # 0.1.0 -> 1.0.0
#   ./scripts/bump_version.sh              # defaults to patch

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_FILE="$REPO_ROOT/VERSION"

# Check if VERSION file exists
if [[ ! -f "$VERSION_FILE" ]]; then
    echo -e "${RED}Error: VERSION file not found at $VERSION_FILE${NC}"
    exit 1
fi

# Read current version
CURRENT_VERSION=$(cat "$VERSION_FILE")
echo -e "${YELLOW}Current version: $CURRENT_VERSION${NC}"

# Parse version (handles X.Y.Z format)
if ! [[ "$CURRENT_VERSION" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo -e "${RED}Error: Invalid version format: $CURRENT_VERSION${NC}"
    echo "Expected format: X.Y.Z (e.g., 0.1.0)"
    exit 1
fi

MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"
PATCH="${BASH_REMATCH[3]}"

# Determine bump type (default: patch)
BUMP_TYPE="${1:-patch}"

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo -e "${RED}Error: Invalid bump type: $BUMP_TYPE${NC}"
        echo "Usage: $0 [major|minor|patch]"
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo -e "${GREEN}New version: $NEW_VERSION${NC}"

# Confirm with user
echo ""
read -p "Proceed with version bump? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}Warning: You have uncommitted changes.${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Commit or stash your changes first."
        exit 1
    fi
fi

# Update VERSION file
echo "$NEW_VERSION" > "$VERSION_FILE"
echo -e "${GREEN}Updated VERSION file${NC}"

# Update version in other files if they exist
# pixi.toml
if [[ -f "$REPO_ROOT/pixi.toml" ]]; then
    if grep -q "^version = " "$REPO_ROOT/pixi.toml"; then
        sed -i "s/^version = .*/version = \"$NEW_VERSION\"/" "$REPO_ROOT/pixi.toml"
        echo -e "${GREEN}Updated pixi.toml${NC}"
    fi
fi

# pyproject.toml
if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
    if grep -q "^version = " "$REPO_ROOT/pyproject.toml"; then
        sed -i "s/^version = .*/version = \"$NEW_VERSION\"/" "$REPO_ROOT/pyproject.toml"
        echo -e "${GREEN}Updated pyproject.toml${NC}"
    fi
fi

# Commit changes
git add VERSION
if [[ -f "$REPO_ROOT/pixi.toml" ]]; then
    # pixi.toml should always exist; git add failing would indicate a real problem.
    git add pixi.toml
fi
if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
    # pyproject.toml should always exist if the file check passed.
    git add pyproject.toml
fi

git commit -m "chore: bump version to $NEW_VERSION

🤖 Generated with [Claude Code](https://claude.com/claude-code)"

echo -e "${GREEN}Committed version bump${NC}"

# Create annotated tag
TAG_NAME="v$NEW_VERSION"
git tag -a "$TAG_NAME" -m "Release $TAG_NAME

🤖 Generated with [Claude Code](https://claude.com/claude-code)"

echo -e "${GREEN}Created tag: $TAG_NAME${NC}"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Version bumped successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Review the changes:"
echo "     git log --oneline -2"
echo "     git show $TAG_NAME"
echo ""
echo "  2. Push commits and tag:"
echo "     git push origin main"
echo "     git push origin $TAG_NAME"
echo ""
echo "  3. GitHub Actions will automatically create the release"
echo ""
echo "Or to push everything at once:"
echo "  git push origin main --tags"
