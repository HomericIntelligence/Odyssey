# Container Usage Guide

This document describes how to use Podman for ml-odyssey development and deployment.

## Quick Start

### Pull Pre-Built Images

```bash
# Pull the latest runtime image
podman pull ghcr.io/HomericIntelligence/Odyssey:main

# Run tests
podman run --rm ghcr.io/HomericIntelligence/Odyssey:main

# Interactive shell
podman run -it --rm ghcr.io/HomericIntelligence/Odyssey:main bash
```

### Local Development

```bash
# Start development environment
just podman-up

# Enter shell
just shell

# Run tests inside container
just test

# Stop environment
just podman-down
```

## Image Variants

| Tag | Dockerfile | Target | Purpose |
| --- | --- | --- | --- |
| `main` | Dockerfile.ci | runtime | Default runtime with tests |
| `main-ci` | Dockerfile.ci | ci | Full CI with pre-commit |
| `main-prod` | Dockerfile.ci | production | Minimal production image |
| `v*` | Dockerfile.ci | production | Release versions |
| `dev` | Dockerfile | development | Local development |

## Building Images

### Local Development Image

```bash
# Build dev image (with your user ID for permissions)
just podman-build

# Rebuild without cache
just podman-rebuild
```

### CI/Production Images

```bash
# Build runtime image
just podman-build-ci runtime

# Build all targets
just podman-build-ci-all

# Build with specific tag
podman build --format docker -f Dockerfile.ci --target production -t my-tag .
```

## Pushing to Registry

```bash
# Login to GHCR
echo $GITHUB_TOKEN | podman login ghcr.io -u USERNAME --password-stdin

# Push specific target
just podman-push runtime

# Push all
just podman-push-all
```

## Caching

Container builds in CI use `actions/cache` on `~/.local/share/containers` keyed
by Dockerfile content for faster builds.

## Security

- Images are scanned with Trivy for vulnerabilities
- SBOM (Software Bill of Materials) generated for each release
- No secrets stored in images

## Troubleshooting

### Permission Issues

```bash
# Rebuild with your user ID
USER_ID=$(id -u) GROUP_ID=$(id -g) podman compose build
```

### Cache Issues

```bash
# Clean all container resources
just podman-clean

# Rebuild without cache
just podman-rebuild
```

### Pixi Lock Mismatch

```bash
# Update lockfile before building
pixi install
git add pixi.lock
```
