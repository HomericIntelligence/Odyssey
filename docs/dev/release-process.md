# Release Process

## Versioning

The project version lives in the `VERSION` file and must stay in sync with
`pyproject.toml` and `mojo.toml`. Run `just check-version-sync` to verify the
three agree; CI enforces this through the `deps/version-sync` check. Versions
follow semantic versioning, currently `0.2.0`.

## Steps

1. Update `VERSION`, `pyproject.toml`, and `mojo.toml` to the new version.
2. Run `just check-version-sync` and confirm it passes.
3. Open a pull request and wait for required checks to go green.
4. Merge to `main`.
5. Tag the merge commit and push the tag: `git tag v<version> && git push origin v<version>`.
6. The `Release` workflow (`.github/workflows/release.yml`) triggers on `v*`
   tags and publishes the release.

A release can also be started manually via `workflow_dispatch`, which takes an
explicit `version` input and an optional `prerelease` flag.

## Checklist

- [ ] `VERSION`, `pyproject.toml`, and `mojo.toml` updated together
- [ ] `just check-version-sync` passes
- [ ] Full CI green on the release pull request
- [ ] Tag matches the version exactly (`v<version>`)
- [ ] Release notes describe user-visible changes
