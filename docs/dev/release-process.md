# Release Process

## Versioning

The authoritative version lives in `pyproject.toml` under `[project] version`.
It must stay in sync with the `VERSION` file and `mojo.toml`.

**Never edit the version files individually.** Use the `just bump-version`
recipe, which updates all three atomically:

```bash
just bump-version 0.2.0
```

Verify the result with `just check-version-sync` (also enforced by the
`deps/version-sync` pre-commit hook and the matching CI check). Versions follow
semantic versioning.

## Steps

1. Bump the version: `just bump-version <version>`. This updates `VERSION`,
   `pyproject.toml`, and `mojo.toml` together.
2. Run `just check-version-sync` and confirm it passes.
3. Open a pull request and wait for the required checks to go green.
4. Merge to `main`.
5. Tag the merge commit and push the tag:
   `git tag v<version> && git push origin v<version>`.
6. The `Release` workflow (`.github/workflows/release.yml`) triggers on `v*`
   tags and publishes the release.

A release can also be started manually via `workflow_dispatch`, which takes an
explicit `version` input and an optional `prerelease` flag.

## Checklist

- [ ] `just bump-version <version>` used; no version file hand-edited
- [ ] `pyproject.toml`, `mojo.toml`, and `VERSION` agree
- [ ] `just check-version-sync` passes
- [ ] Full CI green on the release pull request
- [ ] Tag matches the version exactly (`v<version>`)
- [ ] Release notes describe user-visible changes
