# Installation

## Prerequisites

- Python 3.13 (pinned by `pyproject.toml` as `>=3.13,<3.14`)
- The Mojo 1.0.0 toolchain
- Podman, for the container workflows

Mojo Odyssey's own CI notes that the Mojo 1.0.0 wheels require glibc 2.34 or
newer, so a host older than that must build inside the development container
rather than installing the toolchain directly.

## Clone the Repository

```bash
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey
```

## Start the Development Container

```bash
just podman-preflight
just podman-up
```

`podman-preflight` verifies the host can run the container and reports what to
fix if it cannot. `podman-up` builds the image and starts the `odyssey-dev`
service. Use `just podman-status` to confirm it is up and `just podman-down` to
stop it.

## Verify the Setup

```bash
just podman-run-shell
```

Inside the shell, `mojo --version` should report the toolchain version.

## Steps

1. Install Python 3.13 and Podman.
2. Clone the repository and change into it.
3. Run `just podman-preflight` and resolve anything it reports.
4. Run `just podman-up` to build and start the container.
5. Confirm the toolchain with `mojo --version` inside `just podman-run-shell`.
6. Run `just test` to confirm the test suites pass.
