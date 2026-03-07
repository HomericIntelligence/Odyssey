# Installation

## Prerequisites

- **Platform**: Linux (linux-64) — the only supported platform
- **Git** >= 2.x
- **Pixi** package manager (installation steps below)
- **Docker** (optional) — for the Docker-based workflow

## Installing Pixi

Pixi manages all project dependencies, including Mojo. Install it with:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Restart your shell or source your profile, then verify:

```bash
pixi --version
```

## Cloning the Repository

```bash
git clone https://github.com/HomericIntelligence/ProjectOdyssey.git
cd ProjectOdyssey
```

## Installing Dependencies

Install all dependencies (including Mojo >= 0.26.1) using Pixi:

```bash
pixi install
```

This resolves and installs all packages defined in `pixi.toml`, including Mojo from the
`conda.modular.com/max-nightly` channel.

## Mojo Version Requirements

The project requires **Mojo >= 0.26.1** (< 0.27), as specified in `pixi.toml`:

```toml
[dependencies]
mojo = ">=0.26.1.0.dev2025122805,<0.27"
```

Verify your Mojo version after `pixi install`:

```bash
pixi run mojo --version
```

## Verifying the Installation

Run the following commands to confirm everything is working:

```bash
# Build the shared package
just build

# Run all Mojo tests
just test-mojo
```

Both commands should complete without errors.

## Docker Alternative

If you prefer not to install Pixi natively, a Docker-based workflow is available.

### Pull the Development Image

```bash
docker pull ghcr.io/homericintelligence/projectodyssey:main
```

### Start the Development Environment

```bash
just docker-up
```

### Open a Shell in the Container

```bash
just shell
```

Inside the container, all dependencies including Mojo are pre-installed. Run the same
verification commands:

```bash
just build
just test-mojo
```

### Run Tests Directly (without interactive shell)

```bash
docker run --rm ghcr.io/homericintelligence/projectodyssey:main just test-mojo
```

## Troubleshooting

### `pixi install` fails with channel errors

Ensure your network can reach `https://conda.modular.com/max-nightly` and
`https://conda.anaconda.org/conda-forge`. These channels are required for Mojo and
other dependencies.

### Mojo version mismatch

If `pixi run mojo --version` reports a version outside the `>=0.26.1,<0.27` range,
try clearing the Pixi cache and reinstalling:

```bash
pixi clean
pixi install
```

### `just` command not found

Install the `just` command runner:

```bash
pixi run just --version  # use via pixi if not installed globally
```

Or install it globally following the [just installation guide](https://just.systems/man/en/packages.html).

### Docker container not running

If `just build` or `just test-mojo` fails with a message about the container not running,
start it first:

```bash
just docker-up
```
