# ML Odyssey - Mojo Development Environment
# Multi-stage Dockerfile
#
# Toolchain: uv-managed (Odysseus ADR-018). The Mojo compiler is installed from
# the Modular PyPI index via `uv sync` (pyproject.toml [tool.uv] configures the
# index + prerelease handling) — NOT from conda/pixi anymore.

# ---------------------------
# Stage 0: uv binary (named-stage COPY source, Scylla PR #2054 pattern)
# ---------------------------
# Pinned by version + digest. The digest was verified against the ghcr manifest
# API (HTTP 200) before use — a wrong digest yields 'manifest unknown'. Bump the
# tag AND re-resolve the digest together when upgrading uv.
FROM ghcr.io/astral-sh/uv:0.11.21@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 AS uv

# ---------------------------
# Stage 1: Base image with system deps
# ---------------------------
FROM mcr.microsoft.com/mirror/docker/library/ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libasan8 \
    ca-certificates \
    vim \
    wget \
    uuid \
    sudo \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI (gh) as root
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && mkdir -p -m 755 /etc/apt/sources.list.d \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install just tool (pre-built binary for faster installation)
RUN curl -fsSL https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.14.0

# Install the uv binary from the pinned uv image stage (Scylla PR #2054 pattern).
COPY --from=uv /uv /uvx /usr/local/bin/

# ---------------------------
# Stage 1.5: Create dev user
# ---------------------------
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=dev

RUN groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || \
    groupmod -n ${USER_NAME} $(getent group ${GROUP_ID} | cut -d: -f1) && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} 2>/dev/null || \
    usermod -l ${USER_NAME} -d /home/${USER_NAME} -m $(id -nu ${USER_ID} 2>/dev/null || echo nobody) && \
    chmod 755 /home/${USER_NAME}

# Allow dev user to fix bind-mount ownership at container startup.
# The workspace is bind-mounted as root:root at runtime; entrypoint.sh uses
# sudo chown to reclaim specific subdirs (build/, .venv/, test fixtures).
RUN printf 'dev ALL=(root) NOPASSWD: /bin/mkdir\ndev ALL=(root) NOPASSWD: /bin/chown\ndev ALL=(root) NOPASSWD: /bin/chmod\n' \
    > /etc/sudoers.d/dev-workspace \
    && chmod 440 /etc/sudoers.d/dev-workspace

# Set environment for dev user
ENV HOME=/home/${USER_NAME}
ENV PATH="$HOME/.local/bin:$HOME/.venv/bin:$PATH"

# ---------------------------
# Stage 2: Development environment
# ---------------------------
FROM base AS development

# Re-declare ARG so it's available in this stage (ARGs don't persist across FROM)
ARG USER_NAME=dev

# Switch to dev user
USER ${USER_NAME}
WORKDIR /workspace

# Install Claude Code CLI as the dev user (development-only — production
# stage stays clean per #5328 because it FROM base, not FROM development).
# Installs into ~/.local/bin which is in PATH.
RUN curl -fsSL https://claude.ai/install.sh | bash -s -- stable

# uv cache + Modular home. Keeping the venv inside the image (not /workspace)
# means the runtime bind-mount does not shadow it.
ENV UV_CACHE_DIR=/home/${USER_NAME}/.cache/uv
ENV UV_PROJECT_ENVIRONMENT=/home/${USER_NAME}/.venv
# uv must not try to manage/patch the interpreter it downloads read-only later.
ENV UV_PYTHON_INSTALL_DIR=/home/${USER_NAME}/.local/share/uv/python
RUN mkdir -p $UV_CACHE_DIR $HOME/.modular

# Copy dependency manifests first for layer caching. The venv is built into the
# per-user home ($UV_PROJECT_ENVIRONMENT) so the runtime bind-mount of /workspace
# does not shadow it (mirrors the former detached-pixi-env layout).
COPY --chown=${USER_NAME}:${USER_NAME} pyproject.toml uv.lock .python-version .pre-commit-config.yaml ./

# Install project dependencies incl. the Mojo compiler (cached unless manifests
# change). --no-install-project: only third-party deps + the toolchain are
# baked in here; the workspace source is bind-mounted at runtime.
RUN uv sync --locked --no-install-project

# Ensure pre-commit is available in the uv environment
RUN uv run pre-commit --version

# Install pre-commit hooks (cached unless .pre-commit-config.yaml changes).
# Build context lacks .git, but `pre-commit install-hooks` still requires
# git to be operational inside *some* repository to clone hook envs.
# Initialize a throwaway repo so install-hooks succeeds without `|| true`.
RUN git config --global user.email "build@odyssey.local" && \
    git config --global user.name "build" && \
    git config --global init.defaultBranch main && \
    git init -q . && \
    if [ -d .git ]; then \
        uv run pre-commit install --install-hooks; \
    else \
        uv run pre-commit install-hooks; \
    fi

# Copy entrypoint script for container initialization
COPY --chown=${USER_NAME}:${USER_NAME} docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy the rest of the workspace (invalidates only layers after this)
COPY --chown=${USER_NAME}:${USER_NAME} . .

# Set Python path
ENV PYTHONPATH=/workspace:${PYTHONPATH:-}

# Default shell (uv-managed env is on PATH via $HOME/.venv/bin)
CMD ["/bin/bash"]


# ---------------------------
# Stage 3: CI / Testing
# ---------------------------
FROM development AS ci

CMD ["uv", "run", "pytest", "tests/", "-v"]

# ---------------------------
# Stage 4: Production
# ---------------------------
FROM base AS production

# Re-declare ARG so it's available in this stage (ARGs don't persist across FROM)
ARG USER_NAME=dev

# Copy only what's needed: the uv-managed venv and project source (no dev tools)
COPY --from=development /home/${USER_NAME}/.venv /home/${USER_NAME}/.venv
COPY --from=development /workspace /workspace

ENV ENVIRONMENT=production
ENV PATH="/home/${USER_NAME}/.venv/bin:${PATH}"
USER ${USER_NAME}
WORKDIR /workspace

CMD ["python", "-c", "print('Odyssey production image ready')"]
