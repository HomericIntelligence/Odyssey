# ML Odyssey - Mojo Development Environment
# Multi-stage Dockerfile

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
# sudo chown to reclaim specific subdirs (build/, .pixi/, test fixtures).
RUN printf 'dev ALL=(root) NOPASSWD: /bin/mkdir\ndev ALL=(root) NOPASSWD: /bin/chown\ndev ALL=(root) NOPASSWD: /bin/chmod\n' \
    > /etc/sudoers.d/dev-workspace \
    && chmod 440 /etc/sudoers.d/dev-workspace

# Set environment for dev user
ENV HOME=/home/${USER_NAME}
ENV PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"

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

# Ensure Pixi home and cache directories exist
ENV PIXI_HOME=/home/${USER_NAME}/.pixi
ENV PIXI_CACHE_DIR=/home/${USER_NAME}/.cache/pixi
RUN mkdir -p $PIXI_HOME $PIXI_CACHE_DIR $HOME/.cache/rattler

# Install Pixi as dev user (pinned version for reproducible builds)
ENV PIXI_VERSION=0.65.0
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} bash

# Copy dependency manifests first for layer caching.
# .pixi/config.toml (detached-environments = true) is copied here so that
# pixi install bakes the env into the per-user cache dir rather than
# /workspace/.pixi/envs/. This matches the runtime layout where the workspace
# is bind-mounted and the bind-mount shadows any in-workspace env.
# Note: requirements*.txt are auto-generated lockfiles from pixi.toml (see scripts/sync_requirements.py)
COPY --chown=${USER_NAME}:${USER_NAME} pixi.toml pixi.lock pyproject.toml requirements.txt requirements-dev.txt .pre-commit-config.yaml ./
COPY --chown=${USER_NAME}:${USER_NAME} .pixi/config.toml .pixi/config.toml

# Install project dependencies (cached unless manifests change)
# Env installs into $HOME/.cache/pixi/envs/<workspace-hash>/ per detached-environments config.
RUN pixi install

# Ensure pre-commit is available in Pixi environment
# (pip and pre-commit are already installed by pixi install via pixi.toml;
#  avoid `pip install --upgrade pip` which can fail with directory conflicts)
RUN pixi run pre-commit --version

# Install pre-commit hooks (cached unless .pre-commit-config.yaml changes).
# Build context lacks .git, but `pre-commit install-hooks` still requires
# git to be operational inside *some* repository to clone hook envs.
# Initialize a throwaway repo so install-hooks succeeds without `|| true`.
RUN git config --global user.email "build@projectodyssey.local" && \
    git config --global user.name "build" && \
    git config --global init.defaultBranch main && \
    git init -q . && \
    if [ -d .git ]; then \
        pixi run pre-commit install --install-hooks; \
    else \
        pixi run pre-commit install-hooks; \
    fi

# Copy entrypoint script for container initialization
COPY --chown=${USER_NAME}:${USER_NAME} docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy the rest of the workspace (invalidates only layers after this)
COPY --chown=${USER_NAME}:${USER_NAME} . .

# Set Python path
ENV PYTHONPATH=/workspace:${PYTHONPATH:-}

# Default shell
CMD ["pixi", "shell"]


# ---------------------------
# Stage 3: CI / Testing
# ---------------------------
FROM development AS ci

CMD ["pixi", "run", "pytest", "tests/", "-v"]

# ---------------------------
# Stage 4: Production
# ---------------------------
FROM base AS production

# Re-declare ARG so it's available in this stage (ARGs don't persist across FROM)
ARG USER_NAME=dev

# Copy only what's needed: pixi env and project source (no dev tools)
COPY --from=development /home/${USER_NAME}/.pixi /home/${USER_NAME}/.pixi
COPY --from=development /workspace /workspace

ENV ENVIRONMENT=production
ENV PIXI_HOME=/home/${USER_NAME}/.pixi
ENV PATH="/home/${USER_NAME}/.pixi/bin:${PATH}"
USER ${USER_NAME}
WORKDIR /workspace

CMD ["pixi", "run", "python", "-c", "print('ProjectOdyssey production image ready')"]
