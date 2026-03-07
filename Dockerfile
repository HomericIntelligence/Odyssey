# ML Odyssey - Mojo Development Environment
# Multi-stage Dockerfile

# ---------------------------
# Stage 1: Base image with system deps
# ---------------------------
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ca-certificates \
    vim \
    wget \
    uuid \
    sudo \
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

# Install Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install just tool (pre-built binary, much faster than cargo install)
RUN curl -fsSL https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# ---------------------------
# Stage 1.5: Create dev user
# ---------------------------
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=dev

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME}

# Set environment for dev user
ENV HOME=/home/${USER_NAME}
ENV PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"

# ---------------------------
# Stage 2: Development environment
# ---------------------------
FROM base AS development

# Switch to dev user
USER ${USER_NAME}
WORKDIR /workspace

# Ensure Pixi home and cache directories exist
ENV PIXI_HOME=/home/${USER_NAME}/.pixi
ENV PIXI_CACHE_DIR=/home/${USER_NAME}/.cache/pixi
RUN mkdir -p $PIXI_HOME $PIXI_CACHE_DIR $HOME/.cache/rattler && \
    chmod -R 700 $PIXI_HOME $PIXI_CACHE_DIR $HOME/.cache/rattler

# Install Pixi as dev user (pinned version for reproducible builds)
ENV PIXI_VERSION=0.65.0
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} bash

# Copy dependency manifests first for layer caching
COPY --chown=${USER_NAME}:${USER_NAME} pixi.toml pixi.lock pyproject.toml requirements.txt requirements-dev.txt .pre-commit-config.yaml ./

# Install project dependencies (cached unless manifests change)
RUN pixi install

# Install pre-commit inside Pixi environment
RUN pixi run pip install --upgrade pip pre-commit

# Install pre-commit hooks (cached unless .pre-commit-config.yaml changes)
RUN pixi run pre-commit install --install-hooks || true

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

# Copy only what's needed: pixi env and project source (no dev tools)
COPY --from=development /home/${USER_NAME}/.pixi /home/${USER_NAME}/.pixi
COPY --from=development /workspace /workspace

ENV ENVIRONMENT=production
ENV PIXI_HOME=/home/${USER_NAME}/.pixi
ENV PATH="/home/${USER_NAME}/.pixi/bin:${PATH}"
USER ${USER_NAME}
WORKDIR /workspace

CMD ["pixi", "run", "python", "-c", "print('ProjectOdyssey production image ready')"]
