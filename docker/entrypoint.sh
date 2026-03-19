#!/usr/bin/env bash
set -e

# Ensure the workspace .pixi environment is functional.
# The named volume at /workspace/.pixi shadows the bind-mounted host .pixi/,
# but starts empty on first run. Run pixi install to populate it if needed.
if [ ! -x ".pixi/envs/default/bin/mojo" ]; then
    echo "Initializing pixi environment inside container..."
    pixi install
fi

exec "$@"
