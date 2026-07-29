#!/usr/bin/env bash
# Select the compatible crun shipped on affected GitHub-hosted runners.
#
# Ubuntu runner images can contain both a distribution crun in /usr/bin and a
# newer runner-provided crun in /usr/local/bin.  Podman's default ordering may
# select the incompatible distribution binary.  Override only that affected
# pairing in the current user's containers.conf; never modify runner binaries.

set -euo pipefail

echo "Runner image: ImageOS=${ImageOS:-unknown} ImageVersion=${ImageVersion:-unknown}"
podman --version

if [[ -x /usr/local/bin/crun ]]; then
  echo "/usr/local/bin/crun version:"
  /usr/local/bin/crun --version
else
  echo "/usr/local/bin/crun is unavailable"
fi

if [[ -x /usr/bin/crun ]]; then
  echo "/usr/bin/crun version:"
  /usr/bin/crun --version
else
  echo "/usr/bin/crun is unavailable"
fi

selected_runtime="$(
  podman info --format '{{.Host.OCIRuntime.Path}}'
)"
echo "Podman initially selected OCI runtime: ${selected_runtime:-unknown}"

preferred_runtime="/usr/local/bin/crun"
distribution_runtime="/usr/bin/crun"

if [[ "$selected_runtime" == "$distribution_runtime" && -x "$preferred_runtime" ]]; then
  config_root="${XDG_CONFIG_HOME:-$HOME/.config}"
  runtime_config_dir="$config_root/containers/containers.conf.d"
  runtime_config="$runtime_config_dir/99-odyssey-ci-runtime.conf"

  mkdir -p "$runtime_config_dir"
  {
    echo "[engine.runtimes]"
    echo 'crun = ["/usr/local/bin/crun", "/usr/bin/crun"]'
  } >"$runtime_config"
  echo "Configured rootless crun ordering in $runtime_config"
fi

verified_runtime="$(
  podman info --format '{{.Host.OCIRuntime.Path}}'
)"
echo "Podman selected OCI runtime after bootstrap: ${verified_runtime:-unknown}"

if [[ "$selected_runtime" == "$distribution_runtime" && -x "$preferred_runtime" && "$verified_runtime" != "$preferred_runtime" ]]; then
  echo "ERROR: Podman selected $verified_runtime instead of $preferred_runtime" >&2
  exit 1
fi
