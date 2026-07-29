#!/usr/bin/env bash
#
# Validate the local Podman runtime without changing machine state.
#
# Mojo compilation is memory intensive. The default and constrained profiles
# target nominal 16-GiB and 8-GiB runtimes, respectively. Podman Host.MemTotal
# may be at most 512 MiB lower because the guest reserves memory for itself.
# The rendered Compose model and active engine are authoritative; this script
# never guesses which Podman machine is in use.

set -euo pipefail

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        fail "$name must be a positive integer (received '$value')."
    fi
}

normalize_memory_bytes() {
    python3 - "$1" <<'PY' 2>/dev/null
from decimal import Decimal, InvalidOperation
import re
import sys

value = sys.argv[1]
match = re.fullmatch(
    r"(?i)([0-9]+(?:\.[0-9]+)?|\.[0-9]+)(b|k|kb|ki|kib|m|mb|mi|mib|g|gb|gi|gib)?",
    value,
)
if match is None:
    raise SystemExit(1)

try:
    number = Decimal(match.group(1))
except InvalidOperation:
    raise SystemExit(1)
if number <= 0:
    raise SystemExit(1)

unit = (match.group(2) or "b").lower()
power = {
    "b": 0,
    "k": 1,
    "kb": 1,
    "ki": 1,
    "kib": 1,
    "m": 2,
    "mb": 2,
    "mi": 2,
    "mib": 2,
    "g": 3,
    "gb": 3,
    "gi": 3,
    "gib": 3,
}[unit]
byte_count = number * (Decimal(1024) ** power)
if byte_count != byte_count.to_integral_value():
    raise SystemExit(1)
print(int(byte_count))
PY
}

normalize_cpu_micros() {
    python3 - "$1" <<'PY' 2>/dev/null
from decimal import Decimal, InvalidOperation
import re
import sys

value = sys.argv[1]
if re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", value) is None:
    raise SystemExit(1)
try:
    cpus = Decimal(value)
except InvalidOperation:
    raise SystemExit(1)
micros = cpus * 1_000_000
if cpus <= 0 or micros != micros.to_integral_value():
    raise SystemExit(1)
print(int(micros))
PY
}

resource_remediation() {
    printf 'Increase resources for the host or machine selected by the active '
    printf 'Podman connection, then restart Podman as needed.\n'
    printf 'Inspect the active connection with: podman system connection list\n'
    printf 'Supported profiles:\n'
    printf '  default:     ODYSSEY_MEM_LIMIT=14g ODYSSEY_CPU_LIMIT=6.0 '
    printf 'BUILD_PARALLELISM=1..4 (16-GiB nominal; Host.MemTotal >=15872 MiB)\n'
    printf '  constrained: ODYSSEY_MEM_LIMIT=7g ODYSSEY_CPU_LIMIT=6.0 '
    printf 'BUILD_PARALLELISM=1 (8-GiB nominal; Host.MemTotal >=7680 MiB)\n'
}

if ! command -v podman >/dev/null 2>&1; then
    fail "Podman is not installed. Install Podman before running container recipes."
fi
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 is required to validate Podman and Compose resources."
fi

user_id="${USER_ID:-$(id -u)}"
group_id="${GROUP_ID:-$(id -g)}"
build_parallelism="${BUILD_PARALLELISM:-4}"
compose_memory_limit="${ODYSSEY_MEM_LIMIT:-14g}"
compose_cpu_limit="${ODYSSEY_CPU_LIMIT:-6.0}"

require_positive_integer "USER_ID" "$user_id"
require_positive_integer "GROUP_ID" "$group_id"
require_positive_integer "BUILD_PARALLELISM" "$build_parallelism"
if ! compose_memory_bytes="$(normalize_memory_bytes "$compose_memory_limit")"; then
    fail "ODYSSEY_MEM_LIMIT must be a positive Compose byte value (received '$compose_memory_limit')."
fi
if ! compose_cpu_micros="$(normalize_cpu_micros "$compose_cpu_limit")"; then
    fail "ODYSSEY_CPU_LIMIT must be a positive number (received '$compose_cpu_limit')."
fi

# Compose requires these substitutions even when the caller invokes the script
# directly instead of through Just. Just exports the same values after loading
# .env, so preflight and all subsequent Compose commands share one contract.
export USER_ID="$user_id"
export GROUP_ID="$group_id"
export BUILD_PARALLELISM="$build_parallelism"
export USER_NAME="dev"
export ODYSSEY_MEM_LIMIT="$compose_memory_limit"
export ODYSSEY_CPU_LIMIT="$compose_cpu_limit"

# Query the engine selected by the active connection before inspecting or
# assuming anything about locally configured machines. podman info reports the
# capacity of that reachable engine for both native and machine-backed setups.
if ! podman_info="$(podman info --format json 2>/dev/null)"; then
    {
        printf 'ERROR: Podman engine is not reachable for the active connection.\n'
        printf 'Start that Podman machine or rootless service.\n'
        printf 'Inspect the active connection with: podman system connection list\n'
    } >&2
    exit 1
fi

if ! host_resources="$(
    python3 -c '
import json
import sys

info = json.load(sys.stdin)
host = info.get("host", info.get("Host", {}))
cpus = host.get("cpus", host.get("CPUs"))
memory = host.get("memTotal", host.get("MemTotal"))
if not isinstance(cpus, int) or not isinstance(memory, int):
    raise SystemExit("Podman info omitted host CPU or memory data")
if cpus <= 0 or memory <= 0:
    raise SystemExit("Podman info reported non-positive host resources")
print(f"{cpus}|{memory}")
' <<<"$podman_info"
)"; then
    fail "Could not read CPU and memory capacity from podman info."
fi
IFS='|' read -r runtime_cpus runtime_memory_bytes <<<"$host_resources"
require_positive_integer "Podman host CPU count" "$runtime_cpus"
require_positive_integer "Podman host memory" "$runtime_memory_bytes"
runtime_memory_mib=$((runtime_memory_bytes / 1024 / 1024))
runtime_cpu_micros=$((runtime_cpus * 1000000))

if ! podman compose version >/dev/null 2>&1; then
    fail "Podman Compose is unavailable. Install a Podman Compose provider."
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! compose_config="$(
    cd "$repo_root"
    podman compose config
)" 2>/dev/null; then
    fail "Podman Compose configuration is invalid. Check container identity and resource settings."
fi

# Parse only the stable Compose surface needed here. This deliberately avoids a
# PyYAML dependency in the bootstrap path while still validating the rendered
# model produced by the installed Compose provider.
if ! rendered_limits="$(
    printf '%s\n' "$compose_config" | python3 -c '
import re
import sys

expected = {"odyssey-dev", "odyssey-ci", "odyssey-prod"}
services = {}
current = None
in_services = False
for raw_line in sys.stdin:
    line = raw_line.rstrip()
    if not line or line.lstrip().startswith("#"):
        continue
    indent = len(line) - len(line.lstrip())
    stripped = line.strip()
    if indent == 0:
        in_services = stripped == "services:"
        current = None
        continue
    if not in_services:
        continue
    if indent == 2 and stripped.endswith(":"):
        current = stripped[:-1].strip("\"'\''")
        if current in services:
            raise SystemExit("duplicate rendered Compose service")
        services[current] = {}
        continue
    if current is None or indent != 4:
        continue
    match = re.fullmatch(r"(cpus|mem_limit):\s*(.*)", stripped)
    if match is None:
        continue
    key, value = match.groups()
    if key in services[current]:
        raise SystemExit("duplicate rendered Compose resource")
    services[current][key] = value.strip().strip("\"'\''")

if set(services) != expected:
    raise SystemExit("unexpected or missing rendered Compose services")
for service in sorted(expected):
    resources = services[service]
    if set(resources) != {"cpus", "mem_limit"}:
        raise SystemExit("missing rendered Compose resource")
    print(f"{service}|{resources['\''mem_limit'\'']}|{resources['\''cpus'\'']}")
'
)"; then
    fail "Could not read the rendered Compose resource limits for all services."
fi

rendered_count=0
while IFS='|' read -r service rendered_memory rendered_cpu; do
    [[ -n "$service" ]] || continue
    rendered_count=$((rendered_count + 1))
    if ! rendered_memory_bytes="$(normalize_memory_bytes "$rendered_memory")"; then
        fail "$service rendered Compose memory limit is invalid ('$rendered_memory')."
    fi
    if ! rendered_cpu_micros="$(normalize_cpu_micros "$rendered_cpu")"; then
        fail "$service rendered Compose CPU limit is invalid ('$rendered_cpu')."
    fi
    if ((rendered_memory_bytes != compose_memory_bytes)); then
        fail "$service rendered Compose memory limit contradicts ODYSSEY_MEM_LIMIT."
    fi
    if ((rendered_cpu_micros != compose_cpu_micros)); then
        fail "$service rendered Compose CPU limit contradicts ODYSSEY_CPU_LIMIT."
    fi
done <<<"$rendered_limits"
if ((rendered_count != 3)); then
    fail "Could not read the rendered Compose resource limits for all services."
fi

if ((build_parallelism * 1000000 > compose_cpu_micros)); then
    fail "BUILD_PARALLELISM ($build_parallelism) cannot exceed the effective Compose CPU limit ($compose_cpu_limit)."
fi

gib=$((1024 * 1024 * 1024))
default_memory_bytes=$((14 * gib))
constrained_memory_bytes=$((7 * gib))
required_cpu_micros=6000000
if ((compose_memory_bytes == default_memory_bytes)) \
    && ((compose_cpu_micros == required_cpu_micros)) \
    && ((build_parallelism <= 4)); then
    profile_name="default"
    nominal_memory_gib=16
    minimum_usable_memory_mib=15872
elif ((compose_memory_bytes == constrained_memory_bytes)) \
    && ((compose_cpu_micros == required_cpu_micros)) \
    && ((build_parallelism == 1)); then
    profile_name="constrained"
    nominal_memory_gib=8
    minimum_usable_memory_mib=7680
else
    {
        printf 'ERROR: Resource settings do not match a supported resource profile.\n'
        resource_remediation
    } >&2
    exit 1
fi
minimum_cpus=6
minimum_usable_memory_bytes=$((minimum_usable_memory_mib * 1024 * 1024))

if ((compose_memory_bytes > runtime_memory_bytes)); then
    {
        printf 'ERROR: Compose memory limit (%s bytes) exceeds active Podman engine memory (%s bytes).\n' \
            "$compose_memory_bytes" "$runtime_memory_bytes"
        resource_remediation
    } >&2
    exit 1
fi
if ((compose_cpu_micros > runtime_cpu_micros)); then
    {
        printf 'ERROR: Compose CPU limit (%s) exceeds active Podman engine CPUs (%s).\n' \
            "$compose_cpu_limit" "$runtime_cpus"
        resource_remediation
    } >&2
    exit 1
fi
if ((runtime_cpus < minimum_cpus)) || ((runtime_memory_bytes < minimum_usable_memory_bytes)); then
    {
        printf 'ERROR: The active Podman engine has %s CPUs and %s MiB memory; ' \
            "$runtime_cpus" "$runtime_memory_mib"
        printf 'the %s profile targets %s GiB nominal configured capacity, ' \
            "$profile_name" "$nominal_memory_gib"
        printf 'requires at least %s CPUs, and requires Host.MemTotal of at least %s MiB ' \
            "$minimum_cpus" "$minimum_usable_memory_mib"
        printf '(allowing up to 512 MiB for the Podman guest reservation).\n'
        resource_remediation
    } >&2
    exit 1
fi

printf 'Podman preflight passed (%s CPUs, %s MiB memory, %s build jobs, %s profile).\n' \
    "$runtime_cpus" "$runtime_memory_mib" "$build_parallelism" "$profile_name"
