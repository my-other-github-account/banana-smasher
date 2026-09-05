#!/bin/bash
# Read-only observer. Installed from a pinned canonical commit.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
STATE=${BANANA_CONTROL_STATE:-${HERMES_HOME:-$HOME/.hermes}/state}
exec python3 "$HERE/controller.py" --state "$STATE" --receipt "$STATE/sensor_observation.json"
