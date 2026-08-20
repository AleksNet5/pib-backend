#!/bin/bash
set -Eeuo pipefail

if (( EUID != 0 )); then
    exec sudo -- "$0" "$@"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

motor_container="$(docker compose ps -a -q ros-motors)"
if [[ -n "${motor_container}" ]]; then
    echo "Removing ros-motors for exclusive, no-motion calibration access."
    docker compose rm --stop --force ros-motors
fi

cleanup() {
    echo "ros-motors remains removed until the new zero is verified."
}
trap cleanup EXIT

docker compose run --rm --no-deps ros-motors \
    python3 /app/calibrate_right_upper_arm_zero.py "$@"
