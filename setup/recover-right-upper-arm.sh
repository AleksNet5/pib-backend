#!/bin/bash
set -Eeuo pipefail

if (( EUID != 0 )); then
    exec sudo -- "$0" "$@"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

restart_after_recovery=false
if [[ "${1:-}" == "--restart-motors" ]]; then
    restart_after_recovery=true
    shift
fi

motor_container="$(docker compose ps -a -q ros-motors)"
motors_were_running=false
if [[ -n "${motor_container}" ]] && [[ "$(
    docker inspect --format '{{.State.Running}}' "${motor_container}"
)" == "true" ]]; then
    motors_were_running=true
    echo "Stopping ros-motors to give the recovery tool exclusive bus access."
fi

if [[ -n "${motor_container}" ]]; then
    docker compose rm --stop --force ros-motors
fi

cleanup() {
    if [[ "${restart_after_recovery}" == "true" ]] && \
        [[ "${motors_were_running}" == "true" ]]; then
        echo "Starting ros-motors again."
        docker compose up -d ros-motors
    else
        echo "ros-motors remains removed to prevent a normal clamped target " \
            "from twisting ID 19 again."
    fi
}
trap cleanup EXIT

docker compose run --rm --no-deps ros-motors \
    python3 /app/untwist_right_upper_arm.py "$@"
