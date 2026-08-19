#!/bin/bash
set -Eeuo pipefail

CONTAINER="${PIB_MOTORS_CONTAINER:-multirepo-ros-motors-1}"
CONTAINER_TIMEOUT="${PIB_MOTORS_CONTAINER_TIMEOUT:-180}"

for ((attempt = 1; attempt <= CONTAINER_TIMEOUT; attempt++)); do
    if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" = "true" ]; then
        exec docker exec "${CONTAINER}" \
            /ros_entrypoint.sh \
            python3 /app/system_pose.py \
            "$@"
    fi
    sleep 1
done

echo "system-pose: ERROR: ${CONTAINER} was not running after ${CONTAINER_TIMEOUT}s" >&2
exit 1
