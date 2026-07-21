#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /app/ros2_ws/install/setup.bash

if [ -n "${WAIT_FOR_DEVICES:-}" ]; then
  timeout="${WAIT_FOR_DEVICES_TIMEOUT:-30}"
  for device in ${WAIT_FOR_DEVICES}; do
    for attempt in $(seq 1 "${timeout}"); do
      [ -e "${device}" ] && break
      sleep 1
    done
    [ -e "${device}" ] || {
      echo "Missing required device ${device} after ${timeout}s"
      exit 1
    }
  done
fi

exec "$@"
