#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    --dry-run)
        exec "${SCRIPT_DIR}/run-startup-poweroff-pose.sh" --dry-run
        ;;
    --pose-only)
        exec "${SCRIPT_DIR}/run-startup-poweroff-pose.sh"
        ;;
    "")
        ;;
    *)
        echo "Usage: $0 [--dry-run|--pose-only]" >&2
        exit 2
        ;;
esac

"${SCRIPT_DIR}/run-startup-poweroff-pose.sh"
echo "system-pose: rebooting"
exec sudo -n /usr/bin/systemctl reboot
