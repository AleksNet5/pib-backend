#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_POSE_NAMES = ("start1", "start2", "start3")


def log(message: str) -> None:
    print(f"system-pose: {message}", flush=True)


def get_json(base_url: str, path: str, deadline: float) -> dict:
    last_error = None
    while time.monotonic() < deadline:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            with urlopen(
                f"{base_url.rstrip('/')}{path}",
                timeout=min(5.0, remaining),
            ) as response:
                return json.load(response)
        except (OSError, URLError, ValueError) as error:
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"pib API did not become ready: {last_error}")


def resolve_pose(base_url: str, pose_name: str, timeout: float) -> list[dict]:
    deadline = time.monotonic() + timeout
    poses = get_json(base_url, "/pose", deadline).get("poses", [])
    matches = [
        pose
        for pose in poses
        if str(pose.get("name", "")).casefold() == pose_name.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one pose named {pose_name!r}, found {len(matches)}"
        )

    pose_id = matches[0]["poseId"]
    response = get_json(
        base_url,
        f"/pose/{pose_id}/motor-positions",
        deadline,
    )
    positions = response.get("motorPositions", [])
    if not positions:
        raise RuntimeError(f"pose {pose_name!r} has no motor positions")
    return positions


def clamp_to_motor_limits(
    base_url: str,
    positions: list[dict],
    timeout: float,
) -> tuple[list[tuple[str, float]], list[str]]:
    deadline = time.monotonic() + timeout
    motors = get_json(base_url, "/motor", deadline).get("motors", [])
    limits = {
        motor["name"]: (
            float(motor["rotationRangeMin"]),
            float(motor["rotationRangeMax"]),
        )
        for motor in motors
    }

    targets = []
    adjustments = []
    seen = set()
    for motor_position in positions:
        name = motor_position["motorName"]
        if name in seen:
            raise RuntimeError(f"pose contains duplicate motor {name!r}")
        seen.add(name)
        if name not in limits:
            raise RuntimeError(f"pose references unknown motor {name!r}")

        minimum, maximum = sorted(limits[name])
        requested = float(motor_position["position"])
        target = min(max(requested, minimum), maximum)
        if target != requested:
            adjustments.append(
                f"{name}: {requested / 100:.2f} -> {target / 100:.2f} degrees"
            )
        targets.append((name, target))

    return targets, adjustments


def apply_pose(
    targets: list[tuple[str, float]],
    service_timeout: float,
) -> None:
    import rclpy
    from datatypes.srv import ApplyJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = rclpy.create_node("system_pose_runner")
    try:
        client = node.create_client(
            ApplyJointTrajectory,
            "/apply_joint_trajectory",
        )
        deadline = time.monotonic() + service_timeout
        while not client.wait_for_service(timeout_sec=2.0):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "apply_joint_trajectory service did not become ready"
                )

        trajectory = JointTrajectory()
        for name, position in targets:
            trajectory.joint_names.append(name)
            point = JointTrajectoryPoint()
            point.positions.append(position)
            trajectory.points.append(point)

        request = ApplyJointTrajectory.Request()
        request.joint_trajectory = trajectory
        future = client.call_async(request)
        remaining = max(0.1, deadline - time.monotonic())
        rclpy.spin_until_future_complete(
            node,
            future,
            timeout_sec=remaining,
        )
        if not future.done():
            raise RuntimeError("timed out while applying system pose")
        response = future.result()
        if response is None or not response.successful:
            raise RuntimeError("motor controller rejected the system pose")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def check_motor_service(service_timeout: float) -> None:
    import rclpy
    from datatypes.srv import ApplyJointTrajectory

    rclpy.init()
    node = rclpy.create_node("system_pose_check")
    try:
        client = node.create_client(
            ApplyJointTrajectory,
            "/apply_joint_trajectory",
        )
        deadline = time.monotonic() + service_timeout
        while not client.wait_for_service(timeout_sec=2.0):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "apply_joint_trajectory service did not become ready"
                )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply pib's startup and poweroff pose sequence.",
    )
    parser.add_argument(
        "--pose-name",
        dest="pose_names",
        action="append",
        help=(
            "pose to apply; repeat this option for a sequence "
            "(default: start1, start2, start3)"
        ),
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=180,
        help="seconds to wait for pib-api",
    )
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=120,
        help="seconds to wait for the ROS motor service",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=12,
        help=(
            "seconds to wait after each pose before continuing "
            "(default: 12)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate the pose without moving motors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = os.getenv("FLASK_API_BASE_URL", "http://flask-app:5000")
    try:
        pose_names = args.pose_names or list(DEFAULT_POSE_NAMES)
        sequence = []
        for pose_name in pose_names:
            positions = resolve_pose(base_url, pose_name, args.api_timeout)
            targets, adjustments = clamp_to_motor_limits(
                base_url,
                positions,
                args.api_timeout,
            )
            sequence.append((pose_name, targets))
            log(f"resolved {pose_name!r} with {len(targets)} motor targets")
            for adjustment in adjustments:
                log(f"clamped to configured range: {adjustment}")

        if args.dry_run:
            check_motor_service(args.service_timeout)
            log(
                "dry run successful; all poses, the API, and motor service "
                "are ready"
            )
            log("no motor command was sent")
            return 0

        for index, (pose_name, targets) in enumerate(sequence):
            apply_pose(targets, args.service_timeout)
            log(f"pose {pose_name!r} accepted by motor controller")
            if args.settle_seconds > 0:
                if index + 1 < len(sequence):
                    next_pose_name = sequence[index + 1][0]
                    log(
                        f"waiting {args.settle_seconds:g} seconds before "
                        f"pose {next_pose_name!r}"
                    )
                else:
                    log(
                        f"waiting {args.settle_seconds:g} seconds for the "
                        "final movement to finish"
                    )
                time.sleep(args.settle_seconds)
        log("pose sequence completed")
        return 0
    except Exception as error:
        log(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
