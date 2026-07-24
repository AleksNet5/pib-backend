import os
from typing import Iterable, Tuple

import rclpy
from datatypes.msg import MotorSettings
from datatypes.srv import ApplyMotorSettings, ApplyJointTrajectory, ResetMotorZero
from pib_api_client import motor_client
from pib_motors.bricklet import connected_enumerate
from pib_motors.motor import name_to_motors, motors
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from motors.collision_guard import CollisionGuard, all_arm_names


HAND_MOTOR_NAMES = {
    "thumb_left_opposition",
    "thumb_left_stretch",
    "index_left_stretch",
    "middle_left_stretch",
    "ring_left_stretch",
    "pinky_left_stretch",
    "thumb_right_opposition",
    "thumb_right_stretch",
    "index_right_stretch",
    "middle_right_stretch",
    "ring_right_stretch",
    "pinky_right_stretch",
}


def motor_settings_ros_to_dto(ms: MotorSettings):
    return {
        "name": ms.motor_name,
        "turnedOn": ms.turned_on,
        "pulseWidthMin": ms.pulse_width_min,
        "pulseWidthMax": ms.pulse_width_max,
        "rotationRangeMin": ms.rotation_range_min,
        "rotationRangeMax": ms.rotation_range_max,
        "velocity": ms.velocity,
        "acceleration": ms.acceleration,
        "deceleration": ms.deceleration,
        "period": ms.period,
        "visible": ms.visible,
        "invert": ms.invert,
    }


def as_motor_positions(jt: JointTrajectory) -> Iterable[Tuple[str, int]]:
    """unpacks a jt-message into an iterable of motorname-position-pairs"""
    motor_names = jt.joint_names
    points: list[JointTrajectoryPoint] = jt.points
    positions = (point.positions[0] for point in points)
    return zip(motor_names, positions)


def as_joint_trajectory(motor_name: str, position: int) -> JointTrajectory:
    """converts a motorname and position into a simple jt-message"""
    jt = JointTrajectory()
    jt.joint_names = [motor_name]
    point = JointTrajectoryPoint()
    point.positions.append(position)
    jt.points = [point]
    return jt


def as_arm_pose(positions: dict[str, float]) -> JointTrajectory:
    jt = JointTrajectory()
    jt.joint_names = sorted(positions)
    point = JointTrajectoryPoint()
    point.positions = [positions[name] for name in jt.joint_names]
    jt.points = [point]
    return jt


def as_collision_limits(
    motor_name: str,
    minimum: float,
    maximum: float,
) -> JointTrajectory:
    jt = JointTrajectory()
    jt.joint_names = [motor_name]
    minimum_point = JointTrajectoryPoint()
    minimum_point.positions = [float(minimum)]
    maximum_point = JointTrajectoryPoint()
    maximum_point.positions = [float(maximum)]
    jt.points = [minimum_point, maximum_point]
    return jt


class MotorControl(Node):

    def __init__(self):

        super().__init__("motor_control")

        # Toggle Devmode
        self.declare_parameter("dev", False)
        self.dev = self.get_parameter("dev").value

        # Service for JointTrajectory
        self.srv = self.create_service(
            ApplyJointTrajectory, "apply_joint_trajectory", self.apply_joint_trajectory
        )

        # Publisher for JointTrajectory
        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory, "joint_trajectory", 10
        )
        self.collision_arm_positions_publisher = self.create_publisher(
            JointTrajectory, "collision_arm_positions", 10
        )
        self.collision_joint_limits_publisher = self.create_publisher(
            JointTrajectory, "collision_joint_limits", 10
        )
        self.collision_limit_request_subscription = self.create_subscription(
            String,
            "collision_limit_request",
            self._publish_collision_joint_limits,
            10,
        )

        # Service for MotorSettings
        self.srv = self.create_service(
            ApplyMotorSettings, "apply_motor_settings", self.apply_motor_settings
        )

        self.reset_zero_srv = self.create_service(
            ResetMotorZero, "reset_motor_zero", self.reset_motor_zero
        )

        # Publisher for MotorSettings
        self.motor_settings_publisher = self.create_publisher(
            MotorSettings, "motor_settings", 10
        )

        # load motor-settings if not in dev mode
        if not self.dev:
            for motor in motors:
                successful, motor_settings_dto = motor_client.get_motor_settings(
                    motor.name
                )
                if successful:
                    motor.load_settings(motor_settings_dto)

        self.collision_guard_mode = os.getenv(
            "COLLISION_GUARD_MODE", "off"
        ).lower()
        self.collision_guard = self._create_collision_guard()
        self.arm_positions = (
            self._read_arm_positions() if self.collision_guard is not None else {}
        )
        self.collision_arm_positions_timer = self.create_timer(
            1.0, self._publish_collision_arm_positions
        )
        self._log_initial_collision_state()

        # Log that initialization is complete
        self.get_logger().info("Now Running MOTOR_CONTROL")

    def _create_collision_guard(self):
        if self.collision_guard_mode == "off":
            self.get_logger().info("Collision guard is disabled")
            return None
        if self.collision_guard_mode not in {"monitor", "enforce"}:
            raise ValueError(
                f"invalid COLLISION_GUARD_MODE: {self.collision_guard_mode}"
            )

        config_path = os.getenv(
            "COLLISION_GEOMETRY_PATH",
            "/app/ros2_ws/motors/config/collision_geometry.json",
        )
        step_degrees = float(
            os.getenv("COLLISION_TRAJECTORY_STEP_DEGREES", "1")
        )
        range_step_degrees = float(
            os.getenv("COLLISION_RANGE_STEP_DEGREES", "1")
        )
        guard = CollisionGuard(
            config_path,
            step_degrees=step_degrees,
            range_step_degrees=range_step_degrees,
        )
        self.get_logger().info(
            f"Collision guard started in {self.collision_guard_mode} mode "
            f"with {len(guard.obstacles)} obstacles"
        )
        return guard

    def _read_arm_positions(self) -> dict[str, float]:
        positions = {}
        for motor_name in all_arm_names():
            motor = name_to_motors[motor_name][0]
            position = motor.get_position()
            if motor.invert:
                position *= -1
            positions[motor_name] = float(position)
        self.get_logger().info(f"Collision guard arm positions: {positions}")
        return positions

    def _publish_collision_arm_positions(self) -> None:
        if self.arm_positions:
            self.collision_arm_positions_publisher.publish(
                as_arm_pose(self.arm_positions)
            )

    def _publish_collision_joint_limits(self, request: String) -> None:
        try:
            motor_name = request.data
            if (
                self.collision_guard is None
                or motor_name not in all_arm_names()
                or motor_name not in self.arm_positions
            ):
                return

            self.arm_positions = self._read_arm_positions()
            motor = name_to_motors[motor_name][0]
            minimum, maximum = self.collision_guard.safe_joint_range(
                self.arm_positions,
                motor_name,
                motor.rotation_range_min,
                motor.rotation_range_max,
            )
            self.collision_joint_limits_publisher.publish(
                as_collision_limits(motor_name, minimum, maximum)
            )
        except Exception as error:
            self.get_logger().error(
                f"failed to publish collision limits for {request.data}: {error}"
            )

    def _log_initial_collision_state(self) -> None:
        if self.collision_guard is None:
            return
        collisions = {
            pair: clearance
            for pair, clearance in self.collision_guard.clearances(
                self.arm_positions
            ).items()
            if clearance <= 0
        }
        if collisions:
            self.get_logger().warn(
                "Collision guard current-pose escape handling is active; new or deeper "
                f"intersections will be blocked in enforce mode: {collisions}"
            )
        else:
            self.get_logger().info("Collision guard startup pose is clear")



    def apply_motor_settings(
        self, request: ApplyMotorSettings.Request, response: ApplyMotorSettings.Response
    ) -> ApplyMotorSettings.Response:

        response.settings_applied = True
        response.settings_persisted = True

        motor_settings_ros = request.motor_settings
        motor_settings_dto = motor_settings_ros_to_dto(motor_settings_ros)

        try:
            motors = name_to_motors[request.motor_settings.motor_name]
            for motor in motors:
                motor_settings_dto["name"] = motor.name
                motor_settings_ros.motor_name = motor.name
                applied = motor.apply_settings(motor_settings_dto)
                response.settings_applied &= applied
                if applied or self.dev:
                    persisted, _ = motor_client.update_motor_settings(
                        motor.name, motor_settings_dto
                    )
                    response.settings_persisted &= persisted
                    self.motor_settings_publisher.publish(motor_settings_ros)
                self.get_logger().info(f"updated motor: {str(motor)}")

        except Exception as e:
            response.settings_applied = False
            response.settings_persisted = False
            self.get_logger().warn(
                f"Error while processing motor-settings-message: {str(e)}"
            )

        return response

    def reset_motor_zero(
        self,
        request: ResetMotorZero.Request,
        response: ResetMotorZero.Response,
    ) -> ResetMotorZero.Response:
        motor_name = request.motor_name
        if motor_name not in HAND_MOTOR_NAMES:
            response.successful = False
            response.message = f"zero reset is only allowed for hand motors: {motor_name}"
            self.get_logger().warn(response.message)
            return response

        try:
            response.successful = all(
                motor.reset_zero_position() for motor in name_to_motors[motor_name]
            )
            response.message = (
                "zero position reset succeeded"
                if response.successful
                else "zero position reset failed"
            )
            if response.successful:
                self.joint_trajectory_publisher.publish(
                    as_joint_trajectory(motor_name, 0)
                )
        except Exception as error:
            response.successful = False
            response.message = str(error)

        self.get_logger().info(
            f"reset zero position of {motor_name}: {response.message}"
        )
        return response

    def apply_joint_trajectory(
        self,
        request: ApplyJointTrajectory.Request,
        response: ApplyJointTrajectory.Response,
    ) -> ApplyJointTrajectory.Response:
        jt = request.joint_trajectory
        motor_positions = list(as_motor_positions(jt))
        response.successful = True
        try:
            if self.collision_guard is not None:
                targets = dict(motor_positions)
                if any(name in all_arm_names() for name in targets):
                    self.arm_positions = self._read_arm_positions()
                collision_result = self.collision_guard.evaluate(
                    self.arm_positions, targets
                )
                if not collision_result.allowed:
                    message = (
                        f"Collision guard would block trajectory {targets}: "
                        f"{collision_result.reason}; minimum clearance "
                        f"{collision_result.minimum_clearance_mm:.1f} mm"
                    )
                    if self.collision_guard_mode == "enforce":
                        safe_positions = collision_result.safe_positions or {}
                        clamped_positions = [
                            (
                                motor_name,
                                int(round(safe_positions.get(motor_name, position))),
                            )
                            for motor_name, position in motor_positions
                        ]
                        can_move_to_boundary = any(
                            motor_name in safe_positions
                            and abs(position - self.arm_positions[motor_name]) >= 1
                            for motor_name, position in clamped_positions
                        )
                        if not can_move_to_boundary:
                            response.successful = False
                            self.get_logger().warn(message)
                            return response
                        self.get_logger().warn(
                            f"{message}; clamping to {dict(clamped_positions)}"
                        )
                        motor_positions = clamped_positions
                    else:
                        self.get_logger().warn(f"MONITOR ONLY: {message}")

            for motor_name, position in motor_positions:
                for motor in name_to_motors[motor_name]:
                    self.get_logger().info(
                        f"setting position of {motor.name} to {position}"
                    )
                    successful = motor.set_position(position)
                    self.get_logger().info(
                        f"setting position {'succeeded' if successful else 'failed'}."
                    )
                    response.successful &= successful
                    if successful and motor.name in self.arm_positions:
                        self.arm_positions[motor.name] = float(position)
                    self.joint_trajectory_publisher.publish(
                        as_joint_trajectory(motor.name, position)
                    )
        except Exception as e:
            response.successful = False
            self.get_logger().error(f"error while applying joint-trajectory: {str(e)}")
        return response


def main(args=None):

    rclpy.init(args=args)
    motor_control = MotorControl()
    rclpy.spin(motor_control)
    rclpy.shutdown()
    


if __name__ == "__main__":
    main()
