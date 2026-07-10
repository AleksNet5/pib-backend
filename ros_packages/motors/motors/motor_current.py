import os

import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_msgs.msg import KeyValue


def _current_polling_enabled() -> bool:
    return os.getenv("PIB_MOTOR_CURRENT_ENABLED", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class MotorCurrent(Node):

    def __init__(self):

        super().__init__("motor_current")

        self.enabled = _current_polling_enabled()
        self.motors = []
        self.motor_class = None

        self.declare_parameter("frequency", 4.0)
        self.frequency_ = self.get_parameter("frequency").value

        self.motor_current_publisher = self.create_publisher(
            DiagnosticStatus, "motor_current", 10
        )

        if self.enabled:
            from pib_motors.motor import motors, Motor

            self.motors = motors
            self.motor_class = Motor
            self.timer = self.create_timer(
                1.0 / self.frequency_, self.publish_motor_current
            )
        else:
            self.timer = None
            self.get_logger().info("Motor current polling disabled")

        self.get_logger().info("Now Running MOTOR CURRENT")

    def publish_motor_current(self):

        if not self.enabled or self.motor_class is None:
            return

        for motor in self.motors:
            current = motor.get_current()
            if current == self.motor_class.NO_CURRENT:
                continue
            self.publish_diagnostic_status(motor.name, current)

    def publish_diagnostic_status(self, motor_name: str, current: int) -> None:
        msg = DiagnosticStatus()
        msg.level = DiagnosticStatus.WARN if current >= 1500 else DiagnosticStatus.OK
        msg.name = motor_name
        keyvalue = KeyValue()
        keyvalue.key = motor_name
        keyvalue.value = str(current)
        msg.values = [keyvalue]
        self.motor_current_publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)
    node = MotorCurrent()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
