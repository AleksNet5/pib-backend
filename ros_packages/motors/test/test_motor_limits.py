import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "pib_motors"))

fake_api_client = types.ModuleType("pib_api_client")
fake_api_client.motor_client = types.SimpleNamespace(
    get_all_motors=lambda: (True, {"motors": []})
)
sys.modules.setdefault("pib_api_client", fake_api_client)

from pib_motors.motor import Motor  # noqa: E402


class FakePin:
    def __init__(self):
        self.positions = []

    def set_position(self, position):
        self.positions.append(position)
        return True


class MotorLimitTest(unittest.TestCase):
    def test_non_inverted_logical_position_is_clamped_before_write(self):
        pin = FakePin()
        motor = Motor("test", [pin], invert=False)
        motor.rotation_range_min = -200
        motor.rotation_range_max = 800

        self.assertEqual(motor.logical_position_limits(), (-200, 800))
        self.assertEqual(motor.clamp_logical_position(1600), 800)
        self.assertTrue(motor.set_position(1600))
        self.assertEqual(pin.positions, [800])

    def test_inverted_motor_exposes_inverted_asymmetric_limits(self):
        pin = FakePin()
        motor = Motor("test", [pin], invert=True)
        motor.rotation_range_min = -200
        motor.rotation_range_max = 800

        self.assertEqual(motor.logical_position_limits(), (-800, 200))
        self.assertEqual(motor.clamp_logical_position(-1600), -800)
        self.assertTrue(motor.set_position(-1600))
        self.assertEqual(pin.positions, [800])

    def test_feedback_limit_tolerance_does_not_hide_large_fault(self):
        motor = Motor("test", [FakePin()], invert=False)
        motor.rotation_range_min = -9000
        motor.rotation_range_max = 9000

        self.assertTrue(motor.logical_position_is_in_range(9050, tolerance=100))
        self.assertFalse(
            motor.logical_position_is_in_range(16260, tolerance=100)
        )


if __name__ == "__main__":
    unittest.main()
