import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_ROOT = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))
spec = importlib.util.spec_from_file_location(
    "calibrate_right_upper_arm_zero",
    TOOLS_ROOT / "calibrate_right_upper_arm_zero.py",
)
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class FakePacket:
    def __init__(self, positions=(1976, 1976, 2048, 2048)):
        self.positions = list(positions)
        self.registers = {
            calibrate.STS_TORQUE_ENABLE: 1,
            calibrate.STS_LOCK: 1,
            calibrate.STS_OFS_L: 0,
        }
        self.writes = []
        self.position_writes = []

    def ReadPosSpeed(self, _servo_id):
        position = self.positions.pop(0) if len(self.positions) > 1 else self.positions[0]
        return position, 0, calibrate.COMM_SUCCESS, 0

    def read1ByteTxRx(self, _servo_id, address):
        return self.registers.get(address, 0), calibrate.COMM_SUCCESS, 0

    def read2ByteTxRx(self, _servo_id, address):
        return self.registers.get(address, 0), calibrate.COMM_SUCCESS, 0

    def write1ByteTxRx(self, _servo_id, address, value):
        self.writes.append((address, value))
        self.registers[address] = value
        if address == calibrate.STS_TORQUE_ENABLE and value == 128:
            self.registers[calibrate.STS_OFS_L] = 72
        return calibrate.COMM_SUCCESS, 0

    def WritePosEx(self, _servo_id, position, speed, acceleration):
        self.position_writes.append((position, speed, acceleration))
        return calibrate.COMM_SUCCESS, 0

    def getTxRxResult(self, result):
        return str(result)

    def getRxPacketError(self, error):
        return str(error)


class CalibrateZeroToolTest(unittest.TestCase):
    def test_circular_tick_delta_crosses_encoder_boundary(self):
        self.assertEqual(calibrate.circular_tick_delta(4090, 10), 16)
        self.assertEqual(calibrate.circular_tick_delta(10, 4090), -16)

    @patch.object(calibrate.time, "sleep", return_value=None)
    def test_calibration_sets_midpoint_goal_before_restoring_torque(self, _sleep):
        packet = FakePacket()

        result = calibrate.perform_calibration(
            packet,
            servo_id=19,
            speed=100,
            acceleration=5,
            release_shift_tolerance=8,
            verify_tolerance=25,
            settle_seconds=0.2,
        )

        self.assertEqual(result["before"], 1976)
        self.assertEqual(result["calibrated"], 2048)
        self.assertEqual(result["final"], 2048)
        self.assertEqual(packet.position_writes, [(2048, 100, 5)])
        self.assertEqual(
            packet.writes,
            [
                (calibrate.STS_TORQUE_ENABLE, 0),
                (calibrate.STS_LOCK, 0),
                (calibrate.STS_TORQUE_ENABLE, 128),
                (calibrate.STS_TORQUE_ENABLE, 0),
                (calibrate.STS_LOCK, 1),
                (calibrate.STS_TORQUE_ENABLE, 1),
            ],
        )

    @patch.object(calibrate.time, "sleep", return_value=None)
    def test_shift_while_torque_off_aborts_before_calibration(self, _sleep):
        packet = FakePacket(positions=(1976, 2000, 2000, 2000))

        with self.assertRaisesRegex(RuntimeError, "shifted while torque"):
            calibrate.perform_calibration(
                packet,
                servo_id=19,
                speed=100,
                acceleration=5,
                release_shift_tolerance=8,
                verify_tolerance=25,
                settle_seconds=0.2,
            )

        self.assertEqual(packet.registers[calibrate.STS_OFS_L], 0)
        self.assertEqual(packet.position_writes[-1], (2000, 100, 5))


if __name__ == "__main__":
    unittest.main()
