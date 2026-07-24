import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "pib_motors"))

from pib_motors.bricklet_pin import (  # noqa: E402
    COMM_RX_TIMEOUT,
    COMM_SUCCESS,
    _STSBrickletPin,
)


class FakePacketHandler:
    def __init__(self, torque_results=None, position_results=None):
        self.torque_results = list(torque_results or [(COMM_SUCCESS, 0)])
        self.position_results = list(position_results or [(COMM_SUCCESS, 0)])
        self.position_writes = 0

    def write1ByteTxRx(self, _pin, _address, _value):
        if len(self.torque_results) > 1:
            return self.torque_results.pop(0)
        return self.torque_results[0]

    def WritePosEx(self, _pin, _ticks, _speed, _acc):
        self.position_writes += 1
        if len(self.position_results) > 1:
            return self.position_results.pop(0)
        return self.position_results[0]

    def ReadPosSpeed(self, _pin):
        return 2000, 0, COMM_SUCCESS, 0

    def read1ByteTxRx(self, _pin, _address):
        return 0, COMM_RX_TIMEOUT, 0

    def getTxRxResult(self, result):
        return f"result {result}"

    def getRxPacketError(self, error):
        return f"error {error}" if error else ""


def make_pin(packet_handler):
    pin = object.__new__(_STSBrickletPin)
    pin.pin = 40
    pin.uid = "/dev/test"
    pin.invert = False
    pin._connected = True
    pin._baudrate = 1_000_000
    pin._settings = {"turnedOn": True}
    pin._ph = object()
    pin._pk = packet_handler
    pin._zero_tick = 2000
    pin._last_position = 1234
    pin._last_comm_result = COMM_SUCCESS
    pin._last_servo_error = 0
    pin._last_comm_exception = None
    pin._failed_transactions = 0

    def reconnect():
        pin._connected = True
        return True

    pin.check_connection = reconnect
    return pin


class STSRecoveryTest(unittest.TestCase):
    def test_position_command_retries_after_torque_timeout(self):
        packet_handler = FakePacketHandler(
            torque_results=[(COMM_RX_TIMEOUT, 0), (COMM_SUCCESS, 0)]
        )
        pin = make_pin(packet_handler)

        self.assertTrue(pin.set_position(1000))
        self.assertEqual(packet_handler.position_writes, 1)

    def test_position_command_retries_when_initially_disconnected(self):
        packet_handler = FakePacketHandler()
        pin = make_pin(packet_handler)
        pin._connected = False
        connection_results = iter((False, True))

        def reconnect():
            connected = next(connection_results)
            pin._connected = connected
            if connected:
                pin._record_comm_success()
            else:
                pin._record_comm_failure(COMM_RX_TIMEOUT)
            return connected

        pin.check_connection = reconnect

        self.assertTrue(pin.set_position(1000))
        self.assertEqual(packet_handler.position_writes, 1)

    def test_failed_position_read_uses_last_valid_value(self):
        packet_handler = FakePacketHandler()
        packet_handler.ReadPosSpeed = lambda _pin: (0, 0, COMM_RX_TIMEOUT, 0)
        pin = make_pin(packet_handler)

        self.assertEqual(pin.get_position(), 1234)


if __name__ == "__main__":
    unittest.main()
