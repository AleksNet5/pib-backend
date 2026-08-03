import sys
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "pib_motors"))

from pib_motors.bricklet_pin import (  # noqa: E402
    COMM_RX_TIMEOUT,
    COMM_SUCCESS,
    _STSBrickletPin,
)


class FakePacketHandler:
    def __init__(
        self,
        torque_results=None,
        position_results=None,
        position_reads=None,
        torque_reads=None,
    ):
        self.torque_results = list(torque_results or [(COMM_SUCCESS, 0)])
        self.position_results = list(position_results or [(COMM_SUCCESS, 0)])
        self.position_reads = list(
            position_reads or [(2000, 0, COMM_SUCCESS, 0)]
        )
        self.torque_reads = list(
            torque_reads or [(1, COMM_SUCCESS, 0)]
        )
        self.position_writes = 0
        self.position_targets = []
        self.torque_writes = []

    def write1ByteTxRx(self, _pin, _address, value):
        self.torque_writes.append(value)
        if len(self.torque_results) > 1:
            return self.torque_results.pop(0)
        return self.torque_results[0]

    def WritePosEx(self, _pin, ticks, _speed, _acc):
        self.position_writes += 1
        self.position_targets.append(ticks)
        if len(self.position_results) > 1:
            return self.position_results.pop(0)
        return self.position_results[0]

    def ReadPosSpeed(self, _pin):
        if len(self.position_reads) > 1:
            return self.position_reads.pop(0)
        return self.position_reads[0]

    def read1ByteTxRx(self, _pin, _address):
        if len(self.torque_reads) > 1:
            return self.torque_reads.pop(0)
        return self.torque_reads[0]

    def getTxRxResult(self, result):
        return f"result {result}"

    def getRxPacketError(self, error):
        return f"error {error}" if error else ""


class FakeSerialPort:
    def reset_input_buffer(self):
        pass


class FakePortHandler:
    def __init__(self):
        self.ser = FakeSerialPort()
        self.is_open = True
        self.close_calls = 0
        self.reopen_calls = 0

    def closePort(self):
        self.close_calls += 1
        self.is_open = False

    def setBaudRate(self, _baudrate):
        self.reopen_calls += 1
        self.is_open = True
        return True


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
    pin._has_valid_position = False
    pin._last_comm_result = COMM_SUCCESS
    pin._last_servo_error = 0
    pin._last_comm_exception = None
    pin._failed_transactions = 0
    pin._last_target_ticks = 2300
    pin._has_commanded_position = True
    pin._restore_required = False
    pin._last_recovery_attempt = 0.0
    pin._recovery_attempts = deque()
    pin._recovery_throttled_logged = False
    pin._last_maintenance_received_reply = False

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
        self.assertFalse(pin.has_valid_position())

    def test_watchdog_does_not_write_when_servo_is_healthy(self):
        packet_handler = FakePacketHandler()
        pin = make_pin(packet_handler)

        self.assertTrue(pin.maintain_connection())
        self.assertTrue(pin.has_valid_position())
        self.assertTrue(pin.last_maintenance_received_reply())
        self.assertEqual(packet_handler.torque_writes, [])
        self.assertEqual(packet_handler.position_targets, [])

    def test_watchdog_restores_target_after_torque_turns_off(self):
        packet_handler = FakePacketHandler(
            torque_reads=[(0, COMM_SUCCESS, 0)]
        )
        pin = make_pin(packet_handler)

        self.assertTrue(pin.maintain_connection())
        self.assertEqual(packet_handler.torque_writes, [1])
        self.assertEqual(packet_handler.position_targets, [2000, 2300])
        self.assertFalse(pin._restore_required)

    def test_watchdog_restores_target_after_repeated_connection_loss(self):
        packet_handler = FakePacketHandler(
            position_reads=[
                (0, 0, COMM_RX_TIMEOUT, 0),
                (0, 0, COMM_RX_TIMEOUT, 0),
                (2100, 0, COMM_SUCCESS, 0),
            ],
        )
        pin = make_pin(packet_handler)

        self.assertFalse(pin.maintain_connection())
        self.assertFalse(pin.maintain_connection())
        self.assertTrue(pin.maintain_connection())
        self.assertEqual(packet_handler.torque_writes, [1])
        self.assertEqual(packet_handler.position_targets, [2300])
        self.assertFalse(pin._restore_required)

    def test_watchdog_does_not_reenable_intentionally_disabled_servo(self):
        packet_handler = FakePacketHandler(
            torque_reads=[(0, COMM_SUCCESS, 0)]
        )
        pin = make_pin(packet_handler)
        pin._settings["turnedOn"] = False

        self.assertTrue(pin.maintain_connection())
        self.assertEqual(packet_handler.torque_writes, [])
        self.assertEqual(packet_handler.position_targets, [])

    def test_watchdog_does_not_enable_torque_before_first_position_command(self):
        packet_handler = FakePacketHandler(
            torque_reads=[(0, COMM_SUCCESS, 0)]
        )
        pin = make_pin(packet_handler)
        pin._has_commanded_position = False

        self.assertTrue(pin.maintain_connection())
        self.assertEqual(packet_handler.torque_writes, [])
        self.assertEqual(packet_handler.position_targets, [])

    def test_failed_command_is_restored_when_communication_returns(self):
        packet_handler = FakePacketHandler(
            position_results=[
                (COMM_RX_TIMEOUT, 0),
                (COMM_RX_TIMEOUT, 0),
                (COMM_RX_TIMEOUT, 0),
            ]
        )
        pin = make_pin(packet_handler)

        self.assertFalse(pin.set_position(1000))
        requested_target = pin._last_target_ticks
        self.assertTrue(pin._has_commanded_position)
        self.assertTrue(pin._restore_required)

        packet_handler.position_results = [(COMM_SUCCESS, 0)]
        self.assertTrue(pin.maintain_connection())
        self.assertEqual(packet_handler.position_targets[-1], requested_target)
        self.assertFalse(pin._restore_required)

    def test_failed_probe_does_not_reopen_shared_port(self):
        packet_handler = FakePacketHandler(
            position_reads=[
                (0, 0, COMM_RX_TIMEOUT, 0),
                (0, 0, COMM_RX_TIMEOUT, 0),
            ]
        )
        pin = make_pin(packet_handler)
        port_handler = FakePortHandler()
        pin._ph = port_handler

        self.assertFalse(pin.maintain_connection())
        self.assertFalse(pin.maintain_connection())
        self.assertEqual(port_handler.reopen_calls, 0)

    def test_bus_recovery_reopens_shared_port_once(self):
        pin = make_pin(FakePacketHandler())
        pin.uid = "/dev/test-bus-recovery"
        port_handler = FakePortHandler()
        pin._ph = port_handler

        with patch(
            "pib_motors.bricklet_pin._STS_BUS_REOPEN_DELAY_SECONDS",
            0,
        ):
            self.assertTrue(pin.reopen_shared_bus())
            self.assertFalse(pin.reopen_shared_bus())
        self.assertEqual(port_handler.close_calls, 1)
        self.assertEqual(port_handler.reopen_calls, 1)


if __name__ == "__main__":
    unittest.main()
