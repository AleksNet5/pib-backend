from collections import defaultdict

from motors.STservo_sdk import (
    COMM_SUCCESS,
    STS_MAX_ANGLE_LIMIT_L,
    STS_MIN_ANGLE_LIMIT_L,
    STS_MODE,
    STS_MOVING,
    STS_TORQUE_ENABLE,
)
from motors.right_upper_arm_recovery import (
    POSITION_MODE,
    STEP_MODE,
    RightUpperArmRecovery,
    signed_encoder_delta,
    unwrapped_tick_to_centidegrees,
)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakePacket:
    def __init__(self, raw=2048, torque=1):
        self.raw = raw
        self.registers = defaultdict(int)
        self.registers[STS_MODE] = POSITION_MODE
        self.registers[STS_TORQUE_ENABLE] = torque
        self.registers[STS_MOVING] = 0
        self.words = {
            STS_MIN_ANGLE_LIMIT_L: 0,
            STS_MAX_ANGLE_LIMIT_L: 4095,
        }
        self.position_writes = []
        self.failed_position_reads = 0

    def ReadPosSpeed(self, _servo_id):
        if self.failed_position_reads:
            self.failed_position_reads -= 1
            return 0, 0, -1, 0
        return self.raw, 0, COMM_SUCCESS, 0

    def read1ByteTxRx(self, _servo_id, address):
        return self.registers[address], COMM_SUCCESS, 0

    def read2ByteTxRx(self, _servo_id, address):
        return self.words[address], COMM_SUCCESS, 0

    def write1ByteTxRx(self, _servo_id, address, value):
        self.registers[address] = int(value)
        return COMM_SUCCESS, 0

    def write2ByteTxRx(self, _servo_id, address, value):
        self.words[address] = int(value)
        return COMM_SUCCESS, 0

    def WritePosEx(self, _servo_id, position, _speed, _acceleration):
        position = int(position)
        self.position_writes.append((self.registers[STS_MODE], position))
        if self.registers[STS_MODE] == STEP_MODE:
            self.raw = (self.raw + position) % 4096
        else:
            self.raw = position % 4096
        return COMM_SUCCESS, 0


class FakePin:
    pin = 19
    uid = "/dev/ttyMotor1"

    def __init__(self, packet):
        self._pk = packet
        self._last_target_ticks = None
        self._has_commanded_position = False
        self._restore_required = False
        self._last_maintenance_received_reply = False
        self.positions = []

    def check_connection(self):
        return self._pk is not None

    def _record_position(self, raw):
        self.positions.append(raw)

    def _record_comm_success(self):
        pass

    def _record_comm_failure(self, _result, _error=0):
        pass


def make_recovery(raw=2048, torque=1, **kwargs):
    packet = FakePacket(raw, torque)
    pin = FakePin(packet)
    recovery = RightUpperArmRecovery(pin, FakeLogger(), **kwargs)
    return recovery, packet, pin


def test_signed_encoder_delta_tracks_both_wrap_directions():
    assert signed_encoder_delta(4090, 10) == 16
    assert signed_encoder_delta(10, 4090) == -16


def test_unwrapped_ticks_map_to_id19_logical_positions():
    assert unwrapped_tick_to_centidegrees(2048) == 0
    assert unwrapped_tick_to_centidegrees(1048) == -9000
    assert unwrapped_tick_to_centidegrees(3098) == 9000


def test_unwrapped_position_tracks_decreasing_wrap():
    recovery, _packet, _pin = make_recovery(raw=10)

    recovery._record_sample(10, True, 0.0, infer_decreasing_wrap=False)
    delta, direction, ambiguous = recovery._record_sample(
        4090,
        False,
        0.1,
        infer_decreasing_wrap=False,
    )

    assert delta == -16
    assert direction == "decreasing"
    assert ambiguous is False
    assert recovery.unwrapped_position == -6


def test_startup_with_torque_off_holds_current_position_without_zero_move():
    recovery, packet, _pin = make_recovery(raw=1700, torque=0)

    recovery.tick(now=0.0)

    assert recovery.status.state == "ready"
    assert recovery.motion_locked is False
    assert packet.position_writes == [(POSITION_MODE, 1700)]
    assert packet.registers[STS_TORQUE_ENABLE] == 1


def test_connection_loss_locks_motion_and_unchanged_feedback_unlocks_it():
    recovery, packet, _pin = make_recovery()
    recovery.tick(now=0.0)
    packet.failed_position_reads = 1

    recovery.tick(now=0.1)

    assert recovery.status.state == "connection_lost"
    assert recovery.motion_locked is True

    recovery.tick(now=0.2)

    assert recovery.status.state == "ready"
    assert recovery.motion_locked is False


def test_torque_on_motion_during_short_connection_gap_is_not_recovered():
    recovery, packet, _pin = make_recovery()
    recovery.tick(now=0.0)
    packet.failed_position_reads = 1
    recovery.tick(now=0.1)
    packet.raw = 1800

    recovery.tick(now=0.2)

    assert recovery.status.state == "ready"
    assert recovery.motion_locked is False
    assert packet.registers[STS_MODE] == POSITION_MODE
    assert packet.position_writes == []


def test_monitor_startup_gap_latches_fault_instead_of_guessing_history():
    recovery, packet, _pin = make_recovery()
    packet.failed_position_reads = 1

    recovery.tick(now=0.0)
    recovery.tick(now=0.1)

    assert recovery.status.state == "fault"
    assert recovery.motion_locked is True
    assert "monitoring started" in recovery.status.message


def test_unexpected_increasing_wrap_latches_fault_without_step_movement():
    recovery, packet, _pin = make_recovery(raw=3000)
    recovery.tick(now=0.0)
    # Continue from a near-boundary sample, then inject an increasing wrap.
    recovery.last_raw = 4090
    recovery.unwrapped_position = 4090
    recovery._initialized = True
    packet.raw = 10

    recovery.tick(now=0.1)

    assert recovery.status.state == "fault"
    assert all(mode != STEP_MODE for mode, _position in packet.position_writes)


def test_ambiguous_torque_off_reconnect_latches_fault():
    recovery, packet, _pin = make_recovery()
    recovery.tick(now=0.0)
    packet.failed_position_reads = 1
    recovery.tick(now=0.1)
    packet.registers[STS_TORQUE_ENABLE] = 0

    recovery.tick(now=0.2)

    assert recovery.status.state == "fault"
    assert recovery.motion_locked is True
    assert "ambiguous" in recovery.status.message.lower()


def test_confirmed_decreasing_fall_recovers_to_zero_in_positive_steps():
    recovery, packet, pin = make_recovery(
        raw=2048,
        torque=1,
        step_ticks=100,
    )
    recovery.tick(now=0.0)

    packet.raw = 900
    packet.registers[STS_TORQUE_ENABLE] = 0
    recovery.tick(now=0.1)

    assert recovery.status.state == "recovering"
    assert recovery.motion_locked is True
    assert packet.registers[STS_MODE] == STEP_MODE

    for index in range(30):
        recovery.tick(now=0.2 + index * 0.1)
        if recovery.status.state == "ready":
            break

    relative_steps = [
        position
        for mode, position in packet.position_writes
        if mode == STEP_MODE and position > 0
    ]
    assert relative_steps
    assert all(step > 0 for step in relative_steps)
    assert recovery.status.state == "ready"
    assert recovery.status.raw_position == 2048
    assert recovery.status.unwrapped_position == 2048
    assert packet.registers[STS_MODE] == POSITION_MODE
    assert packet.registers[STS_TORQUE_ENABLE] == 1
    assert pin._last_target_ticks == 2048


def test_collision_guard_can_stop_recovery_before_step_mode():
    checked_paths = []

    def reject_volume_5(start_tick, target_tick):
        checked_paths.append((start_tick, target_tick))
        return False, "wrist_right would enter Volume 5"

    recovery, packet, _pin = make_recovery(
        raw=2048,
        torque=1,
        collision_validator=reject_volume_5,
    )
    recovery.tick(now=0.0)
    packet.raw = 900
    packet.registers[STS_TORQUE_ENABLE] = 0

    recovery.tick(now=0.1)

    assert checked_paths == [(900, 2048)]
    assert recovery.status.state == "fault"
    assert recovery.motion_locked is True
    assert "Volume 5" in recovery.status.message
    assert all(mode != STEP_MODE for mode, _position in packet.position_writes)
