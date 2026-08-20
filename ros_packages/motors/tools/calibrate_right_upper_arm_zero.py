#!/usr/bin/env python3
"""Persist the current physical position as zero for right upper-arm ID 19."""

import argparse
import sys
import time

from motors.STservo_sdk import (
    COMM_SUCCESS,
    PortHandler,
    STS_LOCK,
    STS_MAX_ANGLE_LIMIT_L,
    STS_MIN_ANGLE_LIMIT_L,
    STS_MODE,
    STS_MOVING,
    STS_OFS_L,
    STS_TORQUE_ENABLE,
    sts,
)
from recover_right_upper_arm import (
    DEFAULT_BAUDRATE,
    DEFAULT_DEVICE,
    DEFAULT_SERVO_ID,
    communication_error,
    print_diagnostics,
    read_byte,
    read_diagnostics,
    read_ticks,
    read_unsigned_word,
    write_ticks,
)


POSITION_MODE = 0
MIDPOINT_TICKS = 2048
ENCODER_TICKS = 4096


def checked_write_byte(packet, servo_id: int, address: int, value: int) -> None:
    result, error = packet.write1ByteTxRx(servo_id, address, int(value))
    if result != COMM_SUCCESS:
        raise RuntimeError(
            f"register write at address {address} failed: "
            + communication_error(packet, result, error)
        )


def set_torque(packet, servo_id: int, enabled: bool) -> None:
    checked_write_byte(packet, servo_id, STS_TORQUE_ENABLE, int(enabled))


def circular_tick_delta(start: int, current: int) -> int:
    delta = (int(current) - int(start)) % ENCODER_TICKS
    if delta >= ENCODER_TICKS // 2:
        delta -= ENCODER_TICKS
    return delta


def hold_current_position(
    packet,
    servo_id: int,
    speed: int,
    acceleration: int,
) -> int:
    current = read_ticks(packet, servo_id)
    set_torque(packet, servo_id, False)
    write_ticks(packet, servo_id, current, speed, acceleration)
    return current


def perform_calibration(
    packet,
    servo_id: int,
    speed: int,
    acceleration: int,
    release_shift_tolerance: int,
    verify_tolerance: int,
    settle_seconds: float,
) -> dict[str, int]:
    before = read_ticks(packet, servo_id)
    original_torque = bool(read_byte(packet, servo_id, STS_TORQUE_ENABLE))
    offset_before = read_unsigned_word(packet, servo_id, STS_OFS_L)
    eeprom_unlocked = False
    torque_disabled = False

    try:
        set_torque(packet, servo_id, False)
        torque_disabled = True
        time.sleep(settle_seconds)

        released = read_ticks(packet, servo_id)
        released_delta = circular_tick_delta(before, released)
        if abs(released_delta) > release_shift_tolerance:
            raise RuntimeError(
                "arm shifted while torque was disabled; calibration was not "
                f"written (before={before}, released={released}, "
                f"delta={released_delta})"
            )

        checked_write_byte(packet, servo_id, STS_LOCK, 0)
        eeprom_unlocked = True
        # Use the firmware command directly so this maintenance tool remains
        # correct even when an older ros-motors image contains an old SDK copy.
        checked_write_byte(packet, servo_id, STS_TORQUE_ENABLE, 128)

        time.sleep(settle_seconds)
        # CalibrationOfs writes 128 to the torque register. Explicitly return
        # it to the disabled state before setting the new no-motion hold goal.
        set_torque(packet, servo_id, False)
        calibrated = read_ticks(packet, servo_id)
        if abs(calibrated - MIDPOINT_TICKS) > verify_tolerance:
            raise RuntimeError(
                f"calibration verification failed: expected {MIDPOINT_TICKS} "
                f"+/-{verify_tolerance}, read {calibrated}"
            )

        write_ticks(
            packet,
            servo_id,
            MIDPOINT_TICKS,
            speed,
            acceleration,
        )
        checked_write_byte(packet, servo_id, STS_LOCK, 1)
        eeprom_unlocked = False

        if original_torque:
            set_torque(packet, servo_id, True)
            torque_disabled = False
            time.sleep(settle_seconds)

        final = read_ticks(packet, servo_id)
        final_delta = circular_tick_delta(calibrated, final)
        if abs(final_delta) > verify_tolerance:
            hold_current_position(packet, servo_id, speed, acceleration)
            torque_disabled = True
            if original_torque:
                set_torque(packet, servo_id, True)
                torque_disabled = False
            raise RuntimeError(
                "position changed after torque restore; current position is "
                f"being held (calibrated={calibrated}, final={final})"
            )

        offset_after = read_unsigned_word(packet, servo_id, STS_OFS_L)
        return {
            "before": before,
            "released": released,
            "calibrated": calibrated,
            "final": final,
            "offset_before": offset_before,
            "offset_after": offset_after,
            "torque": int(original_torque),
        }
    finally:
        if eeprom_unlocked:
            try:
                checked_write_byte(packet, servo_id, STS_LOCK, 1)
            except Exception:
                pass
        if torque_disabled:
            try:
                hold_current_position(packet, servo_id, speed, acceleration)
                if original_torque:
                    set_torque(packet, servo_id, True)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persist the present physical position as the 2048 midpoint for "
            "right upper-arm STS servo ID 19. No movement target is sent."
        )
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--servo-id", type=int, default=DEFAULT_SERVO_ID)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--acceleration", type=int, default=5)
    parser.add_argument("--release-shift-tolerance", type=int, default=8)
    parser.add_argument("--verify-tolerance", type=int, default=25)
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.device != DEFAULT_DEVICE or args.servo_id != DEFAULT_SERVO_ID:
        raise ValueError(
            "this calibration tool is restricted to /dev/ttyMotor1, ID 19"
        )
    if not 1 <= args.speed <= 200:
        raise ValueError("speed must be within 1..200")
    if not 1 <= args.acceleration <= 20:
        raise ValueError("acceleration must be within 1..20")
    if not 0 <= args.release_shift_tolerance <= 25:
        raise ValueError("release shift tolerance must be within 0..25 ticks")
    if not 1 <= args.verify_tolerance <= 50:
        raise ValueError("verification tolerance must be within 1..50 ticks")
    if not 0.05 <= args.settle_seconds <= 1.0:
        raise ValueError("settle time must be within 0.05..1.0 seconds")


def main() -> int:
    args = parse_args()
    port = None

    try:
        validate_args(args)
        if not sys.stdin.isatty():
            raise RuntimeError("interactive terminal required")

        port = PortHandler(args.device)
        if not port.openPort():
            raise RuntimeError(f"failed to open {args.device}")
        if not port.setBaudRate(args.baudrate):
            raise RuntimeError(
                f"failed to set {args.device} to {args.baudrate} baud"
            )
        packet = sts(port)

        mode = read_byte(packet, args.servo_id, STS_MODE)
        minimum = read_unsigned_word(
            packet,
            args.servo_id,
            STS_MIN_ANGLE_LIMIT_L,
        )
        maximum = read_unsigned_word(
            packet,
            args.servo_id,
            STS_MAX_ANGLE_LIMIT_L,
        )
        moving = read_byte(packet, args.servo_id, STS_MOVING)
        current = read_ticks(packet, args.servo_id)

        print(
            f"STS ID {args.servo_id} on {args.device}: raw={current}, "
            f"mode={mode}, limits={minimum}..{maximum}, moving={moving}"
        )
        print_diagnostics(read_diagnostics(packet, args.servo_id))

        if mode != POSITION_MODE:
            raise RuntimeError(
                f"ID 19 is in mode {mode}, not position mode 0; no change made"
            )
        if moving:
            raise RuntimeError("ID 19 reports that it is moving; no change made")

        print()
        print("Only right upper-arm STS servo ID 19 will be calibrated.")
        print("Support the arm: torque is disabled for less than one second.")
        print(
            "The shaft will not be commanded to rotate. Its present physical "
            "position will become raw tick 2048 and logical zero."
        )
        confirmation = input("Type SETZERO and press Enter to continue: ")
        if confirmation != "SETZERO":
            print("Cancelled; no register was changed.")
            return 2

        result = perform_calibration(
            packet,
            args.servo_id,
            args.speed,
            args.acceleration,
            args.release_shift_tolerance,
            args.verify_tolerance,
            args.settle_seconds,
        )
        print(
            "Calibration complete without commanded rotation: "
            f"raw {result['before']} -> {result['calibrated']} -> "
            f"{result['final']}; offset 0x{result['offset_before']:04x} -> "
            f"0x{result['offset_after']:04x}."
        )
        print("Right upper-arm logical position is now 0.00 degrees.")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        if port is not None:
            try:
                port.closePort()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
