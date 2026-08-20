#!/usr/bin/env python3
"""Cross the STS encoder wrap while manually untwisting right arm ID 19."""

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
    STS_TORQUE_ENABLE,
    sts,
)
from recover_right_upper_arm import (
    DEFAULT_BAUDRATE,
    DEFAULT_DEVICE,
    DEFAULT_SERVO_ID,
    StopKey,
    communication_error,
    print_diagnostics,
    read_byte,
    read_diagnostics,
    read_ticks,
    read_unsigned_word,
    write_ticks,
)


POSITION_MODE = 0
STEP_MODE = 3
ENCODER_TICKS = 4096


def checked_write_byte(packet, servo_id: int, address: int, value: int) -> None:
    result, error = packet.write1ByteTxRx(servo_id, address, int(value))
    if result != COMM_SUCCESS:
        raise RuntimeError(
            f"register write at address {address} failed: "
            + communication_error(packet, result, error)
        )


def checked_write_word(packet, servo_id: int, address: int, value: int) -> None:
    result, error = packet.write2ByteTxRx(servo_id, address, int(value))
    if result != COMM_SUCCESS:
        raise RuntimeError(
            f"register write at address {address} failed: "
            + communication_error(packet, result, error)
        )


def signed_encoder_delta(start: int, current: int) -> int:
    delta = (int(current) - int(start)) % ENCODER_TICKS
    if delta >= ENCODER_TICKS // 2:
        delta -= ENCODER_TICKS
    return delta


def set_torque(packet, servo_id: int, enabled: bool) -> None:
    checked_write_byte(
        packet,
        servo_id,
        STS_TORQUE_ENABLE,
        int(enabled),
    )


def write_eeprom_configuration(
    packet,
    servo_id: int,
    minimum: int,
    maximum: int,
    mode: int,
) -> None:
    checked_write_byte(packet, servo_id, STS_LOCK, 0)
    try:
        checked_write_word(
            packet,
            servo_id,
            STS_MIN_ANGLE_LIMIT_L,
            minimum,
        )
        checked_write_word(
            packet,
            servo_id,
            STS_MAX_ANGLE_LIMIT_L,
            maximum,
        )
        checked_write_byte(packet, servo_id, STS_MODE, mode)
    finally:
        checked_write_byte(packet, servo_id, STS_LOCK, 1)
    time.sleep(0.05)


def enter_step_mode(packet, args: argparse.Namespace) -> None:
    set_torque(packet, args.servo_id, False)
    write_eeprom_configuration(packet, args.servo_id, 0, 0, STEP_MODE)
    write_ticks(
        packet,
        args.servo_id,
        0,
        args.speed,
        args.acceleration,
    )
    set_torque(packet, args.servo_id, True)

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
    if (mode, minimum, maximum) != (STEP_MODE, 0, 0):
        raise RuntimeError(
            "failed to verify temporary step mode "
            f"(mode={mode}, limits={minimum}..{maximum})"
        )


def restore_position_mode(
    packet,
    args: argparse.Namespace,
    minimum: int,
    maximum: int,
    torque_was_enabled: bool,
    hold_ticks: int,
) -> None:
    set_torque(packet, args.servo_id, False)
    write_eeprom_configuration(
        packet,
        args.servo_id,
        minimum,
        maximum,
        POSITION_MODE,
    )
    hold_ticks %= ENCODER_TICKS
    write_ticks(
        packet,
        args.servo_id,
        hold_ticks,
        args.speed,
        args.acceleration,
    )
    if torque_was_enabled:
        set_torque(packet, args.servo_id, True)

    mode = read_byte(packet, args.servo_id, STS_MODE)
    if mode != POSITION_MODE:
        raise RuntimeError(f"position-mode restore verification failed: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily use STS step mode to keep untwisting right upper-arm "
            "ID 19 across the absolute encoder boundary."
        )
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--servo-id", type=int, default=DEFAULT_SERVO_ID)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--step-ticks", type=int, default=100)
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--acceleration", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--step-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--step-tolerance-ticks", type=int, default=6)
    parser.add_argument("--minimum-progress-ticks", type=int, default=4)
    parser.add_argument(
        "--maximum-travel-ticks",
        type=int,
        default=2048,
        help="maximum relative travel in one run (default: 180 degrees)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.servo_id != DEFAULT_SERVO_ID:
        raise ValueError("this untwist tool is restricted to STS servo ID 19")
    if not 20 <= args.step_ticks <= 150:
        raise ValueError("step ticks must be within 20..150")
    if not 1 <= args.minimum_progress_ticks <= args.step_ticks:
        raise ValueError("minimum progress must be within 1..step ticks")
    if not 1 <= args.step_tolerance_ticks < args.step_ticks:
        raise ValueError("step tolerance must be within 1..step ticks-1")
    if not 1 <= args.speed <= 200:
        raise ValueError("speed must be within 1..200")
    if not 1 <= args.acceleration <= 20:
        raise ValueError("acceleration must be within 1..20")
    if not 1 <= args.maximum_travel_ticks <= 2048:
        raise ValueError("maximum travel must be within 1..2048 ticks")


def main() -> int:
    args = parse_args()
    port = None
    packet = None
    original_minimum = None
    original_maximum = None
    original_torque = False
    step_mode_entered = False
    latest_ticks = 0
    total_travel = 0

    try:
        validate_args(args)
        if not sys.stdin.isatty():
            raise RuntimeError("interactive terminal required for one-key stop")

        port = PortHandler(args.device)
        if not port.openPort():
            raise RuntimeError(f"failed to open {args.device}")
        if not port.setBaudRate(args.baudrate):
            raise RuntimeError(
                f"failed to set {args.device} to {args.baudrate} baud"
            )
        packet = sts(port)

        latest_ticks = read_ticks(packet, args.servo_id)
        original_minimum = read_unsigned_word(
            packet,
            args.servo_id,
            STS_MIN_ANGLE_LIMIT_L,
        )
        original_maximum = read_unsigned_word(
            packet,
            args.servo_id,
            STS_MAX_ANGLE_LIMIT_L,
        )
        original_mode = read_byte(packet, args.servo_id, STS_MODE)
        original_torque = bool(
            read_byte(packet, args.servo_id, STS_TORQUE_ENABLE)
        )

        if original_mode != POSITION_MODE:
            raise RuntimeError(
                f"ID 19 is already in unexpected mode {original_mode}; "
                "no configuration was changed"
            )

        print(
            f"STS ID {args.servo_id} starts at raw {latest_ticks}. "
            "Positive relative steps will continue the confirmed untwisting "
            "direction across the encoder boundary."
        )
        print(
            f"Temporary travel limit: {args.maximum_travel_ticks} ticks "
            f"({args.maximum_travel_ticks * 360 / ENCODER_TICKS:.1f} degrees)."
        )
        print_diagnostics(read_diagnostics(packet, args.servo_id))
        print()
        print("Only right upper-arm STS servo ID 19 will be commanded.")
        print("Press any key or Ctrl+C at any time to stop and hold.")
        confirmation = input("Type UNTWIST and press Enter to begin: ")
        if confirmation != "UNTWIST":
            print("Cancelled; no configuration or movement command was sent.")
            return 2

        step_mode_entered = True
        enter_step_mode(packet, args)
        print("Temporary step mode active. Press any key to stop.", flush=True)

        operator_stopped = False
        with StopKey() as stop_key:
            while total_travel < args.maximum_travel_ticks:
                if stop_key.read() is not None:
                    operator_stopped = True
                    break

                requested_step = min(
                    args.step_ticks,
                    args.maximum_travel_ticks - total_travel,
                )
                step_start = read_ticks(packet, args.servo_id)
                write_ticks(
                    packet,
                    args.servo_id,
                    requested_step,
                    args.speed,
                    args.acceleration,
                )
                deadline = time.monotonic() + args.step_timeout_seconds
                step_progress = 0

                while True:
                    if stop_key.read() is not None:
                        operator_stopped = True
                        break
                    time.sleep(args.poll_seconds)
                    latest_ticks = read_ticks(packet, args.servo_id)
                    delta = signed_encoder_delta(step_start, latest_ticks)
                    if delta < -2:
                        raise RuntimeError(
                            "step mode moved in the wrong encoder direction; "
                            f"start={step_start}, current={latest_ticks}"
                        )
                    step_progress = max(step_progress, delta)
                    moving = read_byte(packet, args.servo_id, STS_MOVING)
                    if step_progress >= requested_step - args.step_tolerance_ticks:
                        break
                    if not moving and step_progress >= args.minimum_progress_ticks:
                        break
                    if time.monotonic() >= deadline:
                        print("Servo status at stalled relative step:")
                        print_diagnostics(
                            read_diagnostics(packet, args.servo_id)
                        )
                        raise RuntimeError(
                            f"relative step stalled after {step_progress}/"
                            f"{requested_step} ticks"
                        )

                if operator_stopped:
                    break
                if step_progress < args.minimum_progress_ticks:
                    raise RuntimeError(
                        f"relative step made only {step_progress} ticks progress"
                    )
                total_travel += step_progress
                print(
                    f"raw={latest_ticks % ENCODER_TICKS:4d}, "
                    f"untwist travel={total_travel:4d} ticks "
                    f"({total_travel * 360 / ENCODER_TICKS:.1f} deg)",
                    flush=True,
                )

        print(
            "Operator stop received."
            if operator_stopped
            else "Per-run travel limit reached."
        )
        return 0
    except KeyboardInterrupt:
        print("\nOperator stop received.")
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        if packet is not None and step_mode_entered:
            try:
                latest_ticks = read_ticks(packet, args.servo_id, attempts=1)
            except Exception:
                pass
            try:
                restore_position_mode(
                    packet,
                    args,
                    original_minimum,
                    original_maximum,
                    original_torque,
                    latest_ticks,
                )
                restored_ticks = read_ticks(packet, args.servo_id)
                print(
                    f"Position mode restored; holding raw "
                    f"{restored_ticks % ENCODER_TICKS}."
                )
            except Exception as restore_error:
                try:
                    set_torque(packet, args.servo_id, False)
                except Exception:
                    pass
                print(
                    "CRITICAL: could not restore position mode; torque was "
                    f"disabled: {restore_error}",
                    file=sys.stderr,
                )
        if port is not None:
            try:
                port.closePort()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
