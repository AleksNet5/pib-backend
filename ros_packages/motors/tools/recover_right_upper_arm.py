#!/usr/bin/env python3
"""Slow, operator-supervised recovery for right upper-arm STS servo ID 19."""

import argparse
import select
import sys
import termios
import time
import tty

from motors.STservo_sdk import (
    COMM_SUCCESS,
    PortHandler,
    STS_ACC,
    STS_CCW_DEAD,
    STS_CW_DEAD,
    STS_GOAL_POSITION_L,
    STS_GOAL_SPEED_L,
    STS_MAX_ANGLE_LIMIT_L,
    STS_MIN_ANGLE_LIMIT_L,
    STS_MODE,
    STS_MOVING,
    STS_PRESENT_CURRENT_L,
    STS_PRESENT_LOAD_L,
    STS_PRESENT_TEMPERATURE,
    STS_PRESENT_VOLTAGE,
    STS_TORQUE_ENABLE,
    sts,
)


DEFAULT_DEVICE = "/dev/ttyMotor1"
DEFAULT_SERVO_ID = 19
DEFAULT_TARGET_TICKS = 3897
DEFAULT_BAUDRATE = 1_000_000

STS_MAX_TORQUE_L = 16
STS_PROTECTION_SWITCH = 19
STS_PUNCH_L = 24
STS_PROTECTION_CURRENT_L = 28
STS_PROTECTION_TORQUE = 34
STS_PROTECTION_TIME = 35
STS_OVERLOAD_TORQUE = 36
STS_TORQUE_LIMIT_L = 48
STS_HARDWARE_ERROR_STATUS = 65


def estimated_centidegrees(ticks: int) -> int:
    if ticks < 2000:
        return round(((ticks - 1000) * (9000 / 1000)) - 9000)
    return round((ticks - 2000) * (9000 / 1050))


def communication_error(packet, result: int, error: int) -> str:
    result_text = packet.getTxRxResult(result)
    error_text = packet.getRxPacketError(error) if error else ""
    details = f"{result_text} (result={result}, error={error})"
    if error_text:
        details += f": {error_text}"
    return details


def read_ticks(packet, servo_id: int, attempts: int = 3) -> int:
    last_result = None
    last_error = 0
    for _attempt in range(attempts):
        ticks, _speed, result, error = packet.ReadPosSpeed(servo_id)
        if result == COMM_SUCCESS:
            return int(ticks)
        last_result = result
        last_error = error
        time.sleep(0.1)
    raise RuntimeError(
        "position read failed: "
        + communication_error(packet, last_result, last_error)
    )


def read_unsigned_word(packet, servo_id: int, address: int) -> int:
    value, result, error = packet.read2ByteTxRx(servo_id, address)
    if result != COMM_SUCCESS:
        raise RuntimeError(
            f"register read at address {address} failed: "
            + communication_error(packet, result, error)
        )
    return int(value)


def read_signed_word(packet, servo_id: int, address: int) -> int:
    value = read_unsigned_word(packet, servo_id, address)
    return int(packet.sts_tohost(value, 15))


def read_byte(packet, servo_id: int, address: int) -> int:
    value, result, error = packet.read1ByteTxRx(servo_id, address)
    if result != COMM_SUCCESS:
        raise RuntimeError(
            f"register read at address {address} failed: "
            + communication_error(packet, result, error)
        )
    return int(value)


def read_diagnostics(packet, servo_id: int) -> dict[str, int | float]:
    current = read_signed_word(packet, servo_id, STS_PRESENT_CURRENT_L)
    voltage = read_byte(packet, servo_id, STS_PRESENT_VOLTAGE)
    raw_load = read_unsigned_word(packet, servo_id, STS_PRESENT_LOAD_L)
    load = raw_load & 0x3FF
    if raw_load & 0x400:
        load *= -1
    return {
        "torque": read_byte(packet, servo_id, STS_TORQUE_ENABLE),
        "mode": read_byte(packet, servo_id, STS_MODE),
        "goal": read_unsigned_word(packet, servo_id, STS_GOAL_POSITION_L),
        "goal_speed": read_unsigned_word(
            packet,
            servo_id,
            STS_GOAL_SPEED_L,
        ),
        "acceleration": read_byte(packet, servo_id, STS_ACC),
        "moving": read_byte(packet, servo_id, STS_MOVING),
        "load": load,
        "current": current,
        "current_ma_estimate": current * 6.5,
        "voltage_v": voltage / 10.0,
        "temperature_c": read_byte(
            packet,
            servo_id,
            STS_PRESENT_TEMPERATURE,
        ),
        "hardware_error": read_byte(
            packet,
            servo_id,
            STS_HARDWARE_ERROR_STATUS,
        ),
        "torque_limit": read_unsigned_word(
            packet,
            servo_id,
            STS_TORQUE_LIMIT_L,
        ),
        "maximum_torque": read_unsigned_word(
            packet,
            servo_id,
            STS_MAX_TORQUE_L,
        ),
        "startup_force": read_unsigned_word(
            packet,
            servo_id,
            STS_PUNCH_L,
        ),
        "cw_deadband": read_byte(packet, servo_id, STS_CW_DEAD),
        "ccw_deadband": read_byte(packet, servo_id, STS_CCW_DEAD),
        "protection_switch": read_byte(
            packet,
            servo_id,
            STS_PROTECTION_SWITCH,
        ),
        "protection_current": read_unsigned_word(
            packet,
            servo_id,
            STS_PROTECTION_CURRENT_L,
        ),
        "protection_torque": read_byte(
            packet,
            servo_id,
            STS_PROTECTION_TORQUE,
        ),
        "protection_time": read_byte(
            packet,
            servo_id,
            STS_PROTECTION_TIME,
        ),
        "overload_torque": read_byte(
            packet,
            servo_id,
            STS_OVERLOAD_TORQUE,
        ),
    }


def print_diagnostics(diagnostics: dict[str, int | float]) -> None:
    print(
        "Servo status: "
        f"torque={diagnostics['torque']}, mode={diagnostics['mode']}, "
        f"goal={diagnostics['goal']}, speed={diagnostics['goal_speed']}, "
        f"acceleration={diagnostics['acceleration']}, "
        f"moving={diagnostics['moving']}, "
        f"load={diagnostics['load']}, current={diagnostics['current']} "
        f"(~{diagnostics['current_ma_estimate']} mA), "
        f"voltage={diagnostics['voltage_v']:.1f} V, "
        f"temperature={diagnostics['temperature_c']} C, "
        f"error=0x{diagnostics['hardware_error']:02x}"
    )
    print(
        "Servo control: "
        f"torque_limit={diagnostics['torque_limit']}/1000, "
        f"maximum_torque={diagnostics['maximum_torque']}/1000, "
        f"startup_force={diagnostics['startup_force']}/1000, "
        f"deadband={diagnostics['cw_deadband']}/"
        f"{diagnostics['ccw_deadband']} ticks"
    )
    print(
        "Servo protection: "
        f"switch=0x{diagnostics['protection_switch']:02x}, "
        f"current={diagnostics['protection_current']} raw, "
        f"protection_torque={diagnostics['protection_torque']}%, "
        f"time={diagnostics['protection_time']} raw, "
        f"overload_torque={diagnostics['overload_torque']}%"
    )


def write_ticks(
    packet,
    servo_id: int,
    ticks: int,
    speed: int,
    acceleration: int,
) -> None:
    result, error = packet.WritePosEx(
        servo_id,
        int(ticks),
        int(speed),
        int(acceleration),
    )
    if result != COMM_SUCCESS:
        raise RuntimeError(
            f"position write to tick {ticks} failed: "
            + communication_error(packet, result, error)
        )


def enable_torque(packet, servo_id: int) -> None:
    result, error = packet.write1ByteTxRx(
        servo_id,
        STS_TORQUE_ENABLE,
        1,
    )
    if result != COMM_SUCCESS:
        raise RuntimeError(
            "torque enable failed: "
            + communication_error(packet, result, error)
        )


class StopKey:
    def __init__(self) -> None:
        self._settings = None

    def __enter__(self):
        self._settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        termios.tcsetattr(
            sys.stdin.fileno(),
            termios.TCSADRAIN,
            self._settings,
        )

    @staticmethod
    def read() -> str | None:
        readable, _writable, _errors = select.select(
            [sys.stdin], [], [], 0
        )
        if not readable:
            return None
        return sys.stdin.read(1)


def hold_current_position(
    packet,
    servo_id: int,
    fallback_ticks: int,
    speed: int,
    acceleration: int,
) -> int:
    try:
        current = read_ticks(packet, servo_id, attempts=1)
    except Exception:
        current = fallback_ticks
    write_ticks(packet, servo_id, current, speed, acceleration)
    return current


def recover(packet, args: argparse.Namespace, starting_ticks: int) -> int:
    current = starting_ticks
    last_observed = current

    # Prime the present position before torque is enabled so a servo-side stale
    # target cannot create a jump.
    write_ticks(
        packet,
        args.servo_id,
        current,
        args.speed,
        args.acceleration,
    )
    enable_torque(packet, args.servo_id)

    print(
        "Recovery started. Press any single key or Ctrl+C to stop and hold.",
        flush=True,
    )

    direction_confirmed = (
        starting_ticks > args.direction_check_until_ticks
    )
    if direction_confirmed:
        print(
            "Increasing direction was confirmed by the previous recovery; "
            "continuing with the controlled target lead. Press any key to stop.",
            flush=True,
        )
    try:
        with StopKey() as stop_key:
            while current < args.target_ticks - args.tolerance_ticks:
                if stop_key.read() is not None:
                    held = hold_current_position(
                        packet,
                        args.servo_id,
                        last_observed,
                        args.speed,
                        args.acceleration,
                    )
                    print(f"Stopped by operator; holding raw tick {held}.")
                    return 130

                next_target = min(
                    current
                    + (
                        args.direction_check_ticks
                        if not direction_confirmed
                        else args.step_ticks
                    ),
                    args.target_ticks,
                )
                required_progress = min(
                    args.minimum_progress_ticks,
                    next_target - current,
                )
                step_start = current
                write_ticks(
                    packet,
                    args.servo_id,
                    next_target,
                    args.speed,
                    args.acceleration,
                )

                deadline = time.monotonic() + args.step_timeout_seconds
                while True:
                    if stop_key.read() is not None:
                        held = hold_current_position(
                            packet,
                            args.servo_id,
                            last_observed,
                            args.speed,
                            args.acceleration,
                        )
                        print(
                            f"Stopped by operator; holding raw tick {held}."
                        )
                        return 130

                    time.sleep(args.poll_seconds)
                    last_observed = read_ticks(packet, args.servo_id)
                    if (
                        last_observed - step_start >= required_progress
                        or args.target_ticks - last_observed
                        <= args.tolerance_ticks
                    ):
                        current = last_observed
                        print(
                            f"raw={current:4d}, estimated="
                            f"{estimated_centidegrees(current) / 100:+.2f} deg",
                            flush=True,
                        )
                        break
                    if time.monotonic() >= deadline:
                        print("Servo status at stalled target:")
                        print_diagnostics(
                            read_diagnostics(packet, args.servo_id)
                        )
                        held = hold_current_position(
                            packet,
                            args.servo_id,
                            last_observed,
                            args.speed,
                            args.acceleration,
                        )
                        raise RuntimeError(
                            f"servo did not reach step target {next_target}; "
                            f"holding raw tick {held}"
                        )

                if not direction_confirmed:
                    print(
                        "Direction check complete after one small step. "
                        "Press c to continue only if the joint started "
                        "untwisting; press any other key to stop and hold.",
                        flush=True,
                    )
                    while True:
                        key = stop_key.read()
                        if key is None:
                            time.sleep(0.05)
                            continue
                        if key.lower() == "c":
                            direction_confirmed = True
                            print(
                                "Continuing recovery. Press any key to stop.",
                                flush=True,
                            )
                            break
                        held = hold_current_position(
                            packet,
                            args.servo_id,
                            last_observed,
                            args.speed,
                            args.acceleration,
                        )
                        print(
                            f"Stopped after direction check; holding raw "
                            f"tick {held}."
                        )
                        return 130
    except KeyboardInterrupt:
        held = hold_current_position(
            packet,
            args.servo_id,
            last_observed,
            args.speed,
            args.acceleration,
        )
        print(f"\nStopped by operator; holding raw tick {held}.")
        return 130
    except Exception:
        try:
            held = hold_current_position(
                packet,
                args.servo_id,
                last_observed,
                args.speed,
                args.acceleration,
            )
            print(f"Recovery aborted; holding raw tick {held}.")
        except Exception as hold_error:
            print(
                f"WARNING: could not send a hold command: {hold_error}",
                file=sys.stderr,
            )
        raise

    final_ticks = hold_current_position(
        packet,
        args.servo_id,
        last_observed,
        args.speed,
        args.acceleration,
    )
    print(
        f"Recovery target reached; holding raw tick {final_ticks} "
        f"({estimated_centidegrees(final_ticks) / 100:+.2f} deg estimated)."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slowly reverse the logged right upper-arm ID 19 movement. "
            "The normal ros-motors container must not be running."
        )
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--servo-id", type=int, default=DEFAULT_SERVO_ID)
    parser.add_argument(
        "--target-ticks",
        type=int,
        default=DEFAULT_TARGET_TICKS,
        help="logged raw position before the unwanted movement",
    )
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--direction-check-ticks", type=int, default=20)
    parser.add_argument("--direction-check-until-ticks", type=int, default=2100)
    parser.add_argument("--step-ticks", type=int, default=100)
    parser.add_argument("--minimum-progress-ticks", type=int, default=4)
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--acceleration", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--step-timeout-seconds", type=float, default=6.0)
    parser.add_argument("--tolerance-ticks", type=int, default=5)
    parser.add_argument("--minimum-start-ticks", type=int, default=1950)
    parser.add_argument("--maximum-start-ticks", type=int, default=3897)
    parser.add_argument("--maximum-travel-ticks", type=int, default=2000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and validate feedback without enabling torque or moving",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.servo_id != DEFAULT_SERVO_ID:
        raise ValueError("this recovery tool is restricted to STS servo ID 19")
    if not 0 <= args.target_ticks <= 4095:
        raise ValueError("target ticks must be within 0..4095")
    if args.direction_check_ticks <= 0 or args.direction_check_ticks > 20:
        raise ValueError("direction check ticks must be within 1..20")
    if not 0 <= args.direction_check_until_ticks <= 4095:
        raise ValueError("direction check threshold must be within 0..4095")
    if args.step_ticks <= 0 or args.step_ticks > 150:
        raise ValueError("step ticks must be within 1..150")
    if not 1 <= args.minimum_progress_ticks <= args.step_ticks:
        raise ValueError(
            "minimum progress ticks must be within 1..step ticks"
        )
    if args.speed <= 0 or args.speed > 200:
        raise ValueError("speed must be within 1..200")
    if args.acceleration <= 0 or args.acceleration > 20:
        raise ValueError("acceleration must be within 1..20")
    if args.maximum_travel_ticks <= 0:
        raise ValueError("maximum travel must be positive")
    if args.maximum_start_ticks < args.minimum_start_ticks:
        raise ValueError("maximum start tick must not be below minimum start tick")


def main() -> int:
    args = parse_args()
    port = None
    try:
        validate_args(args)
        if not args.dry_run and not sys.stdin.isatty():
            raise RuntimeError("interactive terminal required for one-key stop")

        port = PortHandler(args.device)
        if not port.openPort():
            raise RuntimeError(f"failed to open {args.device}")
        if not port.setBaudRate(args.baudrate):
            raise RuntimeError(
                f"failed to set {args.device} to {args.baudrate} baud"
            )
        packet = sts(port)
        current = read_ticks(packet, args.servo_id)
        hardware_minimum = read_unsigned_word(
            packet,
            args.servo_id,
            STS_MIN_ANGLE_LIMIT_L,
        )
        hardware_maximum = read_unsigned_word(
            packet,
            args.servo_id,
            STS_MAX_ANGLE_LIMIT_L,
        )
        diagnostics = read_diagnostics(packet, args.servo_id)
        travel = args.target_ticks - current

        print(
            f"STS ID {args.servo_id} on {args.device}: raw={current}, "
            f"estimated={estimated_centidegrees(current) / 100:+.2f} deg"
        )
        print(
            f"Recovery direction: increasing raw ticks to "
            f"{args.target_ticks}; travel={travel} ticks"
        )
        print(
            f"Servo hardware angle limits: "
            f"{hardware_minimum}..{hardware_maximum} raw ticks"
        )
        print_diagnostics(diagnostics)

        if not hardware_minimum <= current <= hardware_maximum:
            raise RuntimeError(
                f"current tick {current} is outside servo hardware limits "
                f"{hardware_minimum}..{hardware_maximum}"
            )
        if not hardware_minimum <= args.target_ticks <= hardware_maximum:
            raise RuntimeError(
                f"recovery target {args.target_ticks} is outside servo "
                f"hardware limits {hardware_minimum}..{hardware_maximum}; "
                "no movement was sent"
            )

        if current < args.minimum_start_ticks:
            raise RuntimeError(
                f"current tick {current} is below guarded start "
                f"{args.minimum_start_ticks}; refusing to guess the direction"
            )
        if current > args.maximum_start_ticks:
            raise RuntimeError(
                f"current tick {current} is above guarded start "
                f"{args.maximum_start_ticks}; refusing to leave the confirmed "
                "recovery corridor"
            )
        if travel < 0:
            raise RuntimeError(
                "current position is already beyond the recovery target; "
                "no movement was sent"
            )
        if travel > args.maximum_travel_ticks:
            raise RuntimeError(
                f"required travel {travel} exceeds guarded maximum "
                f"{args.maximum_travel_ticks}; no movement was sent"
            )
        if travel <= args.tolerance_ticks:
            print("Motor is already at the recovery target; no movement needed.")
            return 0
        if args.dry_run:
            print("Dry run complete; torque was not enabled and no write was sent.")
            return 0

        print()
        print("Only right upper-arm STS servo ID 19 will be commanded.")
        print("Keep clear of the arm and keep one hand on the keyboard.")
        confirmation = input("Type RECOVER and press Enter to begin: ")
        if confirmation != "RECOVER":
            print("Cancelled; no movement command was sent.")
            return 2

        return recover(packet, args, current)
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
