"""Direction-aware fault recovery for right upper-arm STS servo ID 19."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from motors.STservo_sdk import (
    COMM_SUCCESS,
    STS_LOCK,
    STS_MAX_ANGLE_LIMIT_L,
    STS_MIN_ANGLE_LIMIT_L,
    STS_MODE,
    STS_MOVING,
    STS_TORQUE_ENABLE,
)


ENCODER_TICKS = 4096
POSITION_MODE = 0
STEP_MODE = 3

CollisionValidator = Callable[[int, int], tuple[bool, str]]


def unwrapped_tick_to_centidegrees(tick: int, zero_tick: int = 2048) -> float:
    """Convert ID 19's unwrapped encoder coordinate to its ROS position."""
    delta = int(tick) - int(zero_tick)
    if delta < 0:
        return delta * (9000.0 / 1000.0)
    return delta * (9000.0 / 1050.0)


def signed_encoder_delta(previous: int, current: int) -> int:
    delta = (int(current) - int(previous)) % ENCODER_TICKS
    if delta >= ENCODER_TICKS // 2:
        delta -= ENCODER_TICKS
    return delta


@dataclass
class RecoveryStatus:
    active: bool = False
    state: str = "ready"
    motor_name: str = "upper_arm_right_rotation"
    message: str = "Right upper arm is ready."
    raw_position: int | None = None
    unwrapped_position: int | None = None
    target_position: int = 2048
    progress_percent: int = 0
    direction: str = "increasing"
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RightUpperArmRecovery:
    """Monitor one STS encoder and recover an observed decreasing fall."""

    def __init__(
        self,
        bricklet_pin,
        logger,
        *,
        zero_tick: int = 2048,
        monitor_gap_seconds: float = 0.75,
        trigger_margin_ticks: int = 100,
        step_ticks: int = 60,
        speed: int = 100,
        acceleration: int = 5,
        step_timeout_seconds: float = 5.0,
        step_tolerance_ticks: int = 6,
        minimum_step_progress_ticks: int = 3,
        maximum_recovery_ticks: int = 4600,
        reconnect_wrap_threshold_ticks: int = 200,
        collision_validator: CollisionValidator | None = None,
    ) -> None:
        if bricklet_pin.pin != 19 or bricklet_pin.uid != "/dev/ttyMotor1":
            raise ValueError(
                "right upper-arm recovery is restricted to /dev/ttyMotor1 ID 19"
            )

        self.pin = bricklet_pin
        self.logger = logger
        self.zero_tick = int(zero_tick)
        self.monitor_gap_seconds = float(monitor_gap_seconds)
        self.trigger_margin_ticks = int(trigger_margin_ticks)
        self.step_ticks = int(step_ticks)
        self.speed = int(speed)
        self.acceleration = int(acceleration)
        self.step_timeout_seconds = float(step_timeout_seconds)
        self.step_tolerance_ticks = int(step_tolerance_ticks)
        self.minimum_step_progress_ticks = int(minimum_step_progress_ticks)
        self.maximum_recovery_ticks = int(maximum_recovery_ticks)
        self.reconnect_wrap_threshold_ticks = int(
            reconnect_wrap_threshold_ticks
        )
        self.collision_validator = collision_validator

        # The configured -90 degree command is zero-1000 ticks. Recovery is
        # armed only after another margin beyond that normal command range.
        self.lower_trigger_tick = self.zero_tick - 1000 - self.trigger_margin_ticks

        self.status = RecoveryStatus(target_position=self.zero_tick)
        self.last_raw: int | None = None
        self.unwrapped_position: int | None = None
        self.last_sample_at: float | None = None
        self.last_torque_enabled: bool | None = None
        self.connection_gap = False
        self.gap_reference_unwrapped: int | None = None
        self.observed_fall_before_gap = False
        self.configuration_changed = False
        self.original_minimum = 0
        self.original_maximum = 4095
        self.step_start_unwrapped: int | None = None
        self.step_requested = 0
        self.step_deadline = 0.0
        self.recovery_start_unwrapped: int | None = None
        self.verify_deadline = 0.0
        self._outside_samples = 0
        self._initialized = False

    @property
    def motion_locked(self) -> bool:
        return self.status.active

    def stop_with_fault(self, message: str) -> None:
        """Hold the servo and expose a latched fault to the ROS owner."""
        self._latch_fault(message)

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        try:
            raw, torque_enabled, moving = self._read_sample()
        except Exception as error:
            self._handle_read_failure(error)
            return

        first_sample = not self._initialized
        reconnected = self.connection_gap
        delta, wrap_direction, ambiguous = self._record_sample(
            raw,
            torque_enabled,
            now,
            infer_decreasing_wrap=reconnected and not torque_enabled,
        )

        if first_sample:
            self._initialized = True
            if reconnected:
                self._latch_fault(
                    "Continuous right upper-arm feedback was unavailable when "
                    "monitoring started. The previous wrap direction is unknown, "
                    "so automatic movement was stopped."
                )
                return
            self._initialize_servo(raw, torque_enabled)
            return

        if self.status.state == "fault":
            if self.configuration_changed:
                self._restore_position_mode(raw, enable_torque=True)
            return

        if ambiguous:
            self._latch_fault(
                "The motor reconnected with an ambiguous encoder position. "
                "Automatic movement was stopped."
            )
            return

        if reconnected:
            self.connection_gap = False
            displacement = (
                0
                if self.gap_reference_unwrapped is None
                else self.unwrapped_position - self.gap_reference_unwrapped
            )
            self._log(
                "warning",
                "Right upper-arm feedback returned at raw %d; inferred "
                "unwrapped displacement %d ticks",
                raw,
                displacement,
            )
            if torque_enabled:
                outside_guarded_range = (
                    self.unwrapped_position is not None
                    and self.unwrapped_position < self.lower_trigger_tick
                )
                if wrap_direction is not None or outside_guarded_range:
                    self._latch_fault(
                        "Right upper-arm feedback returned outside its guarded "
                        "range after a connection gap. Automatic movement was "
                        "stopped because the wrap count is uncertain."
                    )
                    return
                self._set_ready(
                    "Connection restored with torque active; position tracking "
                    "continued."
                )
                return
            if displacement >= -2:
                self._latch_fault(
                    "The motor reconnected, but the fall direction could not "
                    "be confirmed. Automatic movement was stopped."
                )
                return
            self._begin_recovery(raw, "connection-loss fall")
            return

        if self.status.state == "recovering":
            self._advance_recovery(raw, moving, now)
            return

        if self.status.state == "verifying":
            self._verify_recovery(raw, moving, now)
            return

        if wrap_direction == "increasing":
            self._latch_fault(
                "The encoder crossed its boundary in the unexpected direction. "
                "Automatic movement was stopped."
            )
            return

        if wrap_direction == "decreasing":
            self._begin_recovery(raw, "observed decreasing encoder wrap")
            return

        if delta < -3:
            self.observed_fall_before_gap = True

        outside_expected_range = (
            self.unwrapped_position is not None
            and self.unwrapped_position < self.lower_trigger_tick
        )
        self._outside_samples = (
            self._outside_samples + 1 if outside_expected_range else 0
        )

        if not torque_enabled:
            self._set_status(
                active=True,
                state="observing",
                message=(
                    "Right upper-arm torque was lost. Tracking the falling "
                    "direction before recovery."
                ),
            )
            if self.unwrapped_position <= self.zero_tick and delta <= 0:
                self._begin_recovery(raw, "torque-loss fall")
            return

        if self._outside_samples >= 2 and delta <= 0:
            self._begin_recovery(raw, "position outside the guarded range")

    def _initialize_servo(self, raw: int, torque_enabled: bool) -> None:
        """Establish a safe baseline without guessing a prior encoder wrap."""
        try:
            mode = self._read_byte(STS_MODE)
            if mode != POSITION_MODE:
                self._latch_fault(
                    f"Right upper-arm servo started in unexpected mode {mode}. "
                    "Automatic movement was stopped."
                )
                return

            upper_trigger_tick = self.zero_tick + 1050 + self.trigger_margin_ticks
            if not self.lower_trigger_tick <= raw <= upper_trigger_tick:
                self._latch_fault(
                    "Right upper-arm feedback started outside its guarded raw "
                    "range. The previous wrap direction is unknown, so automatic "
                    "movement was stopped."
                )
                return

            if not torque_enabled:
                # At process startup there is no history from which to infer a
                # wrap. Hold the observed position instead of moving to zero.
                self._write_position(raw)
                self._set_torque(True)
                self._log(
                    "warning",
                    "Right upper-arm torque was off at monitor startup; holding "
                    "the observed raw position %d",
                    raw,
                )
            self._set_ready("Right upper-arm monitoring is active.")
        except Exception as error:
            self._latch_fault(
                f"Could not establish right upper-arm monitoring: {error}"
            )

    def _read_sample(self) -> tuple[int, bool, bool]:
        if self.pin._pk is None and not self.pin.check_connection():
            raise RuntimeError("STS packet handler is unavailable")
        packet = self.pin._pk

        raw, _speed, result, error = packet.ReadPosSpeed(self.pin.pin)
        if result != COMM_SUCCESS:
            self.pin._record_comm_failure(result, error)
            raise RuntimeError(f"position read failed (result={result}, error={error})")

        torque, result, error = packet.read1ByteTxRx(
            self.pin.pin,
            STS_TORQUE_ENABLE,
        )
        if result != COMM_SUCCESS:
            self.pin._record_comm_failure(result, error)
            raise RuntimeError(f"torque read failed (result={result}, error={error})")

        moving, result, error = packet.read1ByteTxRx(
            self.pin.pin,
            STS_MOVING,
        )
        if result != COMM_SUCCESS:
            self.pin._record_comm_failure(result, error)
            raise RuntimeError(f"moving read failed (result={result}, error={error})")

        self.pin._record_position(int(raw))
        self.pin._record_comm_success()
        self.pin._last_maintenance_received_reply = True
        return int(raw), int(torque) != 0, int(moving) != 0

    def _record_sample(
        self,
        raw: int,
        torque_enabled: bool,
        now: float,
        *,
        infer_decreasing_wrap: bool,
    ) -> tuple[int, str | None, bool]:
        delta = 0
        wrap_direction = None
        ambiguous = False

        if self.last_raw is None or self.unwrapped_position is None:
            self.unwrapped_position = raw
        else:
            raw_difference = raw - self.last_raw
            if infer_decreasing_wrap:
                if raw_difference >= self.reconnect_wrap_threshold_ticks:
                    delta = raw_difference - ENCODER_TICKS
                    wrap_direction = "decreasing"
                elif raw_difference <= -3:
                    delta = raw_difference
                elif abs(raw_difference) <= 2:
                    ambiguous = True
                else:
                    ambiguous = True
            else:
                delta = signed_encoder_delta(self.last_raw, raw)
                if raw_difference > ENCODER_TICKS // 2:
                    wrap_direction = "decreasing"
                elif raw_difference < -(ENCODER_TICKS // 2):
                    wrap_direction = "increasing"
            self.unwrapped_position += delta

        self.last_raw = raw
        self.last_sample_at = now
        self.last_torque_enabled = torque_enabled
        self.status.raw_position = raw
        self.status.unwrapped_position = self.unwrapped_position
        return delta, wrap_direction, ambiguous

    def _handle_read_failure(self, error: Exception) -> None:
        if not self.connection_gap:
            self.connection_gap = True
            self.gap_reference_unwrapped = self.unwrapped_position
            self._log(
                "error",
                "Right upper-arm feedback lost; all robot motion is locked: %s",
                error,
            )
        if self.status.state == "recovering":
            self._set_status(
                active=True,
                state="fault",
                message=(
                    "Connection was lost during recovery. Automatic movement "
                    "is stopped."
                ),
                error=str(error),
            )
            return
        self._set_status(
            active=True,
            state="connection_lost",
            message=(
                "Connection to the right upper arm was lost. Please wait while "
                "pib tracks its position."
            ),
            error=str(error),
        )

    def _begin_recovery(self, raw: int, reason: str) -> None:
        if self.unwrapped_position is None:
            self._latch_fault("No unwrapped encoder position is available.")
            return
        remaining = self.zero_tick - self.unwrapped_position
        if remaining < -self.step_tolerance_ticks:
            self._latch_fault(
                "The arm is on the unexpected side of zero; the safe recovery "
                "direction cannot be guaranteed."
            )
            return
        if remaining > self.maximum_recovery_ticks:
            self._latch_fault(
                f"Required recovery travel {remaining} ticks exceeds the "
                f"{self.maximum_recovery_ticks}-tick safety limit."
            )
            return

        if not self._collision_path_is_allowed(
            self.unwrapped_position,
            self.zero_tick,
        ):
            return

        self._set_status(
            active=True,
            state="preparing",
            message="Please wait. pib is preparing right upper-arm recovery.",
            error="",
        )
        self._log(
            "warning",
            "Starting right upper-arm recovery after %s: raw=%d, "
            "unwrapped=%d, target=%d, travel=%d",
            reason,
            raw,
            self.unwrapped_position,
            self.zero_tick,
            remaining,
        )

        if remaining <= self.step_tolerance_ticks:
            self._restore_position_mode(self.zero_tick, enable_torque=True)
            self._start_verification()
            return

        try:
            self.original_minimum = self._read_word(STS_MIN_ANGLE_LIMIT_L)
            self.original_maximum = self._read_word(STS_MAX_ANGLE_LIMIT_L)
            mode = self._read_byte(STS_MODE)
            if mode != POSITION_MODE:
                raise RuntimeError(f"servo is in unexpected mode {mode}")

            self._set_torque(False)
            self._write_eeprom_configuration(0, 0, STEP_MODE)
            self.configuration_changed = True
            self._write_position(0)
            self._set_torque(True)
            configured_mode = self._read_byte(STS_MODE)
            configured_minimum = self._read_word(STS_MIN_ANGLE_LIMIT_L)
            configured_maximum = self._read_word(STS_MAX_ANGLE_LIMIT_L)
            if (configured_mode, configured_minimum, configured_maximum) != (
                STEP_MODE,
                0,
                0,
            ):
                raise RuntimeError(
                    "temporary step-mode verification failed: "
                    f"mode={configured_mode}, limits="
                    f"{configured_minimum}..{configured_maximum}"
                )
            self.recovery_start_unwrapped = self.unwrapped_position
            self.step_start_unwrapped = None
            self.step_requested = 0
            self._set_status(
                active=True,
                state="recovering",
                message=(
                    "Please wait. pib is recovering the right upper arm in the "
                    "confirmed direction."
                ),
                progress_percent=0,
            )
        except Exception as error:
            self._latch_fault(f"Could not start automatic recovery: {error}")

    def _advance_recovery(self, raw: int, moving: bool, now: float) -> None:
        current = self.unwrapped_position
        if current is None:
            self._latch_fault("Encoder feedback disappeared during recovery.")
            return

        if self.step_start_unwrapped is not None:
            progress = current - self.step_start_unwrapped
            if progress < -2:
                self._latch_fault(
                    "The arm moved opposite to the confirmed recovery direction."
                )
                return
            step_complete = progress >= (
                self.step_requested - self.step_tolerance_ticks
            )
            stopped_with_progress = (
                not moving and progress >= self.minimum_step_progress_ticks
            )
            if step_complete or stopped_with_progress:
                self.step_start_unwrapped = None
                self.step_requested = 0
            elif now >= self.step_deadline or not moving:
                self._latch_fault(
                    "The right upper arm stalled during automatic recovery."
                )
                return
            else:
                self._update_progress()
                return

        remaining = self.zero_tick - current
        if remaining <= self.step_tolerance_ticks:
            self._restore_position_mode(self.zero_tick, enable_torque=True)
            self._start_verification(now)
            return
        if remaining < 0:
            self._latch_fault("Automatic recovery passed the calibrated zero.")
            return

        requested = min(self.step_ticks, remaining)
        if not self._collision_path_is_allowed(current, current + requested):
            return
        try:
            self._write_position(requested)
        except Exception as error:
            self._latch_fault(f"Recovery step command failed: {error}")
            return
        self.step_start_unwrapped = current
        self.step_requested = requested
        self.step_deadline = now + self.step_timeout_seconds
        self._update_progress()

    def _collision_path_is_allowed(self, start_tick: int, target_tick: int) -> bool:
        if self.collision_validator is None:
            return True
        try:
            allowed, reason = self.collision_validator(
                int(start_tick),
                int(target_tick),
            )
        except Exception as error:
            allowed = False
            reason = f"collision validation failed: {error}"
        if allowed:
            return True
        self._latch_fault(
            "Automatic right upper-arm recovery was stopped by the collision "
            f"guard: {reason}"
        )
        return False

    def _start_verification(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.verify_deadline = now + 3.0
        self._set_status(
            active=True,
            state="verifying",
            message="Please wait. pib is verifying the recovered zero position.",
            progress_percent=100,
        )

    def _verify_recovery(self, raw: int, moving: bool, now: float) -> None:
        current = self.unwrapped_position
        if current is not None and abs(current - self.zero_tick) <= 25 and not moving:
            self.pin._last_target_ticks = self.zero_tick
            self.pin._has_commanded_position = True
            self.pin._restore_required = False
            self._set_ready("Right upper-arm recovery completed.")
            self._log(
                "warning",
                "Right upper-arm recovery completed at raw=%d, unwrapped=%d",
                raw,
                current,
            )
            return
        if now >= self.verify_deadline:
            self._latch_fault(
                "The right upper arm did not settle at its calibrated zero."
            )

    def _update_progress(self) -> None:
        if self.recovery_start_unwrapped is None or self.unwrapped_position is None:
            return
        total = self.zero_tick - self.recovery_start_unwrapped
        travelled = self.unwrapped_position - self.recovery_start_unwrapped
        percent = 100 if total <= 0 else round(100 * travelled / total)
        self.status.progress_percent = max(0, min(100, percent))

    def _restore_position_mode(self, hold_ticks: int, enable_torque: bool) -> None:
        packet = self.pin._pk
        if packet is None:
            raise RuntimeError("STS packet handler is unavailable")
        self._set_torque(False)
        if self.configuration_changed or self._read_byte(STS_MODE) != POSITION_MODE:
            self._write_eeprom_configuration(
                self.original_minimum,
                self.original_maximum,
                POSITION_MODE,
            )
        self.configuration_changed = False
        self._write_position(int(hold_ticks) % ENCODER_TICKS)
        if enable_torque:
            self._set_torque(True)

    def _latch_fault(self, message: str) -> None:
        try:
            if self.last_raw is not None and self.pin._pk is not None:
                self._restore_position_mode(self.last_raw, enable_torque=True)
        except Exception as restore_error:
            message = f"{message} Position-mode restore also failed: {restore_error}"
        self._set_status(
            active=True,
            state="fault",
            message=message,
            error=message,
        )
        self._log("error", "Right upper-arm recovery fault: %s", message)

    def _set_ready(self, message: str) -> None:
        self.connection_gap = False
        self.gap_reference_unwrapped = None
        self.observed_fall_before_gap = False
        self._outside_samples = 0
        self._set_status(
            active=False,
            state="ready",
            message=message,
            progress_percent=100,
            error="",
        )

    def _set_status(self, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(self.status, key, value)
        self.status.raw_position = self.last_raw
        self.status.unwrapped_position = self.unwrapped_position

    def _read_byte(self, address: int) -> int:
        value, result, error = self.pin._pk.read1ByteTxRx(self.pin.pin, address)
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"register {address} read failed (result={result}, error={error})"
            )
        return int(value)

    def _read_word(self, address: int) -> int:
        value, result, error = self.pin._pk.read2ByteTxRx(self.pin.pin, address)
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"register {address} read failed (result={result}, error={error})"
            )
        return int(value)

    def _write_byte(self, address: int, value: int) -> None:
        result, error = self.pin._pk.write1ByteTxRx(
            self.pin.pin,
            address,
            int(value),
        )
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"register {address} write failed (result={result}, error={error})"
            )

    def _write_word(self, address: int, value: int) -> None:
        result, error = self.pin._pk.write2ByteTxRx(
            self.pin.pin,
            address,
            int(value),
        )
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"register {address} write failed (result={result}, error={error})"
            )

    def _set_torque(self, enabled: bool) -> None:
        self._write_byte(STS_TORQUE_ENABLE, int(enabled))

    def _write_position(self, position: int) -> None:
        result, error = self.pin._pk.WritePosEx(
            self.pin.pin,
            int(position),
            self.speed,
            self.acceleration,
        )
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"position write failed (result={result}, error={error})"
            )

    def _write_eeprom_configuration(
        self,
        minimum: int,
        maximum: int,
        mode: int,
    ) -> None:
        self._write_byte(STS_LOCK, 0)
        try:
            self._write_word(STS_MIN_ANGLE_LIMIT_L, minimum)
            self._write_word(STS_MAX_ANGLE_LIMIT_L, maximum)
            self._write_byte(STS_MODE, mode)
        finally:
            self._write_byte(STS_LOCK, 1)
        time.sleep(0.05)

    def _log(self, level: str, message: str, *args: Any) -> None:
        method = getattr(self.logger, level, None)
        if method is None and level == "warning":
            method = getattr(self.logger, "warn", None)
        if method is not None:
            method(message % args if args else message)
