import logging
import math
import os
import threading
import time
from collections import deque
from typing import Any, Dict, Tuple

import serial
from motors.STservo_sdk import *
# -------------------------
# STS specifics & utilities
# -------------------------

# STS tick-to-degree factor from your examples (~ 4095 ticks ? 360)
_TICKS_PER_DEG = 11.378

# Reasonable defaults if not supplied via settings
_DEFAULT_BAUD = 1_000_000
_DEFAULT_SPEED = 1000
_DEFAULT_ACCEL = 50
_STS_COMMAND_ATTEMPTS = max(1, int(os.getenv("STS_COMMAND_ATTEMPTS", "3")))
_STS_RETRY_SECONDS = max(0.0, float(os.getenv("STS_RETRY_SECONDS", "0.05")))
_STS_PORT_REOPEN_COOLDOWN_SECONDS = max(
    0.0, float(os.getenv("STS_PORT_REOPEN_COOLDOWN_SECONDS", "1.0"))
)
_STS_BUS_REOPEN_COOLDOWN_SECONDS = max(
    1.0, float(os.getenv("STS_BUS_REOPEN_COOLDOWN_SECONDS", "30.0"))
)
_STS_BUS_REOPEN_DELAY_SECONDS = max(
    0.0, float(os.getenv("STS_BUS_REOPEN_DELAY_SECONDS", "0.5"))
)
_STS_PORT_AUTO_REOPEN_ENABLED = os.getenv(
    "STS_PORT_AUTO_REOPEN_ENABLED", "false"
).strip().lower() in ("1", "true", "yes", "on")
_STS_RECOVERY_FAILURE_THRESHOLD = max(
    1, int(os.getenv("STS_RECOVERY_FAILURE_THRESHOLD", "2"))
)
_STS_RECOVERY_RETRY_SECONDS = max(
    0.0, float(os.getenv("STS_RECOVERY_RETRY_SECONDS", "2.0"))
)
_STS_RECOVERY_MAX_ATTEMPTS = max(
    1, int(os.getenv("STS_RECOVERY_MAX_ATTEMPTS", "3"))
)
_STS_RECOVERY_WINDOW_SECONDS = max(
    1.0, float(os.getenv("STS_RECOVERY_WINDOW_SECONDS", "60.0"))
)
_STS_SLOW_IDS = {18, 19, 20, 21, 38, 39, 40, 41, 50, 51}
_STS_DEFAULT_ZERO_TICK = int(os.getenv("STS_DEFAULT_ZERO_TICK", "2000"))

# Cache one serial port + packet handler per device so multiple motors on the same bus reuse it
_port_cache: Dict[str, Tuple[PortHandler, any]] = {}
_port_last_reopen: Dict[str, float] = {}
_bus_last_reopen: Dict[str, float] = {}
_port_reopen_lock = threading.Lock()
_robstride_bus_cache: Dict[str, "_RobstrideDevice"] = {}


def _csv_env(name: str, default: str) -> set[str]:
    return {value.strip() for value in os.getenv(name, default).split(",") if value.strip()}


def _is_robstride_pin(pin: int, uid: str) -> bool:
    robstride_ports = _csv_env("ROBSTRIDE_PORTS", "/dev/ttyMotor4")
    robstride_ids = {int(value) for value in _csv_env("ROBSTRIDE_MOTOR_IDS", "21,41")}
    return uid in robstride_ports or int(pin) in robstride_ids


def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _sts_zero_tick(pin: int, uid: str) -> int:
    """Return the persistent software midpoint configured for one STS servo."""
    zero_tick = _STS_DEFAULT_ZERO_TICK
    for entry in _csv_env("STS_ZERO_TICK_OVERRIDES", ""):
        try:
            endpoint, tick_text = entry.rsplit("=", 1)
            device, pin_text = endpoint.rsplit(":", 1)
            if device == uid and int(pin_text) == int(pin):
                zero_tick = int(tick_text)
                break
        except ValueError:
            logging.error(
                "Ignoring invalid STS_ZERO_TICK_OVERRIDES entry: %s",
                entry,
            )

    if not 0 <= zero_tick <= 4095:
        raise ValueError(
            f"STS zero tick for {uid} ID {pin} is outside 0..4095: "
            f"{zero_tick}"
        )
    return zero_tick


def _get_or_open_port(device: str, baud: int) -> Tuple[PortHandler, any]:
    """
    Returns (port_handler, packet_handler) for a serial device, opening it if needed.
    """
    if device in _port_cache:
        return _port_cache[device]

    ph = PortHandler(device)
    if not ph.openPort():
        raise RuntimeError(f"Failed to open STS port: {device}")
    if not ph.setBaudRate(baud):
        raise RuntimeError(f"Failed to set STS baudrate {baud} on {device}")

    pk = sts(ph)
    _port_cache[device] = (ph, pk)
    return ph, pk


def _deg_to_ticks(deg: float) -> int:
    # Clamp to valid STS range [0..4095] after conversion
    #ticks = int(round(((deg + 9000) * 4096 / 18000)))
    #return max(0, min(4095, ticks))
    if deg < 0:
        # Map from -9000..0 to 1000..2000
        ticks = int(round(1000 + (deg + 9000) * (1000 / 9000)))
    else:
        # Map from 0..9000 to 2000..3050
        ticks= int(round(2000 + deg * (1050 / 9000)))
    return max(1000, min(3000, ticks))

def _ticks_to_deg(ticks: int) -> float:
    if ticks < 2000:
        return ((float(ticks) - 1000.0) * (9000.0 / 1000.0)) - 9000.0
    return (float(ticks) - 2000.0) * (9000.0 / 1050.0)

def _deg_to_tick_delta(deg: float) -> int:
    if deg < 0:
        return int(round(deg * (1000.0 / 9000.0)))
    return int(round(deg * (1050.0 / 9000.0)))

def _tick_delta_to_deg(ticks: int) -> float:
    if ticks < 0:
        return float(ticks) * (9000.0 / 1000.0)
    return float(ticks) * (9000.0 / 1050.0)


class _STSBrickletPin:
    """
    STS-backed implementation compatible with the original BrickletPin interface.

    Mapping:
      - pin:  STS servo ID (int)
      - uid:  serial device path, e.g., '/dev/ttyMotor0'
      - invert: same semantics as before

    Methods kept: check_connection, apply_settings, get_settings, get_current,
                  is_connected, set_position, get_position
    """

    NO_CURRENT: int = -1

    def __init__(self, pin: int, uid: str, invert: bool) -> None:
        """
        pin  -> STS servo ID
        uid  -> serial device (e.g., '/dev/ttyMotor0')
        """
        self.pin: int = int(pin)             # STS ID
        self.uid: str = uid                  # serial device
        self.invert: bool = invert

        self._connected: bool | None = None
        self._baudrate: int = _DEFAULT_BAUD

        # Persist settings we can honor on STS
        self._settings: Dict[str, Any] = {
            "velocity": _DEFAULT_SPEED,
            "acceleration": _DEFAULT_ACCEL,
            "deceleration": _DEFAULT_ACCEL,  # STS has acc; decel kept for compatibility
            # The following are unsupported in STS; kept to avoid breaking callers
            "pulseWidthMin": None,
            "pulseWidthMax": None,
            "period": None,
            "turnedOn": True,
        }

        # Lazily opened/cached
        self._ph: PortHandler | None = None
        self._pk: any | None = None
        self._zero_tick: int = _sts_zero_tick(self.pin, self.uid)
        self._last_position: int = 0
        self._has_valid_position: bool = False
        self._last_comm_result: int = COMM_SUCCESS
        self._last_servo_error: int = 0
        self._last_comm_exception: str | None = None
        self._failed_transactions: int = 0
        self._last_target_ticks: int | None = None
        self._has_commanded_position: bool = False
        self._restore_required: bool = False
        self._last_recovery_attempt: float = 0.0
        self._recovery_attempts: deque[float] = deque()
        self._recovery_throttled_logged: bool = False
        self._last_maintenance_received_reply: bool = False

        # Attempt initial check/open
        self.check_connection()

    def __str__(self) -> str:
        return f"STS-PIN[ id: {self.pin}, device: {self.uid} ]"

    # ----------------
    # Connection state
    # ----------------
    def _record_comm_failure(
        self,
        result: int,
        error: int = 0,
        exception: Exception | None = None,
    ) -> None:
        self._last_comm_result = result
        self._last_servo_error = error
        self._last_comm_exception = str(exception) if exception is not None else None
        self._failed_transactions += 1
        self._connected = False
        self._has_valid_position = False
        if self._failed_transactions >= _STS_RECOVERY_FAILURE_THRESHOLD:
            self._restore_required = True

    def _record_comm_success(self) -> None:
        if self._failed_transactions:
            logging.info(
                f"STS communication recovered for {self} after "
                f"{self._failed_transactions} failed transaction(s)"
            )
        self._last_comm_result = COMM_SUCCESS
        self._last_servo_error = 0
        self._last_comm_exception = None
        self._failed_transactions = 0
        self._connected = True

    def _record_position(self, ticks: int) -> None:
        self._last_position = int(
            round(_tick_delta_to_deg(int(ticks) - self._zero_tick))
        )
        self._has_valid_position = True

    def has_valid_position(self) -> bool:
        return self._has_valid_position

    def last_maintenance_received_reply(self) -> bool:
        return self._last_maintenance_received_reply

    def _last_failure_description(self) -> str:
        result_text = ""
        servo_error_text = ""
        if self._pk is not None:
            try:
                result_text = self._pk.getTxRxResult(self._last_comm_result)
            except Exception:
                pass
            try:
                servo_error_text = self._pk.getRxPacketError(
                    self._last_servo_error
                )
            except Exception:
                pass

        details = f"res={self._last_comm_result}"
        if result_text:
            details += f" ({result_text})"
        details += f", err={self._last_servo_error}"
        if servo_error_text:
            details += f" ({servo_error_text})"
        if self._last_comm_exception:
            details += f", exception={self._last_comm_exception}"
        return details

    def _prepare_sts_retry(self, failed_attempt: int) -> None:
        if self._ph is None:
            return

        serial_port = getattr(self._ph, "ser", None)
        if serial_port is not None:
            try:
                serial_port.reset_input_buffer()
            except Exception:
                pass

        if failed_attempt < 2:
            return

        if not _STS_PORT_AUTO_REOPEN_ENABLED:
            return

        reopen = getattr(self._ph, "setBaudRate", None)
        if not callable(reopen):
            return

        now = time.monotonic()
        with _port_reopen_lock:
            last_reopen = _port_last_reopen.get(self.uid, 0.0)
            if now - last_reopen < _STS_PORT_REOPEN_COOLDOWN_SECONDS:
                return
            _port_last_reopen[self.uid] = now

        try:
            self._ph.is_using = False
            if reopen(self._baudrate):
                logging.warning(
                    f"Reopened STS serial port {self.uid} after repeated "
                    f"communication failure"
                )
            else:
                logging.error(
                    f"Failed to reopen STS serial port {self.uid} at "
                    f"{self._baudrate} baud"
                )
        except Exception as error:
            logging.error(f"Exception while reopening STS port {self.uid}: {error}")

    def reopen_shared_bus(self) -> bool:
        """Reopen one shared serial port after every configured ID stops replying."""
        if self._ph is None:
            return False

        now = time.monotonic()
        with _port_reopen_lock:
            last_reopen = _bus_last_reopen.get(self.uid, 0.0)
            if now - last_reopen < _STS_BUS_REOPEN_COOLDOWN_SECONDS:
                return False
            _bus_last_reopen[self.uid] = now

            try:
                self._ph.is_using = False
                close_port = getattr(self._ph, "closePort", None)
                if callable(close_port) and bool(
                    getattr(self._ph, "is_open", False)
                ):
                    close_port()
                if _STS_BUS_REOPEN_DELAY_SECONDS:
                    time.sleep(_STS_BUS_REOPEN_DELAY_SECONDS)
                if not self._ph.setBaudRate(self._baudrate):
                    logging.error(
                        f"Failed to reopen unresponsive STS bus {self.uid} at "
                        f"{self._baudrate} baud"
                    )
                    return False
                logging.warning(
                    f"Reopened unresponsive STS bus {self.uid} after a full "
                    "watchdog sweep received no replies"
                )
                return True
            except Exception as error:
                logging.error(
                    f"Exception while reopening unresponsive STS bus "
                    f"{self.uid}: {error}"
                )
                return False

    def check_connection(self) -> bool:
        """Check we can talk to the STS bus and read this ID once."""
        try:
            self._ph, self._pk = _get_or_open_port(self.uid, self._baudrate)
            # Probe by reading pos/speed for this ID
            ticks, _, result, error = self._pk.ReadPosSpeed(self.pin)
            if result == COMM_SUCCESS:
                if self._last_target_ticks is None:
                    self._last_target_ticks = int(ticks)
                self._record_position(ticks)
                self._record_comm_success()
            else:
                self._record_comm_failure(result, error)
        except Exception as error:
            self._record_comm_failure(COMM_TX_FAIL, exception=error)
        return bool(self._connected)

    def is_connected(self) -> bool:
        if self._connected is not True:
            return self.check_connection()
        return bool(self._connected)

    # -------------
    # Settings I/O
    # -------------
    def apply_settings(self, settings_dto: dict[str, Any]) -> bool:
        """
        Apply compatible settings to STS:
          - velocity  -> used with WritePosEx
          - acceleration -> used with WritePosEx
          - deceleration -> stored (STS doesn't separate, we reuse 'acceleration')
          - turnedOn -> stored (STS has torque enable in some models; not used here)
        Other Tinkerforge-specific fields are accepted but ignored safely.
        """
        if not self.is_connected():
            return False

        try:
            # Merge known settings; ignore unknown keys gracefully
            for key in ("velocity", "acceleration", "deceleration", "turnedOn",
                        "pulseWidthMin", "pulseWidthMax", "period"):
                if key in settings_dto:
                    self._settings[key] = settings_dto[key]

            if "turnedOn" in settings_dto:
                return self._set_torque_enabled(bool(settings_dto["turnedOn"]))
            return True
        except Exception as error:
            logging.error(f"Error while applying STS motor settings: {error}")
            return False

    def get_settings(self) -> dict[str, Any]:
        """
        Return the last applied/known settings (best-effort parity with old API).
        """
        return dict(self._settings)

    # --------
    # Telemetry
    # --------
    def get_current(self) -> int:
        try:
            sts_present_current, resC, errC = self._pk.ReadCurrent(self.pin)
            #if res != COMM_SUCCESS:
            #   return 0
            return sts_present_current*10
        except Exception:
            return _STSBrickletPin.NO_CURRENT

    def reset_zero_position(self) -> bool:
        if not self.is_connected():
            return False

        try:
            ticks, _speed, res, err = self._pk.ReadPosSpeed(self.pin)
            if res != COMM_SUCCESS:
                logging.error(f"STS zero read failed (res={res}, err={err})")
                return False
            self._zero_tick = int(ticks)
            self._last_position = 0
            self._has_valid_position = True
            self._last_target_ticks = int(ticks)
            self._restore_required = False
            logging.info(f"Set STS software zero for {self} to tick {self._zero_tick}")
            return True
        except Exception as error:
            logging.error(f"Exception during STS zero position reset: {error}")
            return False

    # -------------
    # Position I/O
    # -------------
    def _set_torque_enabled(self, enabled: bool) -> bool:
        if self._pk is None and not self.check_connection():
            return False
        result, error = self._pk.write1ByteTxRx(
            self.pin, STS_TORQUE_ENABLE, int(enabled)
        )
        if result != COMM_SUCCESS:
            logging.warning(
                f"STS torque {'enable' if enabled else 'disable'} failed for {self} "
                f"(res={result}, err={error})"
            )
            self._record_comm_failure(result, error)
            return False
        self._last_comm_result = COMM_SUCCESS
        self._last_servo_error = 0
        self._last_comm_exception = None
        self._connected = True
        return True

    def _log_failure_diagnostics(self) -> None:
        if self._pk is None:
            return

        values = {}
        for name, address in (
            ("torque", STS_TORQUE_ENABLE),
            ("voltage", STS_PRESENT_VOLTAGE),
            ("temperature", STS_PRESENT_TEMPERATURE),
        ):
            value, result, _error = self._pk.read1ByteTxRx(self.pin, address)
            if result == COMM_SUCCESS:
                values[name] = value

        if values:
            voltage = values.get("voltage")
            if voltage is not None:
                values["voltage_v"] = voltage / 10.0
                del values["voltage"]
            logging.error(f"STS diagnostics after command failure for {self}: {values}")
        else:
            logging.error(f"STS diagnostics unavailable for {self}; servo is not replying")

    def _motion_profile(self) -> tuple[int, int]:
        if self.pin in _STS_SLOW_IDS:
            return 700, 30
        return 3000, 100

    def _begin_recovery_attempt(self) -> bool:
        now = time.monotonic()
        while (
            self._recovery_attempts
            and now - self._recovery_attempts[0] > _STS_RECOVERY_WINDOW_SECONDS
        ):
            self._recovery_attempts.popleft()

        if now - self._last_recovery_attempt < _STS_RECOVERY_RETRY_SECONDS:
            return False

        if len(self._recovery_attempts) >= _STS_RECOVERY_MAX_ATTEMPTS:
            if not self._recovery_throttled_logged:
                logging.error(
                    f"STS automatic recovery throttled for {self} after "
                    f"{len(self._recovery_attempts)} attempts in "
                    f"{_STS_RECOVERY_WINDOW_SECONDS:.0f} seconds"
                )
                self._recovery_throttled_logged = True
            return False

        self._last_recovery_attempt = now
        self._recovery_attempts.append(now)
        self._recovery_throttled_logged = False
        return True

    def maintain_connection(self) -> bool:
        """Probe an enabled STS servo and restore its last successful target."""
        self._last_maintenance_received_reply = False
        if not bool(self._settings.get("turnedOn", True)):
            # An intentionally disabled servo is not evidence that its bus failed.
            self._last_maintenance_received_reply = True
            return True

        if self._pk is None:
            self.check_connection()
            if self._pk is None:
                return False

        try:
            ticks, _speed, result, error = self._pk.ReadPosSpeed(self.pin)
        except Exception as exception:
            self._record_comm_failure(COMM_TX_FAIL, exception=exception)
            self._prepare_sts_retry(self._failed_transactions)
            return False

        if result != COMM_SUCCESS:
            self._record_comm_failure(result, error)
            self._prepare_sts_retry(self._failed_transactions)
            return False

        ticks = int(ticks)
        self._last_maintenance_received_reply = True
        if self._last_target_ticks is None:
            self._last_target_ticks = ticks
        self._record_position(ticks)

        try:
            torque, result, error = self._pk.read1ByteTxRx(
                self.pin, STS_TORQUE_ENABLE
            )
        except Exception as exception:
            self._record_comm_failure(COMM_TX_FAIL, exception=exception)
            self._prepare_sts_retry(self._failed_transactions)
            return False

        if result != COMM_SUCCESS:
            self._record_comm_failure(result, error)
            self._prepare_sts_retry(self._failed_transactions)
            return False

        torque_was_disabled = int(torque) == 0
        if not self._has_commanded_position:
            self._record_comm_success()
            return True
        if torque_was_disabled:
            self._restore_required = True

        self._record_comm_success()
        if not self._restore_required:
            return True
        if not self._begin_recovery_attempt():
            return False

        speed, acceleration = self._motion_profile()
        target_ticks = self._last_target_ticks

        try:
            if torque_was_disabled:
                # Prime the current position before enabling torque so a stale
                # servo-side target cannot cause a jump.
                result, error = self._pk.WritePosEx(
                    self.pin, ticks, speed, acceleration
                )
                if result != COMM_SUCCESS:
                    self._record_comm_failure(result, error)
                    return False

            if not self._set_torque_enabled(True):
                return False

            result, error = self._pk.WritePosEx(
                self.pin, target_ticks, speed, acceleration
            )
            if result != COMM_SUCCESS:
                self._record_comm_failure(result, error)
                self._log_failure_diagnostics()
                return False
        except Exception as exception:
            self._record_comm_failure(COMM_TX_FAIL, exception=exception)
            logging.error(f"STS automatic recovery failed for {self}: {exception}")
            return False

        self._restore_required = False
        self._record_comm_success()
        logging.warning(
            f"STS automatic recovery restored {self} to target tick "
            f"{target_ticks} after "
            f"{'torque-off' if torque_was_disabled else 'communication loss'}"
        )
        return True

    def set_position(self, position: int) -> bool:
        """
        Set target position in DEGREES (kept consistent with your examples).
        If your upstream publishes a different unit, adjust mapping here.
        """
        deg = float(position)
        if self.invert:
            deg *= -1.0

        raw_ticks = self._zero_tick + _deg_to_tick_delta(deg)
        ticks = max(0, min(4095, raw_ticks))
        if ticks != raw_ticks:
            logging.warning(
                f"STS target for {self} is outside raw range "
                f"(zero={self._zero_tick}, position={position}, target={raw_ticks}); "
                f"clamped to {ticks}"
            )
        speed = int(self._settings.get("velocity", _DEFAULT_SPEED))
        acc = int(self._settings.get("acceleration", _DEFAULT_ACCEL))
        '''
        try:
            if self.pin == 41 or self.pin == 40 or self.pin == 50 or self.pin == 51 or self.pin == 41 or self.pin == 18 or self.pin == 38 or self.pin == 20 or self.pin == 21:
                res, err = self._pk.WritePosEx(self.pin, ticks, 1200, 30) #_default/20
            
            if res != COMM_SUCCESS:
                logging.error(f"STS WritePosEx failed (res={res}, err={err})")
                return False
            
            else:
                res, err = self._pk.WritePosEx(self.pin, ticks, 3000, 30) #_default/20
                
        except Exception as e:
            logging.error(f"Exception during WritePosEx: {e}")
            return False
        return True
        '''
        if not bool(self._settings.get("turnedOn", True)):
            logging.warning(f"Ignoring position command while STS torque is disabled for {self}")
            return False

        # Keep the latest requested target even if this command cannot reach
        # the servo. The recovery loop can finish it when communication returns.
        self._last_target_ticks = ticks
        self._has_commanded_position = True

        speed, acc = self._motion_profile()

        last_result = COMM_TX_FAIL
        last_error = 0
        for attempt in range(1, _STS_COMMAND_ATTEMPTS + 1):
            try:
                if self._connected is not True and not self.check_connection():
                    last_result = self._last_comm_result
                    last_error = self._last_servo_error
                elif not self._set_torque_enabled(True):
                    last_result = self._last_comm_result
                    last_error = self._last_servo_error
                else:
                    last_result, last_error = self._pk.WritePosEx(
                        self.pin, ticks, speed, acc
                    )
                    if last_result == COMM_SUCCESS:
                        self._restore_required = False
                        self._record_comm_success()
                        return True
                    self._record_comm_failure(last_result, last_error)
            except Exception as error:
                logging.warning(
                    f"STS position attempt {attempt}/{_STS_COMMAND_ATTEMPTS} "
                    f"raised for {self}: {error}"
                )
                self._record_comm_failure(COMM_TX_FAIL, exception=error)

            if attempt < _STS_COMMAND_ATTEMPTS:
                self._prepare_sts_retry(attempt)
                time.sleep(_STS_RETRY_SECONDS)

        logging.error(
            f"STS WritePosEx failed for {self} after {_STS_COMMAND_ATTEMPTS} attempts "
            f"({self._last_failure_description()})"
        )
        self._log_failure_diagnostics()
        return False
    
    def get_position(self) -> int:
        """
        Read current position in DEGREES (rounded to int for API parity).
        """
        for attempt in range(1, _STS_COMMAND_ATTEMPTS + 1):
            try:
                if self._connected is not True:
                    if self.check_connection():
                        return self._last_position
                else:
                    ticks, _spd, result, error = self._pk.ReadPosSpeed(self.pin)
                    if result == COMM_SUCCESS:
                        self._record_position(ticks)
                        self._record_comm_success()
                        return self._last_position
                    self._record_comm_failure(result, error)
            except Exception as error:
                self._record_comm_failure(COMM_TX_FAIL, exception=error)

            if attempt < _STS_COMMAND_ATTEMPTS:
                self._prepare_sts_retry(attempt)
                time.sleep(_STS_RETRY_SECONDS)

        logging.warning(
            f"STS position read failed for {self}; using last valid position "
            f"{self._last_position} ({self._last_failure_description()})"
        )
        return self._last_position


def _load_robstride_dependencies():
    try:
        import can
        from robstride_dynamics import Motor, ParameterType, RobstrideBus
    except ImportError as error:
        raise RuntimeError(
            "RobStride dependencies are missing. Install python-can and "
            "robstride-dynamics in the ros-motors image."
        ) from error
    return can, Motor, ParameterType, RobstrideBus


def _robstride_parameter(name: str):
    _, _, ParameterType, _ = _load_robstride_dependencies()
    try:
        return getattr(ParameterType, name)
    except AttributeError as error:
        raise RuntimeError(
            f"Installed robstride-dynamics does not define ParameterType.{name}."
        ) from error


class _ATSerialCAN:
    """Minimal python-can-like transport for RobStride's USB-CAN AT protocol."""

    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.serial: serial.Serial | None = None
        self.rx = bytearray()

    def open(self) -> None:
        self.serial = serial.Serial(
            self.port,
            self.baudrate,
            timeout=0,
            write_timeout=1,
        )
        time.sleep(0.10)
        self.serial.reset_input_buffer()

    def clear_pending(self) -> None:
        self.rx.clear()
        if self.serial is not None:
            self.serial.reset_input_buffer()

    def send(self, message) -> None:
        if self.serial is None:
            raise RuntimeError("Serial transport is not open")
        if not message.is_extended_id:
            raise ValueError("RobStride private protocol requires extended CAN IDs")

        encoded_id = (message.arbitration_id << 3) | 0x04
        data = bytes(message.data)
        packet = (
            b"AT"
            + encoded_id.to_bytes(4, "big")
            + bytes([len(data)])
            + data
            + b"\r\n"
        )
        self.serial.write(packet)
        self.serial.flush()

    def _pop_frame(self):
        can, _, _, _ = _load_robstride_dependencies()

        while True:
            start = self.rx.find(b"AT")
            if start < 0:
                if len(self.rx) > 1:
                    del self.rx[:-1]
                return None

            if start:
                del self.rx[:start]

            if len(self.rx) < 7:
                return None

            dlc = self.rx[6]
            if dlc > 8:
                del self.rx[:2]
                continue

            total = 2 + 4 + 1 + dlc + 2
            if len(self.rx) < total:
                return None

            packet = bytes(self.rx[:total])
            del self.rx[:total]

            if packet[-2:] != b"\r\n":
                continue

            encoded_id = int.from_bytes(packet[2:6], "big")
            return can.Message(
                arbitration_id=encoded_id >> 3,
                is_extended_id=True,
                data=packet[7 : 7 + dlc],
            )

    def recv(self, timeout: float | None = None):
        if self.serial is None:
            raise RuntimeError("Serial transport is not open")

        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            frame = self._pop_frame()
            if frame is not None:
                return frame

            waiting = self.serial.in_waiting
            if waiting:
                self.rx.extend(self.serial.read(waiting))
                continue

            if deadline is not None and time.monotonic() >= deadline:
                return None

            time.sleep(0.001)

    def shutdown(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None


def _create_robstride_bus(port: str, motors: dict[str, object], baudrate: int):
    """Create a RobstrideBus subclass after the optional dependency is available."""
    _, _, _, RobstrideBus = _load_robstride_dependencies()

    class RobstrideSerialBus(RobstrideBus):
        def __init__(self, bus_port: str, bus_motors: dict[str, object], bus_baud: int):
            super().__init__(channel=f"serial:{bus_port}", motors=bus_motors)
            self.port = bus_port
            self.baudrate = bus_baud

        def connect(self, handshake: bool = True) -> None:
            if self.is_connected:
                return

            transport = _ATSerialCAN(self.port, self.baudrate)
            transport.open()
            self.channel_handler = transport

        def clear_pending(self) -> None:
            if self.channel_handler is not None:
                self.channel_handler.clear_pending()

    return RobstrideSerialBus(port, motors, baudrate)


class _RobstrideDevice:
    def __init__(self, uid: str, baudrate: int, model: str):
        self.uid = uid
        self.baudrate = baudrate
        self.model = model
        self.motors: dict[str, object] = {}
        self.bus = None
        self.lock = threading.Lock()

    def motor_name(self, motor_id: int) -> str:
        return f"motor_{motor_id}"

    def register_motor(self, motor_id: int) -> str:
        _, Motor, _, _ = _load_robstride_dependencies()
        motor_name = self.motor_name(motor_id)
        if motor_name not in self.motors:
            self.motors[motor_name] = Motor(id=motor_id, model=self.model)
            if self.bus is not None and self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)
            self.bus = None
        return motor_name

    def connect(self):
        if self.bus is None:
            self.bus = _create_robstride_bus(self.uid, self.motors, self.baudrate)
        if not self.bus.is_connected:
            self.bus.connect()
        return self.bus


def _get_robstride_device(uid: str, baudrate: int, model: str) -> _RobstrideDevice:
    if uid not in _robstride_bus_cache:
        _robstride_bus_cache[uid] = _RobstrideDevice(uid, baudrate, model)
    return _robstride_bus_cache[uid]


class _RobstrideBrickletPin:
    NO_CURRENT: int = -1

    def __init__(self, pin: int, uid: str, invert: bool) -> None:
        self.pin = int(pin)
        self.uid = uid
        self.invert = invert
        self._baudrate = int(os.getenv("ROBSTRIDE_BAUD", "921600"))
        self._model = os.getenv("ROBSTRIDE_MODEL", "rs-04")
        self._connected: bool | None = None
        self._last_position: int = 0
        self._has_valid_position: bool = False
        self._settings: Dict[str, Any] = {
            "velocity": float(os.getenv("ROBSTRIDE_DEFAULT_SPEED", "0.06")),
            "acceleration": float(os.getenv("ROBSTRIDE_DEFAULT_ACCEL", "0.15")),
            "deceleration": float(os.getenv("ROBSTRIDE_DEFAULT_ACCEL", "0.15")),
            "pulseWidthMin": None,
            "pulseWidthMax": None,
            "period": None,
            "turnedOn": True,
        }
        self._device = _get_robstride_device(self.uid, self._baudrate, self._model)
        self._motor_name = self._device.register_motor(self.pin)
        self._disable_timer: threading.Timer | None = None
        self._position_target_primed = False
        self._keepalive_thread: threading.Thread | None = None

    def __str__(self) -> str:
        return f"ROBSTRIDE-PIN[ id: {self.pin}, device: {self.uid} ]"

    def _record_position(self, radians: float) -> int:
        logical_radians = self._logical_position(float(radians))
        degrees = math.degrees(logical_radians)
        if self.invert:
            degrees *= -1.0
        self._last_position = int(round(degrees * 100.0))
        self._has_valid_position = True
        return self._last_position

    def has_valid_position(self) -> bool:
        return self._has_valid_position

    def _read_mechanical_position(self, bus, parameter_type) -> float:
        attempts = max(1, int(os.getenv("ROBSTRIDE_READ_ATTEMPTS", "3")))
        retry_delay = max(
            0.0, float(os.getenv("ROBSTRIDE_READ_RETRY_SECONDS", "0.02"))
        )
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                bus.clear_pending()
                return float(bus.read(self._motor_name, parameter_type))
            except Exception as error:
                last_error = error
                if attempt < attempts:
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"could not read mechanical position after {attempts} attempts"
        ) from last_error

    def _uses_shortest_path(self) -> bool:
        motor_ids = {
            int(value)
            for value in _csv_env("ROBSTRIDE_SHORTEST_PATH_IDS", "41")
        }
        return self.pin in motor_ids

    def _logical_position(self, position: float) -> float:
        if not self._uses_shortest_path():
            return position
        return math.atan2(math.sin(position), math.cos(position))

    def _resolve_position_target(self, start: float, requested: float) -> float:
        if not self._uses_shortest_path():
            return requested

        full_turn = 2.0 * math.pi
        nearest_turn = round((start - requested) / full_turn)
        return requested + (nearest_turn * full_turn)

    def _configure_can_timeout(self, bus) -> None:
        timeout = int(os.getenv("ROBSTRIDE_CAN_TIMEOUT", "40000"))
        if timeout <= 0:
            return
        timeout = min(timeout, 100000)
        timeout_parameter = _robstride_parameter("CAN_TIMEOUT")
        attempts = max(1, int(os.getenv("ROBSTRIDE_READ_ATTEMPTS", "3")))
        retry_delay = max(
            0.0, float(os.getenv("ROBSTRIDE_READ_RETRY_SECONDS", "0.02"))
        )
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                bus.clear_pending()
                bus.write(self._motor_name, timeout_parameter, timeout)
                return
            except Exception as error:
                last_error = error
                if attempt < attempts:
                    time.sleep(retry_delay)

        raise RuntimeError(
            f"could not configure CAN timeout after {attempts} attempts"
        ) from last_error

    def _start_keepalive(self) -> None:
        if int(os.getenv("ROBSTRIDE_CAN_TIMEOUT", "40000")) <= 0:
            return
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return

        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name=f"robstride-keepalive-{self.pin}",
            daemon=True,
        )
        self._keepalive_thread.start()

    def _keepalive_loop(self) -> None:
        interval = max(
            0.1, float(os.getenv("ROBSTRIDE_KEEPALIVE_SECONDS", "0.5"))
        )
        communication_failed = False

        while True:
            time.sleep(interval)
            try:
                with self._device.lock:
                    bus = self._device.connect()
                    _, _, ParameterType, _ = _load_robstride_dependencies()
                    self._read_mechanical_position(
                        bus, ParameterType.MECHANICAL_POSITION
                    )
                self._connected = True
                if communication_failed:
                    logging.info(f"RobStride keepalive recovered for {self}")
                    communication_failed = False
            except Exception as error:
                self._connected = False
                self._position_target_primed = False
                if not communication_failed:
                    logging.warning(f"RobStride keepalive failed for {self}: {error}")
                    communication_failed = True

    def check_connection(self) -> bool:
        try:
            with self._device.lock:
                bus = self._device.connect()
                _, _, ParameterType, _ = _load_robstride_dependencies()
                radians = self._read_mechanical_position(
                    bus, ParameterType.MECHANICAL_POSITION
                )
            self._record_position(radians)
            self._connected = True
        except Exception as error:
            logging.warning(f"RobStride check failed for {self}: {error}")
            self._connected = False
        return bool(self._connected)

    def is_connected(self) -> bool:
        if self._connected is not True:
            return self.check_connection()
        return bool(self._connected)

    def apply_settings(self, settings_dto: dict[str, Any]) -> bool:
        for key in (
            "velocity",
            "acceleration",
            "deceleration",
            "turnedOn",
            "pulseWidthMin",
            "pulseWidthMax",
            "period",
        ):
            if key in settings_dto:
                self._settings[key] = settings_dto[key]
        return self.is_connected()

    def get_settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def get_current(self) -> int:
        return _RobstrideBrickletPin.NO_CURRENT

    def reset_zero_position(self) -> bool:
        logging.warning(f"Zero position reset is not supported for {self}")
        return False

    def _motion_speed(self) -> float:
        default = float(os.getenv("ROBSTRIDE_DEFAULT_SPEED", "0.06"))
        velocity = self._settings.get("velocity")
        if velocity is None:
            return default
        return math.radians(float(velocity) / 100.0)

    def _motion_acceleration(self) -> float:
        default = float(os.getenv("ROBSTRIDE_DEFAULT_ACCEL", "0.15"))
        acceleration = self._settings.get("acceleration")
        if acceleration is None:
            return default
        return math.radians(float(acceleration) / 100.0)

    def _move_timeout(self, travel: float, speed: float, acceleration: float) -> float:
        settle = float(os.getenv("ROBSTRIDE_SETTLE_SECONDS", "0.75"))
        factor = float(os.getenv("ROBSTRIDE_DISABLE_WAIT_FACTOR", "2.0"))
        minimum = float(os.getenv("ROBSTRIDE_MIN_DISABLE_SECONDS", "2.0"))

        speed = max(abs(speed), 0.01)
        acceleration = max(abs(acceleration), 0.01)
        travel = abs(travel)

        ramp_distance = speed * speed / acceleration
        if travel >= ramp_distance:
            move_seconds = (travel / speed) + (speed / acceleration)
        else:
            move_seconds = 2.0 * math.sqrt(travel / acceleration)

        return max((move_seconds * factor) + settle, minimum)

    def _schedule_disable(self, wait_seconds: float) -> None:
        if not _bool_env("ROBSTRIDE_DISABLE_AFTER_MOVE", "false"):
            return
        if self._disable_timer is not None:
            self._disable_timer.cancel()

        self._disable_timer = threading.Timer(wait_seconds, self._disable_torque)
        self._disable_timer.daemon = True
        self._disable_timer.start()

    def _disable_torque(self) -> None:
        try:
            with self._device.lock:
                bus = self._device.connect()
                bus.disable(self._motor_name)
        except Exception as error:
            logging.warning(f"Could not disable RobStride torque for {self}: {error}")

    def set_position(self, position: int) -> bool:
        if not self.is_connected():
            return False

        degrees = float(position) / 100.0
        if self.invert:
            degrees *= -1.0
        requested_target = math.radians(degrees)

        try:
            with self._device.lock:
                bus = self._device.connect()
                _, _, ParameterType, _ = _load_robstride_dependencies()
                mode = _robstride_parameter("MODE")
                velocity = _robstride_parameter("PP_VELOCITY_MAX")
                acceleration = _robstride_parameter("PP_ACCELERATION_TARGET")
                position_target = _robstride_parameter("POSITION_TARGET")
                speed = self._motion_speed()
                accel = self._motion_acceleration()
                self._configure_can_timeout(bus)
                start = self._read_mechanical_position(
                    bus, ParameterType.MECHANICAL_POSITION
                )
                target = self._resolve_position_target(start, requested_target)
                prime_target = (
                    _bool_env("ROBSTRIDE_PRIME_POSITION_TARGET", "true")
                    and not self._position_target_primed
                )

                logging.info(
                    "RobStride position command for %s: start=%+.4f rad, "
                    "requested=%+.4f rad, target=%+.4f rad, startup_prime=%s",
                    self,
                    start,
                    requested_target,
                    target,
                    prime_target,
                )

                bus.disable(self._motor_name)
                bus.write(self._motor_name, mode, 1)
                bus.write(self._motor_name, velocity, speed)
                bus.write(self._motor_name, acceleration, accel)

                # Load a no-motion target before enabling PP mode. After a power
                # cycle, enabling first can briefly activate the controller's
                # reset/default target and start a move in the wrong direction.
                if prime_target:
                    bus.write(self._motor_name, position_target, start)

                bus.enable(self._motor_name)
                if prime_target:
                    self._position_target_primed = True
                    time.sleep(
                        float(os.getenv("ROBSTRIDE_PRIME_SETTLE_SECONDS", "0.05"))
                    )
                bus.write(self._motor_name, position_target, target)

            wait_seconds = self._move_timeout(target - start, speed, accel)
            self._schedule_disable(wait_seconds)
            self._start_keepalive()
            return True
        except Exception as error:
            logging.error(f"Error while setting RobStride position for {self}: {error}")
            self._connected = False
            return False

    def get_position(self) -> int:
        if not self.is_connected():
            return self._last_position
        try:
            with self._device.lock:
                bus = self._device.connect()
                _, _, ParameterType, _ = _load_robstride_dependencies()
                radians = self._read_mechanical_position(
                    bus, ParameterType.MECHANICAL_POSITION
                )
            self._connected = True
            return self._record_position(radians)
        except Exception:
            self._connected = False
            return self._last_position


class BrickletPin:
    NO_CURRENT: int = -1

    def __new__(cls, pin: int, uid: str, invert: bool):
        if cls is not BrickletPin:
            return super().__new__(cls)
        if _is_robstride_pin(int(pin), uid):
            return _RobstrideBrickletPin(pin, uid, invert)
        return _STSBrickletPin(pin, uid, invert)
