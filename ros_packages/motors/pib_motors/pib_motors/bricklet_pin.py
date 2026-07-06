import logging
import math
import os
import threading
import time
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

# Cache one serial port + packet handler per device so multiple motors on the same bus reuse it
_port_cache: Dict[str, Tuple[PortHandler, any]] = {}
_robstride_bus_cache: Dict[str, "_RobstrideDevice"] = {}


def _csv_env(name: str, default: str) -> set[str]:
    return {value.strip() for value in os.getenv(name, default).split(",") if value.strip()}


def _is_robstride_pin(pin: int, uid: str) -> bool:
    robstride_ports = _csv_env("ROBSTRIDE_PORTS", "/dev/ttyMotor4")
    robstride_ids = {int(value) for value in _csv_env("ROBSTRIDE_MOTOR_IDS", "21,41")}
    return uid in robstride_ports or int(pin) in robstride_ids


def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


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
    return float(ticks * 18000 / 4096) - 9000


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

        # Attempt initial check/open
        self.check_connection()

    def __str__(self) -> str:
        return f"STS-PIN[ id: {self.pin}, device: {self.uid} ]"

    # ----------------
    # Connection state
    # ----------------
    def check_connection(self) -> bool:
        """Check we can talk to the STS bus and read this ID once."""
        try:
            self._ph, self._pk = _get_or_open_port(self.uid, self._baudrate)
            # Probe by reading pos/speed for this ID
            _, _, res, _err = self._pk.ReadPosSpeed(self.pin)
            self._connected = (res == COMM_SUCCESS)
        except Exception:
            self._connected = False
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

            # Nothing to actively send until a position command (STS sets speed/acc there)
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
    # -------------
    # Position I/O
    # -------------
    def set_position(self, position: int) -> bool:
        """
        Set target position in DEGREES (kept consistent with your examples).
        If your upstream publishes a different unit, adjust mapping here.
        """
        if not self.is_connected():
            return False

        deg = float(position)
        if self.invert:
            deg *= -1.0

        ticks = _deg_to_ticks(deg)
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
        try:
            STS3095_pins = [41, 40, 50, 51, 18, 38, 20, 21, 39, 19]
            if self.pin in STS3095_pins:
                res, err = self._pk.WritePosEx(self.pin, ticks, 700, 30)
            else:
                res, err = self._pk.WritePosEx(self.pin, ticks, 3000, 100)

            if res != COMM_SUCCESS:
                logging.error(f"STS WritePosEx failed (res={res}, err={err})")
                return False

        except Exception as e:
            logging.error(f"Exception during WritePosEx: {e}")
            return False

        return True
    
    def get_position(self) -> int:
        """
        Read current position in DEGREES (rounded to int for API parity).
        """
        if not self.is_connected():
            return 0
        try:
            ticks, _spd, res, _err = self._pk.ReadPosSpeed(self.pin)
            if res != COMM_SUCCESS:
                return 0
            deg = _ticks_to_deg(int(ticks))
            return int(round(deg))
        except Exception:
            return 0


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

    def __str__(self) -> str:
        return f"ROBSTRIDE-PIN[ id: {self.pin}, device: {self.uid} ]"

    def check_connection(self) -> bool:
        try:
            bus = self._device.connect()
            _, _, ParameterType, _ = _load_robstride_dependencies()
            bus.read(self._motor_name, ParameterType.MECHANICAL_POSITION)
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
        if not _bool_env("ROBSTRIDE_DISABLE_AFTER_MOVE", "true"):
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
        target = math.radians(degrees)

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

                try:
                    start = float(
                        bus.read(self._motor_name, ParameterType.MECHANICAL_POSITION)
                    )
                except Exception:
                    start = target

                bus.disable(self._motor_name)
                bus.write(self._motor_name, mode, 1)
                bus.write(self._motor_name, velocity, speed)
                bus.write(self._motor_name, acceleration, accel)
                bus.enable(self._motor_name)
                bus.write(self._motor_name, position_target, target)

            wait_seconds = self._move_timeout(target - start, speed, accel)
            self._schedule_disable(wait_seconds)
            return True
        except Exception as error:
            logging.error(f"Error while setting RobStride position for {self}: {error}")
            self._connected = False
            return False

    def get_position(self) -> int:
        if not self.is_connected():
            return 0
        try:
            bus = self._device.connect()
            _, _, ParameterType, _ = _load_robstride_dependencies()
            radians = bus.read(self._motor_name, ParameterType.MECHANICAL_POSITION)
            degrees = math.degrees(float(radians))
            if self.invert:
                degrees *= -1.0
            return int(round(degrees * 100.0))
        except Exception:
            return 0


class BrickletPin:
    NO_CURRENT: int = -1

    def __new__(cls, pin: int, uid: str, invert: bool):
        if cls is not BrickletPin:
            return super().__new__(cls)
        if _is_robstride_pin(int(pin), uid):
            return _RobstrideBrickletPin(pin, uid, invert)
        return _STSBrickletPin(pin, uid, invert)
