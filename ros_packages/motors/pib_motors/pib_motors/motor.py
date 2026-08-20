import logging
import os
import time
from typing import Any

from pib_motors.bricklet_pin import BrickletPin
from pib_api_client import motor_client


API_RETRY_TIMEOUT_SECONDS = float(os.getenv("PIB_API_RETRY_TIMEOUT_SECONDS", "60"))
API_RETRY_INTERVAL_SECONDS = float(os.getenv("PIB_API_RETRY_INTERVAL_SECONDS", "2"))


class Motor:

    MIN_ROTATION: int = -9000
    MAX_ROTATION: int = 9000

    NO_CURRENT: int = BrickletPin.NO_CURRENT

    def __init__(self, name: str, bricklet_pins: list[BrickletPin], invert: bool):
        self.name: str = name
        self.visible: bool = True
        self.bricklet_pins = bricklet_pins
        self.invert: bool = invert
        self.rotation_range_min: int = Motor.MIN_ROTATION
        self.rotation_range_max: int = Motor.MAX_ROTATION

    def __str__(self):
        return f"MOTOR[ bricklet_pins: {[str(bp) for bp in self.bricklet_pins]}, settings: {self.get_settings()} ]"

    def apply_settings(self, settings_dto: dict[str, Any]) -> bool:
        """apply provided settings to the motor"""
        self.load_settings(settings_dto)

        if not self.bricklet_pins:
            return False

        # Check if current position is outside of new rotation Ranges
        adjusted_position = self._validate_position(self.get_position())
        if adjusted_position != self.get_position():
            self.set_position(adjusted_position)

        return all(bp.apply_settings(settings_dto) for bp in self.bricklet_pins)

    def load_settings(self, settings_dto: dict[str, Any]) -> None:
        """Load stored settings without touching hardware."""
        self.visible = settings_dto["visible"]
        self.invert = settings_dto["invert"]
        self.rotation_range_min = settings_dto["rotationRangeMin"]
        self.rotation_range_max = settings_dto["rotationRangeMax"]

        for bricklet_pin in self.bricklet_pins:
            settings = getattr(bricklet_pin, "_settings", None)
            if settings is None:
                continue
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
                    settings[key] = settings_dto[key]

    def get_settings(self) -> dict[str, Any]:
        """get the current settings of this motor"""
        settings = {
            "visible": self.visible,
            "name": self.name,
            "invert": self.invert,
            "rotationRangeMin": self.rotation_range_min,
            "rotationRangeMax": self.rotation_range_max,
        }
        if not self.bricklet_pins:
            return settings
        settings.update(self.bricklet_pins[0].get_settings())
        return settings

    def set_position(self, position: int) -> bool:
        """sets the position of all bricklet-pins associated with this motor"""
        if not self.bricklet_pins:
            return False
        position = self.clamp_logical_position(position)
        if self.invert:
            position *= -1
        return all(bp.set_position(position) for bp in self.bricklet_pins)

    def logical_position_limits(self) -> tuple[int, int]:
        """Return configured limits in the coordinate system used by ROS."""
        if self.invert:
            return -self.rotation_range_max, -self.rotation_range_min
        return self.rotation_range_min, self.rotation_range_max

    def clamp_logical_position(self, position: int | float) -> int | float:
        """Clamp a ROS position while preserving motor inversion semantics."""
        minimum, maximum = self.logical_position_limits()
        return min(max(position, minimum), maximum)

    def logical_position_is_in_range(
        self,
        position: int | float,
        tolerance: int | float = 0,
    ) -> bool:
        minimum, maximum = self.logical_position_limits()
        return minimum - tolerance <= position <= maximum + tolerance

    def logical_position_distance_to_range(
        self,
        position: int | float,
    ) -> int | float:
        """Return zero in range, otherwise the distance to the nearest limit."""
        minimum, maximum = self.logical_position_limits()
        if position < minimum:
            return minimum - position
        if position > maximum:
            return position - maximum
        return 0

    def logical_target_moves_toward_range(
        self,
        current_position: int | float,
        target_position: int | float,
    ) -> bool:
        """Return whether a target reduces an existing limit violation."""
        return self.logical_position_distance_to_range(
            target_position
        ) < self.logical_position_distance_to_range(current_position)

    def get_position(self) -> int:
        """returns the postion of the motor or '0' if no bricklet-pin is connected"""
        if not self.bricklet_pins:
            return 0
        return self.bricklet_pins[0].get_position()

    def has_valid_position(self) -> bool:
        """Return whether every physical pin has produced real position feedback."""
        return bool(self.bricklet_pins) and all(
            bool(getattr(bp, "has_valid_position", lambda: True)())
            for bp in self.bricklet_pins
        )

    def get_current(self) -> int:
        """returns the maximum current of all bricklet-pins, or NO_CURRENT, if not bricklet-pin is connected"""
        if not self.bricklet_pins:
            return Motor.NO_CURRENT
        return max(bp.get_current() for bp in self.bricklet_pins)

    def check_if_motor_is_connected(self) -> bool:
        """returns 'True' if all bricklet-pins of this motor are connected"""
        return bool(self.bricklet_pins) and all(
            bp.is_connected() for bp in self.bricklet_pins
        )

    def reset_zero_position(self) -> bool:
        """Use the current physical position as this motor's neutral position."""
        return bool(self.bricklet_pins) and all(
            bp.reset_zero_position() for bp in self.bricklet_pins
        )

    def _validate_position(self, position: int) -> int:
        """Check if position is within range, set it to the min/max value if not."""
        position = min(max(position, self.rotation_range_min), self.rotation_range_max)
        return position


def _load_motors_from_api() -> dict[str, Any]:
    """Wait for the Flask API during boot and then return the motor payload."""
    deadline = time.monotonic() + API_RETRY_TIMEOUT_SECONDS
    attempt = 1

    while True:
        successful, response = motor_client.get_all_motors()
        if successful:
            if attempt > 1:
                logging.info("Loaded motors from pib-api after %d attempts.", attempt)
            return response

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"failed to load motors from pib-api after {attempt} attempts..."
            )

        logging.warning(
            "pib-api is not ready yet; retrying motor load in %.1f seconds "
            "(attempt %d).",
            API_RETRY_INTERVAL_SECONDS,
            attempt,
        )
        time.sleep(API_RETRY_INTERVAL_SECONDS)
        attempt += 1


# get data from pib-api
response = _load_motors_from_api()

# list of all available motor-objects
motors: list[Motor] = []
for motor_dto in response["motors"]:
    bricklet_pins = [
        BrickletPin(
            bricklet_pin_dto["pin"],
            bricklet_pin_dto["bricklet"],
            bricklet_pin_dto["invert"],
        )
        for bricklet_pin_dto in motor_dto["brickletPins"]
        if bricklet_pin_dto["bricklet"]
    ]
    motors.append(Motor(motor_dto["name"], bricklet_pins, motor_dto["invert"]))

# maps the name of a (multi-)motor to its associated motor objects
name_to_motors: dict[str, Motor] = {motor.name: [motor] for motor in motors}
name_to_motors["all_fingers_left"] = [
    motor for motor in motors if motor.name.endswith("left_stretch")
]
name_to_motors["all_fingers_right"] = [
    motor for motor in motors if motor.name.endswith("right_stretch")
]
