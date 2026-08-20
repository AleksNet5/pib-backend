import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


Vector = tuple[float, float, float]
Matrix = tuple[Vector, Vector, Vector]
# Real joint feedback can trace a shallow arc before leaving an existing
# safety envelope. Keep that transient well inside the configured 30 mm margin.
ESCAPE_TOLERANCE_MM = 3.0


def _add(a: Vector, b: Vector) -> Vector:
    return tuple(a[i] + b[i] for i in range(3))


def _subtract(a: Vector, b: Vector) -> Vector:
    return tuple(a[i] - b[i] for i in range(3))


def _scale(vector: Vector, factor: float) -> Vector:
    return tuple(value * factor for value in vector)


def _dot(a: Vector, b: Vector) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(vector: Vector) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector) -> Vector:
    length = _length(vector)
    if length < 1e-9:
        raise ValueError("zero-length vector")
    return _scale(vector, 1.0 / length)


def _identity() -> Matrix:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    return tuple(
        tuple(sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(_dot(matrix[row], vector) for row in range(3))


def _axis_angle(axis: Vector, angle: float) -> Matrix:
    x, y, z = _normalize(axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return (
        (
            x * x * complement + cosine,
            x * y * complement - z * sine,
            x * z * complement + y * sine,
        ),
        (
            y * x * complement + z * sine,
            y * y * complement + cosine,
            y * z * complement - x * sine,
        ),
        (
            z * x * complement - y * sine,
            z * y * complement + x * sine,
            z * z * complement + cosine,
        ),
    )


def _euler_xyz(rotation_degrees: Iterable[float]) -> Matrix:
    roll, pitch, yaw = (math.radians(value) for value in rotation_degrees)
    return _matrix_multiply(
        _matrix_multiply(_axis_angle((1, 0, 0), roll), _axis_angle((0, 1, 0), pitch)),
        _axis_angle((0, 0, 1), yaw),
    )


@dataclass(frozen=True)
class Transform:
    rotation: Matrix = _identity()
    translation: Vector = (0.0, 0.0, 0.0)

    def compose(self, child: "Transform") -> "Transform":
        return Transform(
            rotation=_matrix_multiply(self.rotation, child.rotation),
            translation=_add(
                self.translation, _matrix_vector(self.rotation, child.translation)
            ),
        )

    def point(self, local_point: Vector) -> Vector:
        return _add(self.translation, _matrix_vector(self.rotation, local_point))


@dataclass(frozen=True)
class OrientedBox:
    name: str
    center: Vector
    axes: tuple[Vector, Vector, Vector]
    half_extents: Vector


@dataclass(frozen=True)
class PlaneObstacle:
    name: str
    point: Vector
    normal: Vector
    margin: float
    ignored_links: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    box: OrientedBox
    margin: float
    ignored_links: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CollisionResult:
    allowed: bool
    reason: str = ""
    minimum_clearance_mm: float | None = None
    safe_positions: dict[str, float] | None = None


@dataclass(frozen=True)
class JointDefinition:
    motor_name: str
    axis: Vector
    anchor: Vector
    orientation: tuple[float, float, float, float]
    mesh_position: Vector
    mesh_orientation: tuple[float, float, float, float] | None
    mesh_center: Vector
    mesh_half_extents: Vector


ARM_MOTOR_NAMES = {
    "left": (
        "shoulder_vertical_left",
        "shoulder_horizontal_left",
        "upper_arm_left_rotation",
        "elbow_left",
        "lower_arm_left_rotation",
        "wrist_left",
    ),
    "right": (
        "shoulder_vertical_right",
        "shoulder_horizontal_right",
        "upper_arm_right_rotation",
        "elbow_right",
        "lower_arm_right_rotation",
        "wrist_right",
    ),
}

# Physical command directions that differ from the Webots joint axes used by
# the collision geometry.
JOINT_ANGLE_SCALE = {
    "shoulder_vertical_left": -1.0,
}

# Mechanical zero positions that differ from the Webots model. Values are in
# centidegrees so they can be applied directly to motor positions.
JOINT_ANGLE_OFFSET = {
    "elbow_left": 1000.0,
    "upper_arm_right_rotation": 1500.0,
    "elbow_right": 3000.0,
}


_MESH_BOUNDS = {
    "shoulder_vertical": ((61.75, 0.0, 0.0), (55.75, 55.0, 55.0)),
    "shoulder_horizontal": ((65.0, 0.0, -46.25), (47.5, 46.0, 93.75)),
    "upper_arm": ((65.0, 0.0, -186.65), (47.0, 47.0, 46.65)),
    "elbow": ((65.0, 0.0, -251.35), (45.0, 45.0, 52.05)),
    "forearm": ((64.7, 0.5, -388.1), (45.0, 45.0, 84.7)),
    "palm_left": ((-161.3, -0.6, -55.75), (47.8, 13.4, 60.25)),
    "palm_right": ((80.8, 3.45, -500.55), (13.4, 47.85, 60.25)),
}


def _joint(
    motor_name,
    axis,
    anchor,
    orientation,
    mesh_name,
    mesh_position,
    mesh_orientation=None,
):
    center, half_extents = _MESH_BOUNDS[mesh_name]
    return JointDefinition(
        motor_name=motor_name,
        axis=axis,
        anchor=anchor,
        orientation=orientation,
        mesh_position=mesh_position,
        mesh_orientation=mesh_orientation,
        mesh_center=center,
        mesh_half_extents=half_extents,
    )


ARM_DEFINITIONS = {
    "left": (
        _joint(
            "shoulder_vertical_left",
            (1, 0, 0),
            (146.867, 12.566, 835.688),
            (-0.57735, 0.57735, -0.57735, 2.094395),
            "shoulder_vertical",
            (0, 0, -6),
            (0.707107, 0, 0.707107, math.pi),
        ),
        _joint(
            "shoulder_horizontal_left",
            (0, -1, 0),
            (0, -37.495, 59),
            (0.57735, -0.57735, 0.57735, 2.094395),
            "shoulder_horizontal",
            (0, -65, -37.2),
            (-0.57735, -0.57735, 0.57735, 2.094395),
        ),
        _joint(
            "upper_arm_left_rotation",
            (-1, 0, 0),
            (140, 0, -37.2),
            (-0.57735, -0.57735, 0.57735, 2.094395),
            "upper_arm",
            (-65, 0.001, 140),
        ),
        _joint(
            "elbow_left",
            (-1, 0, 0),
            (-37.9, -21, -76.338),
            (0.707107, 0, -0.707107, math.pi),
            "elbow",
            (-138.124, -167.824, 27.2),
            (0.678598, 0.281085, -0.678598, 2.593564),
        ),
        _joint(
            "lower_arm_left_rotation",
            (-0.707107, -0.707107, 0),
            (76.438, 46.74, -37.8),
            (0.678598, 0.281085, -0.678598, 2.593564),
            "forearm",
            (-64.742, -0.469, 303.438),
        ),
        _joint(
            "wrist_left",
            (0, 1, 0),
            (15, 26, -154.4),
            (0, -0.707107, -0.707107, math.pi),
            "palm_left",
            (3, -100, -182.7),
            (0.57735, 0.57735, -0.57735, 2.094395),
        ),
    ),
    "right": (
        _joint(
            "shoulder_vertical_right",
            (-1, 0, 0),
            (-173.133, 12.566, 835.688),
            (-0.57735, -0.57735, 0.57735, 2.094395),
            "shoulder_vertical",
            (0, 0, -6),
            (0.707107, 0, 0.707107, math.pi),
        ),
        _joint(
            "shoulder_horizontal_right",
            (0, -1, 0),
            (0, -37.495, 59),
            (0.57735, -0.57735, 0.57735, 2.094395),
            "shoulder_horizontal",
            (0, 65, -37.2),
            (0.57735, -0.57735, -0.57735, 2.094395),
        ),
        _joint(
            "upper_arm_right_rotation",
            (-1, 0, 0),
            (140, 0, -37.2),
            (-0.57735, -0.57735, 0.57735, 2.094395),
            "upper_arm",
            (-65, 0.001, 140),
        ),
        _joint(
            "elbow_right",
            (-1, 0, 0),
            (-37.9, -21, -76.338),
            (-0.707107, 0, 0.707107, math.pi),
            "elbow",
            (-138.124, -167.824, 27.2),
            (0.678598, 0.281085, -0.678598, 2.593564),
        ),
        _joint(
            "lower_arm_right_rotation",
            (-0.707107, -0.707107, 0),
            (76.438, 46.74, -37.8),
            (0.357407, -0.862856, 0.357407, 1.717772),
            "forearm",
            (-64.742, -0.469, 303.438),
        ),
        _joint(
            "wrist_right",
            (0, 1, 0),
            (15, 25, -154.4),
            (0, -0.707107, -0.707107, math.pi),
            "palm_right",
            (-78.396, 456.264, -25.531),
            (-1, 0, 0, math.pi / 2),
        ),
    ),
}


def _orientation_axis_angle(values) -> Matrix:
    return _axis_angle(tuple(values[:3]), values[3])


def _box_axes(rotation: Matrix) -> tuple[Vector, Vector, Vector]:
    return tuple(tuple(rotation[row][column] for row in range(3)) for column in range(3))


def arm_boxes(side: str, positions: dict[str, float]) -> list[OrientedBox]:
    parent = Transform()
    boxes = []
    for definition in ARM_DEFINITIONS[side]:
        angle = math.radians(
            (
                positions.get(definition.motor_name, 0.0)
                * JOINT_ANGLE_SCALE.get(definition.motor_name, 1.0)
                + JOINT_ANGLE_OFFSET.get(definition.motor_name, 0.0)
            )
            / 100.0
        )
        driver = parent.compose(
            Transform(
                rotation=_axis_angle(definition.axis, angle),
                translation=definition.anchor,
            )
        )
        endpoint = driver.compose(
            Transform(rotation=_orientation_axis_angle(definition.orientation))
        )
        mesh_rotation = (
            _identity()
            if definition.mesh_orientation is None
            else _orientation_axis_angle(definition.mesh_orientation)
        )
        mesh = endpoint.compose(
            Transform(rotation=mesh_rotation, translation=definition.mesh_position)
        )
        boxes.append(
            OrientedBox(
                name=definition.motor_name,
                center=mesh.point(definition.mesh_center),
                axes=_box_axes(mesh.rotation),
                half_extents=definition.mesh_half_extents,
            )
        )
        parent = endpoint
    return boxes


def _projection_radius(box: OrientedBox, axis: Vector) -> float:
    return sum(
        box.half_extents[i] * abs(_dot(axis, box.axes[i])) for i in range(3)
    )


def _plane_clearance(box: OrientedBox, obstacle: PlaneObstacle) -> float:
    center_distance = abs(_dot(obstacle.normal, _subtract(box.center, obstacle.point)))
    return center_distance - _projection_radius(box, obstacle.normal) - obstacle.margin


def _box_clearance(box: OrientedBox, obstacle: BoxObstacle) -> float:
    expanded = OrientedBox(
        name=obstacle.box.name,
        center=obstacle.box.center,
        axes=obstacle.box.axes,
        half_extents=tuple(value + obstacle.margin for value in obstacle.box.half_extents),
    )
    center_delta = _subtract(expanded.center, box.center)
    axes = list(box.axes) + list(expanded.axes)
    axes.extend(_cross(a, b) for a in box.axes for b in expanded.axes)
    separations = []
    for axis in axes:
        if _length(axis) < 1e-7:
            continue
        axis = _normalize(axis)
        separations.append(
            abs(_dot(center_delta, axis))
            - _projection_radius(box, axis)
            - _projection_radius(expanded, axis)
        )
    return max(separations)


class CollisionGuard:
    def __init__(
        self,
        config_path: str | Path,
        step_degrees: float = 1.0,
        range_step_degrees: float = 1.0,
    ):
        self.config_path = Path(config_path)
        self.step_centidegrees = max(1.0, step_degrees * 100.0)
        self.range_step_centidegrees = max(
            self.step_centidegrees, range_step_degrees * 100.0
        )
        self.obstacles = self._load_obstacles()

    def _load_obstacles(self):
        with self.config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        if config.get("coordinate_system", {}).get("unit") != "mm":
            raise ValueError("collision geometry must use millimetres")

        obstacles = []
        for definition in config["obstacles"]:
            margin = float(definition.get("safety_margin_mm", 0))
            ignored_links = frozenset(definition.get("ignored_links", []))
            if definition["type"] == "plane":
                if definition.get("infinite") is not True:
                    raise ValueError(f"plane {definition['name']} must be infinite")
                obstacles.append(
                    PlaneObstacle(
                        name=definition["name"],
                        point=tuple(map(float, definition["point_mm"])),
                        normal=_normalize(tuple(map(float, definition["normal"]))),
                        margin=margin,
                        ignored_links=ignored_links,
                    )
                )
            elif definition["type"] == "box":
                rotation = _euler_xyz(definition["rotation_rpy_deg"])
                obstacles.append(
                    BoxObstacle(
                        name=definition["name"],
                        box=OrientedBox(
                            name=definition["name"],
                            center=tuple(map(float, definition["center_mm"])),
                            axes=_box_axes(rotation),
                            half_extents=tuple(
                                float(value) / 2.0 for value in definition["size_mm"]
                            ),
                        ),
                        margin=margin,
                        ignored_links=ignored_links,
                    )
                )
            else:
                raise ValueError(f"unsupported obstacle type: {definition['type']}")
        return obstacles

    def clearances(
        self,
        positions: dict[str, float],
        sides: Iterable[str] = ("left", "right"),
        *,
        ignored_obstacle_names: Iterable[str] = (),
    ) -> dict[tuple[str, str], float]:
        ignored_obstacle_names = frozenset(ignored_obstacle_names)
        result = {}
        for side in sides:
            for body in arm_boxes(side, positions):
                for obstacle in self.obstacles:
                    if obstacle.name in ignored_obstacle_names:
                        continue
                    if body.name in obstacle.ignored_links:
                        continue
                    if isinstance(obstacle, PlaneObstacle):
                        clearance = _plane_clearance(body, obstacle)
                    else:
                        clearance = _box_clearance(body, obstacle)
                    result[(body.name, obstacle.name)] = clearance
        return result

    @staticmethod
    def _collision_reason(
        side: str,
        clearances: dict[tuple[str, str], float],
        initial_clearances: dict[tuple[str, str], float],
        escape_pairs: frozenset[tuple[str, str]],
        cleared_escape_pairs: set[tuple[str, str]],
    ) -> str | None:
        for pair, clearance in clearances.items():
            body_name, obstacle_name = pair
            if clearance < 0 and (
                pair not in escape_pairs or pair in cleared_escape_pairs
            ):
                return f"{body_name} would enter {obstacle_name}"
            if (
                pair in escape_pairs
                and clearance
                < initial_clearances[pair] - ESCAPE_TOLERANCE_MM
            ):
                return (
                    f"{side} arm would move deeper into "
                    f"{body_name}/{obstacle_name}"
                )

        cleared_escape_pairs.update(
            pair
            for pair in escape_pairs
            if clearances.get(pair, math.inf) > 0
        )
        return None

    def evaluate(
        self,
        current_positions: dict[str, float],
        targets: dict[str, float],
        *,
        ignored_obstacle_names: Iterable[str] = (),
    ) -> CollisionResult:
        ignored_obstacle_names = frozenset(ignored_obstacle_names)
        affected_sides = [
            side
            for side, names in ARM_MOTOR_NAMES.items()
            if any(name in targets for name in names)
        ]
        if not affected_sides:
            return CollisionResult(True)

        target_positions = dict(current_positions)
        target_positions.update(
            {name: value for name, value in targets.items() if name in all_arm_names()}
        )
        maximum_delta = max(
            abs(target_positions[name] - current_positions.get(name, 0.0))
            for side in affected_sides
            for name in ARM_MOTOR_NAMES[side]
        )
        steps = max(1, math.ceil(maximum_delta / self.step_centidegrees))

        initial = self.clearances(
            current_positions,
            affected_sides,
            ignored_obstacle_names=ignored_obstacle_names,
        )
        minimum_clearance = min(initial.values())
        last_safe_positions = dict(current_positions)
        initial_by_side = {
            side: {
                pair: clearance
                for pair, clearance in initial.items()
                if pair[0] in ARM_MOTOR_NAMES[side]
            }
            for side in affected_sides
        }
        escape_pairs = {
            side: frozenset(
                pair
                for pair, clearance in initial_by_side[side].items()
                if clearance <= 0
            )
            for side in affected_sides
        }
        cleared_escape_pairs = {side: set() for side in affected_sides}

        def blocked(reason: str) -> CollisionResult:
            return CollisionResult(
                False,
                reason,
                minimum_clearance,
                {
                    name: last_safe_positions[name]
                    for name in targets
                    if name in all_arm_names()
                },
            )

        for step in range(1, steps + 1):
            fraction = step / steps
            sample = dict(current_positions)
            for side in affected_sides:
                for name in ARM_MOTOR_NAMES[side]:
                    start = current_positions.get(name, 0.0)
                    target = target_positions.get(name, start)
                    sample[name] = start + (target - start) * fraction

            clearances = self.clearances(
                sample,
                affected_sides,
                ignored_obstacle_names=ignored_obstacle_names,
            )
            minimum_clearance = min(minimum_clearance, min(clearances.values()))
            for side in affected_sides:
                side_clearances = {
                    pair: clearance
                    for pair, clearance in clearances.items()
                    if pair[0] in ARM_MOTOR_NAMES[side]
                }
                reason = self._collision_reason(
                    side,
                    side_clearances,
                    initial_by_side[side],
                    escape_pairs[side],
                    cleared_escape_pairs[side],
                )
                if reason is not None:
                    return blocked(reason)

            last_safe_positions = sample

        return CollisionResult(True, minimum_clearance_mm=minimum_clearance)

    def safe_joint_range(
        self,
        current_positions: dict[str, float],
        motor_name: str,
        hard_minimum: float,
        hard_maximum: float,
    ) -> tuple[float, float]:
        """Return the contiguous collision-safe range around the current pose."""
        if motor_name not in all_arm_names():
            return hard_minimum, hard_maximum

        hard_minimum, hard_maximum = sorted(
            (float(hard_minimum), float(hard_maximum))
        )
        current = current_positions[motor_name]
        side = next(
            side
            for side, names in ARM_MOTOR_NAMES.items()
            if motor_name in names
        )
        initial = self.clearances(current_positions, sides=(side,))
        escape_pairs = frozenset(
            pair for pair, clearance in initial.items() if clearance <= 0
        )

        def safe_boundary(target: float) -> float:
            target = float(target)
            delta = target - current
            if abs(delta) < 1.0:
                return target

            coarse_steps = max(
                1,
                math.ceil(abs(delta) / self.range_step_centidegrees),
            )
            last_safe = current
            last_cleared_pairs: set[tuple[str, str]] = set()

            def conservative_boundary(position: float) -> float:
                if position > current:
                    return max(current, position - self.step_centidegrees)
                return min(current, position + self.step_centidegrees)

            for step in range(1, coarse_steps + 1):
                position = current + (delta * step / coarse_steps)
                sample = dict(current_positions)
                sample[motor_name] = position
                clearances = self.clearances(sample, sides=(side,))
                cleared_pairs = set(last_cleared_pairs)
                reason = self._collision_reason(
                    side,
                    clearances,
                    initial,
                    escape_pairs,
                    cleared_pairs,
                )
                if reason is None:
                    last_safe = position
                    last_cleared_pairs = cleared_pairs
                    continue

                fine_delta = position - last_safe
                fine_steps = max(
                    1,
                    math.ceil(abs(fine_delta) / self.step_centidegrees),
                )
                cleared_pairs = set(last_cleared_pairs)
                for fine_step in range(1, fine_steps + 1):
                    fine_position = (
                        last_safe + (fine_delta * fine_step / fine_steps)
                    )
                    fine_sample = dict(current_positions)
                    fine_sample[motor_name] = fine_position
                    fine_clearances = self.clearances(
                        fine_sample, sides=(side,)
                    )
                    fine_reason = self._collision_reason(
                        side,
                        fine_clearances,
                        initial,
                        escape_pairs,
                        cleared_pairs,
                    )
                    if fine_reason is not None:
                        return conservative_boundary(last_safe)
                    last_safe = fine_position
                return conservative_boundary(last_safe)

            return target

        safe_minimum = max(
            hard_minimum,
            min(hard_maximum, safe_boundary(hard_minimum)),
        )
        safe_maximum = max(
            hard_minimum,
            min(hard_maximum, safe_boundary(hard_maximum)),
        )
        if safe_minimum > safe_maximum:
            anchor = max(hard_minimum, min(hard_maximum, current))
            return anchor, anchor
        return safe_minimum, safe_maximum


def all_arm_names() -> set[str]:
    return {name for names in ARM_MOTOR_NAMES.values() for name in names}
