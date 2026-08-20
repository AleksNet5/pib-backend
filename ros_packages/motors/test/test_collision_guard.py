import json
import unittest
from pathlib import Path

from motors.collision_guard import (
    CollisionGuard,
    arm_boxes,
    all_arm_names,
)


CONFIG_PATH = Path(__file__).parents[1] / "config" / "collision_geometry.json"


class CollisionGuardTest(unittest.TestCase):
    def setUp(self):
        self.guard = CollisionGuard(CONFIG_PATH, step_degrees=0.25)
        self.zero = {name: 0 for name in all_arm_names()}
        self.live_pose = {
            "elbow_right": -2846,
            "shoulder_vertical_left": -9727,
            "shoulder_horizontal_left": -8370,
            "wrist_left": -36,
            "shoulder_horizontal_right": 8871,
            "upper_arm_right_rotation": 309,
            "wrist_right": -171,
            "lower_arm_left_rotation": 86,
            "upper_arm_left_rotation": -480,
            "lower_arm_right_rotation": -630,
            "shoulder_vertical_right": -9976,
            "elbow_left": -549,
        }
        self.photo_pose = {
            "shoulder_horizontal_right": 9000,
            "upper_arm_right_rotation": -1791,
            "wrist_left": -651,
            "elbow_left": -342,
            "lower_arm_left_rotation": -18,
            "elbow_right": 405,
            "wrist_right": -180,
            "shoulder_vertical_left": -8300,
            "upper_arm_left_rotation": -189,
            "shoulder_horizontal_left": -8514,
            "shoulder_vertical_right": -8200,
            "lower_arm_right_rotation": -639,
        }
        self.logged_column_crossing_pose = {
            "shoulder_horizontal_right": 2057,
            "lower_arm_right_rotation": -630,
            "wrist_left": -154,
            "elbow_right": 225,
            "lower_arm_left_rotation": -18,
            "wrist_right": -180,
            "elbow_left": -3204,
            "upper_arm_right_rotation": -4716,
            "shoulder_vertical_left": -6298,
            "shoulder_vertical_right": -9000,
            "shoulder_horizontal_left": -8514,
            "upper_arm_left_rotation": -189,
        }
        self.near_bottom_border_pose = {
            "shoulder_vertical_right": -8103,
            "upper_arm_right_rotation": 60,
            "upper_arm_left_rotation": 1903,
            "wrist_left": -9,
            "shoulder_horizontal_right": 8049,
            "shoulder_vertical_left": -8200,
            "elbow_right": -103,
            "wrist_right": -171,
            "lower_arm_left_rotation": 1080,
            "elbow_left": 1920,
            "lower_arm_right_rotation": -567,
            "shoulder_horizontal_left": -7317,
        }

    def test_raised_bottom_volume_intersects_low_photo_pose(self):
        clearances = self.guard.clearances(self.photo_pose)

        self.assertLess(
            clearances[("lower_arm_left_rotation", "Volume 5")], 0
        )
        self.assertLess(clearances[("wrist_left", "Volume 5")], 0)

    def test_safe_command_is_allowed(self):
        result = self.guard.evaluate(self.zero, {"elbow_left": 500})
        self.assertTrue(result.allowed, result.reason)

    def test_bottom_volume_is_raised_another_150mm(self):
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        volume = next(
            obstacle
            for obstacle in config["obstacles"]
            if obstacle["name"] == "Volume 5"
        )

        self.assertEqual(volume["center_mm"][2], 331)
        self.assertEqual(
            volume["center_mm"][2] + (volume["size_mm"][2] / 2),
            461,
        )

    def test_live_pose_maps_both_wrists_inside_rear_boundary(self):
        left = {
            box.name: box.center for box in arm_boxes("left", self.live_pose)
        }["wrist_left"]
        right = {
            box.name: box.center for box in arm_boxes("right", self.live_pose)
        }["wrist_right"]

        self.assertLess(left[1], 125)
        self.assertLess(right[1], 125)

    def test_photo_pose_preserves_physical_arm_asymmetry(self):
        left = {
            box.name: box.center for box in arm_boxes("left", self.photo_pose)
        }["wrist_left"]
        right = {
            box.name: box.center for box in arm_boxes("right", self.photo_pose)
        }["wrist_right"]

        self.assertGreater(right[2] - left[2], 120)
        self.assertLess(abs(abs(right[0]) - abs(left[0])), 20)
        right_collisions = {
            pair
            for pair, clearance in self.guard.clearances(
                self.photo_pose, sides=("right",)
            ).items()
            if clearance <= 0
        }
        self.assertEqual(right_collisions, set())

    def test_elbow_motion_recalculates_downstream_wrist_position(self):
        initial = {
            box.name: box.center for box in arm_boxes("left", self.zero)
        }
        moved_positions = dict(self.zero)
        moved_positions["elbow_left"] = 1000
        moved = {
            box.name: box.center for box in arm_boxes("left", moved_positions)
        }

        self.assertNotEqual(initial["wrist_left"], moved["wrist_left"])

    def test_path_into_restricted_zone_is_blocked(self):
        targets = {
            "shoulder_vertical_left": -9000,
            "shoulder_horizontal_left": -9000,
        }
        result = self.guard.evaluate(
            self.zero,
            targets,
        )
        self.assertFalse(result.allowed)
        self.assertIn("would enter", result.reason)
        self.assertIsNotNone(result.safe_positions)
        self.assertNotEqual(result.safe_positions, targets)
        boundary = self.guard.evaluate(self.zero, result.safe_positions)
        self.assertTrue(boundary.allowed, boundary.reason)

    def test_start1_boundary_is_safe_after_hard_limit_clamp(self):
        current = {
            "shoulder_vertical_left": -9875,
            "upper_arm_right_rotation": -9,
            "lower_arm_right_rotation": 1834,
            "wrist_left": 1629,
            "elbow_right": -1209,
            "elbow_left": 1474,
            "shoulder_horizontal_right": 7963,
            "wrist_right": 7577,
            "shoulder_horizontal_left": -7128,
            "shoulder_vertical_right": -9169,
            "upper_arm_left_rotation": -612,
            "lower_arm_left_rotation": 4543,
        }
        targets = {
            "shoulder_horizontal_left": 0,
            "shoulder_vertical_left": -7200,
            "upper_arm_left_rotation": -9000,
            "elbow_left": 688,
            "lower_arm_left_rotation": 529,
            "wrist_left": 43,
            "shoulder_horizontal_right": -800,
            "shoulder_vertical_right": -9000,
            "upper_arm_right_rotation": 8200,
            "elbow_right": 18,
            "lower_arm_right_rotation": -274,
            "wrist_right": -76,
        }

        result = self.guard.evaluate(current, targets)
        self.assertFalse(result.allowed)
        hard_clamped = {
            name: max(-9000, min(9000, position))
            for name, position in result.safe_positions.items()
        }

        self.assertEqual(hard_clamped["shoulder_vertical_left"], -9000)
        self.assertEqual(hard_clamped["shoulder_vertical_right"], -9000)
        validation = self.guard.evaluate(current, hard_clamped)
        self.assertTrue(validation.allowed, validation.reason)

    def test_starting_inside_zone_can_move_out(self):
        result = self.guard.evaluate(
            self.live_pose,
            {"shoulder_vertical_left": 0},
        )
        self.assertTrue(result.allowed, result.reason)

    def test_starting_inside_zone_cannot_move_deeper(self):
        result = self.guard.evaluate(
            self.live_pose,
            {"upper_arm_right_rotation": 3000},
        )
        self.assertFalse(result.allowed)
        self.assertIn("deeper", result.reason)

    def test_escape_permission_is_scoped_to_exact_link_and_volume(self):
        initial = {
            ("lower_arm_right_rotation", "Volume 6"): -5.0,
            ("wrist_right", "Volume 6"): 2.0,
        }
        reason = self.guard._collision_reason(
            "right",
            {
                ("lower_arm_right_rotation", "Volume 6"): -4.0,
                ("wrist_right", "Volume 6"): -1.0,
            },
            initial,
            frozenset({("lower_arm_right_rotation", "Volume 6")}),
            set(),
        )

        self.assertEqual(reason, "wrist_right would enter Volume 6")

    def test_enclosure_dimensions_and_placement(self):
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        enclosure = config["enclosure"]
        obstacles = {
            obstacle["name"]: obstacle for obstacle in config["obstacles"]
        }

        self.assertEqual(enclosure["clear_size_mm"], [1800, 1000, 1200])
        self.assertEqual(enclosure["clear_center_mm"], [0, -375, 600])
        self.assertEqual(enclosure["rear_wall_y_mm"], 125)
        self.assertEqual(
            obstacles["Enclosure rear wall"]["center_mm"], [0, 150, 600]
        )
        self.assertEqual(
            obstacles["Enclosure front wall"]["center_mm"], [0, -900, 600]
        )

    def test_logged_shoulder_path_stops_before_upper_column(self):
        result = self.guard.evaluate(
            self.logged_column_crossing_pose,
            {"shoulder_horizontal_right": -7400},
        )

        self.assertFalse(result.allowed)
        self.assertIn("would enter", result.reason)
        self.assertGreater(
            result.safe_positions["shoulder_horizontal_right"], -7400
        )
        boundary = self.guard.evaluate(
            self.logged_column_crossing_pose, result.safe_positions
        )
        self.assertTrue(boundary.allowed, boundary.reason)

    def test_current_pose_baseline_clamps_left_upper_arm(self):
        result = self.guard.evaluate(
            self.live_pose, {"upper_arm_left_rotation": 5000}
        )

        self.assertFalse(result.allowed)
        self.assertGreater(result.safe_positions["upper_arm_left_rotation"], -480)
        self.assertLess(result.safe_positions["upper_arm_left_rotation"], 5000)

    def test_dynamic_range_stops_at_collision_boundary(self):
        minimum, maximum = self.guard.safe_joint_range(
            self.photo_pose,
            "upper_arm_right_rotation",
            -9000,
            9000,
        )

        self.assertGreaterEqual(minimum, -9000)
        self.assertLess(maximum, 0)
        self.assertGreaterEqual(
            maximum, self.photo_pose["upper_arm_right_rotation"]
        )
        self.assertTrue(
            self.guard.evaluate(
                self.photo_pose,
                {"upper_arm_right_rotation": minimum},
            ).allowed
        )
        self.assertTrue(
            self.guard.evaluate(
                self.photo_pose,
                {"upper_arm_right_rotation": maximum},
            ).allowed
        )

    def test_dynamic_range_never_exceeds_hard_motor_limits(self):
        minimum, maximum = self.guard.safe_joint_range(
            self.zero,
            "elbow_left",
            -1000,
            1000,
        )

        self.assertGreaterEqual(minimum, -1000)
        self.assertLessEqual(maximum, 1000)
        self.assertIsInstance(minimum, float)
        self.assertIsInstance(maximum, float)

    def test_left_arm_can_reach_the_raised_bottom_border(self):
        minimum, _ = self.guard.safe_joint_range(
            self.near_bottom_border_pose,
            "upper_arm_left_rotation",
            -9000,
            9000,
        )

        self.assertLess(minimum, -1000)
        boundary = self.guard.evaluate(
            self.near_bottom_border_pose,
            {"upper_arm_left_rotation": minimum},
        )
        self.assertTrue(boundary.allowed, boundary.reason)

        beyond_boundary = self.guard.evaluate(
            self.near_bottom_border_pose,
            {"upper_arm_left_rotation": -3000},
        )
        self.assertFalse(beyond_boundary.allowed)
        self.assertIn("Volume 5", beyond_boundary.reason)

    def test_real_wrist_pose_clears_lower_volume(self):
        current = dict(self.zero)
        current.update(
            shoulder_vertical_left=-9000,
            shoulder_horizontal_left=-4851,
            upper_arm_left_rotation=1980,
            elbow_left=2511,
            lower_arm_left_rotation=3206,
            wrist_left=-72,
            shoulder_vertical_right=-9000,
            shoulder_horizontal_right=8511,
            upper_arm_right_rotation=-639,
            elbow_right=2952,
            lower_arm_right_rotation=-90,
            wrist_right=411,
        )

        clearances = self.guard.clearances(current)

        self.assertGreater(clearances[("wrist_left", "Volume 5")], 10)
        self.assertGreater(clearances[("wrist_right", "Volume 5")], 10)


if __name__ == "__main__":
    unittest.main()
