import importlib.util
import sys
import unittest
from pathlib import Path


MOTORS_ROOT = Path(__file__).parents[1]
TOOLS_ROOT = MOTORS_ROOT / "tools"
sys.path.insert(0, str(MOTORS_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

spec = importlib.util.spec_from_file_location(
    "untwist_right_upper_arm",
    TOOLS_ROOT / "untwist_right_upper_arm.py",
)
untwist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(untwist)


class UntwistToolTest(unittest.TestCase):
    def test_encoder_delta_increases_across_wrap(self):
        self.assertEqual(untwist.signed_encoder_delta(4090, 10), 16)

    def test_encoder_delta_detects_reverse_motion_across_wrap(self):
        self.assertEqual(untwist.signed_encoder_delta(10, 4090), -16)

    def test_encoder_delta_without_wrap(self):
        self.assertEqual(untwist.signed_encoder_delta(1000, 1100), 100)
        self.assertEqual(untwist.signed_encoder_delta(1100, 1000), -100)


if __name__ == "__main__":
    unittest.main()
