"""Contract tests for the bounded H3 video request schema.

The tests deliberately exercise the public ``normalize_request`` contract rather
than the HTTP layer.  API and CLI callers can therefore share these checks
without requiring a running model service.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.video_request import normalize_request  # noqa: E402


class NormalizeRequestTests(unittest.TestCase):
    def assertInvalid(self, payload: object) -> None:
        with self.assertRaises(ValueError):
            normalize_request(payload)

    def test_defaults_keep_five_second_single_segment_compatibility(self) -> None:
        result = normalize_request({"prompt": "  a small red fox  "})

        self.assertEqual(result["prompt"], "a small red fox")
        self.assertEqual(result["width"], 864)
        self.assertEqual(result["height"], 480)
        self.assertEqual(result["frames"], 124)
        self.assertEqual(result["steps"], 20)
        self.assertEqual(result["fps"], 24)
        self.assertEqual(result["duration"], 5)
        self.assertEqual(result["segments"], 1)
        self.assertFalse(result["lossless"])
        self.assertGreaterEqual(result["seed"], 0)
        self.assertLessEqual(result["seed"], 2**32 - 1)

    def test_fifteen_second_request_requires_three_segments(self) -> None:
        result = normalize_request(
            {
                "prompt": "a cinematic mountain valley",
                "duration": 15,
                "segments": 3,
                "width": 1152,
                "height": 640,
                "steps": 30,
                "seed": 17,
                "lossless": True,
            }
        )

        self.assertEqual(result["duration"], 15)
        self.assertEqual(result["segments"], 3)
        self.assertEqual(result["width"], 1152)
        self.assertEqual(result["height"], 640)
        self.assertEqual(result["steps"], 30)
        self.assertEqual(result["seed"], 17)
        self.assertTrue(result["lossless"])

    def test_all_supported_orientations_are_accepted(self) -> None:
        for width, height in ((864, 480), (480, 864), (1152, 640), (640, 1152)):
            with self.subTest(width=width, height=height):
                result = normalize_request({"prompt": "ok", "width": width, "height": height})
                self.assertEqual((result["width"], result["height"]), (width, height))

    def test_seed_minus_one_is_replaced_by_an_unsigned_seed(self) -> None:
        result = normalize_request({"prompt": "ok", "seed": -1})
        self.assertIsInstance(result["seed"], int)
        self.assertGreaterEqual(result["seed"], 0)
        self.assertLessEqual(result["seed"], 2**32 - 1)

    def test_unknown_fields_are_ignored_for_forward_compatibility(self) -> None:
        result = normalize_request({"prompt": "ok", "client_request_id": "abc", "future": {}})
        self.assertEqual(result["prompt"], "ok")
        self.assertNotIn("future", result)

    def test_prompt_is_required_and_bounded(self) -> None:
        for payload in (
            {},
            {"prompt": ""},
            {"prompt": "   "},
            {"prompt": 123},
            {"prompt": "x" * 8001},
            None,
            [],
        ):
            with self.subTest(payload_type=type(payload).__name__):
                self.assertInvalid(payload)

        self.assertEqual(len(normalize_request({"prompt": "x" * 8000})["prompt"]), 8000)

    def test_invalid_dimensions_and_sampling_values_are_rejected(self) -> None:
        invalid = (
            {"prompt": "ok", "width": 512, "height": 512},
            {"prompt": "ok", "width": True},
            {"prompt": "ok", "height": "480"},
            {"prompt": "ok", "frames": 360},
            {"prompt": "ok", "frames": True},
            {"prompt": "ok", "steps": 10},
            {"prompt": "ok", "steps": 40},
            {"prompt": "ok", "steps": False},
            {"prompt": "ok", "seed": -2},
            {"prompt": "ok", "seed": 2**32},
            {"prompt": "ok", "seed": 1.5},
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertInvalid(payload)

    def test_duration_and_segment_pairs_are_strict(self) -> None:
        for payload in (
            {"prompt": "ok", "duration": 15, "segments": 1},
            {"prompt": "ok", "duration": 5, "segments": 3},
            {"prompt": "ok", "duration": 10},
            {"prompt": "ok", "segments": 2},
            {"prompt": "ok", "duration": True},
            {"prompt": "ok", "segments": True},
        ):
            with self.subTest(payload=payload):
                self.assertInvalid(payload)

    def test_lossless_must_be_a_real_boolean(self) -> None:
        for value in (0, 1, "true", None, [], {}):
            with self.subTest(value=value):
                self.assertInvalid({"prompt": "ok", "lossless": value})


if __name__ == "__main__":
    unittest.main()
