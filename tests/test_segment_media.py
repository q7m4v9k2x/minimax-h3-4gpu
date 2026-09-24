"""Integration tests for the three-clip H3 media assembler.

The fixture is intentionally tiny (32x24) but has the same 124-frame/24-fps
shape as a production segment.  FFmpeg creates three independently encoded
clips with audio; the test then verifies that the assembled stream contains
all 360 decoded frames in exactly the original order.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.segment_media import join_segments  # noqa: E402


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
FRAME_COUNT = 124
FPS = 24
SEGMENT_SECONDS = FRAME_COUNT / FPS


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required for media integration tests")
class SegmentMediaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="h3-segment-test-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120)

    def _make_segment(self, index: int, color: str) -> Path:
        output = self.root / f"segment-{index}.mp4"
        # Different solid colours make segment boundaries visible in decoded
        # hashes while keeping this fixture fast and deterministic.
        self._run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=32x24:r={FPS}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={440 + index * 110}:sample_rate=48000",
                "-frames:v",
                str(FRAME_COUNT),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-shortest",
                str(output),
            ]
        )
        return output

    def _frame_hashes(self, path: Path) -> list[str]:
        result = self._run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-vsync",
                "0",
                "-f",
                "framemd5",
                "-",
            ]
        )
        hashes: list[str] = []
        for line in result.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            hashes.append(line.rsplit(",", 1)[-1].strip())
        return hashes

    def _probe(self, path: Path) -> dict:
        result = self._run(
            [
                FFPROBE,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
        return json.loads(result.stdout)

    def test_three_segments_join_to_360_frames_in_order(self) -> None:
        segments = [self._make_segment(index, color) for index, color in enumerate(("red", "green", "blue"))]
        expected_hashes = [frame_hash for segment in segments for frame_hash in self._frame_hashes(segment)][:360]
        self.assertEqual(len(expected_hashes), 360)

        progress: list[tuple] = []
        result = join_segments(
            [{"preview": str(path)} for path in segments],
            self.root / "joined",
            progress_cb=lambda *event: progress.append(event),
        )

        joined = Path(result["preview"])
        self.assertTrue(joined.is_file())
        self.assertEqual(result["verification"]["preview"]["frames"], 360)
        self.assertAlmostEqual(result["verification"]["preview"]["duration"], 15.0, places=2)
        self.assertIsNone(result["lossless"])
        self.assertEqual(self._frame_hashes(joined), expected_hashes)

        info = self._probe(joined)
        video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
        audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
        self.assertEqual(int(video["nb_read_frames"]), 360)
        self.assertEqual(video["avg_frame_rate"], "24/1")
        self.assertAlmostEqual(float(video["duration"]), 15.0, places=2)
        self.assertAlmostEqual(float(audio["duration"]), 15.0, delta=0.08)
        self.assertTrue(progress)
        self.assertEqual(progress[-1][0], "encode")

    def test_optional_lossless_output_is_verified(self) -> None:
        segments = [self._make_segment(index, color) for index, color in enumerate(("yellow", "cyan", "magenta"))]
        result = join_segments(
            [{"preview": str(path), "lossless": str(path)} for path in segments],
            self.root / "joined-lossless",
            lossless=True,
        )

        self.assertTrue(Path(result["preview"]).is_file())
        self.assertTrue(Path(result["lossless"]).is_file())
        self.assertEqual(result["verification"]["lossless"]["frames"], 360)
        self.assertAlmostEqual(result["verification"]["lossless"]["duration"], 15.0, places=2)

    def test_missing_segment_fails_before_encoding(self) -> None:
        existing = self._make_segment(0, "red")
        missing = self.root / "does-not-exist.mp4"
        with self.assertRaises(ValueError):
            join_segments(
                [{"preview": str(existing)}, {"preview": str(missing)}, {"preview": str(existing)}],
                self.root / "joined",
            )
        self.assertFalse((self.root / "joined" / "h3-preview.mp4").exists())


if __name__ == "__main__":
    unittest.main()
