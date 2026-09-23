"""Exercise the native preview transport without camera, GUI, or Redis."""
import json
import os
from pathlib import Path
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / ("build-pi-check/Release/face_detector_pi.exe" if os.name == "nt" else "build-pi/face_detector_pi")


@unittest.skipUnless(EXE.is_file(), "Build face_detector_pi first")
class PreviewProtocolTest(unittest.TestCase):
    def run_worker(self, data):
        return subprocess.run([str(EXE), "--config", "deploy/pi/pi.env.example", "--preview-stdio"],
                              input=data, capture_output=True, cwd=ROOT, timeout=30)

    def test_multiple_frames_keep_original_dimensions_and_clear_boxes(self):
        frame = struct.pack("<III", 320, 240, 320*240*3) + bytes(320*240*3)
        result = self.run_worker(frame + frame)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        ready, first, second = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(ready["ready"])
        self.assertEqual((first["frame_width"], first["frame_height"]), (320, 240))
        self.assertEqual(first["bounding_boxes"], [])
        self.assertEqual(second["frame_id"], first["frame_id"]+1)

    def test_reject_invalid_size_before_allocating(self):
        result = self.run_worker(struct.pack("<III", 0xFFFFFFFF, 20, 0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"Invalid preview dimensions", result.stderr)

    def test_truncated_pixels_fail_cleanly(self):
        result = self.run_worker(struct.pack("<III", 20, 20, 1200) + b"short")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"Truncated preview pixels", result.stderr)


if __name__ == "__main__":
    unittest.main()
