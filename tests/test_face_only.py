import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from face_service.app import app
from face_service.hailo_scrfd import decode, preprocess


class FaceOnlyTests(unittest.TestCase):
    def test_top_left_padding_and_rgb(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        frame[20, 30] = (10, 20, 240)
        rgb = preprocess(frame)
        self.assertEqual(rgb.shape, (640, 640, 3))
        self.assertEqual(rgb[20, 30].tolist(), [240, 20, 10])
        self.assertEqual(rgb[500, 30].tolist(), [0, 0, 0])

    def test_raw_heads_return_face_box_and_ignore_padding(self):
        outputs = {}
        for level, side in enumerate((80, 40, 20)):
            outputs[f"scrfd/output_layer{level * 3 + 1}"] = np.full((side, side, 2), -20., np.float32)
            outputs[f"scrfd/output_layer{level * 3 + 2}"] = np.zeros((side, side, 20), np.float32)
            outputs[f"scrfd/output_layer{level * 3 + 3}"] = np.ones((side, side, 8), np.float32)
        outputs["scrfd/output_layer1"][10, 10, :] = 5
        outputs["scrfd/output_layer1"][60, 10, 0] = 5
        rows = decode(outputs, 640, 360)
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0][:4], [72, 72, 16, 16])

    def test_service_is_idle_until_enabled(self):
        with TestClient(app) as client:
            self.assertFalse(client.get("/health").json()["ready"])
            response = client.post("/infer", content=b"not a jpeg")
            self.assertEqual(response.status_code, 400)
            self.assertFalse(client.post("/enabled", json={"enabled": False}).json()["ready"])


if __name__ == "__main__":
    unittest.main()
