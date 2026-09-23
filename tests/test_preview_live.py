import threading
import time
import unittest
from tools.preview_pi import BoxSmoother, LiveDetection, fresh_result, overlay
from unittest.mock import Mock


class PreviewLiveTest(unittest.TestCase):
    def test_overlay_draws_box_but_no_landmark_dots(self):
        cv2 = Mock()
        cv2.FONT_HERSHEY_SIMPLEX = 0
        frame = Mock()
        frame.copy.return_value.shape = (720, 1280, 3)
        result = dict(inference_ms=100, bounding_boxes=[dict(
            bounding_box=dict(x=10, y=10, width=50, height=50),
            confidence_score=0.9, landmarks=[[20, 20]]*5)])
        overlay(cv2, frame, result, 30, 100)
        self.assertGreater(cv2.rectangle.call_count, 0)
        cv2.circle.assert_not_called()

    def test_stale_and_reconnected_results_are_not_displayed(self):
        result = dict(source_epoch=2, source_received=10)
        self.assertIs(fresh_result(result, 10.2, 2, 10.3), result)
        self.assertIsNone(fresh_result(result, 12, 2, 12))
        self.assertIsNone(fresh_result(result, 10.2, 3, 10.3))
        self.assertIsNone(fresh_result(result, 9, 2, 10.3))

    def test_empty_detection_clears_smoothed_boxes(self):
        smoother = BoxSmoother()
        face = dict(bounding_box=dict(x=10, y=10, width=50, height=50))
        smoother.apply([face], 1)
        self.assertEqual(smoother.apply([], 1.1), [])
        self.assertEqual(smoother.boxes, [])

    def test_disjoint_face_does_not_slide_from_old_person(self):
        smoother = BoxSmoother()
        def face(x):
            return dict(bounding_box=dict(x=x, y=10, width=50, height=50))
        smoother.apply([face(0)], 1)
        self.assertEqual(smoother.apply([face(500)], 1.02)[0]["bounding_box"]["x"], 500)

    def test_slow_inference_skips_intermediate_frames(self):
        entered, release, next_entered = threading.Event(), threading.Event(), threading.Event()
        calls = []

        class Camera:
            sequence = 1
            def latest(self, previous):
                sequence = self.sequence
                return (sequence, sequence, time.monotonic(), 1) if sequence != previous else None

        class Detector:
            def detect(self, frame):
                calls.append(frame)
                if len(calls) == 1:
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError("test timed out")
                else:
                    next_entered.set()
                return {"bounding_boxes": []}

        camera = Camera()
        worker = LiveDetection(Detector(), camera, 30)
        try:
            self.assertTrue(entered.wait(1))
            camera.sequence = 2
            camera.sequence = 3
            release.set()
            self.assertTrue(next_entered.wait(1))
        finally:
            release.set()
            worker.close()
        self.assertEqual(calls, [1, 3])


if __name__ == "__main__":
    unittest.main()
