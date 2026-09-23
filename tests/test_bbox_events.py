import unittest
from bbox_events import parse_bbox_event


class BboxEventsTest(unittest.TestCase):
    def event(self, frame=1, boxes=None):
        return dict(camera_id="cam01", frame_id=frame, frame_width=1920,
                    frame_height=1080, bounding_boxes=[] if boxes is None else boxes)

    def test_empty_frame_is_forwarded(self):
        self.assertEqual(parse_bbox_event(self.event())["bounding_boxes"], [])

    def test_frames_are_not_merged(self):
        box = {"bounding_box": dict(x=10, y=20, width=30, height=40), "name": "Unknown"}
        a = parse_bbox_event(self.event(1, [box, box]))
        b = parse_bbox_event(self.event(2))
        self.assertEqual(len(a["bounding_boxes"]), 2)
        self.assertEqual(b["bounding_boxes"], [])
        self.assertNotEqual(a["frame_id"], b["frame_id"])

    def test_invalid_message_does_not_clear_boxes(self):
        bad = [None, {}, self.event(boxes="wrong"), self.event(boxes=[{}])]
        for event in bad:
            with self.subTest(event=event), self.assertRaises(ValueError):
                parse_bbox_event(event)

    def test_invalid_geometry(self):
        for width in (0, -1, float("nan"), float("inf"), True):
            box = {"bounding_box": dict(x=0, y=0, width=width, height=40)}
            with self.subTest(width=width), self.assertRaises(ValueError):
                parse_bbox_event(self.event(boxes=[box]))

    def test_metadata_kept_without_image(self):
        event = self.event()
        event.update(stream_path="cam01", session_id="boot-1", face_image_base64="private")
        parsed = parse_bbox_event(event)
        self.assertEqual(parsed["stream_path"], "cam01")
        self.assertNotIn("face_image_base64", parsed)


if __name__ == "__main__":
    unittest.main()
