import importlib
import json
import os
import unittest
from collections import deque
from unittest.mock import patch

with patch.dict(os.environ, {"BBOX_CHANNEL": "face_detection_queue"}), patch("threading.Thread.start"):
    api = importlib.import_module("api_server")


class PiApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_frame_events_reach_sse_including_clear(self):
        class Request:
            async def is_disconnected(self):
                return False

        response = await api.stream_live_bbox(Request())
        iterator = response.body_iterator
        self.assertIn("connected", await anext(iterator))
        events = [dict(camera_id="cam01", frame_width=640, frame_height=480,
                       frame_id=i, bounding_boxes=boxes)
                  for i, boxes in enumerate([
                      [{"bounding_box": dict(x=5, y=6, width=20, height=30), "name": "Unknown"}], []])]
        messages = [{"type": "message", "data": json.dumps(e)} for e in events]
        with patch.object(api, "redis_messages", return_value=iter(messages)):
            api.bbox_listener()
        for expected in events:
            actual = json.loads((await anext(iterator)).removeprefix("data: "))
            self.assertEqual(actual, expected)
        await iterator.aclose()
        self.assertEqual(api.bbox_subscribers, [])

    async def test_face_crop_does_not_duplicate_bbox_in_frame_mode(self):
        queue = deque()
        api.bbox_subscribers.append(queue)
        message = {"type": "message", "data": json.dumps(dict(
            camera_id="cam01", frame_id=12, session_id="boot", landmarks=[[1, 2]],
            face_image_base64="jpeg", image_format="jpeg_base64_112x112"))}
        try:
            with patch.object(api, "redis_messages", return_value=iter([message])):
                api.redis_listener()
            self.assertEqual(len(queue), 0)
            self.assertEqual(api.latest_result["frame_id"], 12)
            self.assertEqual(api.latest_result["face_image_base64"], "jpeg")
        finally:
            api.bbox_subscribers.remove(queue)


if __name__ == "__main__":
    unittest.main()
