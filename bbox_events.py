"""Frame-level bbox messages from Pi nodes; independent of image payloads."""
import math


def parse_bbox_event(data):
    """Validate the public envelope without turning a corrupt message into a clear event."""
    if not isinstance(data, dict) or not isinstance(data.get("camera_id"), str) or not data["camera_id"]:
        raise ValueError("Missing camera_id")
    for key in ("frame_width", "frame_height"):
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Invalid {key}")
    boxes = data.get("bounding_boxes")
    if not isinstance(boxes, list):
        raise ValueError("bounding_boxes must be an array")
    for entry in boxes:
        if not isinstance(entry, dict) or not isinstance(entry.get("bounding_box"), dict):
            raise ValueError("Invalid bounding_box")
        box = entry["bounding_box"]
        for key in ("x", "y", "width", "height"):
            value = box.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid box {key}")
        if box["width"] <= 0 or box["height"] <= 0:
            raise ValueError("Invalid box dimensions")
    # No regrouping across frame IDs; [] must reach the frontend unchanged.
    fields = ("schema_version", "camera_id", "stream_path", "session_id", "frame_id",
              "received_at_ms", "timestamp_ms", "frame_width", "frame_height", "bounding_boxes")
    return {key: data[key] for key in fields if key in data}
