import requests
import sseclient
import json
import time
from datetime import datetime

STREAM_URL = "http://192.168.77.172:8000/api/v1/faces/stream"
MAX_EVENTS = 10

print(f"[probe] Connecting to SSE stream: {STREAM_URL}")
try:
    resp = requests.get(STREAM_URL, stream=True, headers={"Accept": "text/event-stream"}, timeout=15)
    client = sseclient.SSEClient(resp)
    count = 0
    prev_t = None

    for event in client.events():
        if not event.data:
            continue
        now = time.time()
        try:
            data = json.loads(event.data)
            cam = data.get("camera_id", "?")
            img_b64 = data.get("face_image_base64", "")
            event_ts = data.get("timestamp") or data.get("detected_at") or "?"
            approx_kb = len(img_b64) * 3 / 4 / 1024

            if prev_t is not None:
                gap_ms = (now - prev_t) * 1000
                gap_str = f"{gap_ms:.0f}ms since last"
            else:
                gap_str = "first event"

            wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{wall}] cam={cam}  payload={approx_kb:.1f}KB  event_ts={event_ts}  ({gap_str})")
            prev_t = now
            count += 1
            if count >= MAX_EVENTS:
                break
        except Exception as parse_err:
            print(f"[parse error] {parse_err}")

    print(f"\n[probe] Done. {count} events captured.")
except Exception as e:
    print(f"[probe] Failed to connect: {e}")
