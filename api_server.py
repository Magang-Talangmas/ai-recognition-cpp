import json
import time
import os
import asyncio
import threading
from collections import deque
from dotenv import load_dotenv
import redis as redis_lib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

load_dotenv()

REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", 6379))
REDIS_CH    = os.getenv("REDIS_CHANNEL", "face_preprocessed_queue")
API_PORT    = int(os.getenv("API_PORT", 8000))
API_HOST    = os.getenv("API_HOST", "0.0.0.0")

# ─────────────────────────────────────────────
# In-memory store
# ─────────────────────────────────────────────
MAX_HISTORY     = 50
face_history    = deque(maxlen=MAX_HISTORY)
latest_result   = {}
sse_subscribers = []

latest_video_frame = None
video_subscribers = []
lock            = threading.Lock()


# ─────────────────────────────────────────────
# Redis subscriber (background thread)
# Mendengarkan hasil dari viewer.py / C++ exe
# ─────────────────────────────────────────────
def redis_listener():
    global latest_result
    r      = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe(REDIS_CH)
    print(f"[Redis] Listening on channel: {REDIS_CH}")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            data  = json.loads(message["data"])
            entry = {
                "received_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "camera_id":         data.get("camera_id", "unknown"),
                "timestamp_ms":      data.get("timestamp_ms"),
                "face_index":        data.get("face_index", 0),
                "confidence_score":  data.get("confidence_score"),
                "bounding_box":      data.get("bounding_box"),
                "face_image_base64": data.get("face_image_base64"),
            }
            with lock:
                face_history.append(entry)
                latest_result = entry
                for q in list(sse_subscribers):
                    try:
                        q.append(entry)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Redis] Parse error: {e}")

def video_listener():
    global latest_video_frame
    r      = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0) # raw bytes
    pubsub = r.pubsub()
    pubsub.subscribe("face_video_stream")
    print(f"[Redis] Listening on channel: face_video_stream")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            latest_video_frame = message["data"]
            # Trigger subscribers
            for q in list(video_subscribers):
                try:
                    q.append(latest_video_frame)
                except Exception:
                    pass
        except Exception as e:
            pass

threading.Thread(target=redis_listener, daemon=True).start()
threading.Thread(target=video_listener, daemon=True).start()


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Talangmas Face Preprocessing API",
    description=(
        "HTTP API for the Face Recognition team to consume preprocessed face tensors "
        "produced by the AI preprocessing pipeline (SCRFD + CLAHE + Normalize).\n\n"
        "Flow: viewer.py / C++ exe -> Redis -> api_server.py -> Face Recog Team"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
def health_check():
    """Cek apakah server dan Redis aktif."""
    try:
        redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT).ping()
        redis_status = "connected"
    except Exception:
        redis_status = "disconnected"
    return {
        "status":       "ok",
        "redis":        redis_status,
        "channel":      REDIS_CH,
        "faces_stored": len(face_history),
    }


@app.get("/api/v1/faces/latest", tags=["Faces"], summary="Get latest detected face")
def get_latest_face():
    """
    Ambil hasil preprocessing wajah yang PALING TERAKHIR diterima.
    """
    with lock:
        result = dict(latest_result)
    if not result:
        return JSONResponse(status_code=404, content={"detail": "No face detected yet"})
    return result


@app.get("/api/v1/faces/history", tags=["Faces"], summary="Get face detection history")
def get_face_history(limit: int = 10):
    """
    Ambil histori N wajah terakhir yang sudah dipreprocessing.

    Query params:
    - limit: jumlah entri (max 50, default 10)
    """
    limit = min(limit, MAX_HISTORY)
    with lock:
        entries = list(face_history)[-limit:]
    return {"count": len(entries), "faces": entries}


@app.get("/api/v1/faces/test-array", tags=["Testing"], summary="Get raw array of objects for browser testing")
def get_test_faces_array(limit: int = 10):
    """
    Sama seperti history, namun endpoint ini langsung mereturn **Array of Objects** `[{}, {}]` 
    bukan object `{"faces": []}`. Dibuat khusus agar mudah dipanggil/dibaca langsung dari browser untuk testing.
    """
    limit = min(limit, MAX_HISTORY)
    with lock:
        entries = list(face_history)[-limit:]
    return entries


@app.get("/api/v1/faces/stream", tags=["Faces"], summary="Real-time SSE stream")
async def stream_faces(request: Request):
    """
    Server-Sent Events (SSE): real-time push setiap ada wajah baru terdeteksi.

    Face Recog team connect sekali, data langsung di-push tanpa polling.

    Contoh konsumsi Python:
        import requests, sseclient
        res = requests.get('http://<ip>:8000/api/v1/faces/stream', stream=True)
        for event in sseclient.SSEClient(res).events():
            print(event.data)

    Contoh konsumsi JavaScript:
        const es = new EventSource('http://<ip>:8000/api/v1/faces/stream');
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    q = deque(maxlen=100)
    with lock:
        sse_subscribers.append(q)

    async def event_generator():
        try:
            yield 'data: {"status": "connected"}\n\n'
            while True:
                if await request.is_disconnected():
                    break
                with lock:
                    items = list(q)
                    q.clear()
                for item in items:
                    payload = dict(item)
                    yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.05)
        finally:
            with lock:
                if q in sse_subscribers:
                    sse_subscribers.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/v1/video_feed", tags=["Testing"], summary="MJPEG Video Stream with Bounding Boxes")
async def video_feed(request: Request):
    """
    Menyediakan video live stream (MJPEG) yang bisa dipasang langsung di tag <img> HTML.
    Berisi kotak bounding box hasil deteksi dari program C++.
    """
    q = deque(maxlen=10)
    with lock:
        video_subscribers.append(q)

    async def frame_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                frame_data = None
                with lock:
                    if len(q) > 0:
                        frame_data = q[-1]
                        q.clear()
                
                if frame_data:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                await asyncio.sleep(0.06) # ~15 FPS
        finally:
            with lock:
                if q in video_subscribers:
                    video_subscribers.remove(q)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=================================================")
    print("  Talangmas Face Preprocessing API Server       ")
    print("=================================================")
    print(f"  Listening  : http://{API_HOST}:{API_PORT}")
    print(f"  Swagger UI : http://localhost:{API_PORT}/docs")
    print(f"  Redis      : {REDIS_HOST}:{REDIS_PORT} / {REDIS_CH}")
    print("=================================================")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
