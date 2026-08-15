import cv2
import redis
import json
import threading
import time
import os
import numpy as np
import base64
from dotenv import load_dotenv

load_dotenv()

# Prioritaskan MJPEG HTTP stream dari IP Webcam agar tidak berebut RTSP dengan C++
RTSP_URL = os.getenv("PHONE_IP_CAMERA_URL", "")
if RTSP_URL != "":
    RTSP_URL = RTSP_URL + "/video"
else:
    RTSP_URL = os.getenv("RTSP_URL", "0")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CHANNEL = "face_preprocessed_queue"

print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

latest_faces = []
preprocessed_face_img = None
lock = threading.Lock()

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=6):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw horizontal lines
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
    # Draw vertical lines
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

def redis_listener():
    global preprocessed_face_img
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNEL)
    print(f"Subscribed to Redis channel: {CHANNEL}")
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                
                # Decode preprocessed face
                if 'face_image_base64' in data and data['face_image_base64']:
                    raw_bytes = base64.b64decode(data['face_image_base64'])
                    float_arr = np.frombuffer(raw_bytes, dtype=np.float32)
                    if float_arr.size == 112 * 112 * 3:
                        float_img = float_arr.reshape((112, 112, 3))
                        # Denormalize from [-1, 1] to [0, 255]
                        uint8_img = ((float_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        with lock:
                            preprocessed_face_img = uint8_img
                
                with lock:
                    # Simpan data box yang dikirim oleh program C++
                    latest_faces.append({
                        'box': data['bounding_box'],
                        'conf': data['confidence_score'],
                        'time': time.time()
                    })
            except Exception as e:
                print(f"Error parsing message: {e}")

# Jalankan Redis listener di thread terpisah agar tidak memblokir video
threading.Thread(target=redis_listener, daemon=True).start()

print(f"Opening Stream: {RTSP_URL}")
try:
    source = int(RTSP_URL)
except ValueError:
    source = RTSP_URL

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Gagal membuka stream! Cek kembali koneksi HP Anda.")
    exit(1)

cv2.namedWindow("Real-Time ML Viewer", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal baca frame, retrying...")
        time.sleep(1)
        cap = cv2.VideoCapture(source)
        continue

    current_time = time.time()
    
    with lock:
        # Hanya tampilkan kotak yang baru dideteksi (kurang dari 0.5 detik yang lalu)
        latest_faces = [f for f in latest_faces if current_time - f['time'] < 0.5]
        
        for face in latest_faces:
            box = face['box']
            x = int(box['x'])
            y = int(box['y'])
            w = int(box['width'])
            h = int(box['height'])
            conf = face['conf']
            
            # Gambar kotak deteksi wajah putus-putus warna Cyan/Teal seperti C++
            draw_dashed_rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1, 6)
            cv2.putText(frame, f"Face: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Gambar preprocessed face 112x112 di pojok kanan atas
        if preprocessed_face_img is not None:
            ph, pw = preprocessed_face_img.shape[:2]
            fh, fw = frame.shape[:2]
            if fh >= ph and fw >= pw:
                # Add a small border
                bordered = cv2.copyMakeBorder(preprocessed_face_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 0))
                bph, bpw = bordered.shape[:2]
                frame[10:10+bph, fw-10-bpw:fw-10] = bordered
                cv2.putText(frame, "Preprocessed", (fw-10-bpw, 10+bph+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    cv2.imshow("Real-Time ML Viewer", frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
