import cv2
import redis
import json
import threading
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Gunakan fallback ke 0 (Webcam lokal) jika di .env tidak ada
RTSP_URL = os.getenv("RTSP_URL", "0")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CHANNEL = "face_preprocessed_queue"

print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

latest_faces = []
lock = threading.Lock()

def redis_listener():
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNEL)
    print(f"Subscribed to Redis channel: {CHANNEL}")
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
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
            
            # Gambar kotak deteksi wajah warna hijau
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Real-Time ML Viewer", frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
