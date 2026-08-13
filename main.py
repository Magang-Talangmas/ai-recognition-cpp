import cv2
import time
import asyncio
import redis
import json
import uuid
import requests
import numpy as np
import base64
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load Environment Variables dari file .env
load_dotenv()

# Jika menggunakan C++ core yang sudah di-compile dengan pybind11:
# import face_engine_core
# dummy_engine = face_engine_core.FaceEngine("models/scrfd.onnx", "models/arcface.onnx")

# Konfigurasi dari Environment
RTSP_URL = os.getenv("RTSP_URL_SUBSTREAM", "rtsp://localhost:8554/live")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
COOLDOWN_SECONDS = 60
SIMILARITY_THRESHOLD = 0.65
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:3000/api/recognition/thumbnail")
ML_API_KEY = os.getenv("ML_API_KEY", "")
TEMP_SNAPSHOT_PATH = os.getenv("TEMP_SNAPSHOT_PATH", "./data/snapshots")
ENROLL_DATA_PATH = os.getenv("ENROLL_DATA_PATH", "./data/enrolled")

# Pastikan folder penyimpanan ada
os.makedirs(TEMP_SNAPSHOT_PATH, exist_ok=True)
os.makedirs(ENROLL_DATA_PATH, exist_ok=True)

# Inisialisasi Redis Client (Berfungsi sebagai Middleware Debounce & Message Broker)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# Mock Face DB (Seharusnya diambil dari Backend/DB saat startup)
FACE_DB = {
    "EMP001": np.random.rand(512).astype(np.float32), # Dummy embedding 512d
    "EMP002": np.random.rand(512).astype(np.float32)
}

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

async def process_frame(frame):
    """
    Fungsi asynchronous untuk memproses frame: deteksi, crop, dan embedding.
    """
    # Simulasi latency inferensi C++
    await asyncio.sleep(0.05) 
    
    # TODO: Panggil modul C++ untuk mendeteksi wajah
    # faces = face_engine_core.detect(frame)
    
    # Mock deteksi wajah secara acak untuk keperluan demo
    if np.random.random() > 0.8:
        # Anggap wajah terdeteksi
        # TODO: Panggil C++ engine untuk embedding
        # embedding = face_engine_core.extract_feature(cropped_face)
        detected_embedding = np.random.rand(512).astype(np.float32) 
        
        best_match_id = None
        best_score = 0.0
        
        # Pencocokan dengan Face DB
        for emp_id, db_emb in FACE_DB.items():
            score = cosine_similarity(detected_embedding, db_emb)
            if score > best_score:
                best_score = score
                best_match_id = emp_id
                
        if best_score >= SIMILARITY_THRESHOLD:
            return best_match_id, best_score, frame
        else:
            return "UNKNOWN", best_score, frame
            
    return None, 0.0, None

async def save_thumbnail_async(employee_id, frame):
    """Simpan thumbnail dan mock Hit API Backend"""
    filename = os.path.join(TEMP_SNAPSHOT_PATH, f"{employee_id}_{int(time.time())}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[Async] Snapshot tersimpan di: {filename}")
    
    # Hit API Backend (Mock)
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "employeeId": employee_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "imageBase64": base64_image
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ML_API_KEY}"
        }
        
        # Uncomment baris di bawah jika backend sudah ada
        # response = requests.post(BACKEND_API_URL, json=payload, headers=headers, timeout=3)
        # print(f"[Backend Hit] Response: {response.status_code}")
        print(f"[Backend API Hit] Simulasi POST ke {BACKEND_API_URL}")
    except Exception as e:
        print(f"[Backend API Error] {e}")

async def handle_recognition(employee_id, confidence, frame):
    if employee_id == "UNKNOWN":
        print(f"[Alert] Wajah tidak dikenal terdeteksi!")
        # Bisa dipublish ke topik khusus unknown_faces
        return

    redis_key = f"cooldown:{employee_id}"
    
    # Cek Cooldown di Redis
    try:
        if r.exists(redis_key):
            print(f"[Debounce] Karyawan {employee_id} sedang dalam masa cooldown. Skip event.")
            return
            
        print(f"[Recognition] Karyawan {employee_id} dikenali (Score: {confidence:.2f})")
        
        # Set Cooldown
        r.setex(redis_key, COOLDOWN_SECONDS, "1")
    except redis.exceptions.ConnectionError:
        print(f"[Redis Offline] Gagal terhubung ke Redis, mengabaikan cooldown untuk {employee_id}.")
        print(f"[Recognition] Karyawan {employee_id} dikenali (Score: {confidence:.2f})")
    
    # Trigger Async Task simpan thumbnail
    asyncio.create_task(save_thumbnail_async(employee_id, frame))
    
    # Publish Event ke Message Broker (Redis PubSub)
    event_payload = {
        "eventId": str(uuid.uuid4()),
        "employeeId": employee_id,
        "confidence": float(confidence),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eventType": "auto_recognition"
    }
    
    try:
        r.publish("recognition_events", json.dumps(event_payload))
        print(f"[Message Broker] Event dipublish: {event_payload['eventId']}")
    except redis.exceptions.ConnectionError:
        print(f"[Message Broker Offline] Gagal mem-publish event (Redis down). Payload: {event_payload}")

async def main():
    print(f"Membuka stream RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Gagal membuka RTSP stream. Pastikan MediaMTX berjalan.")
        # Fallback ke webcam lokal untuk testing jika RTSP mati
        print("Fallback ke webcam lokal (0)...")
        cap = cv2.VideoCapture(0)
    
    print("Memulai pemrosesan stream...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame, mencoba reconnect...")
            await asyncio.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue
            
        frame_count += 1
        
        # Proses frame setiap n frame untuk menghemat CPU (misal 5 FPS dari 30 FPS)
        if frame_count % 6 == 0: 
            # Pemrosesan Asynchronous
            employee_id, confidence, processed_frame = await process_frame(frame)
            
            if employee_id:
                await handle_recognition(employee_id, confidence, processed_frame)
                
        # Simulasi UI minimalis (hilangkan di production)
        # cv2.imshow("Stream (Press 'q' to quit)", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
        await asyncio.sleep(0.01) # Yield control to async event loop

if __name__ == "__main__":
    asyncio.run(main())
