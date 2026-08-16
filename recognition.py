import os
import json
import time
import base64
import asyncio
import numpy as np
import requests
import sseclient
import cv2
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
import insightface
from insightface.app import FaceAnalysis
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("Warning: Kredensial Supabase tidak lengkap, upload gambar tidak akan berfungsi.")

DETECTION_STREAM_URL = os.getenv("DETECTION_STREAM_URL", "http://192.168.43.11:8000/api/v1/faces/stream")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:5000/api/v1/live/recognition-events")
ML_API_KEY = os.getenv("ML_API_KEY", "your-secret-api-key-to-auth-requests")
DATA_DIR = "./data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")
SIMILARITY_THRESHOLD = 0.50

# Cache untuk mencegah pengiriman spam ke database dalam interval pendek
last_event_status_cache = {}

print("Memuat model InsightFace (buffalo_l)...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
# Pastikan model recognition tersedia
if 'recognition' not in app.models:
    print("Model recognition tidak ditemukan pada buffalo_l.")
    exit(1)
rec_model = app.models['recognition']

# Memuat data enrollment
if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(LABELS_FILE):
    print("Data enrollment tidak ditemukan! Harap jalankan sync_enroll.py terlebih dahulu.")
    exit(1)

enrolled_embeddings = np.load(EMBEDDINGS_FILE)
with open(LABELS_FILE, "r") as f:
    enrolled_labels = json.load(f)

print(f"Berhasil memuat {len(enrolled_labels)} wajah terdaftar.")

# Koneksi ke DB dihapus, sekarang menggunakan Backend API

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def recognize_face(face_img):
    # Tambahkan padding agar detektor (app.get) punya ruang untuk menemukan dan meluruskan (align) wajah.
    # Ini berfungsi sebagai "re-kalibrasi" karena kalibrasi C++ dari Device 2 kurang presisi.
    padded_img = cv2.copyMakeBorder(face_img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    faces = app.get(padded_img)
    
    if len(faces) == 0:
        # Fallback darurat
        resized = cv2.resize(face_img, (112, 112))
        embedding = rec_model.get_feat(resized)
    else:
        # Ambil wajah yang paling besar di dalam frame
        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        embedding = best_face.embedding

    if embedding is None or len(embedding) == 0:
        return None, 0.0

    embedding = np.array(embedding).flatten()
    
    best_match_id = None
    best_score = 0.0

    for i, enrolled_emb in enumerate(enrolled_embeddings):
        score = cosine_similarity(embedding, enrolled_emb)
        
        if score > best_score:
            best_score = score
            best_match_id = enrolled_labels[i]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_match_id, float(best_score)
    else:
        return "UNKNOWN", float(best_score)


def get_event_type():
    hour = datetime.now().hour
    if 7 <= hour < 16:
        return "CHECK_IN" # Check-in sampai jam 16:00 (4 Sore)
    elif hour >= 17:
        return "CHECK_OUT" # Check-out dari jam 16:00 ke atas
    else:
        return None  # Diabaikan jika di luar jam (00:00-06:59)


def is_action_allowed_by_cooldown(employee_id, event_type):
    now = time.time()
    cache_key = f"{employee_id}_{event_type}"
    
    # 1. Cek cache memori (sebagai Rate Limiter sebelum menembak Backend API)
    if cache_key in last_event_status_cache:
        cached = last_event_status_cache[cache_key]
        if event_type == "CHECK_IN":
            # Check-in: Jeda 10 detik sebelum nembak API lagi (untuk mencegah spam request)
            if (now - cached['time']) < 10: 
                return False
        elif event_type == "CHECK_OUT":
            # Check-out: Upsert cooldown 5 menit (300 detik)
            if (now - cached['time']) < 300:
                return False
                
    return True

def send_to_backend(employee_id, camera_id, confidence, thumbnail_url, event_type):
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ML_API_KEY
        }
        payload = {
            "employeeId": employee_id if employee_id != "UNKNOWN" else None,
            "cameraId": camera_id,
            "confidence": float(confidence * 100),
            "status": "Verified" if employee_id != "UNKNOWN" else "Unknown",
            "thumbnail": thumbnail_url,
            "eventType": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        response = requests.post(BACKEND_API_URL, json=payload, headers=headers, timeout=5)
        
        cache_key = f"{employee_id}_{event_type}"
        
        if response.status_code in [200, 201]:
            print(f"[API] Event {event_type} terkirim. Emp: {employee_id}, Score: {confidence:.2f}")
            # Update cache agar cooldown berjalan
            last_event_status_cache[cache_key] = {'time': time.time()}
        elif response.status_code == 409:
            # Konflik: Karyawan sudah absen hari ini, update cache agar diblokir lagi untuk beberapa detik ke depan
            print(f"[API] Event diabaikan (Duplikat). Emp: {employee_id}")
            last_event_status_cache[cache_key] = {'time': time.time()}
        else:
            print(f"[API Error] Status: {response.status_code}, Body: {response.text}")
            
    except Exception as e:
        print(f"[API Error] Gagal mengirim event ke backend: {e}")

def start_stream_listener():
    print(f"Menghubungkan ke stream SSE: {DETECTION_STREAM_URL} ...")
    headers = {'Accept': 'text/event-stream'}
    
    while True:
        try:
            response = requests.get(DETECTION_STREAM_URL, stream=True, headers=headers, timeout=30)
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if not event.data:
                    continue
                    
                try:
                    data = json.loads(event.data)
                    camera_id = data.get("camera_id", "CAM-01")
                    base64_img = data.get("face_image_base64")
                    
                    if not base64_img:
                        continue
                        
                    image_format = data.get("image_format", "")
                    raw_bytes = base64.b64decode(base64_img)
                    
                    if "float32" in image_format or len(raw_bytes) == 112 * 112 * 3 * 4:
                        face_tensor_1d = np.frombuffer(raw_bytes, dtype=np.float32)
                        face_tensor = face_tensor_1d.reshape((112, 112, 3))
                        face_img = ((face_tensor + 1.0) * 127.5).astype(np.uint8)
                    else:
                        # Format baru kemungkinan JPEG terkompresi
                        np_arr = np.frombuffer(raw_bytes, np.uint8)
                        face_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if face_img is None:
                            print("Gagal mendecode gambar wajah dari stream")
                            continue
                    
                    
                    # Lakukan recognition
                    print(f"[{camera_id}] Menerima wajah, memproses AI...")
                    employee_id, confidence = recognize_face(face_img)
                    
                    if employee_id == "UNKNOWN":
                        print(f"[{camera_id}] Wajah tidak dikenal (Confidence: {confidence:.2f})")
                        continue
                        
                    if employee_id:
                        # Tentukan eventType berdasarkan waktu
                        event_type = get_event_type()
                        
                        # Abaikan jika di luar jam absensi (jam 12:00 - 16:59)
                        if not event_type:
                            print(f"[{camera_id}] Dikenali sebagai {employee_id}, tapi ditolak karena di luar jam absen.")
                            continue
                            
                        # Rate limiter / cooldown
                        if not is_action_allowed_by_cooldown(employee_id, event_type):
                            print(f"[{camera_id}] Dikenali sebagai {employee_id} (Cooldown aktif)")
                            continue
                            
                        print(f"[{camera_id}] Memproses: {employee_id} ({event_type})")
                        
                        # Upload gambar ke Supabase Storage (snapshots)
                        thumbnail_url = None
                        if supabase:
                            try:
                                success, buffer = cv2.imencode('.jpg', face_img)
                                if success:
                                    file_bytes = buffer.tobytes()
                                    file_name = f"snapshots/{uuid.uuid4()}.jpg"
                                    
                                    # Gunakan content-type agar browser bisa merender dengan benar
                                    supabase.storage.from_("recognition").upload(
                                        file_name, 
                                        file_bytes,
                                        {"content-type": "image/jpeg"}
                                    )
                                    thumbnail_url = supabase.storage.from_("recognition").get_public_url(file_name)
                            except Exception as upload_err:
                                print(f"[Supabase Error] Gagal upload gambar: {upload_err}")
                                
                        # Kirim ke API
                        send_to_backend(employee_id, camera_id, confidence, thumbnail_url, event_type)
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"Kesalahan memproses event: {e}")
                    
        except requests.exceptions.RequestException as e:
            print(f"Koneksi stream terputus ({e}). Mencoba ulang dalam 5 detik...")
            time.sleep(5)
            
if __name__ == "__main__":
    start_stream_listener()
