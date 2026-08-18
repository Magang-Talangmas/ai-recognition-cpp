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
import psycopg2
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import insightface
from insightface.app import FaceAnalysis
from supabase import create_client, Client
import threading
from anti_spoof import check_liveness

load_dotenv()

# Kamus boolean per-kamera: True = sedang memproses 1 frame.
# Jika sudah True, frame baru langsung dibuang (latest-wins, no queue).
_camera_busy: dict[str, bool] = {}
_camera_lock = threading.Lock()

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
DATABASE_URL = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
DATA_DIR = "./data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")
SIMILARITY_THRESHOLD = 0.50

# ==================== SCHEDULE LOGIC ====================
employee_schedules = {}
default_schedules = {}

def load_schedules():
    global employee_schedules, default_schedules
    if not DATABASE_URL:
        print("Warning: DATABASE_URL tidak ditemukan, jadwal default akan digunakan.")
        return
        
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Load work_schedules
        cur.execute('SELECT id, "workDays", "checkInTime", "checkOutTime" FROM work_schedules')
        schedules = cur.fetchall()
        
        schedule_map = {}
        for row in schedules:
            sched_id, work_days, check_in, check_out = row
            schedule_map[sched_id] = {
                "checkInTime": check_in,
                "checkOutTime": check_out,
                "workDays": work_days
            }
            if work_days:
                for day in work_days:
                    default_schedules[day] = schedule_map[sched_id]
                    
        # Load employees
        cur.execute('SELECT "employeeId", "scheduleId" FROM employees WHERE status = \'Active\'')
        employees = cur.fetchall()
        
        for row in employees:
            emp_id, sched_id = row
            if sched_id and sched_id in schedule_map:
                employee_schedules[emp_id] = schedule_map[sched_id]
                
        cur.close()
        conn.close()
        print(f"Berhasil memuat jadwal absensi untuk {len(employee_schedules)} karyawan khusus.")
    except Exception as e:
        print(f"[Error] Gagal memuat jadwal dari DB: {e}")

load_schedules()
# ========================================================

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

# Sinkronisasi otomatis: hapus karyawan Inactive, enroll ulang karyawan Active
# Model app dioper agar tidak dimuat 2 kali
from sync_enroll import sync_from_database
sync_from_database(app=app)

# Memuat data enrollment (dibangun oleh sync_from_database di atas)
if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(LABELS_FILE):
    print("[Error] Data enrollment tidak tersedia setelah sinkronisasi. Pastikan ada karyawan aktif dengan foto.")
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
        
        # --- Face Anti-Spoofing Check ---
        orig_bbox = best_face.bbox.copy()
        orig_bbox[0] -= 60
        orig_bbox[1] -= 60
        orig_bbox[2] -= 60
        orig_bbox[3] -= 60
        
        h, w = face_img.shape[:2]
        orig_bbox[0] = max(0, orig_bbox[0])
        orig_bbox[1] = max(0, orig_bbox[1])
        orig_bbox[2] = min(w, orig_bbox[2])
        orig_bbox[3] = min(h, orig_bbox[3])
        
        is_real, liveness_score = check_liveness(face_img, orig_bbox)
        
        # Tolak jika AI mendeteksi palsu, ATAU jika diprediksi asli tapi kurang yakin (< 0.85)
        if not is_real or (is_real and float(liveness_score) < 0.65):
            return "SPOOF", float(liveness_score)
        # --------------------------------
        
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


def get_event_type(employee_id):
    days_id = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    now_utc = datetime.now(timezone.utc)
    now_jkt = now_utc.astimezone(timezone(timedelta(hours=7)))
    day_name = days_id[now_jkt.weekday()]
    
    sched = employee_schedules.get(employee_id)
    if not sched:
        sched = default_schedules.get(day_name)
        
    if not sched:
        # Fallback lama
        current_hour_float = now_jkt.hour + (now_jkt.minute / 60.0)
        if 7 <= current_hour_float < 17:
            return "CHECK_IN" # Check-in dari jam 7:00 sampai 17:29
        elif current_hour_float >= 17.5:
            return "CHECK_OUT" # Check-out dari jam 17:30 ke atas
        else:
            return None  # Diabaikan jika di luar jam

            
    check_in_time = sched["checkInTime"] # e.g. "08:00"
    check_out_time = sched["checkOutTime"] # e.g. "17:00"
    
    cin_h, cin_m = map(int, check_in_time.split(":"))
    cout_h, cout_m = map(int, check_out_time.split(":"))
    
    midpoint_hour = (cin_h + cout_h) / 2.0
    current_hour_float = now_jkt.hour + (now_jkt.minute / 60.0)
    
    if current_hour_float < midpoint_hour:
        if current_hour_float >= cin_h - 2: # Buka absen masuk 2 jam sebelum jam masuk
            return "CHECK_IN"
    else:
        if current_hour_float >= cout_h - 2: # Buka absen pulang 2 jam sebelum jam pulang (jika pulang lebih awal)
            return "CHECK_OUT"
            
    return None

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
        
        response = requests.post(BACKEND_API_URL, json=payload, headers=headers, timeout=15)
        
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

def process_worker(camera_id, face_img):
    try:
        # Lakukan recognition
        print(f"[{camera_id}] Menerima wajah, memproses AI...")
        employee_id, confidence = recognize_face(face_img)
        
        if employee_id == "SPOOF":
            print(f"[{camera_id}] Peringatan: Spoofing / Wajah Palsu terdeteksi! (Liveness: {confidence:.2f})")
            return
            
        if employee_id == "UNKNOWN":
            print(f"[{camera_id}] Wajah tidak dikenal (Confidence: {confidence:.2f})")
            return
            
        if employee_id:
            # Tentukan eventType berdasarkan waktu
            event_type = get_event_type(employee_id)
            
            # Abaikan jika di luar jam absensi
            if not event_type:
                print(f"[{camera_id}] Dikenali sebagai {employee_id}, tapi ditolak karena di luar jam absen.")
                return
                
            # Rate limiter / cooldown
            if not is_action_allowed_by_cooldown(employee_id, event_type):
                print(f"[{camera_id}] Dikenali sebagai {employee_id} (Cooldown aktif)")
                return
                
            print(f"[{camera_id}] Memproses: {employee_id} ({event_type})")
            
            # Pre-emptively update cache
            cache_key = f"{employee_id}_{event_type}"
            last_event_status_cache[cache_key] = {'time': time.time()}
            
            # Upload gambar ke Supabase Storage (snapshots)
            thumb_url = None
            if supabase:
                try:
                    success, buffer = cv2.imencode('.jpg', face_img)
                    if success:
                        file_bytes = buffer.tobytes()
                        file_name = f"snapshots/{uuid.uuid4()}.jpg"
                        
                        supabase.storage.from_("recognition").upload(
                            file_name, 
                            file_bytes,
                            {"content-type": "image/jpeg"}
                        )
                        thumb_url = supabase.storage.from_("recognition").get_public_url(file_name)
                except Exception as upload_err:
                    print(f"[Supabase Error] Gagal upload gambar: {upload_err}")
                    
            # Kirim ke API
            send_to_backend(employee_id, camera_id, confidence, thumb_url, event_type)
    finally:
        with _camera_lock:
            _camera_busy[camera_id] = False

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
                        np_arr = np.frombuffer(raw_bytes, np.uint8)
                        face_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if face_img is None:
                            print("Gagal mendecode gambar wajah dari stream")
                            continue
                    
                    with _camera_lock:
                        if _camera_busy.get(camera_id, False):
                            continue
                        _camera_busy[camera_id] = True

                    threading.Thread(
                        target=process_worker,
                        args=(camera_id, face_img),
                        daemon=True,
                    ).start()
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"Kesalahan memproses event: {e}")
                    
        except requests.exceptions.RequestException as e:
            print(f"Koneksi stream terputus ({e}). Mencoba ulang dalam 5 detik...")
            time.sleep(5)
            
if __name__ == "__main__":
    start_stream_listener()
