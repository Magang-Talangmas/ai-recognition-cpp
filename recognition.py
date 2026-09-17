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
import io
from insightface.app import FaceAnalysis
from minio import Minio
import threading
from anti_spoof import check_liveness

load_dotenv()

# Satu kamera hanya menjalankan satu inference pada satu waktu.
# Event tambahan dibuang agar tidak membentuk antrean tak terbatas.
_camera_busy: dict[str, bool] = {}
_camera_last_started: dict[str, float] = {}
_recent_face_matches: dict[str, list[tuple[float, bytes]]] = {}
_camera_lock = threading.Lock()

# --- MinIO Storage Konfigurasi (Lokal / On-Premise) ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "recognition")

minio_client = None
if MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY:
    try:
        clean_endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        minio_client = Minio(
            clean_endpoint,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_ENDPOINT.startswith("https://")
        )
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
        print(f"[MinIO] Berhasil terhubung ke MinIO lokal (bucket: {MINIO_BUCKET}).")
    except Exception as e:
        print(f"[MinIO Warning] Gagal inisialisasi MinIO ({e}), upload gambar tidak akan berfungsi.")
else:
    print("Warning: Kredensial MinIO tidak lengkap, upload gambar tidak akan berfungsi.")

DETECTION_STREAM_URL = os.getenv("DETECTION_STREAM_URL", "http://192.168.43.11:8000/api/v1/faces/stream")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:5000/api/v1/live/recognition-events")
ML_API_KEY = os.getenv("ML_API_KEY", "your-secret-api-key-to-auth-requests")
DATABASE_URL = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
DATA_DIR = "./data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")
SIMILARITY_THRESHOLD = 0.308 # Calibrated using held-out test data (Panji & Septada)
DRY_RUN = os.getenv("DRY_RUN", "False").lower() in ("true", "1", "yes")
# Batasi embedding Fusion per stream. Bbox tetap diproduksi oleh Detection GPU.
RECOGNITION_MAX_FPS = max(0.1, float(os.getenv("RECOGNITION_MAX_FPS", "3")))
RECOGNITION_SAME_FACE_COOLDOWN_SECONDS = max(
    0.0, float(os.getenv("RECOGNITION_SAME_FACE_COOLDOWN_SECONDS", "5"))
)
# 16x16 average hash; ambang kecil agar hanya crop wajah yang hampir sama yang dilewati.
RECOGNITION_FACE_HASH_MAX_DISTANCE = max(
    0, int(os.getenv("RECOGNITION_FACE_HASH_MAX_DISTANCE", "12"))
)

# ==================== SCHEDULE LOGIC ====================
employee_schedules = {}
default_schedules = {}
employee_names = {}

def load_schedules():
    global employee_schedules, default_schedules, employee_names
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
        cur.execute('SELECT "employeeId", "scheduleId", name FROM employees WHERE status = \'Active\'')
        employees = cur.fetchall()
        
        for row in employees:
            emp_id, sched_id, name = row
            employee_names[emp_id] = name or emp_id
            if sched_id and sched_id in schedule_map:
                employee_schedules[emp_id] = schedule_map[sched_id]
                
        cur.close()
        conn.close()
        print(f"Berhasil memuat jadwal absensi untuk {len(employee_schedules)} karyawan khusus.")
    except Exception as e:
        print(f"[Error] Gagal memuat jadwal dari DB: {e}")

load_schedules()
# ========================================================

def employee_label(employee_id):
    """Format yang mudah dibaca untuk log tanpa mengubah ID yang dikirim ke backend."""
    return f"{employee_names.get(employee_id, 'Karyawan tidak diketahui')} ({employee_id})"

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


def face_fingerprint(face_img):
    """Fingerprint murah untuk mengenali crop wajah yang hampir sama tanpa inference."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    return np.packbits(small >= small.mean()).tobytes()


def hash_distance(first, second):
    return sum((left ^ right).bit_count() for left, right in zip(first, second))


def is_same_recent_face(camera_id, fingerprint):
    if RECOGNITION_SAME_FACE_COOLDOWN_SECONDS == 0:
        return False

    now = time.monotonic()
    with _camera_lock:
        recent = [
            entry for entry in _recent_face_matches.get(camera_id, [])
            if now - entry[0] < RECOGNITION_SAME_FACE_COOLDOWN_SECONDS
        ]
        _recent_face_matches[camera_id] = recent
        return any(
            hash_distance(fingerprint, previous_fingerprint) <= RECOGNITION_FACE_HASH_MAX_DISTANCE
            for _, previous_fingerprint in recent
        )


def remember_recent_face(camera_id, fingerprint):
    if RECOGNITION_SAME_FACE_COOLDOWN_SECONDS == 0:
        return

    now = time.monotonic()
    with _camera_lock:
        recent = [
            entry for entry in _recent_face_matches.get(camera_id, [])
            if now - entry[0] < RECOGNITION_SAME_FACE_COOLDOWN_SECONDS
        ]
        recent.append((now, fingerprint))
        _recent_face_matches[camera_id] = recent[-20:]


def try_reserve_camera_slot(camera_id):
    """Terima paling banyak RECOGNITION_MAX_FPS inference per kamera."""
    now = time.monotonic()
    minimum_interval = 1.0 / RECOGNITION_MAX_FPS
    with _camera_lock:
        if _camera_busy.get(camera_id, False):
            return False
        if now - _camera_last_started.get(camera_id, 0.0) < minimum_interval:
            return False
        _camera_busy[camera_id] = True
        _camera_last_started[camera_id] = now
        return True

def recognize_face(face_img):
    # --- Langsung ke Recognition (Skip Double Detection) ---
    # Wajah dari C++ sudah di-crop dan align ke 112x112
    resized = cv2.resize(face_img, (112, 112))
    
    h, w = face_img.shape[:2]
    # Skip liveness check for 112x112 tightly cropped images from embedding fusion
    if h == 112 and w == 112:
        is_real, liveness_score = True, 1.0
    else:
        # Liveness butuh bbox original, kita asumsikan seluruh crop adalah wajah
        bbox_for_liveness = [0, 0, w, h]
        is_real, liveness_score = check_liveness(face_img, bbox_for_liveness)
    
    if not is_real or (is_real and float(liveness_score) < 0.65):
        return "SPOOF", float(liveness_score)

    embedding = rec_model.get_feat(resized)

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

def process_worker(camera_id, face_img, fingerprint):
    try:
        # Lakukan recognition
        print(f"[{camera_id}] Menerima wajah, memproses AI...")
        employee_id, confidence = recognize_face(face_img)
        
        # --- NEW ALERT FOR EMBEDDING FUSION ---
        if employee_id and employee_id != "SPOOF":
            endpoint_ip = DETECTION_STREAM_URL.split("/api")[0] if "api" in DETECTION_STREAM_URL else "http://192.168.1.101"
            print(f"[ALERT] Embedding Fusion successfully processed data from pre-processing endpoint ({endpoint_ip})")
        # --------------------------------------
        
        if employee_id == "SPOOF":
            print(f"[{camera_id}] Peringatan: Spoofing / Wajah Palsu terdeteksi! (Liveness: {confidence:.2f})")
            return
            
        if employee_id == "UNKNOWN":
            print(f"[{camera_id}] Wajah tidak dikenal (Confidence: {confidence:.2f})")
            return
            
        if employee_id:
            # Setelah identitas diketahui, simpan fingerprint supaya crop orang yang
            # sama tidak dihitung ulang selama cooldown pendek.
            remember_recent_face(camera_id, fingerprint)
            display_name = employee_label(employee_id)
            # Tentukan eventType berdasarkan waktu
            event_type = get_event_type(employee_id)
            
            # Abaikan jika di luar jam absensi
            if not event_type:
                print(f"[{camera_id}] Dikenali sebagai {display_name}, tapi ditolak karena di luar jam absen.")
                return
                
            # Rate limiter / cooldown
            if not is_action_allowed_by_cooldown(employee_id, event_type):
                print(f"[{camera_id}] Dikenali sebagai {display_name} (Cooldown aktif)")
                return
                
            print(f"[{camera_id}] Memproses: {display_name} ({event_type})")
            
            # Pre-emptively update cache
            cache_key = f"{employee_id}_{event_type}"
            last_event_status_cache[cache_key] = {'time': time.time()}
            
            # --- DRY_RUN: Hanya log, jangan kirim ke Supabase atau Backend ---
            if DRY_RUN:
                print(f"🧪 [DRY_RUN] Embedding Fusion match berhasil! Employee: {display_name}, Score: {confidence:.4f}, Event: {event_type}")
                print(f"🧪 [DRY_RUN] Data TIDAK dikirim ke Supabase/Backend (mode testing aktif).")
                return
            # -----------------------------------------------------------------
            
            # Upload gambar ke MinIO Storage lokal (snapshots)
            thumb_url = None
            if minio_client:
                try:
                    success, buffer = cv2.imencode('.jpg', face_img)
                    if success:
                        file_bytes = buffer.tobytes()
                        file_name = f"snapshots/{uuid.uuid4()}.jpg"
                        minio_client.put_object(
                            bucket_name=MINIO_BUCKET,
                            object_name=file_name,
                            data=io.BytesIO(file_bytes),
                            length=len(file_bytes),
                            content_type="image/jpeg"
                        )
                        base_pub = MINIO_PUBLIC_URL.rstrip('/')
                        thumb_url = f"{base_pub}/{MINIO_BUCKET}/{file_name}"
                except Exception as upload_err:
                    print(f"[MinIO Error] Gagal upload gambar snapshot: {upload_err}")
                    
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
                        # Tensor dari C++ (embedding fusion) kemungkinan dalam format CHW (Channel, Height, Width)
                        face_tensor_chw = face_tensor_1d.reshape((3, 112, 112))
                        # Ubah ke HWC (Height, Width, Channel) untuk OpenCV dan InsightFace
                        face_tensor = np.transpose(face_tensor_chw, (1, 2, 0))
                        face_img = ((face_tensor + 1.0) * 127.5).astype(np.uint8)
                        
                        # Opsional: Jika gambar dari C++ adalah RGB, sedangkan InsightFace (cv2) butuh BGR, kita balik channelnya
                        # C++ embedding fusion (NCNN/ONNX) umumnya pakai RGB tensor.
                        face_img = face_img[:, :, ::-1] # Konversi RGB to BGR
                    else:
                        np_arr = np.frombuffer(raw_bytes, np.uint8)
                        face_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if face_img is None:
                            print("Gagal mendecode gambar wajah dari stream")
                            continue
                    
                    fingerprint = face_fingerprint(face_img)
                    if is_same_recent_face(camera_id, fingerprint):
                        continue
                    if not try_reserve_camera_slot(camera_id):
                        continue

                    threading.Thread(
                        target=process_worker,
                        args=(camera_id, face_img, fingerprint),
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
