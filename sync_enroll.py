import os
import cv2
import json
import shutil
import numpy as np
import requests
import psycopg2
from dotenv import load_dotenv
from fusion_utils import fuse_embeddings
from minio_utils import get_image_from_url

load_dotenv()

DATABASE_URL = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL") or os.getenv("TEAM_DATABASE_URL")
DATA_DIR = "./data"
ENROLL_DIR = os.path.join(DATA_DIR, "enrolled")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")

os.makedirs(ENROLL_DIR, exist_ok=True)

def sync_from_database(app=None):
    """
    Sinkronisasi data wajah dari database ke lokal.
    - Hanya mengambil karyawan dengan status 'Active'.
    - Menghapus folder lokal karyawan yang sudah Inactive / tidak ada di DB.
    - Membangun ulang embeddings.npy dan labels.json.

    Parameter:
        app: Instance FaceAnalysis yang sudah di-load (opsional).
             Jika None, fungsi ini akan menginisialisasi modelnya sendiri.
    """
    if not DATABASE_URL:
        print("[Sync] DATABASE_URL / DIRECT_URL tidak ditemukan di .env, melewati sinkronisasi.")
        return

    print("[Sync] Menghubungkan ke database untuk mengambil data karyawan aktif...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT \"employeeId\", photos FROM employees WHERE photos IS NOT NULL AND status = 'Active'")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Sync] Gagal mengambil data dari database: {e}")
        return

    active_employee_ids = {row[0] for row in rows}
    print(f"[Sync] Ditemukan {len(active_employee_ids)} karyawan aktif di database.")

    # --- Pembersihan: Hapus folder lokal karyawan yang sudah Inactive / dihapus ---
    if os.path.isdir(ENROLL_DIR):
        for folder_name in os.listdir(ENROLL_DIR):
            folder_path = os.path.join(ENROLL_DIR, folder_name)
            if os.path.isdir(folder_path) and folder_name not in active_employee_ids:
                print(f"[Sync] Menghapus data lokal karyawan tidak aktif: {folder_name}")
                shutil.rmtree(folder_path)

    # --- Inisialisasi model jika belum disediakan ---
    if app is None:
        from insightface.app import FaceAnalysis
        print("[Sync] Memuat model InsightFace untuk ekstraksi embedding...")
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings = []
    labels = []

    for row in rows:
        employee_id = row[0]
        photos = row[1]

        if not photos or not isinstance(photos, list):
            continue

        print(f"[Sync] Memproses karyawan: {employee_id} ({len(photos)} foto)")

        # Buat folder khusus untuk employee ini
        emp_dir = os.path.join(ENROLL_DIR, employee_id)
        os.makedirs(emp_dir, exist_ok=True)

        # Buat file emp_id.json
        emp_json_path = os.path.join(emp_dir, "emp_id.json")
        with open(emp_json_path, "w") as f:
            json.dump({"employeeId": employee_id}, f)

        employee_embs = []
        employee_weights = []

        for photo_url in photos:
            if not isinstance(photo_url, str) or not photo_url.startswith("http"):
                continue

            filename = photo_url.split("/")[-1]
            local_path = os.path.join(emp_dir, filename)

            # Download gambar jika belum ada
            if not os.path.exists(local_path):
                success = get_image_from_url(photo_url, local_path)
                if not success:
                    continue

            # Baca gambar dengan OpenCV
            img = cv2.imread(local_path)
            if img is None:
                print(f"  -> Gagal membaca gambar {filename}")
                continue

            # Ekstrak embedding menggunakan InsightFace
            faces = app.get(img)
            if len(faces) == 0:
                print(f"  -> Tidak ada wajah terdeteksi pada gambar {filename}")
                continue

            # Ambil wajah dengan bounding box terbesar (mengabaikan wajah bocor di background)
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = best_face.embedding
            
            # Bobot = Luas Kotak Wajah * Tingkat Keyakinan Detektor
            area = (best_face.bbox[2] - best_face.bbox[0]) * (best_face.bbox[3] - best_face.bbox[1])
            weight = best_face.det_score * area
            
            employee_embs.append(embedding)
            employee_weights.append(float(weight))
            print(f"  -> Berhasil mengekstrak embedding dari {filename} (Bobot: {weight:.0f})")

        if employee_embs:
            fused_emb = fuse_embeddings(employee_embs, weights=employee_weights)
            upsert_query = """
            INSERT INTO employee_embeddings (employee_id, fused_embedding, source_photo_count, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (employee_id) DO UPDATE SET 
                fused_embedding = EXCLUDED.fused_embedding,
                source_photo_count = EXCLUDED.source_photo_count,
                updated_at = EXCLUDED.updated_at
            """
            try:
                conn2 = psycopg2.connect(DATABASE_URL)
                cur2 = conn2.cursor()
                cur2.execute(upsert_query, (employee_id, fused_emb.tolist(), len(employee_embs)))
                conn2.commit()
                cur2.close()
                conn2.close()
                print(f"  -> Berhasil menyimpan fused embedding ke database (sumber: {len(employee_embs)} foto)")
            except Exception as e:
                print(f"  -> Gagal menyimpan embedding ke database: {e}")
        else:
            print(f"  -> Peringatan: Tidak ada wajah yang bisa di-fuse untuk {employee_id}")

    # Simpan embedding dan label ke file lokal dari database
    print("[Sync] Membangun ulang cache lokal dari employee_embeddings...")
    try:
        conn3 = psycopg2.connect(DATABASE_URL)
        cur3 = conn3.cursor()
        cur3.execute("SELECT employee_id, fused_embedding FROM employee_embeddings")
        db_rows = cur3.fetchall()
        for r in db_rows:
            labels.append(r[0])
            embeddings.append(np.array(r[1], dtype=np.float32))
        cur3.close()
        conn3.close()
    except Exception as e:
        print(f"[Sync] Gagal menarik cache lokal: {e}")

    if embeddings:
        # Backup old format before overwrite
        if os.path.exists(EMBEDDINGS_FILE) and not os.path.exists(EMBEDDINGS_FILE + ".bak"):
            shutil.copy(EMBEDDINGS_FILE, EMBEDDINGS_FILE + ".bak")
            shutil.copy(LABELS_FILE, LABELS_FILE + ".bak")

        np.save(EMBEDDINGS_FILE, np.array(embeddings))
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f)
        print(f"[Sync] Selesai: {len(embeddings)} vektor wajah disimpan ({len(active_employee_ids)} karyawan aktif).")
    else:
        print("[Sync] Tidak ada wajah yang berhasil diproses.")

if __name__ == "__main__":
    print("Memulai proses sinkronisasi wajah...")
    sync_from_database()
