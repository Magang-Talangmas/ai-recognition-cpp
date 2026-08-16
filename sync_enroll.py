import os
import cv2
import json
import numpy as np
import requests
import psycopg2
from dotenv import load_dotenv
import insightface
from insightface.app import FaceAnalysis

load_dotenv()

DATABASE_URL = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
DATA_DIR = "./data"
ENROLL_DIR = os.path.join(DATA_DIR, "enrolled")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")

os.makedirs(ENROLL_DIR, exist_ok=True)

# Inisialisasi InsightFace
# Menggunakan model buffalo_l yang memiliki akurasi tinggi untuk pengenalan wajah
print("Loading InsightFace model (buffalo_l)...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 untuk GPU (jika ada), -1 untuk CPU

def sync_from_database():
    if not DATABASE_URL:
        print("DATABASE_URL / DIRECT_URL tidak ditemukan di .env")
        return

    print("Terhubung ke database untuk mengambil data karyawan...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute('SELECT "employeeId", photos FROM employees WHERE photos IS NOT NULL')
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Gagal mengambil data dari database: {e}")
        return

    embeddings = []
    labels = []

    for row in rows:
        employee_id = row[0]
        photos = row[1]
        
        if not photos or not isinstance(photos, list):
            continue
            
        print(f"Memproses karyawan: {employee_id} ({len(photos)} foto)")
        
        # Buat folder khusus untuk employee ini
        emp_dir = os.path.join(ENROLL_DIR, employee_id)
        os.makedirs(emp_dir, exist_ok=True)
        
        # Buat file emp_id.json
        emp_json_path = os.path.join(emp_dir, "emp_id.json")
        with open(emp_json_path, "w") as f:
            json.dump({"employeeId": employee_id}, f)
            
        for i, photo_url in enumerate(photos):
            if not isinstance(photo_url, str) or not photo_url.startswith("http"):
                continue
                
            filename = photo_url.split("/")[-1]
            local_path = os.path.join(emp_dir, filename)
            
            # Download gambar jika belum ada
            if not os.path.exists(local_path):
                try:
                    response = requests.get(photo_url, stream=True)
                    response.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    print(f"  -> Gagal mendownload {photo_url}: {e}")
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
                
            # Ambil wajah pertama yang terdeteksi (asumsi foto profil hanya ada 1 wajah)
            embedding = faces[0].embedding
            
            embeddings.append(embedding)
            labels.append(employee_id)
            print(f"  -> Berhasil mengekstrak embedding dari {filename}")

    # Simpan embedding dan label ke file lokal
    if embeddings:
        np.save(EMBEDDINGS_FILE, np.array(embeddings))
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f)
        print(f"Berhasil menyimpan {len(embeddings)} vektor wajah ke lokal.")
    else:
        print("Tidak ada wajah yang berhasil diproses.")

if __name__ == "__main__":
    print("Memulai proses sinkronisasi wajah...")
    sync_from_database()
