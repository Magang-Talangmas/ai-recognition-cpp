import os
import cv2
import json
import numpy as np
import shutil
import psycopg2
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from fusion_utils import fuse_embeddings

load_dotenv()
DATABASE_URL = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL") or os.getenv("TEAM_DATABASE_URL")

DATA_DIR = "./data"
# User says they put it in data/enroll
ENROLL_DIR = os.path.join(DATA_DIR, "enroll")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")

def enroll_from_local_and_upload():
    if not os.path.exists(ENROLL_DIR):
        print(f"Error: Folder '{ENROLL_DIR}' tidak ditemukan.")
        return

    if not DATABASE_URL:
        print("Error: DATABASE_URL tidak ditemukan di .env")
        return

    print("[Enroll] Memuat model InsightFace...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings = []
    labels = []

    print("[Enroll] Menghubungkan ke database...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
    except Exception as e:
        print(f"[Enroll] Gagal koneksi ke database: {e}")
        return

    for emp_id in os.listdir(ENROLL_DIR):
        emp_dir = os.path.join(ENROLL_DIR, emp_id)
        if not os.path.isdir(emp_dir):
            continue
        
        print(f"\n[Enroll] Memproses kelas/karyawan: {emp_id}")
        employee_embs = []
        for filename in os.listdir(emp_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            filepath = os.path.join(emp_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"  -> Gagal membaca gambar {filename}")
                continue
            
            faces = app.get(img)
            if len(faces) > 0:
                employee_embs.append(faces[0].embedding)
                print(f"  -> Sukses mengekstrak wajah dari {filename}")
            else:
                print(f"  -> Wajah tidak terdeteksi pada {filename}")
        
        if employee_embs:
            fused = fuse_embeddings(employee_embs)
            embeddings.append(fused)
            labels.append(emp_id)
            print(f"  -> Selesai fuse {len(employee_embs)} foto untuk {emp_id}.")
            
            # Upload ke Supabase
            upsert_query = """
            INSERT INTO employee_embeddings (employee_id, fused_embedding, source_photo_count, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (employee_id) DO UPDATE SET 
                fused_embedding = EXCLUDED.fused_embedding,
                source_photo_count = EXCLUDED.source_photo_count,
                updated_at = EXCLUDED.updated_at
            """
            try:
                cur.execute(upsert_query, (emp_id, json.dumps(fused.tolist()), len(employee_embs)))
                conn.commit()
                print(f"  -> [DB] Berhasil upload/update embedding untuk {emp_id} ke Supabase.")
            except Exception as e:
                print(f"  -> [DB] Gagal upload embedding untuk {emp_id}: {e}")
                conn.rollback()

        else:
            print(f"  -> Peringatan: Tidak ada wajah valid untuk {emp_id}.")
    
    cur.close()
    conn.close()

    if embeddings:
        # Backup file lama dan replace (hapus old embedding local)
        if os.path.exists(EMBEDDINGS_FILE):
            shutil.copy(EMBEDDINGS_FILE, EMBEDDINGS_FILE + ".bak")
            shutil.copy(LABELS_FILE, LABELS_FILE + ".bak")
            
        np.save(EMBEDDINGS_FILE, np.array(embeddings))
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f)
        print(f"\n[Enroll] MANTAP! {len(labels)} kelas berhasil di-enroll, dikirim ke Supabase, dan disimpan di lokal.")
    else:
        print("\n[Enroll] Tidak ada wajah yang berhasil diproses.")

if __name__ == "__main__":
    print("Memulai proses Enroll dari folder lokal dan Upload ke Database...")
    enroll_from_local_and_upload()

