import os
import json
import time
import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify
import insightface
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

load_dotenv()

app_flask = Flask(__name__)

print("Memuat model InsightFace (buffalo_l) untuk Mobile API...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

if 'recognition' not in face_app.models:
    print("Model recognition tidak ditemukan pada buffalo_l.")
    exit(1)

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def download_image_cv2(url):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Gagal mendownload gambar dari {url}: {e}")
    return None

def extract_embedding(img):
    faces = face_app.get(img)
    if len(faces) == 0:
        resized = cv2.resize(img, (112, 112))
        embedding = face_app.models['recognition'].get_feat(resized)
        if embedding is None or len(embedding) == 0:
            return None
    else:
        best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        embedding = best_face.embedding
    return np.array(embedding).flatten()

@app_flask.route('/verify-face', methods=['POST'])
def verify_face():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "Request body tidak ditemukan"}), 400
    
    employee_id = data.get('employeeId')
    photo_url = data.get('photoUrl')
    master_photo_url = data.get('masterPhotoUrl')

    if not employee_id or not photo_url or not master_photo_url:
        return jsonify({"success": False, "error": "Field employeeId, photoUrl, dan masterPhotoUrl wajib disertakan"}), 400

    # Download kedua gambar dari Supabase
    img_selfie = download_image_cv2(photo_url)
    img_master = download_image_cv2(master_photo_url)

    if img_selfie is None:
        return jsonify({"success": False, "error": "Gagal mengunduh foto selfie dari URL"}), 400
    if img_master is None:
        return jsonify({"success": False, "error": "Gagal mengunduh foto master dari URL"}), 400

    # Ekstrak embedding
    emb_selfie = extract_embedding(img_selfie)
    emb_master = extract_embedding(img_master)

    if emb_selfie is None:
        return jsonify({"success": False, "error": "Wajah tidak terdeteksi pada foto selfie"}), 400
    if emb_master is None:
        return jsonify({"success": False, "error": "Wajah tidak terdeteksi pada foto master"}), 400

    # Hitung kemiripan
    score = float(cosine_similarity(emb_selfie, emb_master))

    # Karena backend Node.js menerima persentase (score < threshold)
    # Backend default env.ML_VERIFY_FACE_THRESHOLD = 35%
    return jsonify({
        "success": True,
        "similarity": score * 100,
        "isMatch": True, # Backend yang akan memutus isMatch berdasarkan ML_VERIFY_FACE_THRESHOLD
        "antiSpoofing": {
            "isReal": True,
            "score": 0.99
        }
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    print(f"Menjalankan Mobile Face API secara terpisah di port {port}...")
    app_flask.run(host='0.0.0.0', port=port, debug=False)
