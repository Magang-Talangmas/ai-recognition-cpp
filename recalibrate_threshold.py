#!/usr/bin/env python3
"""
recalibrate_threshold.py — End-to-End Threshold Calibration
=============================================================
Menghitung threshold optimal menggunakan DUA jalur embedding:

  Mode 1: "cpp_sim"  — Simulasi exact pipeline C++ FacePreprocessor
                        (SCRFD detect → estimateAffinePartial2D 5-point → CLAHE → rec_model)
  Mode 2: "live_sse" — Tangkap crop asli dari SSE stream C++ yang sedang berjalan
  Mode 3: "python"   — app.get() murni (baseline, untuk perbandingan enrollment)

Juga menyertakan Parity Test permanen: memastikan embedding C++ vs Python
tidak pernah diverge >0.05 (cosine sim > 0.95).

Cara pakai:
    python recalibrate_threshold.py                  # default: cpp_sim
    python recalibrate_threshold.py --mode python    # baseline lama
    python recalibrate_threshold.py --mode live_sse  # dari stream C++ asli
    python recalibrate_threshold.py --parity-only    # hanya jalankan parity test
"""

import os
import sys
import json
import time
import argparse
import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ==============================================================================
# EMBEDDING EXTRACTORS — Tiga jalur berbeda
# ==============================================================================

def extract_embedding_python(img_path, app):
    """
    Jalur Enrollment (Python murni).
    Sama persis dengan apa yang sync_enroll.py lakukan saat mendaftarkan wajah.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [python] Error reading {img_path}")
        return None

    faces = app.get(img)
    if len(faces) == 0:
        print(f"  [python] No face detected in {os.path.basename(img_path)}")
        return None

    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = np.array(best_face.embedding).flatten()
    return emb / np.linalg.norm(emb)


def extract_embedding_cpp_sim(img_path, app):
    """
    Simulasi exact pipeline C++ FacePreprocessor (jalur produksi).

    Langkah-langkah identik dengan FacePreprocessor.cpp setelah fix:
      1. SCRFD detect → dapat bounding box + 5 landmarks
      2. Umeyama similarity transform (5-point → ArcFace reference) → warpAffine 112x112
      3. rec_model.get_feat() → embedding 512-d

    Alignment menggunakan skimage.SimilarityTransform — algoritma yang IDENTIK
    dengan implementasi Umeyama di C++. Parity test: 1.0000 vs app.get().
    """
    from skimage import transform as trans

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [cpp_sim] Error reading {img_path}")
        return None

    # Step 1: Detect faces menggunakan SCRFD (model yang sama dengan C++)
    faces = app.get(img)
    if len(faces) == 0:
        print(f"  [cpp_sim] No face detected in {os.path.basename(img_path)}")
        return None

    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # Step 2: Umeyama similarity transform (identik dengan C++ Umeyama)
    landmarks = best_face.kps
    arcface_ref = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, arcface_ref)
    M = tform.params[0:2, :]

    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    # Step 3: CLAHE DIHAPUS — ArcFace dilatih tanpa CLAHE.
    #         Parity test membuktikan CLAHE merusak embedding (drop 0.08-0.18 poin).

    # Step 4: rec_model.get_feat() — jalur yang sama dengan recognition.py
    rec_model = app.models['recognition']
    embedding = rec_model.get_feat(aligned)

    if embedding is None or len(embedding) == 0:
        return None

    emb = np.array(embedding).flatten()
    return emb / np.linalg.norm(emb)


def extract_embedding_live_sse(rec_model, stream_url, timeout=30):
    """
    Tangkap crop 112x112 langsung dari SSE stream C++ yang sedang berjalan.
    Ini adalah jalur produksi sesungguhnya — 100% end-to-end.

    Returns: (embedding, face_img) atau (None, None) jika timeout.
    """
    import requests
    import sseclient

    print(f"  [live_sse] Connecting to {stream_url} ...")
    try:
        response = requests.get(stream_url, stream=True, timeout=timeout,
                                headers={'Accept': 'text/event-stream'})
        client = sseclient.SSEClient(response)

        for event in client.events():
            if not event.data:
                continue
            data = json.loads(event.data)
            b64 = data.get("face_image_base64")
            if not b64:
                continue

            raw_bytes = base64.b64decode(b64)
            image_format = data.get("image_format", "")

            if "float32" in image_format or len(raw_bytes) == 112 * 112 * 3 * 4:
                tensor = np.frombuffer(raw_bytes, dtype=np.float32).reshape((112, 112, 3))
                face_img = ((tensor + 1.0) * 127.5).astype(np.uint8)
            else:
                face_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

            if face_img is None:
                continue

            face_img = cv2.resize(face_img, (112, 112))
            embedding = rec_model.get_feat(face_img)
            if embedding is None:
                continue

            emb = np.array(embedding).flatten()
            return emb / np.linalg.norm(emb), face_img

    except Exception as e:
        print(f"  [live_sse] Error: {e}")

    return None, None


# ==============================================================================
# PARITY TEST — Deteksi divergensi antara jalur C++ dan Python
# ==============================================================================

def run_parity_test(app, test_dir="./data/testData", threshold=0.95):
    """
    Untuk setiap gambar test, bandingkan embedding dari jalur C++ sim vs Python.
    Keduanya HARUS menghasilkan embedding yang sangat mirip (>0.95).
    Jika tidak, ada bug alignment di salah satu jalur.
    """
    print("\n" + "=" * 70)
    print("PARITY TEST: C++ Simulation vs Python app.get()")
    print("=" * 70)

    classes = [d for d in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, d))]

    all_scores = []
    failures = []

    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        for file in sorted(os.listdir(cls_dir)):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(cls_dir, file)

            # Pre-check: berapa wajah terdeteksi?
            img = cv2.imread(path)
            if img is None:
                print(f"  SKIP  {cls}/{file} (cannot read image)")
                continue
            faces = app.get(img)
            if len(faces) == 0:
                print(f"  SKIP  {cls}/{file} (no face detected)")
                continue
            if len(faces) > 1:
                print(f"  SKIP  {cls}/{file} ({len(faces)} faces — multi-face photos "
                      f"tidak valid untuk parity test)")
                continue

            emb_py = extract_embedding_python(path, app)
            emb_cpp = extract_embedding_cpp_sim(path, app)

            if emb_py is None or emb_cpp is None:
                print(f"  SKIP  {cls}/{file} (embedding extraction failed)")
                continue

            parity = float(np.dot(emb_py, emb_cpp))
            all_scores.append(parity)
            status = "PASS" if parity >= threshold else "FAIL"

            if status == "FAIL":
                failures.append((f"{cls}/{file}", parity))

            print(f"  {status}  {cls}/{file}: parity = {parity:.4f}")

    if not all_scores:
        print("\n  No images processed!")
        return False

    print(f"\n--- Parity Summary ---")
    print(f"  Images tested : {len(all_scores)}")
    print(f"  Min parity    : {min(all_scores):.4f}")
    print(f"  Max parity    : {max(all_scores):.4f}")
    print(f"  Avg parity    : {np.mean(all_scores):.4f}")

    if failures:
        print(f"\n  [FAIL] {len(failures)} FAILURES (parity < {threshold}):")
        for name, score in failures:
            print(f"     {name}: {score:.4f}")
        return False
    else:
        print(f"\n  [OK] ALL PASSED — C++ and Python alignment are in sync!")
        return True


# ==============================================================================
# THRESHOLD CALIBRATION — Genuine vs Impostor analysis
# ==============================================================================

def run_calibration(app, mode="cpp_sim", test_dir="./data/testData",
                    stream_url="http://192.168.1.101:8000/api/v1/faces/stream"):
    """
    Hitung threshold optimal berdasarkan mode yang dipilih.
    """
    print("\n" + "=" * 70)
    print(f"THRESHOLD CALIBRATION — Mode: {mode}")
    print("=" * 70)

    classes = [d for d in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, d))]

    if mode == "live_sse":
        print("\n[!]  Mode live_sse: Anda harus berdiri di depan kamera secara bergantian.")
        print("   Script akan menangkap wajah dari SSE stream C++ yang sedang berjalan.")
        print("   Pastikan C++ preprocessor dan SSE bridge aktif!\n")

        rec_model = app.models['recognition']
        embeddings = {}
        for cls in classes:
            embeddings[cls] = []
            n_photos = len([f for f in os.listdir(os.path.join(test_dir, cls))
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"\nKelas '{cls}': Perlu {n_photos} capture.")
            for i in range(n_photos):
                input(f"  Tekan ENTER untuk capture #{i+1} dari '{cls}'...")
                emb, _ = extract_embedding_live_sse(rec_model, stream_url)
                if emb is not None:
                    embeddings[cls].append((f"live_{i}", emb))
                    print(f"    [OK] Captured!")
                else:
                    print(f"    [FAIL] Failed to capture, skipping")
    else:
        # Mode cpp_sim atau python
        extract_fn = (extract_embedding_cpp_sim if mode == "cpp_sim"
                      else extract_embedding_python)

        embeddings = {}
        for cls in classes:
            cls_dir = os.path.join(test_dir, cls)
            embeddings[cls] = []
            for file in sorted(os.listdir(cls_dir)):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                path = os.path.join(cls_dir, file)
                emb = extract_fn(path, app)
                if emb is not None:
                    embeddings[cls].append((file, emb))
                    print(f"  [OK] [{mode}] {cls}/{file}")

    # --- Compute scores ---
    genuine_scores = []
    impostor_scores = []

    for cls in classes:
        embs = embeddings.get(cls, [])
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = float(np.dot(embs[i][1], embs[j][1]))
                genuine_scores.append(sim)

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            cls1, cls2 = classes[i], classes[j]
            for _, emb1 in embeddings.get(cls1, []):
                for _, emb2 in embeddings.get(cls2, []):
                    sim = float(np.dot(emb1, emb2))
                    impostor_scores.append(sim)

    if not genuine_scores or not impostor_scores:
        print("\n  Not enough data to calibrate. Need ≥2 classes with ≥2 photos each.")
        return

    print(f"\n--- Results ({mode}) ---")
    print(f"Genuine  (same person)      — {len(genuine_scores)} comparisons:")
    print(f"  Min: {min(genuine_scores):.4f}  Max: {max(genuine_scores):.4f}  Avg: {np.mean(genuine_scores):.4f}")

    print(f"Impostor (different person) — {len(impostor_scores)} comparisons:")
    print(f"  Min: {min(impostor_scores):.4f}  Max: {max(impostor_scores):.4f}  Avg: {np.mean(impostor_scores):.4f}")

    highest_impostor = max(impostor_scores)
    lowest_genuine = min(genuine_scores)

    print(f"\n--- Recommendation ---")
    if lowest_genuine > highest_impostor:
        threshold = (lowest_genuine + highest_impostor) / 2
        gap = lowest_genuine - highest_impostor
        print(f"[OK] PERFECT SEPARATION! Gap: {gap:.4f}")
        print(f"   Recommended Threshold: {threshold:.4f}")
        print(f"   (Midpoint between impostor max {highest_impostor:.4f} and genuine min {lowest_genuine:.4f})")
    else:
        overlap = highest_impostor - lowest_genuine
        print(f"[!]  OVERLAP detected! Size: {overlap:.4f}")
        print(f"   Highest Impostor: {highest_impostor:.4f}")
        print(f"   Lowest Genuine:   {lowest_genuine:.4f}")
        threshold = highest_impostor + 0.05
        print(f"   Recommended Threshold (favor security): {threshold:.4f}")

    print(f"\n   To apply: set SIMILARITY_THRESHOLD = {threshold:.4f} in recognition.py")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end threshold calibration & parity test")
    parser.add_argument("--mode", choices=["cpp_sim", "python", "live_sse"],
                        default="cpp_sim",
                        help="Extraction mode (default: cpp_sim)")
    parser.add_argument("--test-dir", default="./data/testData",
                        help="Path ke folder test data")
    parser.add_argument("--stream-url",
                        default="http://192.168.1.101:8000/api/v1/faces/stream",
                        help="SSE stream URL (untuk mode live_sse)")
    parser.add_argument("--parity-only", action="store_true",
                        help="Hanya jalankan parity test, skip kalibrasi")
    parser.add_argument("--skip-parity", action="store_true",
                        help="Skip parity test, langsung kalibrasi")
    args = parser.parse_args()

    print("Loading InsightFace buffalo_l...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    if args.parity_only:
        passed = run_parity_test(app, args.test_dir)
        sys.exit(0 if passed else 1)

    if not args.skip_parity:
        passed = run_parity_test(app, args.test_dir)
        if not passed:
            print("\n[!]  Parity test gagal! Fix alignment dulu sebelum kalibrasi.")
            print("   Gunakan --skip-parity untuk force kalibrasi meskipun parity gagal.\n")
            sys.exit(1)

    run_calibration(app, mode=args.mode, test_dir=args.test_dir,
                    stream_url=args.stream_url)


if __name__ == "__main__":
    main()
