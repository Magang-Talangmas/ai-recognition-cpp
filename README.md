# Talangmas AI Attendance - Face Preprocessor Service

This repository contains the high-performance C++ backend for the **Talangmas AI Attendance** system. It is designed to capture RTSP video streams from CCTV cameras, detect faces using the **SCRFD** model via ONNX Runtime, preprocess (align, crop, fix brightness, blur check) the faces, and publish the resulting tensor data to a **Redis** queue for downstream facial recognition.

## 🌟 Fitur Utama & Pipeline (Sesuai Diagram Alur)

Aplikasi ini mengimplementasikan *Enterprise AI Pipeline* secara modular dan asinkron:

1. **Raw frame (RTSP)**
   - Menggunakan Python `cv2` backend (lewat *Windows Named Pipe*) untuk menangkap stream CCTV RTSP secara tangguh dan mentransfer frame mentah (raw) ke C++ secara real-time. Ini menembus tembok limitasi decoding `CAP_FFMPEG` di OpenCV C++ untuk Windows.
2. **Frame Validation**
   - Dilengkapi dengan *corrupt check*. Jika frame video gagal di-decode atau rusak akibat *packet loss* di jaringan, sistem akan mengabaikannya dengan aman tanpa menyebabkan aplikasi crash.
3. **Pilar Preprocessing Paralel**
   - **Blur check (Laplacian variance)**: Sistem mengukur ketajaman wajah menggunakan varian dari filter Laplacian. Wajah yang terlalu buram akan diabaikan.
   - **Brightness fix (Histogram equalize)**: Sistem secara otomatis menerapkan algoritma **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) pada wajah yang gelap atau terlalu terang agar struktur wajah terekspos jelas.
   - **Resize + Format (Standard resolution)**: Memanfaatkan 5 titik landmarks (mata, hidung, mulut) dari SCRFD untuk meluruskan (align) wajah, lalu memotong (crop) presisi ke standar resolusi AI: **112x112 pixel Float32 Tensor**.
4. **Publish ke Queue**
   - Data tensor wajah yang sudah tervalidasi dan siap uji dipaketkan bersama `Camera_ID` dan *Timestamp*, lalu dikirim ke **Redis** (`face_preprocessed_queue`) agar tim *Face Recognition* (contoh: ArcFace) dapat mengkonsumsinya seketika.

---

## 🏎️ Asynchronous Multi-Threading (Zero-Lag UI)

Untuk menjamin pemutaran video langsung (*Live View*) dari CCTV tidak macet/patah-patah saat CPU bekerja keras memikirkan AI, program ini memecah beban menjadi dua Thread terpisah (Pola *Producer-Consumer*):
- **UI & Video Thread**: Menarik video dari CCTV dan me-render kotak putus-putus (*Dashed Bounding Box* warna Cyan) secepat kecepatan asli CCTV tanpa hambatan.
- **Inference Thread**: Bersembunyi di latar belakang, memotong frame terbaru, menjalankan AI ONNX SCRFD, dan menyetorkan koordinat wajah ke *UI Thread*.

---

## ⚙️ Konfigurasi Mudah via `.env`

Konfigurasi aplikasi kini bisa diganti kapan saja tanpa perlu kompilasi ulang (Rebuild). Cukup edit file `.env` di direktori utama:

```properties
# Contoh isi .env
RTSP_URL=rtsp://192.168.77.171:8554/stream
REDIS_URL=tcp://127.0.0.1:6379
REDIS_CHANNEL=face_preprocessed_queue
CAMERA_ID=cam_01
```

---

## 🛠️ Persyaratan Sistem (Prerequisites)

- **OS**: Windows 10/11 (64-bit)
- **Compiler**: MSVC (Visual Studio 2022 Build Tools)
- **CMake**: Versi 3.20 atau lebih baru
- **Vcpkg**: Package manager untuk menginstall dependensi C++ (`opencv`, `onnxruntime`, `hiredis`).
- **Python 3**: (Miniconda direkomendasikan) dengan library `opencv-python`.
- **Redis Server**: Server lokal (`127.0.0.1:6379`) atau sesuaikan di `.env`.

---

## 🏗️ Cara Membangun (Build)

1. Pastikan Anda sudah berada di folder root proyek:
   ```powershell
   cd D:\ai-recognition-cpp
   ```

2. Generate file build menggunakan CMake:
   ```powershell
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="vcpkg/scripts/buildsystems/vcpkg.cmake"
   ```

3. Kompilasi (Build) proyek untuk mode Release:
   ```powershell
   cmake --build build --config Release
   ```

---

## 🚀 Cara Menjalankan (Run)

1. Pastikan konfigurasi di file `.env` sudah benar, dan server **Redis** menyala.
2. Jalankan executable yang sudah di-build:
   ```powershell
   .\build\Release\face_preprocessor.exe
   ```
3. Jendela **"Talangmas AI Attendance - Live View"** akan otomatis terbuka. Jika Anda membesarkan ukuran jendela (Maximize/Fullscreen), video akan otomatis *merentang/scaling* menyesuaikan ukuran layar Anda. Garis putus-putus (Cyan) akan mengikuti pergerakan wajah.
4. Untuk menghentikan program, tekan `Ctrl+C` di PowerShell Anda, klik Close (`X`) di UI, atau tekan tombol `q` di keyboard.

---

## 📡 Integrasi dengan Tim Face Recognition (Python)

Sistem ini **TIDAK** menggunakan Endpoint API (HTTP POST/GET) demi mengejar kecepatan real-time tanpa overhead. Kami menggunakan arsitektur **Redis Pub/Sub** sebagai *Message Broker*.

Tim *Face Recognition* hanya perlu membuat skrip Python yang "berlangganan" (subscribe) ke antrean Redis yang sama, lalu menangkap dan me-reshape Tensor Float32 tersebut untuk langsung dimasukkan ke model (misal: ArcFace).

### Format Payload JSON di Redis:
```json
{
  "camera_id": "cam_01",
  "timestamp_ms": 1723623594000,
  "bounding_box": { "x": 450, "y": 120, "width": 210, "height": 210 },
  "confidence_score": 0.987542,
  "image_format": "float32_raw_112x112x3",
  "face_image_base64": "v/z8Pj...<base64>.../Pz8/Pw=="
}
```

### Skrip Python Minimal (Untuk Tim Face Recog)
Kalian cukup menggunakan library `redis` dan `numpy` untuk me-rekonstruksi *raw bytes* kembali menjadi gambar (tensor) berukuran 112x112 pixel.

```python
import redis
import json
import base64
import numpy as np

# 1. Konek ke server Redis
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
pubsub = r.pubsub()
pubsub.subscribe('face_preprocessed_queue')

print("Menunggu stream wajah dari C++...")

# 2. Dengarkan data yang masuk secara real-time
for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        
        # 3. Decode base64 ke bytes mentah
        raw_bytes = base64.b64decode(data["face_image_base64"])
        
        # 4. Ubah bytes jadi Numpy array Float32 (1D)
        face_tensor_1d = np.frombuffer(raw_bytes, dtype=np.float32)
        
        # 5. Reshape jadi format gambar (112 x 112 x 3)
        face_tensor = face_tensor_1d.reshape((112, 112, 3))
        
        print(f"Wajah diterima dari {data['camera_id']} - Siap diproses model!")
        
        # Contoh eksekusi ArcFace:
        # embedding = arcface_model.predict(np.expand_dims(face_tensor, axis=0))
```

---

## 📁 Struktur Direktori

```text
ai-recognition-cpp/
│
├── .env                        # File konfigurasi yang bisa diedit langsung
├── CMakeLists.txt              # Konfigurasi build CMake
├── README.md                   # Dokumentasi ini
├── rtsp_proxy.py               # Script proxy Python (otomatis berjalan)
│
├── models/                     
│   └── scrfd_2.5g_kps.onnx     # Model AI SCRFD (ONNX)
│
└── src/
    ├── main.cpp                # File UI & Manajemen Multi-Threading Asynchronous
    ├── BrokerPublisher.hpp/cpp # Logika pengiriman tensor wajah ke Redis
    ├── FacePreprocessor.hpp/cpp# Logika deteksi, validasi blur, perbaikan brightness, crop
    └── StreamReader.hpp/cpp    # Logika penangkapan frame CCTV (via Named Pipe)
```

**Dikembangkan oleh Tim Developer Antigravity untuk Talangmas.**
