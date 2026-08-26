# AI Recognition - Media Server Pipeline

Repositori ini berfungsi sebagai **Pusat Infrastruktur Streaming (Media Server)** untuk mendistribusikan *feed* video dari IP Camera (CCTV) ke seluruh tim terkait (Machine Learning / Pre-processing dan Frontend Dashboard) dalam sistem *AI Face Recognition*.

Infrastruktur ini didesain menggunakan **MediaMTX** (Go-based media server) dengan latensi ultra-rendah, serta **FFmpeg & OpenCV** (C/C++ backend) untuk melakukan *ingest* video yang sangat efisien.

---

## 🏗 Arsitektur Sistem

1. **Ingest (Streaming In):** Skrip `publisher.py` menarik RTSP stream mentah dari IP Camera.
2. **Preprocessing (Aspect Ratio):** OpenCV secara otomatis menyesuaikan rasio kamera menjadi standar **16:9** (menambahkan *pillarbox/letterbox* hitam) agar seragam di semua klien, tanpa mengubah bentuk (*stretch*) wajah.
3. **Media Server:** FFmpeg mem-*publish* video tersebut ke server lokal **MediaMTX** (`rtsp://localhost:8554/stream`).
4. **Distribusi (Streaming Out):** MediaMTX secara *real-time* memecah *stream* tersebut ke dalam berbagai protokol (RTSP untuk backend AI, WebRTC & HLS untuk Frontend).

---

## 🛠 Persyaratan Sistem (Prerequisites)

Pastikan mesin/server Anda telah memiliki perangkat lunak berikut:
- **Docker & Docker Compose** (Untuk menjalankan MediaMTX & Redis)
- **Python 3.8+**
- **FFmpeg** terinstal di OS Anda dan sudah masuk ke dalam *Environment Variables* (Path).

---

## ⚙️ Cara Instalasi & Menjalankan Server

### 1. Konfigurasi Lingkungan
Buat atau salin file `.env` di folder *root* proyek ini, sesuaikan kredensial IP Camera Anda:

```env
# Konfigurasi Kamera RTSP
RTSP_USER=admin
RTSP_PASS=rahasia123
RTSP_HOST=192.168.77.209
RTSP_PORT=554
RTSP_STREAM_PATH=/cam/realmonitor?channel=1&subtype=1
```

### 2. Instalasi Dependensi Python
Instal pustaka Python yang dibutuhkan (OpenCV dan Dotenv):
```bash
pip install -r requirements.txt
```

### 3. Menghidupkan Media Server
Jalankan *container* Docker untuk MediaMTX:
```bash
docker-compose up -d mediamtx
```

### 4. Memulai Siaran (Publisher)
Jalankan skrip untuk menarik video dari kamera dan mengirimkannya ke Media Server:
```bash
python publisher.py
```
*(Biarkan terminal ini tetap berjalan. Anda akan melihat preview lokal dan log FPS)*.

---

## 📡 API Endpoints (Untuk Tim Lain)

Silakan berikan URL berikut kepada tim yang membutuhkan akses *stream* (*Catatan: Ganti `192.168.77.171` dengan IP Address LAN/Publik dari komputer yang menjalankan Media Server ini*).

### 🤖 Untuk Tim AI / Machine Learning
Didesain untuk diproses di *backend* (menggunakan OpenCV Python/C++) dengan latensi serendah mungkin tanpa membebani browser.
- **RTSP Endpoint:** `rtsp://192.168.77.171:8554/stream`

*(Contoh kode *consumer* dapat dilihat pada file `stream_consumer_example.py`)*.

### 💻 Untuk Tim Frontend (Dashboard Web)
Didesain untuk di-*embed* langsung ke dalam HTML/Browser klien.
- **WebRTC Endpoint (Utama):** `http://192.168.77.171:8889/stream/`
  *(Latensi < 1 detik. Wajib digunakan agar tampilan video sinkron dengan notifikasi Face Recognition dari backend).*
- **HLS Endpoint (Fallback):** `http://192.168.77.171:8888/stream/`
  *(Gunakan protokol ini jika WebRTC diblokir oleh jaringan klien. Sangat stabil namun memiliki latensi 3-5 detik).*

---

## 📝 Catatan Penting
- **Bounding Box Kamera:** Skrip di repositori ini **TIDAK** menambahkan kotak wajah (*bounding box*) hijau/cyan apa pun. Jika *stream* akhir menampilkan kotak deteksi wajah bawaan, fitur *IVS/Face Detection* pada IP Camera asli harus dimatikan lewat Web UI Kamera.
