# AI Recognition - Media Server Pipeline (Stream Branch)

Repositori `ai-stream` ini berfungsi murni sebagai **Pusat Infrastruktur Streaming (Media Server) & Control Plane**.

> **PENTING:** Repositori ini **TIDAK** menangani deteksi wajah (Face Recognition). Logika deteksi wajah, Bounding Box, dan Redis di-*handle* sepenuhnya oleh repositori terpisah (`ai-detection`).

## 🏗 Arsitektur Sistem

1. **Ingest (Streaming In):**
   - **Kamera HP:** Menggunakan aplikasi PRISM Live / Larix Broadcaster untuk melakukan *push* RTMP langsung ke MediaMTX (port 1935).
   - **Kamera CCTV:** MediaMTX akan melakukan *pull* RTSP secara langsung dari IP CCTV fisik.
2. **Media Server:** MediaMTX secara *real-time* memecah *stream* tersebut ke dalam berbagai protokol (WebRTC & HLS untuk Frontend).
3. **Control Plane:** Aplikasi FastAPI (`server.py`) bertindak sebagai *Control Plane* untuk mengecek status dan mengelola konfigurasi/worker (opsional, port default 8011).

## 🛠 Persyaratan Sistem (Prerequisites)

- **Docker & Docker Compose** (Dikelola oleh Server/DevOps di `docker-compose.yml` utama)
- **Python 3.10+**
- **PM2** (Untuk menjalankan Control Plane di latar belakang)

## ⚙️ Cara Instalasi & Menjalankan Server

### 1. Konfigurasi Lingkungan
Salin file `.env.example` menjadi `.env`.

```bash
cp .env.example .env
```
Pastikan `CONTROL_PLANE_PORT` diatur (default `8011`).

### 2. Instalasi Dependensi Python
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Menjalankan Control Plane via PM2
```bash
pm2 start ecosystem.config.js --env production
pm2 save
```

*(Catatan: Jangan menjalankan `docker-compose up -d` dari repositori ini di Server Production. MediaMTX harus dikelola oleh docker-compose utama).*

## 📡 API Endpoints 

### Untuk Tim Frontend (Dashboard Web)
Didesain untuk di-*embed* langsung ke dalam HTML/Browser klien menggunakan WebRTC.
- **WebRTC Endpoint:** `http://<IP_SERVER>:8889/<camera_id>/whep`
- **HLS Endpoint:** `http://<IP_SERVER>:8888/<camera_id>/index.m3u8`

### Untuk Tim AI (ai-detection)
Didesain untuk diproses di *backend* menggunakan OpenCV dengan latensi serendah mungkin.
- **RTSP Endpoint:** `rtsp://<IP_SERVER>:8554/<camera_id>`
