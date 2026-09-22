# Talangmas AI Attendance - Recognition Fusion Worker

This repository contains the Python-based AI worker for the **Talangmas AI Attendance** system. It is designed to consume an SSE (Server-Sent Events) stream of pre-processed faces from the C++ Detection Server, perform facial recognition using InsightFace, and record attendance events to the database.

## 🌟 Arsitektur Face Recognition Fusion

Modul pengenalan wajah (Recognition) ini berjalan sebagai *service* Python mandiri yang didesain untuk deployment *bare-metal* menggunakan **PM2** di server produksi. 

### 1. `sync_enroll.py` (Sinkronisasi Wajah)
Script ini berfungsi untuk melakukan sinkronisasi data wajah (enrollment) karyawan yang ada di database ke penyimpanan lokal server AI.
- **Fungsionalitas**: Mengekstrak vektor fitur (*embedding*) menggunakan **InsightFace (Buffalo_L)**, dan menyimpannya secara lokal ke dalam file `data/embeddings.npy` beserta labelnya di `data/labels.json`.
- **Kapan digunakan?**: Otomatis dipanggil saat worker berjalan, atau bisa dijalankan manual saat inisialisasi awal server AI agar server mengenali wajah-wajah tersebut.

### 2. `recognition.py` (Service Utama Face Recognition)
Script ini merupakan *worker* utama yang berjalan di latar belakang untuk melakukan pengenalan wajah secara *real-time*.
- **Fungsionalitas**: 
  - Melakukan koneksi ke stream (SSE) dari module pendeteksi wajah (Detection) pada endpoint yang dikonfigurasi (`DETECTION_STREAM_URL`).
  - Menerima dan melakukan dekode stream gambar berformat `float32` (*base64*).
  - Melakukan inferensi menggunakan model **InsightFace** untuk mendapatkan *embedding* dari wajah yang masuk.
  - Membandingkan wajah tersebut dengan data wajah karyawan.
  - Mengunggah bukti foto absensi ke MinIO.
  - Menembak API Backend utama.

### 3. `mobile_api.py` (Service API Tambahan)
Menyediakan API untuk interaksi aplikasi *mobile* jika diperlukan.

---

## ⚙️ Konfigurasi Mudah via `.env`

Ubah konfigurasi di file `.env` di direktori utama:

```properties
DETECTION_STREAM_URL=http://localhost:8000/api/v1/faces/stream
BACKEND_API_URL=http://localhost:5000/api/v1/live/recognition-events
ML_API_KEY=your-secret-api-key
DATABASE_URL=postgresql://user:pass@host:5432/db
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password
MINIO_BUCKET=recognition
```

---

## 🛠️ Persyaratan Sistem (Prerequisites)

- **OS**: Windows / Linux
- **Python 3**: Direkomendasikan menggunakan environment yang bersih (Miniconda atau venv).
- **PM2**: Node.js Process Manager (`npm install -g pm2`).

---

## 🚀 Cara Menjalankan (Deployment via PM2)

### Memeriksa upload MinIO

Saat branch `feat/recognition-fusion` di-push, job deploy menjalankan tes tulis/baca
MinIO dengan kredensial `.env` **sebelum** me-restart worker. Objek tes memakai
nama tetap `healthchecks/ai-recognition-upload-smoke.txt`, sehingga tidak
menumpuk pada setiap deploy. Tes ini memeriksa koneksi serta izin bucket;
keberhasilan pengenalan wajah tetap perlu diperiksa dari event nyata.
Job berikutnya memastikan PID worker tetap hidup setelah runner menyelesaikan
proses pembersihan job deploy.

Di server, tes yang sama bisa dijalankan manual:

```bash
cd /home/popos/projek-anakmagang/ai-recognition
.venv/bin/python check_minio_upload.py smoke
```

Setelah log worker menampilkan `Memproses: <nama> (<event>)` dan
`[MinIO] Snapshot tersimpan: recognition/snapshots/<id>.jpg`, periksa objek
terbaru tanpa menampilkan foto atau kredensial:

```bash
.venv/bin/python check_minio_upload.py recent --minutes 10
```

Wajah `UNKNOWN`, di luar jam absensi, terkena cooldown, atau `DRY_RUN=true`
tidak mengunggah snapshot.

Kita menggunakan **PM2** sebagai *process manager* agar aplikasi otomatis me-restart jika terjadi *crash* dan otomatis menyala saat server *reboot*.

1. **Install Dependencies Python**
   Pastikan Anda sudah berada di root direktori proyek, lalu jalankan:
   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan Aplikasi dengan PM2**
   Gunakan file konfigurasi `ecosystem.config.js` yang telah disediakan:
   ```bash
   pm2 start ecosystem.config.js
   ```

3. **Simpan Konfigurasi PM2**
   Agar aplikasi otomatis menyala saat server mati/reboot:
   ```bash
   pm2 save
   pm2 startup
   ```

4. **Monitoring Log**
   Untuk melihat log *real-time* dari worker AI:
   ```bash
   pm2 logs
   ```
   Atau buka file log spesifik di dalam folder `logs/`.

---

## 📁 Struktur Direktori

```text
ai-recognition-cpp/
│
├── .env                        # File konfigurasi utama
├── ecosystem.config.js         # Konfigurasi deployment PM2 (Pengganti Docker)
├── README.md                   # Dokumentasi ini
├── sync_enroll.py              # Script sinkronisasi data wajah
├── recognition.py              # Script utama AI Recognition Worker
├── mobile_api.py               # Script API untuk Mobile
├── requirements.txt            # Python dependencies
│
└── data/                       
    ├── embeddings.npy          # File tensor embedding seluruh wajah
    └── labels.json             # Pemetaan index embedding ke ID Karyawan
```

**Dikembangkan oleh Tim Developer Antigravity untuk Talangmas.**
