# Raspberry Pi 5: node detection dan preprocessing

Branch `raspberry-pi5` menyediakan executable terpisah `face_detector_pi`.
Kontrak untuk tim FE dan face recognition: [API_HANDOFF.md](API_HANDOFF.md).
Target: Raspberry Pi 5 dengan OS Linux ARM64, satu stream, inferensi CPU.
Recognition, database absensi, FaceFusion, API, dan video untuk browser tetap di server.
Ini implementasi untuk diuji di perangkat, bukan klaim bahwa deployment Pi sudah selesai.

## Melihat kamera dan bounding box di laptop

Jalankan preview lokal untuk memeriksa wajah yang terdeteksi dan confidence pada
video RTSP. Titik landmark tidak digambar. Preview memanggil executable `face_detector_pi` melalui
mode `--preview-stdio`, sehingga model, decoding SCRFD, NMS, dan alignment memakai
kode C++ yang sama. Python membaca stream dan menggambar bbox terbaru di atas video
live. Tidak perlu Redis, API, atau FE untuk pengujian visual ini.
Mode ini tidak publish hasil recognition/absensi dan tidak mengubah service Pi.

Pada komputer Windows pengembangan yang sudah disiapkan:

```powershell
cd D:\ai-recognition-cpp
.\output\pi-validation-venv\Scripts\python.exe tools/preview_pi.py
```

Kamera default `rtsp://192.168.77.100:8554/cam01` diambil dari konfigurasi contoh.
Gunakan `--config .env.pi` untuk konfigurasi lain. Tekan **Q** atau **Esc**, atau
tutup jendela untuk berhenti. Capture, inference, dan tampilan bekerja terpisah:
video ditampilkan maksimal 30 FPS, detection ditargetkan maksimal 10 FPS pada
laptop, tanpa menunggu inference untuk menggambar frame berikutnya. Keduanya bisa
diatur dengan `--display-fps 30 --detection-fps 10`. Nilai tersebut batas, bukan
jaminan throughput. Konfigurasi service Pi tetap 2 FPS untuk mengendalikan beban.

Hanya frame terbaru diambil setiap kali detector siap; tidak ada antrean inference.
Kotak diperhalus dengan interpolasi singkat 40 ms, dan hasil berumur lebih dari
1 detik atau berasal dari sesi koneksi lama dibuang. Jika frame baru tidak diterima
selama 1 detik, tampilan menjadi status reconnect. Landmark tetap dihitung karena
dibutuhkan alignment; yang dihapus hanya tampilan titik kuningnya.

Karena video tidak menunggu detector, bbox berasal dari frame sedikit lebih lama
daripada video yang sedang ditampilkan. `age` mengukur umur hasil sejak frame
diterima aplikasi, bukan latency kamera-ke-layar. Log tiap 5 detik menampilkan FPS
video yang dirender, FPS detection aktual, waktu inference, dan umur hasil.
Delay kamera, encoder, jaringan, dan buffer decoder tidak tercakup pada angka ini.

Untuk clone baru, build executable dahulu, kemudian siapkan Python:

```bash
python -m venv .venv-preview
# Windows: .venv-preview/Scripts/python.exe; Linux: .venv-preview/bin/python
.venv-preview/bin/python -m pip install opencv-python
.venv-preview/bin/python tools/preview_pi.py --exe build-pi/face_detector_pi
```

Pada Windows gunakan interpreter `Scripts/python.exe` dan path executable Windows
hasil build sendiri melalui `--exe`. Folder environment dan binary lokal tidak
disertakan dalam Git. Preview GUI memerlukan desktop; pada Pi headless jalankan
tes tanpa GUI atau lakukan preview di laptop.

Tes terbatas dan penyimpanan snapshot lokal (opsional):

```powershell
.\output\pi-validation-venv\Scripts\python.exe tools/preview_pi.py --headless --frames 5 --snapshot output/preview.jpg
```

Secara default tidak ada rekaman atau snapshot yang disimpan. Pengujian visual
di laptop membuktikan model dapat memproses stream; tidak membuktikan throughput Pi
atau integrasi Redis/FE. Kotak bisa tidak muncul pada wajah kecil, tertutup, atau
menoleh jauh; periksa variasi pose dan pencahayaan sebelum menilai akurasi.

## Alur yang dijalankan

1. Media stream menyediakan `rtsp://192.168.77.100:8554/cam01`.
2. OpenCV dengan FFmpeg membaca RTSP melalui TCP dan decode ke BGR.
3. Thread capture menyimpan satu frame terbaru; frame lama diganti, bukan ditumpuk.
4. Loop inference mengambil frame baru yang belum diproses dan belum kedaluwarsa.
5. SCRFD 2.5G memproses resize/padding 640×640, RGB, normalisasi input, lalu
   menghasilkan confidence, bounding box, dan lima landmark. Filter confidence 0.5
   dan NMS 0.4 mengikuti implementasi sebelumnya.
6. Bounding box dikembalikan ke koordinat frame sumber. Alignment lima landmark
   menghasilkan crop wajah 112×112. Blur filter dan CLAHE tidak diaktifkan oleh jalur ini.
7. Satu pesan bbox per frame dikirim ke `face_detection_queue`, termasuk
   `bounding_boxes: []` ketika tidak ada wajah. Setiap crop dikirim sebagai JPEG
   Base64 ke `face_preprocessed_queue`, dengan metadata yang menghubungkan keduanya.
8. API server meneruskan bbox melalui `/api/v1/live-bboxes` ke FE. Tim recognition
   subscribe crop langsung dari Redis atau menggunakan `/api/v1/faces/stream`.

Pi tidak membuat embedding, nama karyawan, keputusan absensi, atau video face swap.
Nama pada bbox detection adalah `Unknown`, bukan hasil penilaian recognition.
Crop berupa gambar JPEG, bukan tensor float atau embedding. Tim recognition tetap
perlu decode dan menerapkan normalisasi yang sesuai dengan modelnya; jangan deteksi
dan alignment ulang tanpa kebutuhan.

## Pengaturan ringan dan batasnya

| Pengaturan awal | Nilai / perilaku |
|---|---|
| Kamera | Satu proses, satu `cam01` |
| Eksekusi model | ONNX Runtime CPU, tanpa CUDA |
| Thread inference/OpenCV | 1 |
| Input detector | 640×640 |
| Crop | 112×112, JPEG quality 90 |
| `PI_INFERENCE_FPS` | 2: jeda 500 ms setelah inference; FPS aktual lebih rendah |
| Buffer aplikasi | Satu frame terbaru, umur maksimum 2 detik |
| Reconnect RTSP | Backoff 1–10 detik |
| Reconnect Redis | 3 detik; tidak menyimpan backlog |
| Preview JPEG/MJPEG | Tidak dibuat di Pi |

Frame decode masih mengikuti stream sumber; membatasi inference tidak otomatis
menurunkan biaya decode. Jika CPU/suhu tinggi, turunkan FPS/resolusi **substream
di sumber** dahulu, sambil memastikan ukuran wajah masih cukup untuk deteksi.
Mulai satu kamera, gunakan Ethernet, catu daya yang memadai dan pendingin aktif.
Jangan menambah kamera atau meningkatkan FPS sebelum mengukur suhu, RAM, latency,
akurasi wajah kecil, dan kondisi orang berjalan pada perangkat sebenarnya.

## Build di Pi

Gunakan Raspberry Pi OS 64-bit atau Linux ARM64 yang menyediakan OpenCV >=4.5.2
dengan backend FFmpeg. `uname -m` harus mengeluarkan `aarch64`.
Binary Windows/x64 dan SDK CUDA lama tidak dapat disalin untuk dijalankan di Pi.

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl pkg-config libopencv-dev libhiredis-dev libssl-dev
git clone --branch raspberry-pi5 https://github.com/Magang-Talangmas/ai-recognition-cpp.git
cd ai-recognition-cpp
curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz -o /tmp/ort-pi.tgz
sudo tar -xzf /tmp/ort-pi.tgz -C /opt
export ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-aarch64-1.20.1
bash deploy/pi/build.sh
cp deploy/pi/pi.env.example .env.pi
```

Build mengunduh dependency C++ sehingga memerlukan internet. Script menjalankan
tes kontrak payload dan smoke test model asli tanpa koneksi kamera/Redis.
SDK harus tetap tersedia di lokasi yang sama saat executable dijalankan.

Edit `.env.pi`, terutama `PI_REDIS_URL`, menjadi Redis **yang sama dengan yang
digunakan API dan tim recognition**. Nilai localhost hanya contoh bootstrap;
bukan alamat server kantor yang sudah diketahui. Gunakan format URI redis++
`tcp://HOST:6379` (opsional kredensial sesuai konfigurasi server).
Variabel environment mengalahkan isi file. Jangan commit kredensial.

```bash
./build-pi/face_detector_pi --config .env.pi --validate-config
./build-pi/face_detector_pi --config .env.pi --check-model
./build-pi/face_detector_pi --config .env.pi
```

Validasi konfigurasi memeriksa parameter, bukan konektivitas. Smoke test hanya
menguji model dengan gambar kosong, bukan akurasi pada wajah nyata.
Pastikan Pi dapat mengakses kamera port 8554 dan Redis melalui jaringan kantor.
Hentikan foreground dengan Ctrl+C sebelum mengaktifkan service.

## Service otomatis

Setelah foreground berhasil, salin hasil build ke lokasi service. Perintah ini
untuk instalasi awal; jangan menyalin konfigurasi di atas instalasi aktif tanpa review.

```bash
sudo useradd --system --user-group --no-create-home --shell /usr/sbin/nologin face-detection
sudo install -d /opt/face-detection/build-pi /opt/face-detection/models
sudo install -m 755 build-pi/face_detector_pi /opt/face-detection/build-pi/
sudo install -m 644 models/scrfd_2.5g_kps.onnx /opt/face-detection/models/
sudo install -d -m 750 -o root -g face-detection /etc/face-detection
sudo install -m 640 -o root -g face-detection .env.pi /etc/face-detection/pi.env
sudo install -m 644 deploy/pi/face-detection-pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now face-detection-pi
sudo journalctl -u face-detection-pi -f
```

Gunakan path model default relatif terhadap `/opt/face-detection`, atau path absolut
yang dapat dibaca user service. SDK ONNX berada di `/opt`, bukan home karena service
mengaktifkan `ProtectHome`. Verifikasi `ldd /opt/face-detection/build-pi/face_detector_pi`
tidak mengandung `not found`. Service restart jika proses gagal; reconnect stream
dan Redis juga dilakukan di dalam aplikasi.

## Kontrak server, FE, dan recognition

Pada **API server** gunakan versi `api_server.py` dan `bbox_events.py` dari branch ini:

```dotenv
REDIS_HOST=ALAMAT_REDIS_KANTOR
REDIS_PORT=6379
REDIS_CHANNEL=face_preprocessed_queue
BBOX_CHANNEL=face_detection_queue
```

API contoh saat ini menghubungkan Redis melalui host/port tanpa kredensial; jika
Redis kantor memerlukan autentikasi/TLS, integrasi API tersebut perlu disesuaikan
sebelum deployment. Akses Redis harus terbatas pada jaringan yang berwenang.

`BBOX_CHANNEL` bersifat opt-in. Jika tidak diset, API tetap memakai agregasi per-face
lama. Jika diset, sumber bbox seluruh kamera API itu harus mengirim pesan per-frame;
gunakan instance API terpisah untuk campuran node lama yang belum mengirim channel ini.

Contoh satu pesan bbox (koordinat piksel frame sumber):

```json
{
  "schema_version": "1.0", "camera_id": "cam01", "stream_path": "cam01",
  "session_id": "boot-unique", "frame_id": 25,
  "received_at_ms": 1800000000000, "timestamp_ms": 1800000000100,
  "frame_width": 1280, "frame_height": 720,
  "bounding_boxes": [{"face_index": 0, "name": "Unknown", "confidence_score": 0.95,
    "bounding_box": {"x": 100, "y": 80, "width": 120, "height": 150},
    "landmarks": [[130,120],[185,120],[157,150],[135,185],[179,185]]}]
}
```

Pesan crop menggunakan metadata frame yang sama, ditambah `face_index`,
`bounding_box`, `confidence_score`, `landmarks`,
`image_format: "jpeg_base64_112x112"`, dan `face_image_base64`.
Identitas unik hasil adalah `(camera_id, session_id, frame_id, face_index)`.

FE menonton video melalui layanan streaming server (misalnya WebRTC/HLS), lalu
menggambar overlay dari SSE `/api/v1/live-bboxes`. Samakan `camera_id`, skalakan
koordinat dengan ukuran video yang benar, perhitungkan letterbox/object-fit, ganti
seluruh daftar bbox saat event baru, dan hapus overlay ketika daftar kosong atau
event sudah basi. Event awal `{"status":"connected"}` bukan bbox.
Endpoint `/api/v1/video_feed/cam01` tidak mendapat preview dari node Pi ini.
Menjalankan detector saja tidak otomatis membuat kotak pada player FE.

`frame_id` adalah urutan decode lokal; `received_at_ms` waktu Pi selesai menerima
frame, **bukan PTS kamera**. SSE dan video browser punya latency berbeda sehingga
overlay tidak dijamin presisi per-frame. Sinkronisasi waktu Pi/server dan TTL overlay
perlu diuji; sinkronisasi video presisi memerlukan kontrak timestamp tambahan.
Frame asli juga tidak disimpan/dikirim oleh node ini: FaceFusion perlu jalur frame
asli tersendiri dan penyelarasan yang disepakati dengan tim recognition/streaming.

Redis yang dipakai adalah Pub/Sub, bukan antrean persisten. Subscriber yang offline
kehilangan pesan; kegagalan publish dapat menghasilkan sebagian batch saja.
Sistem absensi tetap memerlukan deduplikasi dan penyimpanan transaksi di server.

## Pengujian dan penerimaan deployment

- CTest: kontrak Base64/JPEG, metadata, bbox kosong/multi-face, validasi config,
  dan model SCRFD asli pada gambar kosong.
- Python: `python -m unittest discover -s tests -p 'test_*.py' -v`
  (install `fastapi uvicorn redis python-dotenv httpx` di environment tes).
- Workflow `Pi CPU validation` menjalankan build ARM64 dan tes; tidak deploy ke server.
- Uji di Pi: satu wajah dan banyak wajah, orang meninggalkan gambar, stream/Redis
  diputus lalu tersambung, restart service, pemakaian RAM/CPU/suhu stabil, crop
  diterima recognition dan overlay FE cocok dengan kamera.
- Catat FPS aktual, latency, suhu dan throttling selama minimal satu sesi kerja
  representatif sebelum menentukan target produksi. Tidak ada klaim benchmark Pi
  sebelum perangkat dan stream benar-benar diuji.

Alamat/IP SSH Pi dan alamat Redis kantor dibutuhkan pada tahap deployment dan
pengujian jaringan, bukan untuk membuat branch dan menyiapkan implementasi ini.
