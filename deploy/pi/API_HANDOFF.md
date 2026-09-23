# Kontrak API detection untuk tim FE dan face recognition

Dokumen ini menjelaskan API yang **sudah tersedia** pada branch `raspberry-pi5`.
`BASE_URL` adalah alamat HTTP server yang menjalankan `api_server.py`, misalnya
`http://<IP-SERVER-API>:8000`. Alamat tersebut belum ditetapkan di repository.
IP RTSP MediaMTX/kamera **bukan** alamat API. Service Pi mengirim hasil ke Redis;
API server membaca Redis yang sama dan meneruskannya ke klien.

## Konfigurasi integrasi

Pada Pi:

```dotenv
PI_CAMERA_ID=cam01
PI_STREAM_PATH=cam01
PI_BBOX_CHANNEL=face_detection_queue
PI_FACE_CHANNEL=face_preprocessed_queue
PI_REDIS_URL=tcp://<IP-REDIS>:6379
```

Pada server API:

```dotenv
REDIS_HOST=<IP-REDIS>
REDIS_PORT=6379
REDIS_CHANNEL=face_preprocessed_queue
BBOX_CHANNEL=face_detection_queue
API_HOST=0.0.0.0
API_PORT=8000
```

Kedua proses harus mengarah ke **Redis yang sama**. Konfigurasi API di atas
mengasumsikan Redis TCP tanpa autentikasi/TLS; implementasi `api_server.py`
sekarang memakai `REDIS_HOST`/`REDIS_PORT` saja. Bila server kantor memakai
autentikasi atau TLS, dukungan tersebut harus ditambahkan sebelum digunakan.
Jangan mengirim kredensial RTSP kepada tim FE atau recognition; mereka hanya
membutuhkan `BASE_URL` API yang dapat dijangkau dari jaringan masing-masing.

## Untuk tim FE: bbox per frame

`GET {BASE_URL}/api/v1/live-bboxes` adalah stream **Server-Sent Events (SSE)**
bertipe `text/event-stream`. Sambungkan sekali dan baca setiap pesan `data:`.
Pesan pertama adalah `{"status":"connected"}`; abaikan sebagai data bbox.

Contoh event ketika ada wajah:

```json
{
  "schema_version": "1.0",
  "camera_id": "cam01",
  "stream_path": "cam01",
  "session_id": "1800000000000-123456",
  "frame_id": 25,
  "received_at_ms": 1800000000000,
  "timestamp_ms": 1800000000100,
  "frame_width": 1280,
  "frame_height": 720,
  "bounding_boxes": [
    {
      "face_index": 0,
      "name": "Unknown",
      "confidence_score": 0.83,
      "bounding_box": {"x": 300, "y": 180, "width": 90, "height": 110},
      "landmarks": [[320, 215], [365, 215], [342, 240], [325, 263], [360, 263]]
    }
  ]
}
```

Ketika tidak ada wajah, server mengirim event baru dengan
`"bounding_boxes": []`. FE harus **mengganti** daftar kotak kamera tersebut,
bukan menambahkan ke daftar sebelumnya. Koordinat `x`, `y`, `width`, `height`
adalah piksel pada frame berukuran `frame_width` × `frame_height`.
Skalakan ke ukuran video yang tampil dan perhitungkan `object-fit`/letterbox.
Pilih kamera memakai `camera_id` atau `stream_path`; SSE ini bisa memuat beberapa
kamera sekaligus. `name` masih `Unknown` sampai sistem recognition mengirim
identitas melalui kontrak lain.

Contoh browser:

```js
const events = new EventSource(`${BASE_URL}/api/v1/live-bboxes`);
events.onmessage = ({data}) => {
  const event = JSON.parse(data);
  if (event.status === "connected" || event.camera_id !== "cam01") return;
  drawBoxes(event.bounding_boxes, event.frame_width, event.frame_height);
};
```

Video diambil dari endpoint WebRTC/HLS yang diberikan tim MediaMTX. API ini
hanya menyediakan metadata bbox; `/api/v1/video_feed/cam01` adalah jalur MJPEG
lama dan node Pi tidak menerbitkan preview ke sana. FE perlu menghapus kotak
jika event berhenti atau sudah basi (mulai dari batas 1 detik, lalu ukur bersama
latency video sebenarnya). `frame_id` adalah nomor decode lokal Pi, bukan
identitas frame WebRTC/HLS; sinkronisasi presisi memerlukan timestamp video
yang disepakati bersama tim MediaMTX.

## Untuk tim face recognition: satu crop per wajah

`GET {BASE_URL}/api/v1/faces/stream` adalah SSE. Setelah event awal
`{"status":"connected"}`, setiap event berisi **satu wajah**. Crop yang dikirim
adalah gambar JPEG 112×112 dalam Base64, hasil alignment lima landmark.
Ini **belum** embedding atau identitas karyawan.

Contoh event:

```json
{
  "received_at": "2026-09-23T11:45:23Z",
  "camera_id": "cam01",
  "stream_path": "cam01",
  "session_id": "1800000000000-123456",
  "frame_id": 25,
  "face_index": 0,
  "timestamp_ms": 1800000000100,
  "received_at_ms": 1800000000000,
  "frame_width": 1280,
  "frame_height": 720,
  "confidence_score": 0.83,
  "bounding_box": {"x": 300, "y": 180, "width": 90, "height": 110},
  "landmarks": [[320, 215], [365, 215], [342, 240], [325, 263], [360, 263]],
  "image_format": "jpeg_base64_112x112",
  "face_image_base64": "<BASE64-JPEG>"
}
```

Decode `face_image_base64` sebagai JPEG, lalu jalankan preprocessing input
sesuai model recognition. Gunakan `(camera_id, session_id, frame_id, face_index)`
sebagai kunci wajah dalam sesi, bukan `frame_id` saja. Tim recognition menentukan
embedding, hasil pencocokan, dan identitas karyawan; API detection ini belum
menyediakan endpoint untuk menulis hasil recognition atau keputusan absensi.

Endpoint pendukung:

| Endpoint | Respons saat ini |
|---|---|
| `GET /api/v1/faces/latest` | JSON wajah terakhir; `404` jika belum ada wajah |
| `GET /api/v1/faces/history?limit=10` | JSON `{"count": N, "faces": [...]}`; memori maksimum 50 wajah |
| `GET /health` | Status API dan koneksi Redis; periksa nilai field `redis` |
| `GET /docs` | Dokumentasi Swagger yang dibuat FastAPI |

Tes koneksi SSE dari terminal:

```bash
curl -N http://<IP-SERVER-API>:8000/api/v1/live-bboxes
curl -N http://<IP-SERVER-API>:8000/api/v1/faces/stream
```

API membaca Redis Pub/Sub: pesan yang dikirim saat subscriber mati tidak
diputar ulang. Riwayat hanya berada di memori proses API dan hilang saat restart;
endpoint `/health` dapat menjawab HTTP 200 walaupun field `redis` bernilai
`disconnected`. Untuk absensi yang wajib tahan kehilangan pesan, tim server perlu
penyimpanan/transaksi atau message broker persisten tersendiri. SSE mengirim
hasil inference, bukan setiap frame video. Ketika tidak ada wajah, FE menerima
bbox kosong, sedangkan stream recognition tidak mengirim crop.

## Status pengujian

Payload bbox dan crop, SSE FE, serta mode CPU model diuji secara lokal.
Alamat server API, Redis kantor, autentikasi, routing jaringan FE, dan pemutaran
video MediaMTX harus diuji saat deployment; dokumen ini belum menyatakan bahwa
endpoint sudah dapat diakses dari perangkat tim lain.
