# SCRFD Face Only untuk proyek PPE

Folder `face_service/` adalah microservice deteksi wajah yang terpisah dari
proses recognition demo pada `main.py`. Model `models/scrfd.hef` dikompilasi
dari `BE/models/face.onnx` proyek PPE untuk **Hailo-8** dengan DFC 3.34.0.
SHA-256 HEF: `255072095def732c4ccc5629cb763d995707a5b249f9dac182e677b807ff50ec`.
Model ini hanya mendeteksi kotak wajah; tidak melakukan ArcFace, pencocokan
identitas, penyimpanan wajah, atau pembuatan insiden APD.

## Menjalankan di Pi dengan Hailo-8

HailoRT 4.x (`hailo_platform`) harus terpasang pada Python yang menjalankan
service. Karena APD dan wajah memakai satu chip dari proses berbeda, aktifkan
HailoRT multi-process service pada Pi:

```bash
sudo systemctl enable --now hailort.service
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn face_service.app:app --host 127.0.0.1 --port 8001 --workers 1
```

Gunakan `--host 127.0.0.1`: endpoint kendali ini ditujukan hanya untuk backend
PPE di mesin yang sama. Model baru dibuka di Hailo ketika diaktifkan dan
dilepas ketika dimatikan.

Di `.env` backend PPE, atur `PPE_FACE_BACKEND=service` dan
`PPE_FACE_SERVICE_URL=http://127.0.0.1:8001`. Backend memakai
`GET /health`, `POST /enabled` dengan JSON `{"enabled": true/false}`,
dan `POST /infer` dengan body JPEG; hasilnya `{"rows": [[x,y,w,h,score], ...]}`.
Sakelar Face Only di halaman Pengaturan PPE mengendalikan status ini.
`FACE_HEF_PATH` dapat mengganti lokasi default `models/scrfd.hef`.

HEF menerima RGB uint8 640×640 yang sudah di-resize tanpa memperbesar gambar
kecil dan dipad hitam di kiri atas. Normalisasi `(RGB-127.5)/128` sudah berada
di dalam HEF. Microservice mengubah output score, bbox dan landmark mentah
pada stride 8/16/32 menjadi kotak wajah dengan sigmoid dan NMS di CPU.
Sembilan output dipetakan berdasarkan `output_layer1` sampai `output_layer9`;
urutan channel diverifikasi terhadap ONNX asli. Ketepatan numerik hasil
inferensi dan penggunaan dua model pada satu Hailo-8 masih perlu diuji di Pi.

Panduan Hailo untuk beberapa proses:
https://github.com/hailo-ai/hailo-apps/blob/main/doc/user_guide/running_parallel.md
