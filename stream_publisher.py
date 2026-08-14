import cv2
import subprocess
import time
import sys
import os
from dotenv import load_dotenv

# Load variable dari .env
load_dotenv()

# Konfigurasi RTSP Target (MediaMTX)
RTSP_URL = 'rtsp://localhost:8554/stream'

# Ambil kredensial kamera dari .env
RTSP_USER = os.getenv("RTSP_USER", "admin")
RTSP_PASS = os.getenv("RTSP_PASS", "")
RTSP_HOST = os.getenv("RTSP_HOST", "")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
RTSP_STREAM_PATH = os.getenv("RTSP_STREAM_PATH", "/cam/realmonitor?channel=1&subtype=1")

# Rangkai URL Kamera
CAMERA_SOURCE = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_HOST}:{RTSP_PORT}{RTSP_STREAM_PATH}"

print(f"Menghubungkan ke sumber kamera: rtsp://{RTSP_USER}:***@{RTSP_HOST}:{RTSP_PORT}{RTSP_STREAM_PATH}")

# Inisialisasi Kamera dengan URL dari .env
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    sys.exit(1)

# Mengambil properti kamera
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30 # Default fps jika tidak terbaca

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Kalkulasi Padding 16:9
target_ratio = 16 / 9
current_ratio = width / height

pad_top = pad_bottom = pad_left = pad_right = 0
out_width = width
out_height = height

if current_ratio < target_ratio:
    # Terlalu kotak, tambah pinggiran hitam di kiri-kanan (pillarbox)
    out_width = int(height * target_ratio)
    if out_width % 2 != 0: out_width += 1
    pad_total = out_width - width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
elif current_ratio > target_ratio:
    # Terlalu lebar, tambah pinggiran hitam di atas-bawah (letterbox)
    out_height = int(width / target_ratio)
    if out_height % 2 != 0: out_height += 1
    pad_total = out_height - height
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top

print(f"Resolusi asli: {width}x{height}. Setelah ditambah pinggiran hitam 16:9 menjadi {out_width}x{out_height} @ {fps} FPS")
print(f"Mem-publish stream ke {RTSP_URL}...")

# Konfigurasi perintah FFmpeg untuk push stream
# Mengambil input RAW dari stdin (pipe) dan mengonversinya ke format H.264 untuk RTSP
command = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f"{out_width}x{out_height}",
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    RTSP_URL
]

# Jalankan proses FFmpeg
try:
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
except FileNotFoundError:
    print("Error: FFmpeg tidak ditemukan. Pastikan FFmpeg sudah terinstal di sistem Anda.")
    sys.exit(1)

print("Streaming dimulai. Tekan 'q' pada jendela video untuk berhenti.")

# Buat window yang bisa di-resize dan menjaga rasio aspek (agar di tengah)
cv2.namedWindow('Streaming Out (Publisher)', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame dari kamera.")
        break
    
    # Tambahkan pinggiran hitam agar 16:9
    if pad_left > 0 or pad_top > 0:
        padded_frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padded_frame = frame

    # Optional: Tampilkan frame secara lokal untuk mengecek
    cv2.imshow('Streaming Out (Publisher)', padded_frame)
    
    # Tulis frame ke stdin FFmpeg untuk dipublish
    try:
        process.stdin.write(padded_frame.tobytes())
    except BrokenPipeError:
        print("Error: Koneksi FFmpeg terputus.")
        break
    
    # Keluar jika menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()
if process.stdin:
    process.stdin.close()
process.wait()
print("Streaming dihentikan.")
