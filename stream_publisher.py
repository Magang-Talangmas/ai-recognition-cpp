import cv2
import subprocess
import time
import sys
import os
import threading
import queue
from dotenv import load_dotenv

# Load variable dari .env
load_dotenv()

class VideoStreamWidget:
    def __init__(self, src):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=1)
        self.stopped = False
        
        if self.capture.isOpened():
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    self.stopped = True
                    break
                
                # If queue is full, discard the old frame and put the new one
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put((ret, frame))
            else:
                self.stopped = True

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def get(self, prop):
        return self.capture.get(prop)

    def isOpened(self):
        return self.capture.isOpened()

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.capture.isOpened():
            self.capture.release()

class FFmpegWriter:
    def __init__(self, process):
        self.process = process
        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            try:
                frame_bytes = self.q.get(timeout=0.1)
                self.process.stdin.write(frame_bytes)
                self.process.stdin.flush()
            except queue.Empty:
                pass
            except BrokenPipeError:
                print("Error: Koneksi FFmpeg terputus.")
                self.stopped = True
                break
            except Exception as e:
                print("FFmpeg writer error:", e)
                self.stopped = True
                break

    def write(self, frame_bytes):
        if self.stopped:
            return
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(frame_bytes)

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)


# Konfigurasi RTSP Target (MediaMTX)
RTSP_URL = 'rtsp://localhost:8554/stream'

# Ambil tipe kamera aktif dari .env
ACTIVE_CAMERA = os.getenv("ACTIVE_CAMERA", "cctv").lower().strip()

if ACTIVE_CAMERA == "cctv":
    # Konfigurasi CCTV RTSP
    RTSP_USER = os.getenv("RTSP_USER", "admin")
    RTSP_PASS = os.getenv("RTSP_PASS", "")
    RTSP_HOST = os.getenv("RTSP_HOST", "")
    RTSP_PORT = os.getenv("RTSP_PORT", "554")
    RTSP_STREAM_PATH = os.getenv("RTSP_STREAM_PATH", "/cam/realmonitor?channel=1&subtype=1")
    CAMERA_SOURCE = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_HOST}:{RTSP_PORT}{RTSP_STREAM_PATH}"
    print(f"Menghubungkan ke sumber kamera CCTV: rtsp://{RTSP_USER}:***@{RTSP_HOST}:{RTSP_PORT}{RTSP_STREAM_PATH}")

elif ACTIVE_CAMERA == "phone_usb" or ACTIVE_CAMERA == "phone":
    # Konfigurasi USB/Webcam
    camera_index = os.getenv("PHONE_CAMERA_INDEX", "0")
    try:
        CAMERA_SOURCE = int(camera_index)
    except ValueError:
        CAMERA_SOURCE = 0
    print(f"Menghubungkan ke USB/Webcam Kamera HP dengan index: {CAMERA_SOURCE}")

elif ACTIVE_CAMERA == "phone_ipwebcam":
    # Konfigurasi IP Webcam
    CAMERA_SOURCE = os.getenv("PHONE_IP_CAMERA_URL", "")
    if not CAMERA_SOURCE:
        print("Error: PHONE_IP_CAMERA_URL belum di-set di .env")
        sys.exit(1)
    print(f"Menghubungkan ke IP Webcam Kamera HP: {CAMERA_SOURCE}")

else:
    print(f"Error: Konfigurasi ACTIVE_CAMERA '{ACTIVE_CAMERA}' tidak valid.")
    sys.exit(1)

# Inisialisasi Kamera dengan class yang menggunakan threading
cap = VideoStreamWidget(CAMERA_SOURCE)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    sys.exit(1)

# Mengambil properti kamera
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30 # Default fps jika tidak terbaca

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Kalkulasi Crop 16:9 (Center Crop)
target_ratio = 16 / 9
current_ratio = width / height

crop_y1 = 0
crop_y2 = height
crop_x1 = 0
crop_x2 = width

if current_ratio < target_ratio:
    # Terlalu kotak (contoh 4:3), potong atas-bawah agar jadi 16:9
    new_height = int(width / target_ratio)
    crop_total = height - new_height
    crop_y1 = crop_total // 2
    crop_y2 = height - (crop_total - crop_y1)
elif current_ratio > target_ratio:
    # Terlalu lebar, potong kiri-kanan agar jadi 16:9
    new_width = int(height * target_ratio)
    crop_total = width - new_width
    crop_x1 = crop_total // 2
    crop_x2 = width - (crop_total - crop_x1)

out_width = crop_x2 - crop_x1
out_height = crop_y2 - crop_y1

print(f"Resolusi asli: {width}x{height} @ {fps} FPS")
print(f"Setelah di-crop ke 16:9 menjadi {out_width}x{out_height}")
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
    ffmpeg_writer = FFmpegWriter(process)
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
    
    # Crop frame agar 16:9 (Menghilangkan blank space di atas/bawah dari kamera bawaan)
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Optional: Tampilkan frame secara lokal untuk mengecek
    cv2.imshow('Streaming Out (Publisher)', cropped_frame)
    
    # Tulis frame ke stdin FFmpeg untuk dipublish
    ffmpeg_writer.write(cropped_frame.tobytes())
    if ffmpeg_writer.stopped:
        break
    
    # Keluar jika menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()
ffmpeg_writer.release()
if process.stdin:
    process.stdin.close()
process.wait()
print("Streaming dihentikan.")
