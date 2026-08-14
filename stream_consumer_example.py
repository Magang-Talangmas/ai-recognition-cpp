import cv2
import time

# URL RTSP dari MediaMTX yang disediakan oleh tim Media Server
# Tim ML dan FE bisa mengubah URL ini sesuai kebutuhan (misal stream, stream_2)
RTSP_URL = 'rtsp://localhost:8554/stream'

print(f"Menghubungkan ke stream: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)

# Mengatur ukuran buffer sekecil mungkin agar meminimalisir latency (opsional)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print(f"Error: Tidak dapat menarik stream dari {RTSP_URL}")
    print("Pastikan MediaMTX menyala dan ada publisher yang memancarkan stream ke URL tersebut.")
    exit(1)

print("Berhasil terhubung! Menarik frame...")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Koneksi terputus atau stream berhenti.")
        break

    # Di sini tim ML akan melakukan pemrosesan mereka (misal: face detection)
    # ...
    
    # Untuk contoh ini, kita hanya menampilkan framenya
    cv2.imshow('Streaming In (Consumer / Tim ML)', frame)

    # Hitung FPS
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        print(f"Menerima stream dengan kecepatan ~{30/elapsed:.2f} FPS")
        start_time = time.time()

    # Keluar jika menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Selesai.")
