import cv2
import struct
import time
import sys

# Ambil URL dari argumen pertama, kalau tidak ada pakai fallback
rtsp_url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://192.168.77.171:8554/stream"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("[Python Proxy] Gagal membuka RTSP Stream")
    sys.exit(1)

print("[Python Proxy] RTSP Terbuka. Menunggu C++ connect ke Named Pipe...")

try:
    # Buka named pipe yang dibuat oleh C++ (pipe harus sudah dibuat oleh C++ duluan)
    pipe = open(r'\\.\pipe\rtsp_pipe', 'wb')
    print("[Python Proxy] Terhubung ke Named Pipe!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Python Proxy] Frame kosong, reconnecting...")
            time.sleep(1)
            cap.open(rtsp_url)
            continue
            
        # Encode ke JPG untuk menghemat bandwidth pipe
        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
            
        # Kirim ukuran buffer (4 bytes integer) lalu data JPG-nya
        size = len(buf)
        pipe.write(struct.pack('<I', size))
        pipe.write(buf.tobytes())
        pipe.flush()
        
        # Jeda dikit biar gak bikin CPU 100%
        time.sleep(0.01)

except Exception as e:
    print("[Python Proxy] Error:", e)
finally:
    cap.release()
