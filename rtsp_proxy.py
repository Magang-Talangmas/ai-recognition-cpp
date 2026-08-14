import cv2
import struct
import time
import sys

# Ambil URL dari argumen pertama, kalau tidak ada pakai fallback
rtsp_url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://192.168.77.171:8554/stream"

# Jika input berupa angka tunggal (0, 1, 2), ubah jadi integer untuk Webcam USB/DroidCam
video_source = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url
cap = cv2.VideoCapture(video_source)

# Set buffer size sekecil mungkin agar tidak ada antrean frame (delay/lag)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
            
        # Encode ke JPG untuk menghemat bandwidth pipe (kualitas 70 agar lebih cepat)
        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
            
        # Kirim ukuran buffer (4 bytes integer) lalu data JPG-nya
        size = len(buf)
        pipe.write(struct.pack('<I', size))
        pipe.write(buf.tobytes())
        pipe.flush()
        
        # HAPUS time.sleep(0.01) di sini! cap.read() sudah blocking sesuai FPS asli kamera.
        # Adanya sleep sebelumnya membuat Windows menambah delay ~15ms ekstra per frame.

except Exception as e:
    print("[Python Proxy] Error:", e)
finally:
    cap.release()
