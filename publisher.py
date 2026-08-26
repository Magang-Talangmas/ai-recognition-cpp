#!/usr/bin/env python3
"""
Multi-Camera Launcher — 1 .env untuk banyak kamera
====================================================
Baca satu file .env, loop lewat CAM1_..CAMN_, lalu jalankan pipeline
yang sesuai per kamera:
  - type=cctv  -> FFmpeg pure copy (tanpa OpenCV, CPU nyaris 0%)
  - type=iriun -> OpenCV bridge + Queue(maxsize=1) -> pipe ke FFmpeg

Cara pakai:
    cp .env.example .env   # lalu edit sesuai kamera kamu
    pip install python-dotenv opencv-python --break-system-packages
    python3 multi_camera_launcher.py
"""

import os
import sys
import subprocess
import threading
import queue
import signal
import time

try:
    from dotenv import load_dotenv
except ImportError:
    print("Butuh python-dotenv. Install: pip install python-dotenv --break-system-packages")
    sys.exit(1)

load_dotenv()

running_processes = []   # subprocess.Popen (ffmpeg untuk jalur cctv & iriun)
running_threads = []     # thread untuk jalur iriun
stop_event = threading.Event()


def load_camera_configs():
    """Baca semua CAM{N}_* dari .env berdasarkan CAM_COUNT."""
    count = int(os.getenv("CAM_COUNT", "0"))
    if count == 0:
        print("CAM_COUNT belum diset atau 0. Cek file .env kamu.")
        sys.exit(1)

    cameras = []
    for i in range(1, count + 1):
        prefix = f"CAM{i}_"
        source = os.getenv(prefix + "SOURCE")
        if not source:
            print(f"[WARNING] {prefix}SOURCE kosong, kamera {i} dilewati.")
            continue
        cameras.append({
            "id": i,
            "name": os.getenv(prefix + "NAME", f"camera_{i}"),
            "type": os.getenv(prefix + "TYPE", "cctv").lower(),
            "source": source,
            "path": os.getenv(prefix + "PATH", f"cam{i}"),
        })
    return cameras


def mediamtx_rtsp_url(path: str) -> str:
    host = os.getenv("MEDIAMTX_HOST", "localhost")
    port = os.getenv("MEDIAMTX_RTSP_PORT", "8554")
    return f"rtsp://{host}:{port}/{path}"


def start_cctv_camera(cam: dict):
    """Jalur CCTV/IP Camera: FFmpeg murni, -c:v copy, tanpa decode/encode."""
    target = mediamtx_rtsp_url(cam["path"])
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", cam["source"],
        "-c:v", "copy",
        "-c:a", "copy",
        "-f", "rtsp",
        target,
    ]
    print(f"[{cam['name']}] Jalur CCTV (pure copy) -> {target}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    running_processes.append((cam["name"], proc))


def start_iriun_camera(cam: dict):
    """Jalur Iriun/USB: OpenCV capture (Queue maxsize=1, drop frame lama)
    -> pipe raw bytes ke FFmpeg -> FFmpeg encode H.264 ultrafast/zerolatency."""
    try:
        import cv2
    except ImportError:
        print("Butuh opencv-python untuk jalur iriun. Install: pip install opencv-python --break-system-packages")
        return

    source = cam["source"]
    # source dari .env bisa berupa index device (angka) atau path
    cam_index = int(source) if source.isdigit() else source

    cap = cv2.VideoCapture(cam_index)
    # Set buffer internal OpenCV ke 1 untuk meminimalkan delay pada network stream (seperti IP Webcam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[{cam['name']}] Gagal membuka kamera index/source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    target = mediamtx_rtsp_url(cam["path"])
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "rtsp",
        target,
    ]
    print(f"[{cam['name']}] Jalur Iriun (OpenCV bridge) -> {target} ({width}x{height}@{fps})")
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    running_processes.append((cam["name"], ffmpeg_proc))

    frame_queue = queue.Queue(maxsize=1)

    def capture_loop():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # drop frame lama: kalau queue penuh, buang isi lama, taruh yang baru
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        cap.release()

    def writer_loop():
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                print(f"[{cam['name']}] Pipe ke FFmpeg terputus.")
                break

    t1 = threading.Thread(target=capture_loop, daemon=True)
    t2 = threading.Thread(target=writer_loop, daemon=True)
    t1.start()
    t2.start()
    running_threads.extend([t1, t2])


def shutdown(signum=None, frame=None):
    print("\nMenghentikan semua kamera...")
    stop_event.set()
    for name, proc in running_processes:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"  - {name}: dihentikan")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    cameras = load_camera_configs()
    print(f"Ditemukan {len(cameras)} kamera aktif dari .env\n")

    for cam in cameras:
        if cam["type"] == "cctv":
            start_cctv_camera(cam)
        elif cam["type"] == "iriun":
            start_iriun_camera(cam)
        else:
            print(f"[WARNING] Tipe kamera '{cam['type']}' tidak dikenal untuk {cam['name']}, dilewati.")

    print("\nSemua kamera dijalankan. Tekan Ctrl+C untuk berhenti.")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
