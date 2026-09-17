import requests
import time
import subprocess
import os
from dotenv import load_dotenv

# Load kredensial dari .env
load_dotenv()
API_USER = os.getenv("RTSP_USER", "admin")
API_PASS = os.getenv("RTSP_PASS", "bismillah123")

# Konfigurasi MediaMTX API
MEDIAMTX_API_URL = os.getenv("MEDIAMTX_API_URL", "http://127.0.0.1:9997/v3/paths/list")
MEDIAMTX_RTSP_BASE = os.getenv("MEDIAMTX_RTSP_BASE", "rtsp://127.0.0.1:8554")

# Path ke program C++ Face Preprocessor
if os.name == 'nt':
    CPP_EXECUTABLE = os.path.join("build", "Release", "face_preprocessor.exe")
else:
    CPP_EXECUTABLE = os.path.join("build", "face_preprocessor")

# Dictionary untuk melacak kamera yang sedang berjalan: { name: subprocess_object }
active_cameras = {}

def get_active_paths():
    """Mengambil daftar kamera aktif dari MediaMTX API dengan Authentication."""
    try:
        # Coba akses tanpa password sesuai kata Tim Device 1
        response = requests.get(MEDIAMTX_API_URL, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            active_names = []
            if "items" in data:
                for item in data["items"]:
                    if "name" in item:
                        active_names.append(item["name"])
            return active_names
        else:
            print(f"[Supervisor] API Menolak Akses. Kode Status: {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"[Supervisor] Gagal menghubungi API MediaMTX: {e}")
        return []

def run_supervisor():
    print("=================================================")
    print("  AI Supervisor (Auto-Discovery Camera) AKTIF!   ")
    print("=================================================")
    print(f"Target API: {MEDIAMTX_API_URL}")
    print(f"Login API: User='{API_USER}', Password='{API_PASS}'")
    print("Menunggu kamera nyala...\n")

    try:
        while True:
            current_paths = get_active_paths()
            
            # 1. Cek apakah ada kamera BARU yang menyala
            for path_name in current_paths:
                if path_name not in active_cameras:
                    print(f"[+] Kamera Baru Terdeteksi: {path_name}")
                    
                    cam_id = path_name.replace("/", "_").replace("\\", "_")
                    rtsp_url = f"{MEDIAMTX_RTSP_BASE}/{path_name}"
                    
                    print(f"    Membuka stream: {rtsp_url}")
                    cmd = [CPP_EXECUTABLE, "--url", rtsp_url, "--cam", cam_id]
                    
                    proc = subprocess.Popen(cmd)
                    active_cameras[path_name] = proc

            # 2. Cek apakah ada kamera LAMA yang mati/hilang dari daftar
            lost_cameras = []
            for path_name in active_cameras.keys():
                if path_name not in current_paths:
                    lost_cameras.append(path_name)
                    
            for path_name in lost_cameras:
                print(f"[-] Kamera Mati/Terputus: {path_name}")
                proc = active_cameras[path_name]
                proc.terminate()
                del active_cameras[path_name]
                print(f"    Menutup jendela deteksi untuk {path_name}.")

            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n[Supervisor] Mematikan semua kamera...")
        for path_name, proc in active_cameras.items():
            proc.terminate()
        print("Supervisor berhenti.")

if __name__ == "__main__":
    run_supervisor()
