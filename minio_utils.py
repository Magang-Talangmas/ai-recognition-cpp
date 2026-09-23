import os
import requests
import cv2
import numpy as np
from minio import Minio
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# --- Konfigurasi MinIO ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

minio_client = None
if MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY:
    try:
        clean_endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        minio_client = Minio(
            clean_endpoint,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_ENDPOINT.startswith("https://")
        )
        print(f"[MinIO] Client siap untuk endpoint: {clean_endpoint}")
    except Exception as e:
        print(f"[MinIO Error] Gagal inisialisasi client: {e}")

def get_image_from_url(url, local_path=None):
    """
    Mendownload gambar. Mendukung URL HTTP biasa dan URL MinIO private.
    Jika local_path diisi, gambar akan disimpan ke disk dan me-return boolean.
    Jika local_path None, akan me-return numpy array (cv2 image).
    """
    is_minio_url = False
    if minio_client:
        clean_endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        hostname = clean_endpoint.split(":")[0] # Ambil IP/Host tanpa port
        # Deteksi jika URL mengandung hostname MinIO
        if hostname in url:
            is_minio_url = True

    try:
        if is_minio_url:
            from urllib.parse import unquote
            parsed = urlparse(url)
            
            # Abaikan kata '/browser/' jika itu URL dari UI MinIO
            path_raw = unquote(parsed.path)
            if path_raw.startswith('/browser/'):
                path_raw = path_raw.replace('/browser/', '/', 1)
                
            path_parts = path_raw.strip("/").split("/")
            bucket_name = path_parts[0]
            object_name = "/".join(path_parts[1:])
            
            response = minio_client.get_object(bucket_name, object_name)
            content = response.read()
            response.close()
            response.release_conn()
        else:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            content = resp.content
            
        if local_path:
            with open(local_path, "wb") as f:
                f.write(content)
            return True
        else:
            image_array = np.asarray(bytearray(content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"  -> Gagal mengambil gambar dari {url}: {e}")
        return None if not local_path else False
