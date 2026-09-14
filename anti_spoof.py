import numpy as np

try:
    from uniface.spoofing import MiniFASNet
    spoofer = MiniFASNet()
    print("Berhasil memuat model MiniFASNet untuk Face Anti-Spoofing.")
except ImportError:
    print("Warning: uniface tidak terinstall. Jalankan `pip install uniface`.")
    spoofer = None
except Exception as e:
    print(f"Gagal memuat MiniFASNet: {e}")
    spoofer = None

def check_liveness(image: np.ndarray, bbox: list) -> tuple[bool, float]:
    """
    Mengecek apakah wajah yang terdeteksi adalah asli atau palsu (spoof).
    Mengembalikan (is_real: bool, confidence: float).
    Jika model gagal dimuat, sistem akan otomatis melakukan fail-open (mengembalikan True).
    """
    if spoofer is None:
        return True, 1.0
        
    try:
        # MiniFASNet menerima BGR image dan bbox koordinat
        result = spoofer.predict(image, bbox)
        return result.is_real, result.confidence
    except Exception as e:
        print(f"[Anti-Spoofing Error] Gagal memprediksi liveness: {e}")
        # Fail-open agar tidak merusak sistem absensi jika terjadi error crop
        return True, 1.0
