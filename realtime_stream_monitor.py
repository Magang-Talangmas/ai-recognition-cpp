import psutil
import time
import os

def get_publisher_processes():
    # Cari proses python yang menjalankan stream_publisher.py dan semua proses ffmpeg
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            name = proc.info['name'].lower()
            if cmd and 'stream_publisher.py' in ' '.join(cmd):
                processes.append((proc, "Python (Publisher)"))
            elif 'ffmpeg' in name:
                processes.append((proc, "FFmpeg (Encoder/Copy)"))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def main():
    print("Mencari proses publisher (Python & FFmpeg)...")
    time.sleep(1) # Beri waktu psutil menghitung baseline CPU
    
    try:
        while True:
            procs = get_publisher_processes()
            if not procs:
                print("Tidak ada proses stream_publisher atau ffmpeg yang berjalan.", end='\r')
                time.sleep(2)
                continue

            os.system('cls' if os.name == 'nt' else 'clear')
            print("="*50)
            print("REALTIME CPU MONITOR (Stream Publisher)")
            print("="*50)
            
            total_cpu = 0
            for proc, label in procs:
                try:
                    # Ambil CPU usage dibagi jumlah core agar hasilnya persentase global (0-100%)
                    cpu_percent = proc.cpu_percent(interval=None) / psutil.cpu_count()
                    total_cpu += cpu_percent
                    print(f"[{label}] PID: {proc.pid: <6} | CPU: {cpu_percent:>5.1f}% | Mem: {proc.memory_info().rss / 1024 / 1024:>5.1f} MB")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            print("-" * 50)
            print(f"TOTAL CPU USAGE : {total_cpu:>5.1f}%")
            if total_cpu < 2.0:
                print("STATUS          : SANGAT RINGAN (Mendekati 0% - Pure Copy / Idle)")
            elif total_cpu < 10.0:
                print("STATUS          : RINGAN (Encoding Ringan / Ultra-Low Latency)")
            else:
                print("STATUS          : BERAT (Butuh Optimasi / Hardware Acceleration)")
            
            print("Tekan Ctrl+C untuk keluar...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nMonitor dihentikan.")

if __name__ == "__main__":
    main()
