#include "StreamReader.hpp"
#include <iostream>
#include <chrono>
#include <windows.h>
#include <fstream>
#include <chrono>

StreamReader::StreamReader(const std::string& rtsp_url, const std::string& camera_id) 
    : rtsp_url_(rtsp_url), camera_id_(camera_id), is_running_(false), has_new_frame_(false) {
}

StreamReader::~StreamReader() {
    stop();
}

void StreamReader::start() {
    if (is_running_) return;
    is_running_ = true;
    capture_thread_ = std::thread(&StreamReader::captureLoop, this);
}

void StreamReader::stop() {
    if (is_running_) {
        is_running_ = false;
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
    }
    if (capture_.isOpened()) {
        capture_.release();
    }
}

std::optional<cv::Mat> StreamReader::getLatestFrame() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_new_frame_ && !latest_frame_.empty()) {
        has_new_frame_ = false;
        return latest_frame_.clone();
    }
    return std::nullopt;
}

void StreamReader::reconnect() {
    // Dengan Named Pipe, Python proxy yang handle reconnect.
    // Di C++, reconnect() tidak melakukan apa-apa.
    std::cout << "[StreamReader] RTSP Reconnect ditangani oleh Python Proxy..." << std::endl;
}

void StreamReader::captureLoop() {
    std::cout << "[StreamReader] Starting capture loop for: " << rtsp_url_ << std::endl;
    
    // Jalankan proxy Python di background dengan ID kamera
    std::cout << "[StreamReader] Memulai Python RTSP Proxy..." << std::endl;
    // Gunakan path absolut untuk keamanan dan spesifik Miniconda Python
    std::string cmd = "start /B D:\\MiniConda\\python.exe D:\\ai-recognition-cpp\\rtsp_proxy.py \"" + rtsp_url_ + "\" " + camera_id_;
    system(cmd.c_str());

    // Buka Named Pipe dinamis per kamera
    std::string pipe_name = "\\\\.\\pipe\\rtsp_pipe_" + camera_id_;
    HANDLE hPipe = CreateNamedPipeA(
        pipe_name.c_str(),
        PIPE_ACCESS_INBOUND,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        1, 1024 * 1024, 1024 * 1024, 0, NULL
    );

    if (hPipe == INVALID_HANDLE_VALUE) {
        std::cerr << "[StreamReader] Gagal membuat Named Pipe!" << std::endl;
        return;
    }

    std::cout << "[StreamReader] Menunggu Python Proxy terhubung ke Pipe..." << std::endl;
    bool connected = ConnectNamedPipe(hPipe, NULL) ? true : (GetLastError() == ERROR_PIPE_CONNECTED);
    if (!connected) {
        std::cerr << "[StreamReader] Python Proxy gagal terhubung!" << std::endl;
        CloseHandle(hPipe);
        return;
    }
    
    std::cout << "[StreamReader] Proxy terhubung! Memulai stream..." << std::endl;

    while (is_running_) {
        DWORD bytesRead;
        uint32_t size = 0;
        
        // Baca ukuran (4 bytes)
        if (!ReadFile(hPipe, &size, 4, &bytesRead, NULL) || bytesRead != 4) {
            std::cerr << "[StreamReader] Gagal membaca ukuran frame dari Pipe" << std::endl;
            break;
        }

        if (size == 0 || size > 1024 * 1024 * 10) {
            std::cerr << "[StreamReader] Ukuran frame tidak valid: " << size << std::endl;
            break;
        }
        
        // Baca data JPG
        std::vector<uchar> buf(size);
        DWORD totalRead = 0;
        while (totalRead < size) {
            DWORD chunkRead = 0;
            if (!ReadFile(hPipe, buf.data() + totalRead, size - totalRead, &chunkRead, NULL) || chunkRead == 0) {
                break;
            }
            totalRead += chunkRead;
        }

        if (totalRead != size) {
            std::cerr << "[StreamReader] Gagal membaca seluruh data JPG dari Pipe" << std::endl;
            break;
        }
        
        // Decode frame
        cv::Mat frame = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "[StreamReader] Gagal mendecode JPG!" << std::endl;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_frame_ = frame.clone();
            has_new_frame_ = true;
        }
    }
    
    CloseHandle(hPipe);
    std::cout << "[StreamReader] Capture loop stopped." << std::endl;
}
