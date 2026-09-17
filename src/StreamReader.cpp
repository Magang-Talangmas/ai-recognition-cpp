#include "StreamReader.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#endif

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
    std::cout << "[StreamReader] RTSP Reconnect ditangani oleh Python Proxy..." << std::endl;
}

void StreamReader::captureLoop() {
    std::cout << "[StreamReader] Starting capture loop for: " << rtsp_url_ << std::endl;
    std::cout << "[StreamReader] Memulai Python RTSP Proxy..." << std::endl;

#ifdef _WIN32
    std::string cmd = "start /B python rtsp_proxy.py \"" + rtsp_url_ + "\" " + camera_id_;
    system(cmd.c_str());

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
#else
    std::string pipe_name = "/tmp/rtsp_pipe_" + camera_id_;
    unlink(pipe_name.c_str());
    if (mkfifo(pipe_name.c_str(), 0666) == -1) {
        std::cerr << "[StreamReader] Gagal membuat mkfifo: " << strerror(errno) << std::endl;
        return;
    }

    const char* configured_python = std::getenv("PYTHON_EXECUTABLE");
    std::string python = configured_python ? configured_python : "./.venv/bin/python";
    std::string cmd = python + " rtsp_proxy.py \"" + rtsp_url_ + "\" " + camera_id_ + " &";
    if (system(cmd.c_str()) != 0) {
        std::cerr << "[StreamReader] Gagal menjalankan rtsp_proxy.py" << std::endl;
        unlink(pipe_name.c_str());
        return;
    }

    std::cout << "[StreamReader] Menunggu Python Proxy terhubung ke FIFO..." << std::endl;
    int fd = open(pipe_name.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[StreamReader] Gagal membuka FIFO: " << strerror(errno) << std::endl;
        return;
    }
#endif
    
    std::cout << "[StreamReader] Proxy terhubung! Memulai stream..." << std::endl;

    while (is_running_) {
        uint32_t size = 0;
        
#ifdef _WIN32
        DWORD bytesRead;
        if (!ReadFile(hPipe, &size, 4, &bytesRead, NULL) || bytesRead != 4) {
            std::cerr << "[StreamReader] Gagal membaca ukuran frame dari Pipe" << std::endl;
            break;
        }
#else
        ssize_t bytesRead = read(fd, &size, 4);
        if (bytesRead <= 0) break;
#endif

        if (size == 0 || size > 1024 * 1024 * 10) {
            std::cerr << "[StreamReader] Ukuran frame tidak valid: " << size << std::endl;
            break;
        }
        
        std::vector<uchar> buf(size);
        uint32_t totalRead = 0;
        while (totalRead < size) {
#ifdef _WIN32
            DWORD chunkRead = 0;
            if (!ReadFile(hPipe, buf.data() + totalRead, size - totalRead, &chunkRead, NULL) || chunkRead == 0) {
                break;
            }
#else
            ssize_t chunkRead = read(fd, buf.data() + totalRead, size - totalRead);
            if (chunkRead <= 0) break;
#endif
            totalRead += chunkRead;
        }

        if (totalRead != size) {
            std::cerr << "[StreamReader] Gagal membaca seluruh data JPG" << std::endl;
            break;
        }
        
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
    
#ifdef _WIN32
    CloseHandle(hPipe);
#else
    close(fd);
    unlink(pipe_name.c_str());
#endif
    std::cout << "[StreamReader] Capture loop stopped." << std::endl;
}
