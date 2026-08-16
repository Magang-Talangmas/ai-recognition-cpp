#include "StreamReader.hpp"
#include "FacePreprocessor.hpp"
#include "BrokerPublisher.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>

std::atomic<bool> keep_running(true);

void signalHandler(int signum) {
    std::cout << "\n[Main] Interrupt signal (" << signum << ") received. Shutting down..." << std::endl;
    keep_running = false;
}

// Fungsi bantuan untuk menggambar kotak putus-putus (dashed rectangle)
void drawDashedRectangle(cv::Mat& img, cv::Rect rect, const cv::Scalar& color, int thickness = 1, int dash_length = 8) {
    // Top edge
    for (int x = rect.x; x < rect.x + rect.width; x += dash_length * 2) {
        cv::line(img, cv::Point(x, rect.y), cv::Point(std::min(x + dash_length, rect.x + rect.width), rect.y), color, thickness);
    }
    // Bottom edge
    for (int x = rect.x; x < rect.x + rect.width; x += dash_length * 2) {
        cv::line(img, cv::Point(x, rect.y + rect.height), cv::Point(std::min(x + dash_length, rect.x + rect.width), rect.y + rect.height), color, thickness);
    }
    // Left edge
    for (int y = rect.y; y < rect.y + rect.height; y += dash_length * 2) {
        cv::line(img, cv::Point(rect.x, y), cv::Point(rect.x, std::min(y + dash_length, rect.y + rect.height)), color, thickness);
    }
    // Right edge
    for (int y = rect.y; y < rect.y + rect.height; y += dash_length * 2) {
        cv::line(img, cv::Point(rect.x + rect.width, y), cv::Point(rect.x + rect.width, std::min(y + dash_length, rect.y + rect.height)), color, thickness);
    }
}

#include <fstream>
#include <sstream>
#include <map>

// Helper untuk membaca file .env
std::map<std::string, std::string> loadEnv(const std::string& path) {
    std::map<std::string, std::string> env;
    std::ifstream file(path);
    if (!file.is_open()) return env;
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto delimiterPos = line.find("=");
        if (delimiterPos != std::string::npos) {
            std::string key = line.substr(0, delimiterPos);
            std::string value = line.substr(delimiterPos + 1);
            // Hapus whitespace jika ada
            key.erase(key.find_last_not_of(" \n\r\t") + 1);
            value.erase(0, value.find_first_not_of(" \n\r\t\"'"));
            value.erase(value.find_last_not_of(" \n\r\t\"'") + 1);
            env[key] = value;
        }
    }
    return env;
}

int main(int argc, char** argv) {
    // Register signal handlers for graceful shutdown
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Load .env file
    std::map<std::string, std::string> env = loadEnv(".env");

    // Configuration with fallback to default values if .env is missing
    std::string rtsp_url = env.count("RTSP_URL") ? env["RTSP_URL"] : "rtsp://192.168.77.171:8554/stream";
    std::string redis_url = env.count("REDIS_URL") ? env["REDIS_URL"] : "tcp://127.0.0.1:6379";
    std::string redis_channel = env.count("REDIS_CHANNEL") ? env["REDIS_CHANNEL"] : "face_preprocessed_queue";
    std::string camera_id = env.count("CAMERA_ID") ? env["CAMERA_ID"] : "cam_01";
    
    // Model paths for ONNX
    std::string scrfd_model_path = "models/scrfd_2.5g_kps.onnx";

    std::cout << "=================================================" << std::endl;
    std::cout << "  Face Recognition Input Preprocessing Service   " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "RTSP URL       : " << rtsp_url << std::endl;
    std::cout << "Redis URL      : " << redis_url << std::endl;
    std::cout << "Redis Channel  : " << redis_channel << std::endl;

    // Initialize modules
    StreamReader reader(rtsp_url);
    FacePreprocessor preprocessor(scrfd_model_path);
    BrokerPublisher publisher(redis_url, redis_channel, camera_id);

    // Mutex dan variabel untuk sinkronisasi thread inferensi
    std::mutex mtx;
    cv::Mat inference_frame;
    std::vector<PreprocessedFace> latest_faces;
    std::atomic<bool> new_frame_for_inference(false);

    // Start async reading from MediaMTX
    reader.start();

    const std::string WINDOW_NAME = "Talangmas AI Attendance - Live View";
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

    // Thread khusus untuk Inferensi AI agar tidak membuat video lag
    std::thread inference_thread([&]() {
        auto last_inference_time = std::chrono::steady_clock::now();
        
        while (keep_running) {
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_inference_time).count();
            
            // --- FRAME SKIPPING LOGIC ---
            // Hanya proses maksimal 5 gambar per detik (1000ms / 5 = 200ms)
            if (time_since_last < 200) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            
            cv::Mat frame_to_process;
            bool should_process = false;
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (new_frame_for_inference) {
                    frame_to_process = inference_frame.clone();
                    new_frame_for_inference = false;
                    should_process = true;
                }
            }
            
            if (should_process && !frame_to_process.empty()) {
                auto faces = preprocessor.process(frame_to_process);
                last_inference_time = std::chrono::steady_clock::now(); // Catat waktu proses terakhir
                
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    latest_faces = faces; // Simpan hasil terbaru untuk digambar di Main Thread
                }
                
                if (!faces.empty()) {
                    std::cout << "[Inference Thread] Detected & preprocessed " << faces.size() << " valid face(s)." << std::endl;
                    publisher.publish(faces);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    });

    // Main processing loop (Hanya untuk UI dan membaca frame agar sangat mulus)
    while (keep_running) {
        auto opt_frame = reader.getLatestFrame();
        
        if (opt_frame.has_value()) {
            cv::Mat frame = opt_frame.value();
            
            // Kirim frame ke thread inferensi jika sudah siap menerima yang baru
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!new_frame_for_inference) {
                    inference_frame = frame.clone();
                    new_frame_for_inference = true;
                }
            }
            
            cv::Mat display_frame = frame.clone();
            
            // Ambil kotak bounding box terakhir yang ditemukan AI
            std::vector<PreprocessedFace> faces_to_draw;
            {
                std::lock_guard<std::mutex> lock(mtx);
                faces_to_draw = latest_faces;
            }
            
            for (const auto& face : faces_to_draw) {
                // Gambar kotak putus-putus warna Cyan/Teal
                drawDashedRectangle(display_frame, face.bounding_box, cv::Scalar(255, 255, 0), 1, 6);
            }
            
            cv::imshow(WINDOW_NAME, display_frame);
            
            // Tunggu 1 milidetik agar window OpenCV sempat merender gambar
            if (cv::waitKey(1) == 'q') {
                keep_running = false;
            }
            
        } else {
            // Sleep briefly to yield CPU if no new frame is available yet
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    inference_thread.join();

    std::cout << "Stopping reader..." << std::endl;
    reader.stop();
    std::cout << "Service stopped gracefully." << std::endl;

    return 0;
}
