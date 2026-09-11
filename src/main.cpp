#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "FacePreprocessor.hpp"
#include "BrokerPublisher.hpp"
#include "StreamReader.hpp"

using json = nlohmann::json;

// Fungsi untuk menggambar Bounding Box ala Sci-Fi (Patah-patah di sudut)
// Fungsi untuk menggambar Bounding Box kotak putus-putus
void drawDashedRect(cv::Mat& img, const cv::Rect& rect, const cv::Scalar& color, int thickness = 3, int dash_length = 15) {
    int x1 = rect.x, y1 = rect.y;
    int x2 = rect.x + rect.width, y2 = rect.y + rect.height;

    // Garis Atas
    for (int x = x1; x < x2; x += dash_length * 2) {
        cv::line(img, cv::Point(x, y1), cv::Point(std::min(x + dash_length, x2), y1), color, thickness);
    }
    // Garis Bawah
    for (int x = x1; x < x2; x += dash_length * 2) {
        cv::line(img, cv::Point(x, y2), cv::Point(std::min(x + dash_length, x2), y2), color, thickness);
    }
    // Garis Kiri
    for (int y = y1; y < y2; y += dash_length * 2) {
        cv::line(img, cv::Point(x1, y), cv::Point(x1, std::min(y + dash_length, y2)), color, thickness);
    }
    // Garis Kanan
    for (int y = y1; y < y2; y += dash_length * 2) {
        cv::line(img, cv::Point(x2, y), cv::Point(x2, std::min(y + dash_length, y2)), color, thickness);
    }
}

int main(int argc, char** argv) {
    std::cout << "=================================================" << std::endl;
    std::cout << "  Talangmas AI Preprocessing Pipeline (GPU)      " << std::endl;
    std::cout << "=================================================" << std::endl;

    std::string rtsp_url = "rtsp://127.0.0.1:8554/stream";
    std::string camera_id = "cam_01";
    std::string redis_channel = "face_preprocessed_queue";
    std::string redis_url = "tcp://127.0.0.1:6379";

    bool is_cam_overridden = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--url" && i + 1 < argc) rtsp_url = argv[++i];
        else if (arg == "--cam" && i + 1 < argc) { camera_id = argv[++i]; is_cam_overridden = true; }
        else if (arg == "--redis" && i + 1 < argc) redis_channel = argv[++i];
    }

    if (!is_cam_overridden) {
        size_t last_slash = rtsp_url.find_last_of('/');
        if (last_slash != std::string::npos && last_slash + 1 < rtsp_url.length()) {
            camera_id = rtsp_url.substr(last_slash + 1);
        }
    }

    std::string scrfd_model_path = "models/scrfd_2.5g_kps.onnx";
    
    FacePreprocessor preprocessor(scrfd_model_path);
    BrokerPublisher publisher(redis_url, redis_channel, camera_id);
    StreamReader reader(rtsp_url, camera_id);

    std::cout << "Initializing stream for: " << camera_id << std::endl;
    std::cout << "URL: " << rtsp_url << std::endl;

    // Agar window OpenCV bisa di full-screen tanpa ada sisa ruang abu-abu
    cv::namedWindow("Face Detection - " + camera_id, cv::WINDOW_NORMAL);

    reader.start();

    auto last_inference_time = std::chrono::steady_clock::now();
    auto last_video_publish = std::chrono::steady_clock::now();
    
    // Menyimpan memori wajah terakhir agar kotak tidak kedap-kedip (flickering) di frame yang tidak diproses AI
    std::vector<PreprocessedFace> last_faces;

    while (true) {
        auto opt_frame = reader.getLatestFrame();
        if (!opt_frame.has_value()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        cv::Mat frame = opt_frame.value();
        cv::Mat display_frame = frame.clone(); // Untuk ditampilkan di layar dan dikirim ke Redis

        auto now = std::chrono::steady_clock::now();
        auto time_since_last_inference = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_inference_time).count();

        // 50 FPS Throttling for Inference
        if (time_since_last_inference >= 20) {
            last_faces = preprocessor.process(frame); // Simpan hasil deteksi ke memori
            last_inference_time = std::chrono::steady_clock::now();
            
            if (!last_faces.empty()) {
                std::cout << "[Inference Thread] Detected & preprocessed " << last_faces.size() << " valid face(s)." << std::endl;
                publisher.publish(last_faces);
            }
        }

        // GAMBAR BOUNDING BOX DI SETIAP FRAME (Menggunakan data terakhir)
        // Ini kunci agar stream video terlihat 100% smooth dan kotak tidak hilang-timbul
        for (const auto& face : last_faces) {
            // Warna Cyan (255, 255, 0) di OpenCV BGR
            drawDashedRect(display_frame, face.bounding_box, cv::Scalar(255, 255, 0), 3, 15);
            
            // Background hitam untuk teks agar mudah dibaca
            std::string text = "Conf: " + std::to_string(face.confidence_score).substr(0,4);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::rectangle(display_frame, 
                cv::Point(face.bounding_box.x, face.bounding_box.y - 20), 
                cv::Point(face.bounding_box.x + textSize.width, face.bounding_box.y), 
                cv::Scalar(0, 0, 0), cv::FILLED);
                
            cv::putText(display_frame, text, 
                cv::Point(face.bounding_box.x, face.bounding_box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        }

        // 50 FPS Throttling for Video Publish ke Redis
        auto time_since_last_vid = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_video_publish).count();
        if (time_since_last_vid >= 20) {
            cv::Mat small_frame;
            cv::resize(display_frame, small_frame, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
            publisher.publishVideoFrame(small_frame);
            last_video_publish = std::chrono::steady_clock::now();
        }

        cv::imshow("Face Detection - " + camera_id, display_frame);
        if (cv::waitKey(1) == 27) break; // Tekan ESC untuk keluar
    }

    reader.stop();
    cv::destroyAllWindows();
    return 0;
}
