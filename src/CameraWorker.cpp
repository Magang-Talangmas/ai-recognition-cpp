#include "CameraWorker.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

CameraWorker::CameraWorker(
    const std::string& rtsp_url, 
    const std::string& camera_id, 
    const std::string& redis_url, 
    const std::string& redis_channel,
    std::shared_ptr<FacePreprocessor> preprocessor
) : rtsp_url_(rtsp_url), 
    camera_id_(camera_id), 
    preprocessor_(preprocessor), 
    keep_running_(false) 
{
    reader_ = std::make_unique<StreamReader>(rtsp_url_, camera_id_);
    publisher_ = std::make_unique<BrokerPublisher>(redis_url, redis_channel, camera_id_);
}

CameraWorker::~CameraWorker() {
    stop();
}

void CameraWorker::start() {
    if (keep_running_) return;
    
    keep_running_ = true;
    reader_->start();
    
    worker_thread_ = std::thread(&CameraWorker::run, this);
    std::cout << "[CameraWorker] Started thread for " << camera_id_ << std::endl;
}

void CameraWorker::stop() {
    if (!keep_running_) return;
    
    keep_running_ = false;
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    reader_->stop();
    std::cout << "[CameraWorker] Stopped thread for " << camera_id_ << std::endl;
}

void CameraWorker::run() {
    auto last_inference_time = std::chrono::steady_clock::now();
    auto last_video_publish = std::chrono::steady_clock::now();
    
    while (keep_running_) {
        auto opt_frame = reader_->getLatestFrame();
        
        if (opt_frame.has_value()) {
            cv::Mat frame = opt_frame.value();
            auto now = std::chrono::steady_clock::now();
            auto time_since_last_inference = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_inference_time).count();
            
            // --- FRAME SKIPPING LOGIC (AI Inference) ---
#ifdef USE_GPU
            if (time_since_last_inference >= 20) { // Max 50 FPS
#else
            if (time_since_last_inference >= 200) { // Max 5 FPS
#endif
                auto faces = preprocessor_->process(frame);
                last_inference_time = std::chrono::steady_clock::now();
                
                if (!faces.empty()) {
                    publisher_->publish(faces);
                }
            }

            // --- VIDEO STREAM PUBLISH LOGIC ---
            auto time_since_last_vid = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_video_publish).count();
            if (time_since_last_vid >= 20) { // Max 50 FPS
                cv::Mat small_frame;
                cv::resize(frame, small_frame, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
                publisher_->publishVideoFrame(small_frame);
                last_video_publish = std::chrono::steady_clock::now();
            }
            
            // Yield to avoid 100% CPU on empty iterations
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
