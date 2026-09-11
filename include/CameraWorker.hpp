#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <chrono>
#include "StreamReader.hpp"
#include "BrokerPublisher.hpp"
#include "FacePreprocessor.hpp"

class CameraWorker {
public:
    CameraWorker(
        const std::string& rtsp_url, 
        const std::string& camera_id, 
        const std::string& redis_url, 
        const std::string& redis_channel,
        std::shared_ptr<FacePreprocessor> preprocessor
    );
    
    ~CameraWorker();

    void start();
    void stop();

private:
    void run();

    std::string rtsp_url_;
    std::string camera_id_;
    std::shared_ptr<FacePreprocessor> preprocessor_;
    
    std::unique_ptr<StreamReader> reader_;
    std::unique_ptr<BrokerPublisher> publisher_;
    
    std::atomic<bool> keep_running_;
    std::thread worker_thread_;
};
