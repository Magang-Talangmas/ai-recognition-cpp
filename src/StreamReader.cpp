#include "StreamReader.hpp"
#include <iostream>
#include <vector>

StreamReader::StreamReader(const std::string& redis_url, const std::string& camera_id) 
    : redis_url_(redis_url), camera_id_(camera_id), is_running_(false), has_new_frame_(false) {
    channel_name_ = "camera:" + camera_id_ + ":frames";
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
}

std::optional<cv::Mat> StreamReader::getLatestFrame() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_new_frame_ && !latest_frame_.empty()) {
        has_new_frame_ = false;
        return latest_frame_.clone();
    }
    return std::nullopt;
}

void StreamReader::captureLoop() {
    std::cout << "[StreamReader] Starting Redis subscriber loop for channel: " << channel_name_ << std::endl;
    
    while (is_running_) {
        try {
            auto redis = sw::redis::Redis(redis_url_);
            auto sub = redis.subscriber();
            
            sub.on_message([this](std::string channel, std::string msg) {
                this->processMessage(channel, msg);
            });

            sub.subscribe(channel_name_);
            std::cout << "[StreamReader] Subscribed to Redis channel successfully!" << std::endl;

            while (is_running_) {
                try {
                    // Consume messages with a timeout so we can periodically check `is_running_`
                    sub.consume();
                } catch (const sw::redis::TimeoutError& e) {
                    continue;
                } catch (const sw::redis::Error& e) {
                    std::cerr << "[StreamReader] Redis error during consume: " << e.what() << std::endl;
                    break; // Break the inner loop to reconnect
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[StreamReader] Failed to connect/subscribe to Redis: " << e.what() << std::endl;
            // Sleep before retrying
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    
    std::cout << "[StreamReader] Subscriber loop stopped." << std::endl;
}

void StreamReader::processMessage(const std::string& channel, const std::string& msg) {
    if (channel != channel_name_) return;
    
    // Decode JPG buffer
    std::vector<uchar> buf(msg.begin(), msg.end());
    cv::Mat frame = cv::imdecode(buf, cv::IMREAD_COLOR);
    
    if (frame.empty()) {
        std::cerr << "[StreamReader] Failed to decode JPG from Redis!" << std::endl;
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_frame_ = frame.clone();
        has_new_frame_ = true;
    }
}
