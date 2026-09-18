#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <optional>
#include <sw/redis++/redis++.h>

class StreamReader {
public:
    /**
     * @brief Construct a new Stream Reader object
     * 
     * @param redis_url URL of the Redis server (e.g., tcp://127.0.0.1:6379)
     * @param camera_id ID of the camera to subscribe to
     */
    explicit StreamReader(const std::string& redis_url, const std::string& camera_id);
    ~StreamReader();

    /**
     * @brief Starts the background thread for asynchronous frame receiving
     */
    void start();

    /**
     * @brief Stops the background thread and disconnects
     */
    void stop();

    /**
     * @brief Retrieves the most recent frame from the stream.
     * Drops older frames to avoid lag.
     * 
     * @return std::optional<cv::Mat> The latest frame, if available.
     */
    std::optional<cv::Mat> getLatestFrame();

private:
    void captureLoop();
    void processMessage(const std::string& channel, const std::string& msg);

    std::string redis_url_;
    std::string camera_id_;
    std::string channel_name_;
    
    std::atomic<bool> is_running_;
    std::thread capture_thread_;
    
    std::mutex mutex_;
    cv::Mat latest_frame_;
    bool has_new_frame_;
};
