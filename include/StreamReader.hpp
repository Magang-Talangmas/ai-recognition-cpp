#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <optional>

class StreamReader {
public:
    /**
     * @brief Construct a new Stream Reader object
     * 
     * @param rtsp_url URL of the RTSP stream (e.g., MediaMTX)
     */
    explicit StreamReader(const std::string& rtsp_url, const std::string& camera_id);
    ~StreamReader();

    /**
     * @brief Starts the background thread for asynchronous frame capturing
     */
    void start();

    /**
     * @brief Stops the background thread and releases the stream
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
    void reconnect();

    std::string rtsp_url_;
    cv::VideoCapture capture_;
    std::string camera_id_;
    std::atomic<bool> is_running_;
    std::thread capture_thread_;
    
    std::mutex mutex_;
    cv::Mat latest_frame_;
    bool has_new_frame_;
};
