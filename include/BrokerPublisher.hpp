#pragma once

#include <sw/redis++/redis++.h>
#include <string>
#include <vector>
#include "FacePreprocessor.hpp"

class BrokerPublisher {
public:
    /**
     * @brief Construct a new Broker Publisher
     * 
     * @param redis_url Redis connection URL (e.g., "tcp://127.0.0.1:6379")
     * @param channel Redis Pub/Sub channel to publish to
     * @param camera_id Identifier for the camera source
     */
    BrokerPublisher(const std::string& redis_url, const std::string& channel, const std::string& camera_id);
    ~BrokerPublisher();

    /**
     * @brief Asynchronously publish the preprocessed faces to Redis Pub/Sub
     * 
     * @param faces List of preprocessed faces
     */
    void publish(const std::vector<PreprocessedFace>& faces);

    /**
     * @brief Encode and publish the full video frame for MJPEG stream
     * 
     * @param frame The full frame with drawn bounding boxes
     */
    void publishVideoFrame(const cv::Mat& frame);

private:
    std::string encodeBase64(const cv::Mat& image);

    sw::redis::Redis redis_;
    std::string channel_;
    std::string camera_id_;
};
