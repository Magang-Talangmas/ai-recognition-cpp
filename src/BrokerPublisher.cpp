#include "BrokerPublisher.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <future>

using json = nlohmann::json;

// Simple Base64 encoder helper
static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::string base64_encode(const unsigned char* buf, unsigned int bufLen) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (bufLen--) {
        char_array_3[i++] = *(buf++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }
    return ret;
}

BrokerPublisher::BrokerPublisher(const std::string& redis_url, const std::string& channel, const std::string& camera_id)
    : redis_(redis_url), channel_(channel), camera_id_(camera_id) {
}

BrokerPublisher::~BrokerPublisher() {
}

std::string BrokerPublisher::encodeBase64(const cv::Mat& image) {
    if (image.empty()) return "";
    
    // Convert to standard JPEG format
    std::vector<uchar> buf;
    cv::imencode(".jpg", image, buf);
    
    return base64_encode(buf.data(), buf.size());
}

void BrokerPublisher::publish(const std::vector<PreprocessedFace>& faces) {
    if (faces.empty()) return;

    // Get current timestamp in ms
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    std::vector<PreprocessedFace> faces_copy = faces;
    
    std::async(std::launch::async, [this, faces_copy, ms]() {
        int face_idx = 0;
        for (const auto& face : faces_copy) {
            json j;
            j["camera_id"] = camera_id_;
            j["timestamp_ms"] = ms;
            j["face_index"] = face_idx++;
            j["bounding_box"] = {
                {"x", face.bounding_box.x},
                {"y", face.bounding_box.y},
                {"width", face.bounding_box.width},
                {"height", face.bounding_box.height}
            };
            j["confidence_score"] = face.confidence_score;
            j["face_image_base64"] = encodeBase64(face.face_image);
            j["image_format"] = "jpeg_base64_112x112"; // Update metadata
            
            std::string payload = j.dump();
            
            try {
                redis_.publish(channel_, payload);
                // std::cout << "[BrokerPublisher] Published face data to channel: " << channel_ << std::endl;
            } catch (const sw::redis::Error& e) {
                std::cerr << "[BrokerPublisher] Redis publish error: " << e.what() << std::endl;
            }
        }
    });
}

void BrokerPublisher::publishVideoFrame(const cv::Mat& frame) {
    if (frame.empty()) return;

    // We do this asynchronously to avoid blocking the main UI thread
    cv::Mat frame_copy = frame.clone();
    std::async(std::launch::async, [this, frame_copy]() {
        try {
            // Compress frame to JPEG (lower quality to save redis bandwidth)
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(60); // 60% quality

            std::vector<uchar> buf;
            cv::imencode(".jpg", frame_copy, buf, compression_params);
            
            // Publish raw binary bytes directly to redis dynamically per camera
            std::string payload(buf.begin(), buf.end());
            std::string video_channel = "face_video_stream_" + camera_id_;
            redis_.publish(video_channel, payload);
        } catch (const std::exception& e) {
            std::cerr << "[BrokerPublisher] Video frame publish error: " << e.what() << std::endl;
        }
    });
}
