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
    
    // Convert raw memory of float32 Mat to Base64
    size_t data_size = image.total() * image.elemSize();
    return base64_encode(image.ptr(), data_size);
}

void BrokerPublisher::publish(const std::vector<PreprocessedFace>& faces) {
    if (faces.empty()) return;

    // Get current timestamp in ISO 8601 or unix ms
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    // We run the serialization and publishing asynchronously to not block the main preprocessing thread
    std::vector<PreprocessedFace> faces_copy = faces;
    
    std::async(std::launch::async, [this, faces_copy, ms]() {
        for (const auto& face : faces_copy) {
            json j;
            j["camera_id"] = camera_id_;
            j["timestamp_ms"] = ms;
            j["bounding_box"] = {
                {"x", face.bounding_box.x},
                {"y", face.bounding_box.y},
                {"width", face.bounding_box.width},
                {"height", face.bounding_box.height}
            };
            j["confidence_score"] = face.confidence_score;
            j["face_image_base64"] = encodeBase64(face.face_image);
            j["image_format"] = "float32_raw_112x112x3"; // metadata describing the base64 content
            
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
