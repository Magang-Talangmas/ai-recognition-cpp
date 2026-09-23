#pragma once
#include "FacePreprocessor.hpp"
#include "PiConfig.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>

namespace pi {
using Json = nlohmann::json;
inline std::string base64(const std::vector<uchar>& bytes) {
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t v = (uint32_t(bytes[i]) << 16) |
            (i + 1 < bytes.size() ? uint32_t(bytes[i + 1]) << 8 : 0) |
            (i + 2 < bytes.size() ? uint32_t(bytes[i + 2]) : 0);
        result += alphabet[(v >> 18) & 63];
        result += alphabet[(v >> 12) & 63];
        result += i + 1 < bytes.size() ? alphabet[(v >> 6) & 63] : '=';
        result += i + 2 < bytes.size() ? alphabet[v & 63] : '=';
    }
    return result;
}
inline Json metadata(const Config& c, const cv::Size& size, const std::string& session,
                     uint64_t frameId, int64_t receivedMs, int64_t processedMs) {
    return {{"schema_version", "1.0"}, {"camera_id", c.camera}, {"stream_path", c.stream},
        {"session_id", session}, {"frame_id", frameId}, {"received_at_ms", receivedMs},
        {"timestamp_ms", processedMs}, {"frame_width", size.width}, {"frame_height", size.height}};
}
inline Json box(const PreprocessedFace& f) {
    return {{"x", f.bounding_box.x}, {"y", f.bounding_box.y},
        {"width", f.bounding_box.width}, {"height", f.bounding_box.height}};
}
inline Json landmarks(const PreprocessedFace& f) {
    Json points = Json::array();
    for (const auto& p : f.landmarks) points.push_back({p.x, p.y});
    return points;
}
inline Json detectionMessage(Json meta, const std::vector<PreprocessedFace>& faces) {
    meta["bounding_boxes"] = Json::array();
    int index = 0;
    for (const auto& f : faces) meta["bounding_boxes"].push_back({
        {"face_index", index++}, {"bounding_box", box(f)}, {"confidence_score", f.confidence_score},
        {"landmarks", landmarks(f)}, {"name", "Unknown"}});
    return meta; // Includes an empty array when a frame contains no faces.
}
inline Json faceMessage(Json meta, const PreprocessedFace& f, int index, int quality) {
    cv::Mat pixels;
    f.face_image.convertTo(pixels, CV_8U, 127.5, 127.5);
    std::vector<uchar> jpeg;
    if (!cv::imencode(".jpg", pixels, jpeg, {cv::IMWRITE_JPEG_QUALITY, quality}))
        throw std::runtime_error("Cannot encode face JPEG");
    meta["face_index"] = index;
    meta["bounding_box"] = box(f);
    meta["confidence_score"] = f.confidence_score;
    meta["landmarks"] = landmarks(f);
    meta["image_format"] = "jpeg_base64_112x112";
    meta["face_image_base64"] = base64(jpeg);
    return meta;
}
} // namespace pi
