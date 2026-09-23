#pragma once
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace pi {
inline std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

struct Config {
    std::string rtsp, redis, camera, stream, model, faceChannel, bboxChannel;
    double fps = 2;
    int threads = 1, timeoutMs = 3000, maxAgeMs = 2000, jpegQuality = 90;
};

inline Config loadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open config file: " + path);
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto at = line.find('=');
        if (at == std::string::npos) throw std::runtime_error("Invalid config line (expected KEY=VALUE)");
        auto value = trim(line.substr(at + 1));
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
            (value.front() == '\'' && value.back() == '\''))) value = value.substr(1, value.size() - 2);
        values[trim(line.substr(0, at))] = value;
    }
    auto get = [&](const std::string& key, const std::string& fallback) {
        const char* env = std::getenv(key.c_str());
        return env ? std::string(env) : (values.count(key) ? values[key] : fallback);
    };
    auto number = [&](const std::string& key, const std::string& fallback, double low, double high, bool integer) {
        const auto value = get(key, fallback);
        size_t end = 0;
        double n;
        try { n = std::stod(value, &end); }
        catch (...) { throw std::runtime_error("Invalid numeric setting: " + key); }
        if (end != value.size() || !std::isfinite(n) || n < low || n > high || (integer && std::floor(n) != n))
            throw std::runtime_error("Setting outside supported range: " + key);
        return n;
    };
    Config c;
    c.rtsp = get("PI_RTSP_URL", "rtsp://192.168.77.100:8554/cam01");
    c.camera = get("PI_CAMERA_ID", "cam01");
    c.stream = get("PI_STREAM_PATH", "cam01");
    c.redis = get("PI_REDIS_URL", "tcp://127.0.0.1:6379");
    c.model = get("PI_MODEL_PATH", "models/scrfd_2.5g_kps.onnx");
    c.faceChannel = get("PI_FACE_CHANNEL", "face_preprocessed_queue");
    c.bboxChannel = get("PI_BBOX_CHANNEL", "face_detection_queue");
    c.fps = number("PI_INFERENCE_FPS", "2", 0.2, 10, false);
    c.threads = static_cast<int>(number("PI_INFERENCE_THREADS", "1", 1, 4, true));
    c.timeoutMs = static_cast<int>(number("PI_IO_TIMEOUT_MS", "3000", 500, 10000, true));
    c.maxAgeMs = static_cast<int>(number("PI_MAX_FRAME_AGE_MS", "2000", 100, 10000, true));
    c.jpegQuality = static_cast<int>(number("PI_JPEG_QUALITY", "90", 50, 100, true));
    if (c.rtsp.rfind("rtsp://", 0) != 0 && c.rtsp.rfind("rtsps://", 0) != 0)
        throw std::runtime_error("PI_RTSP_URL must be an RTSP URL, not an HTTP link");
    if (c.camera.empty() || c.stream.empty() || c.model.empty() || c.faceChannel.empty() || c.bboxChannel.empty())
        throw std::runtime_error("Camera, stream, model and channels must not be empty");
    if (c.faceChannel == c.bboxChannel) throw std::runtime_error("Face and bbox channels must differ");
    return c;
}
} // namespace pi
