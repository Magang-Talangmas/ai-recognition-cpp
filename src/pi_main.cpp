#include "FacePreprocessor.hpp"
#include "PiConfig.hpp"
#include "PiPayload.hpp"
#include "PiPreview.hpp"
#include <sw/redis++/redis++.h>
#include <sw/redis++/redis_uri.h>
#include <algorithm>
#include <opencv2/videoio/registry.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

namespace {
volatile std::sig_atomic_t stopped = 0;
void stopSignal(int) { stopped = 1; }
using Clock = std::chrono::steady_clock;
int64_t wallMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
void pauseMs(int ms, const std::atomic<bool>& running) {
    for (int i = 0; i < ms && running && !stopped; i += 50)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
}
struct Frame {
    cv::Mat image;
    uint64_t id = 0;
    int64_t receivedMs = 0;
    Clock::time_point received;
};
class Capture {
public:
    explicit Capture(const pi::Config& config) : config_(config), thread_([this] { run(); }) {}
    ~Capture() { running_ = false; if (thread_.joinable()) thread_.join(); }
    bool latest(Frame& result, uint64_t lastId) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (frame_.image.empty() || frame_.id == lastId) return false;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - frame_.received).count() > config_.maxAgeMs)
            return false;
        result = frame_; // Refcount keeps this immutable frame alive while capture replaces its own slot.
        return true;
    }
private:
    void run() {
        int delay = 1000;
        uint64_t id = 0;
        while (running_ && !stopped) {
            try {
                cv::VideoCapture cap;
                const std::vector<int> params = {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, config_.timeoutMs,
                    cv::CAP_PROP_READ_TIMEOUT_MSEC, config_.timeoutMs};
                if (cap.open(config_.rtsp, cv::CAP_FFMPEG, params)) {
                    std::cout << "[Capture] Connected camera=" << config_.camera << std::endl;
                    while (running_ && !stopped) {
                        cv::Mat decoded;
                        if (!cap.read(decoded) || decoded.empty()) break;
                        delay = 1000;
                        Frame next{decoded, ++id, wallMs(), Clock::now()};
                        std::lock_guard<std::mutex> guard(mutex_);
                        frame_ = std::move(next);
                    }
                }
            } catch (const cv::Exception&) {
                std::cerr << "[Capture] Decode/open error" << std::endl;
            }
            { std::lock_guard<std::mutex> guard(mutex_); frame_ = {}; }
            if (!running_ || stopped) break;
            std::cerr << "[Capture] Stream unavailable; reconnect in " << delay << " ms" << std::endl;
            pauseMs(delay, running_);
            delay = std::min(delay * 2, 10000);
        }
    }
    pi::Config config_;
    std::atomic<bool> running_{true};
    std::mutex mutex_;
    Frame frame_;
    std::thread thread_;
};
}

int main(int argc, char** argv) {
    std::signal(SIGINT, stopSignal);
    std::signal(SIGTERM, stopSignal);
    try {
        std::string configPath = ".env.pi";
        bool checkModel = false, validate = false, previewStdio = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) configPath = argv[++i];
            else if (arg == "--check-model") checkModel = true;
            else if (arg == "--validate-config") validate = true;
            else if (arg == "--preview-stdio") previewStdio = true;
            else if (arg == "--help") {
                std::cout << "face_detector_pi [--config PATH] [--validate-config | --check-model | --preview-stdio]\n";
                return 0;
            } else throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (int(checkModel) + int(validate) + int(previewStdio) > 1)
            throw std::runtime_error("Choose only one diagnostic mode");
        // Keep stdout exclusively for replies; model diagnostics go to stderr.
        std::ostream previewOutput(std::cout.rdbuf());
        if (previewStdio) std::cout.rdbuf(std::cerr.rdbuf());
        const auto c = pi::loadConfig(configPath);
        std::cout << "[Pi] CPU; camera=" << c.camera << "; inference limit=" << c.fps
            << " FPS; threads=" << c.threads << "; preview=off" << std::endl;
        if (validate) return 0;
        cv::setNumThreads(c.threads);
        FacePreprocessor model(c.model, c.threads);
        if (previewStdio) return runPiPreview(model, c, previewOutput);
        if (checkModel) {
            auto faces = model.process(cv::Mat::zeros(480, 640, CV_8UC3));
            std::cout << "[Pi] CPU model smoke test completed; faces=" << faces.size() << std::endl;
            return 0;
        }
        if (!cv::videoio_registry::hasBackend(cv::CAP_FFMPEG))
            throw std::runtime_error("OpenCV FFmpeg backend unavailable; install an FFmpeg-enabled OpenCV build");
        // TCP is the default for this RTSP application; an operator can override it.
        if (!std::getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS")) {
#ifdef _WIN32
            _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp");
#else
            setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 0);
#endif
        }
        auto options = sw::redis::Uri(c.redis).connection_options();
        options.connect_timeout = std::chrono::milliseconds(c.timeoutMs);
        options.socket_timeout = std::chrono::milliseconds(c.timeoutMs);
        std::unique_ptr<sw::redis::Redis> redis;
        auto retryAt = Clock::now();
        auto nextInference = Clock::now();
        auto nextLog = Clock::now();
        const auto interval = std::chrono::milliseconds(static_cast<int>(1000.0 / c.fps));
        std::random_device random;
        const std::string session = std::to_string(wallMs()) + "-" + std::to_string(random());
        Capture capture(c);
        uint64_t lastId = 0, processed = 0, publishErrors = 0;
        while (!stopped) {
            const auto now = Clock::now();
            if (now >= nextLog) {
                std::cout << "[Pi] processed=" << processed << " last_frame=" << lastId
                    << " publish_errors=" << publishErrors << " redis=" << (redis ? "ready" : "retrying") << std::endl;
                nextLog = now + std::chrono::seconds(10);
            }
            if (!redis) {
                if (now < retryAt) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
                try {
                    redis = std::make_unique<sw::redis::Redis>(options);
                    redis->ping();
                    std::cout << "[Pi] Redis ready" << std::endl;
                } catch (const sw::redis::Error&) {
                    redis.reset();
                    retryAt = Clock::now() + std::chrono::seconds(3);
                    std::cerr << "[Pi] Redis unavailable; retrying (no frame backlog)" << std::endl;
                    continue;
                }
            }
            Frame frame;
            if (now < nextInference || !capture.latest(frame, lastId)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;
            }
            lastId = frame.id;
            const auto start = Clock::now();
            auto faces = model.process(frame.image);
            ++processed;
            nextInference = Clock::now() + interval; // Deliberate idle time leaves CPU headroom.
            const auto meta = pi::metadata(c, frame.image.size(), session, frame.id, frame.receivedMs, wallMs());
            try {
                redis->publish(c.bboxChannel, pi::detectionMessage(meta, faces).dump());
                int index = 0;
                for (const auto& face : faces)
                    redis->publish(c.faceChannel, pi::faceMessage(meta, face, index++, c.jpegQuality).dump());
            } catch (const sw::redis::Error&) {
                ++publishErrors;
                redis.reset();
                retryAt = Clock::now() + std::chrono::seconds(3);
                std::cerr << "[Pi] Publish failed; batch may be partial and is not replayed" << std::endl;
            }
            if (!faces.empty()) std::cout << "[Pi] faces=" << faces.size() << " frame=" << frame.id
                << " processing_and_publish_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count() << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Pi] Fatal: " << e.what() << std::endl;
        return 1;
    }
}
