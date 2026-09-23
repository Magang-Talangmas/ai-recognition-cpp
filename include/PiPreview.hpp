#pragma once
#include "FacePreprocessor.hpp"
#include "PiPayload.hpp"
#include <iostream>
#include <chrono>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// Local diagnostic protocol: little-endian uint32 width, height, byte count,
// then packed BGR pixels. One JSON reply per frame, no Redis side effects.
inline int runPiPreview(FacePreprocessor& model, const pi::Config& config, std::ostream& output) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    output << pi::Json({{"ready", true}, {"fps", config.fps}, {"rtsp_url", config.rtsp}}).dump() << std::endl;
    uint64_t id = 0;
    while (true) {
        unsigned char header[12];
        std::cin.read(reinterpret_cast<char*>(header), sizeof(header));
        if (std::cin.gcount() == 0 && std::cin.eof()) return 0;
        if (std::cin.gcount() != sizeof(header)) throw std::runtime_error("Truncated preview header");
        auto read32 = [&](int offset) {
            return uint32_t(header[offset]) | (uint32_t(header[offset+1]) << 8) |
                (uint32_t(header[offset+2]) << 16) | (uint32_t(header[offset+3]) << 24);
        };
        const auto width = read32(0), height = read32(4), bytes = read32(8);
        if (!width || !height || width > 4096 || height > 4096 || bytes != uint64_t(width)*height*3)
            throw std::runtime_error("Invalid preview dimensions or byte count (maximum 4096x4096)");
        cv::Mat frame(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
        std::cin.read(reinterpret_cast<char*>(frame.data), bytes);
        if (std::cin.gcount() != bytes) throw std::runtime_error("Truncated preview pixels");
        const auto start = std::chrono::steady_clock::now();
        const auto faces = model.process(frame);
        auto message = pi::detectionMessage(pi::metadata(config, frame.size(), "local-preview", ++id, 0, 0), faces);
        message["inference_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        output << message.dump() << std::endl;
    }
}
