#include "PiConfig.hpp"
#include "PiPayload.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

void require(bool condition, const char* text) {
    if (!condition) throw std::runtime_error(text);
}
int main() {
    try {
        require(pi::base64({}) == "", "empty base64");
        require(pi::base64({'f'}) == "Zg==", "base64 padding 2");
        require(pi::base64({'f','o'}) == "Zm8=", "base64 padding 1");
        require(pi::base64({'f','o','o'}) == "Zm9v", "base64 group");
        pi::Config c;
        c.camera = "cam01"; c.stream = "cam01";
        auto meta = pi::metadata(c, {1920,1080}, "session-a", 7, 100, 200);
        auto none = pi::detectionMessage(meta, {});
        require(none["bounding_boxes"].empty(), "empty detections must clear FE");
        require(none["frame_id"] == 7 && none["camera_id"] == "cam01", "frame correlation");
        PreprocessedFace f;
        f.bounding_box = {10,20,60,80}; f.confidence_score = 0.9f;
        f.landmarks = {{20,30},{50,30},{35,40},{25,60},{45,60}};
        f.face_image = cv::Mat(112,112,CV_32FC3,cv::Scalar(0,0,0));
        auto two = pi::detectionMessage(meta, {f,f});
        require(two["bounding_boxes"].size() == 2, "preserve multiple faces in one frame");
        auto face = pi::faceMessage(meta, f, 1, 90);
        require(face["image_format"] == "jpeg_base64_112x112", "recognition image contract");
        require(face["landmarks"].size() == 5 && face["face_index"] == 1, "face metadata");
        require(face["face_image_base64"].get<std::string>().rfind("/9j/",0) == 0, "JPEG bytes, not float tensor");
        const auto file = std::filesystem::temp_directory_path() / "pi-config-test.env";
        { std::ofstream out(file); out << "PI_INFERENCE_FPS=0\n"; }
        bool rejected = false;
        try { pi::loadConfig(file.string()); } catch (const std::exception&) { rejected = true; }
        require(rejected, "reject zero FPS");
        { std::ofstream out(file); out << "PI_RTSP_URL=https://example.invalid/cam01\n"; }
        rejected = false;
        try { pi::loadConfig(file.string()); } catch (const std::exception&) { rejected = true; }
        require(rejected, "reject HTTP camera URL");
        std::filesystem::remove(file);
        std::cout << "Pi payload and configuration contracts passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
