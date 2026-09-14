#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

struct PreprocessedFace {
    cv::Mat face_image;         // 112x112 CV_32FC3 (normalized)
    cv::Rect bounding_box;      // Original coordinates in frame
    float confidence_score;     // Detection confidence
    std::vector<cv::Point2f> landmarks; // 5 facial landmarks from SCRFD
};

class FacePreprocessor {
public:
    /**
     * @brief Construct a new Face Preprocessor
     * 
     * @param scrfd_model_path Path to the SCRFD ONNX model (.onnx)
     */
    explicit FacePreprocessor(const std::string& scrfd_model_path);
    ~FacePreprocessor();

    /**
     * @brief Process a single frame: Detect (SCRFD), Align (5 landmarks), Crop, Enhance, Normalize
     * 
     * @param frame The input BGR frame from the camera
     * @return std::vector<PreprocessedFace> List of processed faces
     */
    std::vector<PreprocessedFace> process(const cv::Mat& frame);

private:
    // Configurable thresholds
    double blur_threshold_ = 50.0; // Lowered for modern cameras
    int target_size_ = 112;
    float conf_threshold_ = 0.5f;
    float nms_threshold_ = 0.4f;

    // ONNX Runtime components
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "FacePreprocessor"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    // Model specific metadata
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    // Helper methods for Preprocessing Pipeline
    bool isBlurry(const cv::Mat& image);
    cv::Mat applyCLAHE(const cv::Mat& image);
    cv::Mat normalizeImage(const cv::Mat& image);
    
    // SCRFD Implementation Methods
    std::vector<PreprocessedFace> detectFacesSCRFD(const cv::Mat& frame);
    cv::Mat alignFace5Points(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
};
