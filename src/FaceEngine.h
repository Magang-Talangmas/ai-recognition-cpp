#ifndef FACE_ENGINE_H
#define FACE_ENGINE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
// #include <onnxruntime_cxx_api.h> // Will be included in actual implementation

struct FaceDetectionResult {
    float x1, y1, x2, y2;
    float confidence;
    std::vector<cv::Point2f> landmarks;
};

class FaceEngine {
public:
    FaceEngine(const std::string& scrfd_model_path, const std::string& arcface_model_path);
    ~FaceEngine();

    // Detects faces in an image (Returns bounding boxes and landmarks)
    std::vector<FaceDetectionResult> detect(const cv::Mat& image);

    // Aligns and extracts the face based on landmarks
    cv::Mat alignFace(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks);

    // Extracts features (embedding) from an aligned face
    std::vector<float> extractFeature(const cv::Mat& aligned_face);

    // Calculates cosine similarity between two embeddings
    float calculateSimilarity(const std::vector<float>& emb1, const std::vector<float>& emb2);

private:
    // Pointers to ONNX Runtime sessions would be here
    // Ort::Env env;
    // Ort::Session scrfd_session;
    // Ort::Session arcface_session;
    
    // Internal helper methods
    void preprocess(const cv::Mat& image, cv::Mat& blob);
};

#endif // FACE_ENGINE_H
