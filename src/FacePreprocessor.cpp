#include "FacePreprocessor.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/dnn.hpp> // Required for dnn::NMSBoxes

FacePreprocessor::FacePreprocessor(const std::string& scrfd_model_path) {
    // 1. Konfigurasi ONNX Runtime Session
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Di Windows, ONNX Runtime mewajibkan string path dalam bentuk wide-char (wstring)
#ifdef _WIN32
    std::string model_path = "D:\\ai-recognition-cpp\\models\\scrfd_2.5g_kps.onnx";
    std::wstring w_model_path(model_path.begin(), model_path.end());
    const wchar_t* model_path_ptr = w_model_path.c_str();
#else
    const char* model_path_ptr = scrfd_model_path.c_str();
#endif

    try {
        // Load model .onnx ke memori
        session_ = std::make_unique<Ort::Session>(env_, model_path_ptr, session_options_);
    } catch (const Ort::Exception& e) {
        std::cerr << "[FacePreprocessor] ERROR: Gagal me-load ONNX model SCRFD: " << e.what() << std::endl;
        std::cerr << "Pastikan file model .onnx berada di path yang benar!" << std::endl;
        return;
    }

    // 2. Alokasi nama In/Out nodes agar dinamis sesuai model
    size_t num_input_nodes = session_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(i, allocator_);
        input_names_str_.push_back(input_name_ptr.get());
    }
    for (const auto& str : input_names_str_) input_node_names_.push_back(str.c_str());

    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        output_names_str_.push_back(output_name_ptr.get());
        
        // --- DEBUG PRINT ---
        auto type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << "[DEBUG ONNX] Output " << i << ": " << output_name_ptr.get() << " Shape: [";
        for (auto dim : tensor_info.GetShape()) std::cout << dim << ", ";
        std::cout << "]" << std::endl;
    }
    for (const auto& str : output_names_str_) output_node_names_.push_back(str.c_str());
    
    std::cout << "[FacePreprocessor] Berhasil memuat model ONNX SCRFD!" << std::endl;
}

FacePreprocessor::~FacePreprocessor() {}

bool FacePreprocessor::isBlurry(const cv::Mat& image) {
    cv::Mat gray, laplacian;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image;
    
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    double variance = stddev.val[0] * stddev.val[0];
    return variance < blur_threshold_;
}

cv::Mat FacePreprocessor::applyCLAHE(const cv::Mat& image) {
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::merge(lab_planes, lab_image);
    cv::Mat clahe_bgr;
    cv::cvtColor(lab_image, clahe_bgr, cv::COLOR_Lab2BGR);
    return clahe_bgr;
}

cv::Mat FacePreprocessor::normalizeImage(const cv::Mat& image) {
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC3);
    return (float_image / 127.5f) - 1.0f;
}

cv::Mat FacePreprocessor::alignFace5Points(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 5) return frame.clone();
    
    // Algoritma Alignment menggunakan sudut mata kiri dan kanan (Titik 0 dan 1)
    cv::Point2f left_eye = landmarks[0];
    cv::Point2f right_eye = landmarks[1];
    
    double dy = right_eye.y - left_eye.y;
    double dx = right_eye.x - left_eye.x;
    double angle = std::atan2(dy, dx) * 180.0 / CV_PI;
    
    cv::Point2f center((left_eye.x + right_eye.x) / 2.0f, (left_eye.y + right_eye.y) / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cv::Mat aligned;
    cv::warpAffine(frame, aligned, rot_mat, frame.size(), cv::INTER_CUBIC);
    return aligned;
}

std::vector<PreprocessedFace> FacePreprocessor::detectFacesSCRFD(const cv::Mat& frame) {
    std::vector<PreprocessedFace> faces;
    if (!session_) return faces;

    // --- TAHAP 1: PRE-INFERENCE (Resize + Keep Ratio) ---
    int inpWidth = 640;
    int inpHeight = 640;
    int srch = frame.rows, srcw = frame.cols;
    int newh = inpHeight, neww = inpWidth;
    int padh = 0, padw = 0;
    
    cv::Mat dstimg;
    float hw_scale = (float)srch / srcw;
    if (hw_scale > 1) {
        newh = inpHeight;
        neww = int(inpWidth / hw_scale);
        cv::resize(frame, dstimg, cv::Size(neww, newh), 0, 0, cv::INTER_AREA);
        padw = int((inpWidth - neww) * 0.5);
        cv::copyMakeBorder(dstimg, dstimg, 0, 0, padw, inpWidth - neww - padw, cv::BORDER_CONSTANT, 0);
    } else {
        newh = (int)(inpHeight * hw_scale);
        neww = inpWidth;
        cv::resize(frame, dstimg, cv::Size(neww, newh), 0, 0, cv::INTER_AREA);
        padh = (int)(inpHeight - newh) * 0.5;
        cv::copyMakeBorder(dstimg, dstimg, padh, inpHeight - newh - padh, 0, 0, cv::BORDER_CONSTANT, 0);
    }

    cv::Mat rgb;
    cv::cvtColor(dstimg, rgb, cv::COLOR_BGR2RGB);
    
    cv::Mat float_image;
    // Normalisasi SCRFD: (pixel - 127.5) / 128.0
    rgb.convertTo(float_image, CV_32FC3, 1.0/128.0, -127.5/128.0); 

    std::vector<cv::Mat> chw;
    for (int i = 0; i < 3; ++i) chw.emplace_back(cv::Size(inpWidth, inpHeight), CV_32FC1);
    cv::split(float_image, chw);
    
    std::vector<float> input_tensor_values;
    for (int i = 0; i < 3; ++i) {
        input_tensor_values.insert(input_tensor_values.end(), (float*)chw[i].datastart, (float*)chw[i].dataend);
    }

    std::vector<int64_t> input_shape = {1, 3, inpHeight, inpWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // --- TAHAP 2: INFERENSI ONNX RUNTIME ---
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, 
        input_node_names_.data(), 
        &input_tensor, 
        1, 
        output_node_names_.data(), 
        output_node_names_.size()
    );

    // --- TAHAP 3: POST-PROCESSING TENSORS (Decoding Anchors) ---
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> landmarks_list;
    
    float ratioh = (float)frame.rows / newh;
    float ratiow = (float)frame.cols / neww;
    const float strides[3] = {8.0f, 16.0f, 32.0f};

    for (int n = 0; n < 3; n++) {
        int num_grid_x = (int)(inpWidth / strides[n]);
        int num_grid_y = (int)(inpHeight / strides[n]);
        
        // Output dari onnx model ini dikelompokkan berdasarkan tipe fitur:
        // Indices 0, 1, 2 = scores untuk stride 8, 16, 32
        // Indices 3, 4, 5 = bboxes untuk stride 8, 16, 32
        // Indices 6, 7, 8 = kps untuk stride 8, 16, 32
        const float* pdata_score = output_tensors[n].GetTensorData<float>();
        const float* pdata_bbox  = output_tensors[n + 3].GetTensorData<float>();
        const float* pdata_kps   = output_tensors[n + 6].GetTensorData<float>();

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                for (int k = 0; k < 2; k++) { // 2 anchors per grid
                    if (pdata_score[0] > conf_threshold_) {
                        const int xmin = (int)(((j - pdata_bbox[0]) * strides[n] - padw) * ratiow);
                        const int ymin = (int)(((i - pdata_bbox[1]) * strides[n] - padh) * ratioh);
                        const int width = (int)((pdata_bbox[2] + pdata_bbox[0]) * strides[n] * ratiow);
                        const int height = (int)((pdata_bbox[3] + pdata_bbox[1]) * strides[n] * ratioh);
                        
                        confidences.push_back(pdata_score[0]);
                        boxes.push_back(cv::Rect(xmin, ymin, width, height));
                        
                        std::vector<cv::Point2f> landmark;
                        for (int l = 0; l < 10; l += 2) {
                            float lx = ((j + pdata_kps[l]) * strides[n] - padw) * ratiow;
                            float ly = ((i + pdata_kps[l + 1]) * strides[n] - padh) * ratioh;
                            landmark.push_back(cv::Point2f(lx, ly));
                        }
                        landmarks_list.push_back(landmark);
                    }
                    pdata_score++;
                    pdata_bbox += 4;
                    pdata_kps += 10;
                }
            }
        }
    }

    // --- TAHAP 4: NON-MAXIMUM SUPPRESSION (NMS) ---
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
    
    for (int idx : indices) {
        PreprocessedFace face;
        face.bounding_box = boxes[idx];
        face.confidence_score = confidences[idx];
        face.landmarks = landmarks_list[idx];
        faces.push_back(face);
    }

    return faces;
}

std::vector<PreprocessedFace> FacePreprocessor::process(const cv::Mat& frame) {
    std::vector<PreprocessedFace> results;
    
    // Deteksi Wajah dan Landmarks dari Model SCRFD
    auto detected_faces = detectFacesSCRFD(frame);
    
    for (const auto& raw_face : detected_faces) {
        // 1. Align wajah agar lurus menggunakan 5 landmarks
        cv::Mat aligned = alignFace5Points(frame, raw_face.landmarks);
        
        // 2. Cek apakah gambar telalu blur
        // if (isBlurry(aligned)) continue; // Tunda dulu pengecekan blur selama masa testing
        
        // 3. Potong (Crop) & Resize ke 112x112
        cv::Rect box = raw_face.bounding_box;
        box.x = std::max(0, box.x);
        box.y = std::max(0, box.y);
        box.width = std::min(aligned.cols - box.x, box.width);
        box.height = std::min(aligned.rows - box.y, box.height);
        
        if (box.width <= 0 || box.height <= 0) continue;
        
        cv::Mat face_crop = aligned(box);
        cv::Mat resized;
        cv::resize(face_crop, resized, cv::Size(target_size_, target_size_));
        
        // 4. Seimbangkan Kecerahan
        cv::Mat enhanced = applyCLAHE(resized);
        
        // 5. Normalisasi piksel dari (0 - 255) menjadi (-1.0 - 1.0) tipe Float32
        cv::Mat normalized = normalizeImage(enhanced);
        
        PreprocessedFace result = raw_face;
        result.face_image = normalized;
        results.push_back(result);
    }
    
    return results;
}
