#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FaceEngine.h"

// Note: To pass cv::Mat seamlessly between C++ and Python (numpy array), 
// ndarray converter headers like pybind11-OpenCV would be used in full implementation.

namespace py = pybind11;

PYBIND11_MODULE(face_engine_core, m) {
    m.doc() = "C++ Core for Face Recognition (SCRFD & ArcFace)";

    py::class_<FaceDetectionResult>(m, "FaceDetectionResult")
        .def(py::init<>())
        .def_readwrite("x1", &FaceDetectionResult::x1)
        .def_readwrite("y1", &FaceDetectionResult::y1)
        .def_readwrite("x2", &FaceDetectionResult::x2)
        .def_readwrite("y2", &FaceDetectionResult::y2)
        .def_readwrite("confidence", &FaceDetectionResult::confidence);

    py::class_<FaceEngine>(m, "FaceEngine")
        .def(py::init<const std::string&, const std::string&>())
        // In real implementation, pybind11 needs converter for cv::Mat <-> numpy array
        // .def("detect", &FaceEngine::detect) 
        // .def("align_face", &FaceEngine::alignFace)
        // .def("extract_feature", &FaceEngine::extractFeature)
        .def("calculate_similarity", &FaceEngine::calculateSimilarity);
}
