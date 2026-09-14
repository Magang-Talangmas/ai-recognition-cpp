cmake_minimum_required(VERSION 3.10)
project(ai-recognition-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Pybind11
# find_package(pybind11 REQUIRED)

# OpenCV
# find_package(OpenCV REQUIRED)

# ONNX Runtime (Assume installed locally or in 3rdparty folder)
# find_package(onnxruntime REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# pybind11_add_module(face_engine_core 
#     src/bindings.cpp 
#     src/FaceEngine.cpp
# )

# target_link_libraries(face_engine_core PRIVATE 
#     ${OpenCV_LIBS} 
#     ${ONNXRUNTIME_LIBS}
# )
