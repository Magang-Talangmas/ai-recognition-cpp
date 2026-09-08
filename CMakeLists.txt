cmake_minimum_required(VERSION 3.14)
project(FaceRecognitionPreprocessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

option(USE_GPU "Enable NVIDIA GPU (CUDA) support" ON)

if(USE_GPU)
    message(STATUS "MIGRASI GPU DIAKTIFKAN! Menggunakan prebuilt ONNX Runtime GPU...")
    # Arahkan CMake ke folder onnxruntime-gpu hasil download
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-gpu/include)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-gpu/lib)
    add_compile_definitions(USE_GPU)
else()
    message(STATUS "Berjalan di mode CPU.")
    find_package(onnxruntime CONFIG REQUIRED)
endif()

# Threading
find_package(Threads REQUIRED)

# Fetch nlohmann/json automatically
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# For Redis, we expect redis++ and hiredis to be installed on the system
# vcpkg install hiredis redis-plus-plus
find_path(HIREDIS_INCLUDE_DIRS hiredis/hiredis.h)
find_library(HIREDIS_LIBRARIES hiredis)

find_path(REDIS_PLUS_PLUS_INCLUDE_DIRS sw/redis++/redis++.h)
find_library(REDIS_PLUS_PLUS_LIBRARIES redis++)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${HIREDIS_INCLUDE_DIRS}
    ${REDIS_PLUS_PLUS_INCLUDE_DIRS}
)

add_executable(face_preprocessor
    src/main.cpp
    src/StreamReader.cpp
    src/FacePreprocessor.cpp
    src/BrokerPublisher.cpp
)

if(USE_GPU)
    target_link_libraries(face_preprocessor PRIVATE
        ${OpenCV_LIBS}
        ${REDIS_PLUS_PLUS_LIBRARIES}
        ${HIREDIS_LIBRARIES}
        nlohmann_json::nlohmann_json
        Threads::Threads
        onnxruntime
        onnxruntime_providers_cuda
    )
else()
    target_link_libraries(face_preprocessor PRIVATE
        ${OpenCV_LIBS}
        ${REDIS_PLUS_PLUS_LIBRARIES}
        ${HIREDIS_LIBRARIES}
        nlohmann_json::nlohmann_json
        Threads::Threads
        onnxruntime::onnxruntime
    )
endif()
