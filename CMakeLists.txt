cmake_minimum_required(VERSION 3.14)
project(FaceRecognitionPreprocessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

option(USE_GPU "Enable NVIDIA GPU (CUDA) support" ON)

if(USE_GPU)
    message(STATUS "MIGRASI GPU DIAKTIFKAN! Menggunakan prebuilt ONNX Runtime GPU...")
    if(UNIX)
        include(FetchContent)
        FetchContent_Declare(onnxruntime_gpu
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.16.1/onnxruntime-linux-x64-gpu-1.16.1.tgz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(onnxruntime_gpu)
        include_directories(${onnxruntime_gpu_SOURCE_DIR}/include)
        link_directories(${onnxruntime_gpu_SOURCE_DIR}/lib)
    else()
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-gpu/include)
        link_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-gpu/lib)
    endif()
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

# Fetch cpp-httplib automatically (For HTTP API Requests)
FetchContent_Declare(httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.14.1.tar.gz
)
FetchContent_MakeAvailable(httplib)

# Hiredis is provided by the operating system on Linux.
find_path(HIREDIS_INCLUDE_DIRS hiredis/hiredis.h)
find_library(HIREDIS_LIBRARIES hiredis)

if(NOT HIREDIS_INCLUDE_DIRS OR NOT HIREDIS_LIBRARIES)
    message(FATAL_ERROR "hiredis development files are required (install libhiredis-dev).")
endif()

# Pop!_OS does not provide a redis-plus-plus development package. Fetch a pinned
# version during the build and link it against the system hiredis package.
if(UNIX)
    FetchContent_Declare(redis_plus_plus
        GIT_REPOSITORY https://github.com/sewenew/redis-plus-plus.git
        GIT_TAG 1.3.13
        GIT_SHALLOW TRUE
    )
    set(REDIS_PLUS_PLUS_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(REDIS_PLUS_PLUS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(redis_plus_plus)
    set(REDIS_PLUS_PLUS_LIBRARIES redis++_static)
    # redis-plus-plus generates hiredis_features.h during its own build.
    set(REDIS_PLUS_PLUS_INCLUDE_DIRS ${redis_plus_plus_BINARY_DIR}/src)
else()
    find_path(REDIS_PLUS_PLUS_INCLUDE_DIRS sw/redis++/redis++.h)
    find_library(REDIS_PLUS_PLUS_LIBRARIES redis++)
    if(NOT REDIS_PLUS_PLUS_INCLUDE_DIRS OR NOT REDIS_PLUS_PLUS_LIBRARIES)
        message(FATAL_ERROR "redis-plus-plus development files are required.")
    endif()
endif()

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

        httplib


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

        httplib


        Threads::Threads
        onnxruntime::onnxruntime
    )
endif()
