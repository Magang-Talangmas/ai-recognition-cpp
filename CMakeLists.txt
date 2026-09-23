cmake_minimum_required(VERSION 3.20)
project(FaceRecognitionPreprocessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

option(USE_GPU "Enable NVIDIA GPU (CUDA) support" ON)
option(BUILD_PI_NODE "Build the single-camera headless RTSP CPU node" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Native CPU ONNX Runtime SDK (include/ and lib/)")
if(BUILD_PI_NODE AND USE_GPU)
    message(FATAL_ERROR "The Pi node requires -DUSE_GPU=OFF")
endif()
if(USE_GPU AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    message(FATAL_ERROR "This CUDA package is x64. For Pi use -DUSE_GPU=OFF -DBUILD_PI_NODE=ON")
endif()

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
    if(ONNXRUNTIME_ROOT)
        find_path(ORT_INCLUDE onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
        find_library(ORT_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
        add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}")
    else()
        find_package(onnxruntime CONFIG REQUIRED)
    endif()
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

if(BUILD_PI_NODE)
    if(OpenCV_VERSION VERSION_LESS "4.5.2")
        message(FATAL_ERROR "Pi RTSP timeouts require OpenCV >= 4.5.2 with FFmpeg support")
    endif()
    add_executable(face_detector_pi src/pi_main.cpp src/FacePreprocessor.cpp)
    target_link_libraries(face_detector_pi PRIVATE ${OpenCV_LIBS}
        ${REDIS_PLUS_PLUS_LIBRARIES} ${HIREDIS_LIBRARIES}
        nlohmann_json::nlohmann_json Threads::Threads onnxruntime::onnxruntime)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(pi_contract_tests tests/pi_contract_tests.cpp)
        target_link_libraries(pi_contract_tests PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json onnxruntime::onnxruntime)
        add_test(NAME pi_contract COMMAND pi_contract_tests)
        add_test(NAME pi_model_smoke COMMAND face_detector_pi --config ${CMAKE_SOURCE_DIR}/deploy/pi/pi.env.example --check-model)
        set_tests_properties(pi_model_smoke PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endif()

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
