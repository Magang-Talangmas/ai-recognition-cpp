#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo 'Use a 64-bit ARM OS (aarch64) on Raspberry Pi 5.' >&2
  exit 1
fi
: "${ONNXRUNTIME_ROOT:?Set ONNXRUNTIME_ROOT to the extracted Linux aarch64 CPU SDK directory}"
test -f "$ONNXRUNTIME_ROOT/include/onnxruntime_cxx_api.h"
test -f "$ONNXRUNTIME_ROOT/lib/libonnxruntime.so"
test -f models/scrfd_2.5g_kps.onnx
cmake -S . -B build-pi -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GPU=OFF -DBUILD_PI_NODE=ON -DBUILD_TESTING=ON \
  -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build build-pi --target face_detector_pi pi_contract_tests --parallel 2
ctest --test-dir build-pi --output-on-failure
