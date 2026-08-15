FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    libhiredis-dev \
    && rm -rf /var/lib/apt/lists/*

# Install redis-plus-plus
RUN cd /tmp && \
    git clone https://github.com/sewenew/redis-plus-plus.git && \
    cd redis-plus-plus && \
    mkdir build && cd build && \
    cmake -DREDIS_PLUS_PLUS_CXX_STANDARD=17 .. && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/redis-plus-plus

# Install ONNX Runtime
RUN cd /tmp && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.3.tgz && \
    cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/onnxruntime*

# Copy source code
WORKDIR /app
COPY . /app

# Build the project
RUN mkdir -p build && cd build && rm -rf * && \
    cmake .. && \
    make -j$(nproc)

CMD ["./build/face_preprocessor"]
