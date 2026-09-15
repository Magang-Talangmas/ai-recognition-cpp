FROM python:3.10-slim

WORKDIR /app

# Install dependensi sistem yang dibutuhkan OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install paket Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    redis \
    numpy \
    requests \
    python-dotenv \
    opencv-python-headless \
    onnxruntime

# Copy kode aplikasi
COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]
