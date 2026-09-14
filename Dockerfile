FROM python:3.10-slim

WORKDIR /app

# Install dependensi compiler dan OpenCV sistem yang dibutuhkan InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependensi python dari requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code
COPY . .

EXPOSE 5001
