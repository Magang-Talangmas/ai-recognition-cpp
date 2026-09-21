import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
TARGET_FPS = max(1, int(os.getenv("STREAM_TARGET_FPS", "15")))
JPEG_QUALITY = max(1, min(100, int(os.getenv("STREAM_JPEG_QUALITY", "70"))))

NVDEC_DECODERS = {"h264": "h264_cuvid", "hevc": "hevc_cuvid"}


def init_redis():
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
        client.ping()
        print(f"[Redis] Connected to {REDIS_HOST}:{REDIS_PORT}")
        return client
    except Exception as error:
        print(f"[Redis] Failed to connect: {error}")
        return None


def probe_stream(rtsp_url):
    command = [
        "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
        "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height",
        "-of", "json", rtsp_url,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("video stream tidak ditemukan")

    stream = streams[0]
    codec = stream.get("codec_name")
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if not codec or width <= 0 or height <= 0:
        raise RuntimeError(f"metadata video tidak valid: {stream}")
    if codec not in NVDEC_DECODERS:
        raise RuntimeError(f"codec {codec} belum didukung NVDEC worker")
    return codec, width, height


def start_nvdec_decoder(rtsp_url, codec):
    """Decode RTSP with NVIDIA NVDEC, then transfer BGR frames for JPEG/Redis."""
    decoder = NVDEC_DECODERS[codec]
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", "tcp", "-hwaccel", "cuda", "-c:v", decoder,
        "-i", rtsp_url, "-an", "-vf", f"fps={TARGET_FPS}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    print(f"[NVDEC] Starting {decoder} at {TARGET_FPS} FPS")
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, bufsize=0)


def read_exact(pipe, size):
    data = bytearray()
    while len(data) < size:
        chunk = pipe.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return data


def stop_decoder(process):
    if not process or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description="RTSP NVDEC worker for Media Server")
    parser.add_argument("camera_id", help="ID kamera")
    parser.add_argument("rtsp_url", help="RTSP URL kamera")
    args = parser.parse_args()

    redis_client = init_redis()
    if not redis_client:
        sys.exit(1)

    channel_name = f"camera:{args.camera_id}:frames"
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    while True:
        decoder_process = None
        try:
            codec, width, height = probe_stream(args.rtsp_url)
            print(f"[Worker {args.camera_id}] {codec} {width}x{height} -> Redis {channel_name}")
            decoder_process = start_nvdec_decoder(args.rtsp_url, codec)
            frame_bytes = width * height * 3

            while decoder_process.poll() is None:
                raw_frame = read_exact(decoder_process.stdout, frame_bytes)
                if raw_frame is None:
                    raise RuntimeError("FFmpeg berhenti mengirim frame")

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                success, jpeg = cv2.imencode(".jpg", frame, encode_params)
                if success:
                    redis_client.publish(channel_name, jpeg.tobytes())
        except KeyboardInterrupt:
            print(f"[Worker {args.camera_id}] Stopping worker...")
            return
        except Exception as error:
            print(f"[Worker {args.camera_id}] NVDEC stream error: {error}; reconnecting in 3 seconds...")
            time.sleep(3)
        finally:
            stop_decoder(decoder_process)


if __name__ == "__main__":
    main()
