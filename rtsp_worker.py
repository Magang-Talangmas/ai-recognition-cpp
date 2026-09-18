import sys
import time
import cv2
import argparse
import os
import redis
from dotenv import load_dotenv

load_dotenv()

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

def init_redis():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
        r.ping()
        print(f"[Redis] Connected to {REDIS_HOST}:{REDIS_PORT}")
        return r
    except Exception as e:
        print(f"[Redis] Failed to connect: {e}")
        return None

def connect_camera(rtsp_url, camera_id):
    print(f"[Worker {camera_id}] Attempting to connect to stream: {rtsp_url}")
    # Try connecting to the stream with hardware acceleration
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_ANY, [
        cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,
        # Optional: you can force CUDA specifically if ANY doesn't pick it up:
        # cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_CUDA
    ])
    
    # Set small buffer to avoid lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def main():
    parser = argparse.ArgumentParser(description="RTSP Worker for Media Server")
    parser.add_argument("camera_id", type=str, help="ID of the camera")
    parser.add_argument("rtsp_url", type=str, help="RTSP URL to connect to")
    args = parser.parse_args()

    camera_id = args.camera_id
    rtsp_url = args.rtsp_url
    channel_name = f"camera:{camera_id}:frames"

    redis_client = init_redis()
    if not redis_client:
        print(f"[Worker {camera_id}] Redis is required. Exiting...")
        sys.exit(1)

    cap = connect_camera(rtsp_url, camera_id)
    
    # Framerate limiter config (e.g. max 15 FPS to save network/Redis bandwidth)
    TARGET_FPS = 15
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    last_processed_time = 0

    try:
        while True:
            if not cap.isOpened():
                print(f"[Worker {camera_id}] Stream disconnected. Reconnecting in 3 seconds...")
                time.sleep(3)
                cap = connect_camera(rtsp_url, camera_id)
                continue

            ret, frame = cap.read()
            if not ret:
                print(f"[Worker {camera_id}] No frame received. Reconnecting...")
                cap.release()
                continue
                
            current_time = time.time()
            if (current_time - last_processed_time) >= FRAME_INTERVAL:
                # Encode frame to JPEG (Quality 70% for balance of speed and size)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                success, buffer = cv2.imencode('.jpg', frame, encode_param)
                
                if success:
                    # Publish to Redis Pub/Sub
                    redis_client.publish(channel_name, buffer.tobytes())
                    last_processed_time = current_time

    except KeyboardInterrupt:
        print(f"[Worker {camera_id}] Stopping worker...")
    finally:
        if cap and cap.isOpened():
            cap.release()
        print(f"[Worker {camera_id}] Cleanup complete.")

if __name__ == "__main__":
    main()
