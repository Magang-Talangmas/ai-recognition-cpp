import sys
import time
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="RTSP Worker for Media Server")
    parser.add_argument("camera_id", type=str, help="ID of the camera")
    parser.add_argument("rtsp_url", type=str, help="RTSP URL to connect to")
    args = parser.parse_args()

    camera_id = args.camera_id
    rtsp_url = args.rtsp_url

    print(f"[Worker {camera_id}] Starting RTSP reader for {rtsp_url}")
    
    # Try connecting to the stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[Worker {camera_id}] Failed to open stream: {rtsp_url}")
        sys.exit(1)
        
    print(f"[Worker {camera_id}] Successfully connected to stream. Reading frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[Worker {camera_id}] Stream disconnected or ended.")
                break
                
            # Here we would send the frame to the AI node (e.g. via Redis pubsub or SSE)
            # For now, we just simulate work by sleeping slightly
            # We don't want to flood logs, so we won't print every frame
            time.sleep(0.033) # simulate 30fps processing
            
    except KeyboardInterrupt:
        print(f"[Worker {camera_id}] Stopping worker...")
    finally:
        cap.release()
        print(f"[Worker {camera_id}] Cleanup complete.")

if __name__ == "__main__":
    main()
