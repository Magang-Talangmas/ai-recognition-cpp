import os
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Media Server Control Plane")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv()
control_plane_port = int(os.getenv("CONTROL_PLANE_PORT", "8011"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CameraConfig(BaseModel):
    camera_id: str
    rtsp_url: str

@app.get("/")
def read_root():
    return {"status": "Media Server Control Plane is Running"}

@app.post("/cameras/start")
def start_camera_worker(config: CameraConfig):
    """
    Start a PM2 worker for a specific camera.
    In the future, the rtsp_url will be fetched from Supabase.
    For MediaMTX, rtsp_url might just be rtsp://localhost:8554/stream/{camera_id}
    """
    worker_name = f"cam_{config.camera_id}"
    
    # Check if already running
    try:
        check = subprocess.run(["pm2", "jlist"], capture_output=True, text=True)
        if worker_name in check.stdout:
            # It's running, maybe restart?
            subprocess.run(["pm2", "restart", worker_name])
            return {"status": "restarted", "worker_name": worker_name}
    except Exception as e:
        pass

    # Start the worker via PM2
    try:
        # Example: pm2 start rtsp_worker.py --name "cam_1" -- "cam_1" "rtsp://localhost:8554/live/cam_1"
        cmd = [
            "pm2", "start", "rtsp_worker.py",
            "--interpreter", sys.executable,
            "--name", worker_name,
            "--", config.camera_id, config.rtsp_url,
        ]
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        
        return {"status": "started", "worker_name": worker_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cameras/stop/{camera_id}")
def stop_camera_worker(camera_id: str):
    """ Stop a PM2 worker for a specific camera. """
    worker_name = f"cam_{camera_id}"
    try:
        cmd = ["pm2", "stop", worker_name]
        subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
        # Optionally delete from PM2 list
        subprocess.run(["pm2", "delete", worker_name], cwd=PROJECT_DIR, capture_output=True, text=True)
        return {"status": "stopped", "worker_name": worker_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cameras/status")
def list_cameras():
    """ List all PM2 workers """
    try:
        # pm2 jlist returns JSON string of all processes
        result = subprocess.run(["pm2", "jlist"], shell=True, capture_output=True, text=True)
        import json
        processes = json.loads(result.stdout)
        workers = []
        for p in processes:
            if p.get("name", "").startswith("cam_"):
                workers.append({
                    "camera_id": p["name"].replace("cam_", ""),
                    "status": p.get("pm2_env", {}).get("status"),
                    "memory": p.get("monit", {}).get("memory"),
                    "cpu": p.get("monit", {}).get("cpu")
                })
        return {"workers": workers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=control_plane_port,
        reload=False,
    )
