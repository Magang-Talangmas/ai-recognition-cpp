"""Local Face Only microservice for the PPE backend (Hailo-8 SCRFD HEF)."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from face_service.hailo_scrfd import HailoScrfd


DEFAULT_HEF = Path(__file__).resolve().parents[1] / "models" / "scrfd.hef"
HEF_PATH = Path(os.getenv("FACE_HEF_PATH", str(DEFAULT_HEF)))
engine = HailoScrfd(HEF_PATH)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    engine.close()


app = FastAPI(title="SCRFD Face Only", lifespan=lifespan)


class EnabledRequest(BaseModel):
    enabled: bool


@app.get("/health")
def health():
    return {"ready": engine.ready, "model": HEF_PATH.name}


@app.post("/enabled")
def set_enabled(payload: EnabledRequest):
    try:
        if payload.enabled:
            engine.open()
        else:
            engine.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ready": engine.ready}


@app.post("/infer")
async def infer(request: Request):
    body = await request.body()
    if not body or len(body) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="JPEG kosong atau terlalu besar")
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="JPEG tidak valid")
    try:
        rows = await run_in_threadpool(engine.detect, frame)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"rows": rows}
