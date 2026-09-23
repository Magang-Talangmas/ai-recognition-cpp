"""Live camera preview using the actual C++ Pi detector, without Redis or FE."""
import argparse
import json
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


class Detector:
    def __init__(self, executable, config):
        self.process = subprocess.Popen(
            [str(executable), "--config", str(config), "--preview-stdio"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.replies = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self.settings = self.reply()
            if not self.settings.get("ready"):
                raise RuntimeError("Detector did not send its ready message")
        except Exception:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def reply(self):
        try:
            line = self.replies.get(timeout=30)
        except queue.Empty:
            raise RuntimeError("Detector did not respond within 30 seconds") from None
        if line is None:
            raise RuntimeError("C++ detector stopped; see its terminal error above")
        return json.loads(line)

    def detect(self, frame):
        height, width = frame.shape[:2]
        if width > 4096 or height > 4096:
            raise RuntimeError("Preview supports up to 4096x4096; use a smaller substream")
        self.process.stdin.write(struct.pack("<III", width, height, frame.nbytes))
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()
        return self.reply()

    def close(self):
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


class Camera:
    def __init__(self, cv2, url):
        self.cv2, self.url = cv2, url
        self.lock = threading.Lock()
        self.frame, self.sequence, self.received = None, 0, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        cv2 = self.cv2
        while not self.stop.is_set():
            cap = cv2.VideoCapture()
            try:
                # Allow startup to reach a complete H.264 keyframe before retrying.
                if cap.open(self.url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                                                       cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000]):
                    print("[Preview] RTSP connected", flush=True)
                    while not self.stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            break
                        with self.lock:
                            self.frame = frame
                            self.sequence += 1
                            self.received = time.monotonic()
            except cv2.error:
                print("[Preview] Camera decode error", flush=True)
            finally:
                cap.release()
                with self.lock:
                    self.frame = None
            if not self.stop.is_set():
                print("[Preview] Stream unavailable; retry in 3 seconds", flush=True)
                self.stop.wait(3)

    def latest(self, previous):
        with self.lock:
            if self.frame is None or self.sequence == previous or time.monotonic()-self.received > 2:
                return None
            return self.frame, self.sequence

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)


def overlay(cv2, frame, result):
    canvas = frame.copy()
    for face in result["bounding_boxes"]:
        box = face["bounding_box"]
        x, y, w, h = (round(box[key]) for key in ("x", "y", "width", "height"))
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(canvas, f"Face {face['confidence_score']:.2f}", (x, max(20, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for px, py in face["landmarks"]:
            cv2.circle(canvas, (round(px), round(py)), 3, (0, 200, 255), -1)
    title = f"C++ SCRFD CPU | faces={len(result['bounding_boxes'])} | {result['inference_ms']:.0f} ms | Q / Esc: exit"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1000), 38), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "deploy/pi/pi.env.example")
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N processed frames; 0 = continuous")
    parser.add_argument("--snapshot", type=Path, help="Save last annotated frame locally (opt-in)")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    import cv2
    import numpy as np
    executable = args.exe or ROOT / ("build-pi-check/Release/face_detector_pi.exe" if os.name == "nt" else "build-pi/face_detector_pi")
    if not executable.is_file():
        parser.error(f"Build the detector first; executable not found: {executable}")
    detector = Detector(executable.resolve(), args.config.resolve())
    camera = Camera(cv2, detector.settings["rtsp_url"])
    window = "Pi detector - live camera"
    canvas, last, processed = None, 0, 0
    next_detection = 0
    last_success = time.monotonic()
    try:
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1100, 650)
        print("[Preview] Same C++ model as Pi; local visual test, Redis publishing disabled", flush=True)
        while True:
            now = time.monotonic()
            latest = camera.latest(last) if now >= next_detection else None
            if latest:
                frame, last = latest
                result = detector.detect(frame)
                canvas = overlay(cv2, frame, result)
                processed += 1
                last_success = time.monotonic()
                next_detection = last_success + 1 / detector.settings["fps"]
                print(f"[Preview] frame={last} faces={len(result['bounding_boxes'])} inference_ms={result['inference_ms']:.1f}", flush=True)
                if args.frames and processed >= args.frames:
                    break
            if now-last_success > 3:
                canvas = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for camera / reconnecting...", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if args.headless:
                if now-last_success > 40:
                    raise RuntimeError("No usable camera frame within 40 seconds; check RTSP/network")
                time.sleep(0.02)
            else:
                if canvas is not None:
                    cv2.imshow(window, canvas)
                if cv2.waitKey(20) & 0xFF in (27, ord("q")):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        if args.snapshot and processed and canvas is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), canvas):
                raise RuntimeError("Could not save preview snapshot")
            print(f"[Preview] Saved {args.snapshot}")
    finally:
        camera.close()
        detector.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        raise SystemExit(f"[Preview] {error}")
