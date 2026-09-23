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
        self.epoch = 0
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
                    with self.lock:
                        self.epoch += 1
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
            return self.frame, self.sequence, self.received, self.epoch

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)


class LiveDetection:
    """One inference at a time, always taking the latest frame; never a FIFO backlog."""
    def __init__(self, detector, camera, fps):
        self.detector, self.camera, self.interval = detector, camera, 1 / fps
        self.lock = threading.Lock()
        self.result, self.error, self.count = None, None, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        previous = 0
        try:
            while not self.stop.is_set():
                latest = self.camera.latest(previous)
                if latest is None:
                    self.stop.wait(0.005)
                    continue
                frame, previous, received, epoch = latest
                start = time.monotonic()
                result = self.detector.detect(frame)
                result.update(source_received=received, source_epoch=epoch,
                              source_sequence=previous, ready_at=time.monotonic())
                with self.lock:
                    self.result = result
                    self.count += 1
                # Period includes inference time, instead of adding a long rest afterwards.
                self.stop.wait(max(0, self.interval-(time.monotonic()-start)))
        except Exception as error:
            with self.lock:
                self.error = error

    def latest(self):
        with self.lock:
            if self.error:
                raise self.error
            return self.result, self.count

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            self.detector.process.kill()
            self.thread.join(timeout=3)


def fresh_result(result, received, epoch, now):
    # Never keep boxes across reconnects, changed geometry, or >1s-old source frames.
    if result is None or result["source_epoch"] != epoch or now-result["source_received"] > 1:
        return None
    if result["source_received"] > received:
        return None
    return result


class BoxSmoother:
    """Short display interpolation; no identity recognition or long-lived tracks."""
    def __init__(self):
        self.boxes = []
        self.updated = None

    @staticmethod
    def iou(a, b):
        intersection = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
            0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
        union = a[2]*a[3]+b[2]*b[3]-intersection
        return intersection/union if union > 0 else 0

    def apply(self, faces, now):
        import math
        alpha = 1 if self.updated is None else 1-math.exp(-max(0, now-self.updated)/0.04)
        self.updated = now
        available = list(self.boxes)
        boxes, result = [], []
        for face in faces:
            target = [face["bounding_box"][key] for key in ("x", "y", "width", "height")]
            scores = [self.iou(target, old) for old in available]
            if scores and max(scores) > 0.2:
                old = available.pop(scores.index(max(scores)))
                target = [a+(b-a)*alpha for a, b in zip(old, target)]
            boxes.append(target)
            result.append(dict(face, bounding_box=dict(zip(("x", "y", "width", "height"), target))))
        self.boxes = boxes
        return result


def overlay(cv2, frame, result, fps, age_ms):
    canvas = frame.copy()
    for face in result["bounding_boxes"]:
        box = face["bounding_box"]
        x, y, w, h = (round(box[key]) for key in ("x", "y", "width", "height"))
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(canvas, f"Face {face['confidence_score']:.2f}", (x, max(20, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    title = f"Video {fps:.1f} FPS | faces={len(result['bounding_boxes'])} | detect {result['inference_ms']:.0f} ms | age {age_ms:.0f} ms | Q: exit"
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
    parser.add_argument("--detection-fps", type=float, default=10, help="Laptop preview detection ceiling (default 10)")
    parser.add_argument("--display-fps", type=float, default=30, help="Video display ceiling (default 30)")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if not 0.2 <= args.detection_fps <= 30 or not 1 <= args.display_fps <= 60:
        parser.error("Detection FPS must be 0.2..30 and display FPS 1..60")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    executable = args.exe or ROOT / ("build-pi-check/Release/face_detector_pi.exe" if os.name == "nt" else "build-pi/face_detector_pi")
    if not executable.is_file():
        parser.error(f"Build the detector first; executable not found: {executable}")
    detector = Detector(executable.resolve(), args.config.resolve())
    camera = Camera(cv2, detector.settings["rtsp_url"])
    detection = LiveDetection(detector, camera, args.detection_fps)
    smoother = BoxSmoother()
    window = "Pi detector - live camera"
    canvas, last, processed = None, 0, 0
    shown, previous_count, display_rate, detection_rate = 0, 0, 0, 0
    meter_at = time.monotonic()
    last_success = time.monotonic()
    try:
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1100, 650)
        print("[Preview] Same C++ model as Pi; local visual test, Redis publishing disabled", flush=True)
        print(f"[Preview] Independent video ceiling={args.display_fps:g} FPS; detection ceiling={args.detection_fps:g} FPS", flush=True)
        while True:
            now = time.monotonic()
            result, processed = detection.latest()
            latest = camera.latest(last)
            if latest:
                frame, last, received, epoch = latest
                current = fresh_result(result, received, epoch, now)
                if current and (current["frame_width"], current["frame_height"]) != (frame.shape[1], frame.shape[0]):
                    current = None
                age_ms = (now-current["source_received"])*1000 if current else 0
                current = dict(current) if current else {"bounding_boxes": [], "inference_ms": 0}
                current["bounding_boxes"] = smoother.apply(current["bounding_boxes"], now)
                canvas = overlay(cv2, frame, current, display_rate, age_ms)
                shown += 1
                last_success = time.monotonic()
                if args.frames and processed >= args.frames:
                    break
                if now-meter_at >= 5:
                    display_rate = shown/(now-meter_at)
                    detection_rate = (processed-previous_count)/(now-meter_at)
                    print(f"[Preview] video_fps={display_rate:.1f} detection_fps={detection_rate:.1f} "
                          f"faces={len(current['bounding_boxes'])} inference_ms={current['inference_ms']:.1f} "
                          f"result_age_ms={age_ms:.1f}", flush=True)
                    shown, previous_count, meter_at = 0, processed, now
            if now-last_success > 1:
                canvas = np.zeros((480, 800, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for camera / reconnecting...", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if args.headless:
                if now-last_success > 40:
                    raise RuntimeError("No usable camera frame within 40 seconds; check RTSP/network")
            else:
                if canvas is not None:
                    cv2.imshow(window, canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            time.sleep(max(0, 1/args.display_fps-(time.monotonic()-now)))
        if args.snapshot and processed and canvas is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), canvas):
                raise RuntimeError("Could not save preview snapshot")
            print(f"[Preview] Saved {args.snapshot}")
    finally:
        detection.close()
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
