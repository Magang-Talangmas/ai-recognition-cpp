"""Run the nine raw SCRFD heads compiled in scrfd.hef on Hailo-8."""

from pathlib import Path
from threading import Event, RLock

import cv2
import numpy as np


STRIDES = (8, 16, 32)


def preprocess(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, 640.0 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size != (width, height):
        image = cv2.resize(image, size)
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:size[1], :size[0]] = image
    return np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def _head(outputs: dict[str, np.ndarray], index: int, side: int, channels: int) -> np.ndarray:
    suffix = f"/output_layer{index}"
    matches = [value for name, value in outputs.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Output HEF {suffix} tidak ditemukan: {list(outputs)}")
    value = np.asarray(matches[0])
    if value.shape != (side, side, channels):
        raise ValueError(f"Bentuk {suffix} tidak sesuai: {value.shape}")
    return value


def decode(outputs: dict[str, np.ndarray], width: int, height: int) -> list[list[float]]:
    """Decode Hailo NHWC heads to the same xywh+score rows as face.onnx."""
    boxes, scores = [], []
    for level, stride in enumerate(STRIDES):
        side = 640 // stride
        # Compiled HN mapping: score, landmarks, bbox for every scale.
        logits = _head(outputs, level * 3 + 1, side, 2).reshape(-1)
        confidence = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        distances = _head(outputs, level * 3 + 3, side, 8).reshape(-1, 4) * stride
        selected = np.flatnonzero(confidence >= 0.5)
        if not len(selected):
            continue
        centers = np.stack(np.mgrid[:side, :side][::-1], axis=-1)
        centers = np.repeat(centers.reshape(-1, 2) * stride, 2, axis=0)
        xy = centers[selected]
        d = distances[selected]
        x1 = np.maximum(0.0, xy[:, 0] - d[:, 0])
        y1 = np.maximum(0.0, xy[:, 1] - d[:, 1])
        x2 = np.minimum(float(width), xy[:, 0] + d[:, 2])
        y2 = np.minimum(float(height), xy[:, 1] + d[:, 3])
        valid = (x2 > x1) & (y2 > y1)
        for pos in np.flatnonzero(valid):
            boxes.append([float(x1[pos]), float(y1[pos]),
                          float(x2[pos] - x1[pos]), float(y2[pos] - y1[pos])])
            scores.append(float(confidence[selected[pos]]))
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)
    return [boxes[int(i)] + [scores[int(i)]] for i in np.asarray(keep).reshape(-1)]


class HailoScrfd:
    """One configured VDevice, serialized calls and explicit device release."""

    def __init__(self, hef_path: Path):
        self.hef_path = hef_path
        self._lock = RLock()
        self._vdevice = None
        self._configured = None
        self._model = None
        self._shapes = {}

    @property
    def ready(self) -> bool:
        return self._configured is not None

    def open(self) -> None:
        with self._lock:
            if self.ready:
                return
            if not self.hef_path.is_file():
                raise FileNotFoundError(f"HEF Face Only tidak ada: {self.hef_path}")
            from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.multi_process_service = True
            params.group_id = "SHARED"
            device = VDevice(params)
            try:
                model = device.create_infer_model(str(self.hef_path))
                model.set_batch_size(1)
                model.input().set_format_type(FormatType.UINT8)
                for output in model.outputs:
                    output.set_format_type(FormatType.FLOAT32)
                shapes = {output.name: tuple(output.shape) for output in model.outputs}
                if len(shapes) != 9:
                    raise ValueError(f"SCRFD HEF harus punya 9 output, ditemukan {list(shapes)}")
                configured = model.configure()
            except Exception:
                device.release()
                raise
            self._vdevice, self._model = device, model
            self._configured, self._shapes = configured, shapes

    def close(self) -> None:
        with self._lock:
            configured, device = self._configured, self._vdevice
            self._configured = self._model = self._vdevice = None
            self._shapes = {}
            try:
                if configured is not None:
                    configured.shutdown()
            finally:
                if device is not None:
                    device.release()

    def detect(self, image: np.ndarray) -> list[list[float]]:
        with self._lock:
            if not self.ready:
                raise RuntimeError("Microservice Face Only nonaktif")
            rgb = preprocess(image)
            bindings = self._configured.create_bindings(output_buffers={
                name: np.empty(shape, np.float32) for name, shape in self._shapes.items()
            })
            bindings.input().set_buffer(rgb)
            complete = Event()
            errors = []

            def finished(info):
                if getattr(info, "exception", None) is not None:
                    errors.append(info.exception)
                complete.set()

            self._configured.wait_for_async_ready(timeout_ms=5000, frames_count=1)
            self._configured.run_async([bindings], finished)
            if not complete.wait(5):
                raise TimeoutError("Inferensi Hailo Face Only melewati 5 detik")
            if errors:
                raise RuntimeError(f"Inferensi Hailo gagal: {errors[0]}")
            outputs = {name: bindings.output(name).get_buffer() for name in self._shapes}
            height, width = image.shape[:2]
            scale = min(1.0, 640.0 / max(width, height))
            return decode(outputs, max(1, round(width * scale)), max(1, round(height * scale)))
