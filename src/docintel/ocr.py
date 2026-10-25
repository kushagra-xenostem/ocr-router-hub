"""OCR with OnnxTR (docTR models on ONNX Runtime): word-level text, confidence and geometry on CPU."""
import io
import hashlib
import threading
from PIL import Image
from typing import Protocol


class OcrEngine(Protocol):
    def recognize(self, image: Image.Image) -> dict: ...


def image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


class OnnxTrEngine:
    """Returns {"text", "conf", "lines": [{"text", "conf", "bbox": [x0, y0, x1, y1] relative 0-1}]}."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from onnxtr.models import ocr_predictor

            # straight-page mode handles typical scan skew (a few degrees); orientation
            # detection misread slightly rotated scans as upside down in testing
            self._model = ocr_predictor(assume_straight_pages=True)
        return self._model

    def recognize(self, image: Image.Image) -> dict:
        import numpy as np

        with self._lock:  # ONNX sessions are shared; serialising keeps memory predictable
            result = self._load()([np.asarray(image.convert("RGB"))])
        lines = []
        for block in result.pages[0].blocks:
            for line in block.lines:
                words = sorted((w for w in line.words if w.value.strip()), key=lambda w: _points(w.geometry)[0][0])
                if not words:
                    continue
                xs = [p[0] for w in words for p in _points(w.geometry)]
                ys = [p[1] for w in words for p in _points(w.geometry)]
                lines.append({
                    "text": " ".join(w.value for w in words),
                    "conf": round(sum(w.confidence for w in words) / len(words), 4),
                    "bbox": [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)],
                })
        lines.sort(key=lambda ln: (round(ln["bbox"][1], 2), ln["bbox"][0]))
        conf = round(sum(ln["conf"] for ln in lines) / len(lines), 4) if lines else 0.0
        return {"text": "\n".join(ln["text"] for ln in lines), "conf": conf, "lines": lines}


def _points(geometry) -> list:
    g = list(geometry)
    if len(g) == 2 and len(g[0]) == 2 and not hasattr(g[0][0], "__len__"):
        (x0, y0), (x1, y1) = g
        return [(x0, y0), (x1, y1)]
    return [tuple(p) for p in g]
