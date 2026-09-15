"""OpenCV YuNet detection and SFace embeddings (CPU, BGR input)."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .storage import validate_embedding

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def verified_models(directory=MODEL_DIR):
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    paths = []
    for entry in manifest["models"]:
        path = Path(directory) / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise RuntimeError("Recognition models are missing. Run: python scripts/download_models.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("A recognition model failed verification. Run: python scripts/download_models.py")
        paths.append(path)
    return paths


def best_match(feature, enrolled, threshold=0.363):
    """Return a candidate ID and cosine similarity; scores are not probabilities."""
    if not -1 <= threshold <= 1:
        raise ValueError("Similarity threshold must be between -1 and 1.")
    feature = validate_embedding(feature)
    if not enrolled:
        return None, None
    ids, vectors = zip(*enrolled)
    matrix = np.stack([validate_embedding(vector) for vector in vectors])
    scores = matrix @ feature / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(feature))
    index = int(np.argmax(scores))
    score = float(np.clip(scores[index], -1, 1))
    return (ids[index] if score >= threshold else None), score


class FaceEngine:
    def __init__(self, model_dir=MODEL_DIR):
        detection, recognition = verified_models(model_dir)
        self.detector = cv2.FaceDetectorYN.create(str(detection), "", (320, 320), 0.9)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognition), "")

    def detect(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("No camera image is available.")
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else list(faces)

    def encode(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        return validate_embedding(self.recognizer.feature(aligned).copy())

    def single_face(self, frame):
        faces = self.detect(frame)
        if len(faces) != 1:
            raise ValueError(f"Found {len(faces)} faces. Please use a clear image of exactly one person.")
        return self.encode(frame, faces[0])


class Camera:
    def __init__(self, index=0):
        self.index = index
        self.capture = None

    def read(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"Camera {self.index} could not open. Check camera permissions or restart with --camera 1. ID lookup and photo import are still available.")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.close()
            raise RuntimeError("The camera stopped returning frames. Reconnect it and try again.")
        return frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
