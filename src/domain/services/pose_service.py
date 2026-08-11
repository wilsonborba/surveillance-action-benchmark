from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

from src.core.logs import get_logger
from src.core.settings import load_settings
from src.core.utils.video import image_to_bgr, open_capture
from src.domain.services.tracker_service import CentroidTracker


LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_torchvision_pose_model():
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


class PoseExtractor:
    def __init__(self, backend: str | None = None, max_persons: int | None = None, score_threshold: float = 0.7) -> None:
        settings = load_settings()
        self.backend = backend or settings.default_pose_backend
        self.max_persons = max_persons or settings.default_max_persons
        self.score_threshold = score_threshold
        self.tracker = CentroidTracker()
        if self.backend != "torchvision":
            raise ValueError(f"Unsupported pose backend for this MVP: {self.backend}")
        self.model, self.device = _load_torchvision_pose_model()

    def extract_detections(self, frame: np.ndarray) -> List[dict]:
        rgb = frame[:, :, ::-1].copy()
        tensor = to_tensor(rgb).to(self.device)
        with torch.no_grad():
            output = self.model([tensor])[0]
        detections: List[dict] = []
        for score, label, box, keypoints, keypoint_scores in zip(
            output["scores"].detach().cpu().numpy(),
            output["labels"].detach().cpu().numpy(),
            output["boxes"].detach().cpu().numpy(),
            output["keypoints"].detach().cpu().numpy(),
            output["keypoints_scores"].detach().cpu().numpy(),
        ):
            if int(label) != 1 or float(score) < self.score_threshold:
                continue
            keypoints_xy = keypoints[:, :2]
            kp = np.concatenate([keypoints_xy, keypoint_scores[:, None]], axis=1).astype(np.float32)
            detections.append(
                {
                    "score": float(score),
                    "bbox": box.astype(float).tolist(),
                    "keypoints": kp,
                }
            )
        detections = sorted(detections, key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0)
        return self.tracker.update(detections)

    def detections_to_slots(self, detections: List[dict], frame_shape: Tuple[int, int, int]) -> np.ndarray:
        height, width = frame_shape[:2]
        slots = np.zeros((self.max_persons, 17, 3), dtype=np.float32)
        detections = sorted(detections, key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0)[: self.max_persons]
        for index, detection in enumerate(detections):
            kp = detection["keypoints"].copy()
            kp[:, 0] = kp[:, 0] / max(width, 1)
            kp[:, 1] = kp[:, 1] / max(height, 1)
            kp[:, 2] = np.clip(kp[:, 2], 0.0, 1.0)
            slots[index] = kp
        return slots

    def extract_cached_slots(self, source: Path) -> Tuple[np.ndarray, float, int, List[np.ndarray]]:
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image = image_to_bgr(source)
            detections = self.extract_detections(image)
            slots = self.detections_to_slots(detections, image.shape)[None, ...]
            boxes = [np.array([det["bbox"] for det in detections[: self.max_persons]], dtype=np.float32)]
            return slots, 1.0, 1, boxes

        cap = open_capture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        fps = cap.get(5) or 30.0
        frames = []
        boxes_per_frame = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = self.extract_detections(frame)
            frames.append(self.detections_to_slots(detections, frame.shape))
            boxes_per_frame.append(np.array([det["bbox"] + [det.get("track_id", -1)] for det in detections[: self.max_persons]], dtype=np.float32))
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames extracted from: {source}")
        return np.stack(frames, axis=0), float(fps), len(frames), boxes_per_frame
