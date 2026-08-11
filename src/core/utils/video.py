from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def resize_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return resized


def create_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def draw_hud(frame: np.ndarray, lines: Iterable[str], alpha: float = 0.85) -> np.ndarray:
    overlay = frame.copy()
    height = 28 + 24 * len(list(lines))
    cv2.rectangle(overlay, (10, 10), (540, height), (20, 20, 20), -1)
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    y = 36
    for line in lines:
        cv2.putText(blended, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        y += 24
    return blended


def frame_to_seconds(frame_index: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return frame_index / fps


def open_capture(source: Path) -> cv2.VideoCapture:
    return cv2.VideoCapture(str(source))


def image_to_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not load image: {path}")
    return image
