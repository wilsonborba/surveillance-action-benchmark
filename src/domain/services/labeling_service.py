from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd

from src.core.logs import get_logger
from src.core.settings import load_settings
from src.core.utils.fs import slugify
from src.core.utils.video import draw_hud, image_to_bgr, open_capture, resize_for_display
from src.dal.local.labels import load_labels, resolve_labels_path, save_labels
from src.dal.local.predictions import load_predictions


LOGGER = get_logger(__name__)


@dataclass
class LabelingArtifacts:
    labels_path: Path
    rows_saved: int


class LabelingService:
    def __init__(self) -> None:
        self.settings = load_settings()

    def review(
        self,
        source: Path,
        task_type: str,
        labels_path: Optional[Path] = None,
        predictions_path: Optional[Path] = None,
    ) -> LabelingArtifacts:
        resolved_labels = resolve_labels_path(labels_path, self.settings.labels_dir, source, task_type)
        labels = load_labels(resolved_labels)
        predictions = load_predictions(predictions_path) if predictions_path is not None and predictions_path.exists() else pd.DataFrame()
        source_id = slugify(source.stem)
        marked_start = None
        current_label = task_type
        review_status = "reviewed"

        def append_segment(frame_start: int, frame_end: int, label: str) -> None:
            nonlocal labels
            row = {
                "source_id": source_id,
                "source_path": str(source),
                "task_type": task_type,
                "split": "eval",
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "label": label,
                "subject_id": "",
                "review_status": review_status,
                "reviewer": "manual-cli",
                "notes": "",
            }
            labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)

        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            frame = image_to_bgr(source)
            frames = [(0, frame)]
            fps = 1.0
        else:
            cap = open_capture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open source: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            def _iter_frames():
                frame_index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield frame_index, frame
                    frame_index += 1
                cap.release()
            frames = _iter_frames()

        paused = False
        last_frame = None
        last_frame_index = 0
        for frame_index, frame in frames:
            last_frame = frame
            last_frame_index = frame_index
            rendered = frame.copy()
            if not predictions.empty:
                scoped = predictions[predictions["frame_index"] == frame_index]
                if not scoped.empty:
                    score = float(scoped["positive_score"].max())
                    label_guess = scoped.sort_values("positive_score", ascending=False).iloc[0]["label"]
                    cv2.putText(rendered, f"prediction: {label_guess} {score:.2f}", (18, rendered.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            start_text = "none" if marked_start is None else str(marked_start)
            rendered = draw_hud(
                rendered,
                [
                    f"labeling task: {task_type}",
                    f"frame: {frame_index}",
                    f"segment start: {start_text}",
                    f"current label: {current_label}",
                    "keys: [ mark start | ] save segment | 1 positive | 0 normal | u uncertain | q quit",
                ],
            )
            display = resize_for_display(rendered, self.settings.default_preview_width, self.settings.default_preview_height)
            cv2.imshow("Label Review", display)
            key = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("["):
                marked_start = frame_index
            elif key == ord("]") and marked_start is not None:
                append_segment(marked_start, frame_index, current_label)
                save_labels(resolved_labels, labels)
                marked_start = None
            elif key == ord("1"):
                current_label = task_type
            elif key == ord("0"):
                current_label = "normal"
            elif key == ord("u"):
                current_label = "uncertain"
        cv2.destroyAllWindows()
        if marked_start is not None and last_frame is not None:
            append_segment(marked_start, last_frame_index, current_label)
        save_labels(resolved_labels, labels)
        return LabelingArtifacts(labels_path=resolved_labels, rows_saved=len(labels))
