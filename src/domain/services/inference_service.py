from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from src.core.logs import get_logger
from src.core.settings import load_settings
from src.core.utils.fs import ensure_dir, slugify, timestamp_slug, write_json
from src.core.utils.video import create_video_writer, draw_hud, frame_to_seconds, image_to_bgr, open_capture, resize_for_display
from src.dal.local.predictions import empty_predictions_frame, save_predictions
from src.domain.models.sequence_lstm import SequenceLSTMClassifier
from src.domain.models.stgcn import STGCNBinaryClassifier
from src.domain.services.pose_service import PoseExtractor


LOGGER = get_logger(__name__)


@dataclass
class InferenceArtifacts:
    run_dir: Path
    predictions_path: Path
    video_path: Optional[Path]
    frame_count: int
    fps: float
    source_id: str


class SequenceInferenceService:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, architecture: str, in_channels: int = 5) -> torch.nn.Module:
        if architecture == "lstm":
            return SequenceLSTMClassifier(input_dim=in_channels * 17 * self.settings.default_max_persons)
        if architecture == "stgcn":
            return STGCNBinaryClassifier(in_channels=in_channels)
        raise ValueError(f"Unsupported architecture: {architecture}")

    def prepare_input(self, architecture: str, window: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(window).float()
        if architecture == "lstm":
            tensor = tensor.reshape(1, tensor.shape[0], -1)
        else:
            tensor = tensor.permute(3, 0, 2, 1).unsqueeze(0)
        return tensor.to(self.device)

    def build_features(self, slots_window: np.ndarray) -> np.ndarray:
        deltas = np.zeros_like(slots_window[..., :2])
        deltas[1:] = slots_window[1:, :, :, :2] - slots_window[:-1, :, :, :2]
        return np.concatenate([slots_window, deltas], axis=-1).astype(np.float32)

    def resolve_weights(self, task_type: str, architecture: str, explicit: Optional[Path]) -> Path:
        if explicit is not None:
            return explicit
        base = self.settings.output_runs_dir / "training" / task_type / architecture
        candidates = sorted(base.glob("*/best.pt"))
        if not candidates:
            raise FileNotFoundError(f"No trained weights found for {task_type}/{architecture}")
        return candidates[-1]

    def run(
        self,
        task_type: str,
        architecture: str,
        source: Path,
        weights_path: Optional[Path],
        preview: bool,
        save_video: bool,
        threshold: Optional[float] = None,
    ) -> InferenceArtifacts:
        extractor = PoseExtractor()
        model = self.build_model(architecture).to(self.device)
        resolved_weights = self.resolve_weights(task_type, architecture, weights_path)
        model.load_state_dict(torch.load(resolved_weights, map_location=self.device))
        model.eval()

        threshold = threshold if threshold is not None else self.settings.default_positive_threshold
        run_dir = self.settings.output_runs_dir / "inference" / task_type / architecture / timestamp_slug()
        ensure_dir(run_dir)
        source_id = slugify(source.stem)
        predictions_rows = []

        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image = image_to_bgr(source)
            slots_window = np.repeat(extractor.detections_to_slots(extractor.extract_detections(image), image.shape)[None, ...], self.settings.default_sequence_window_size, axis=0)
            frames = [(0, image)]
            fps = 1.0
            writer = create_video_writer(run_dir / f"{source_id}_{architecture}.mp4", 1.0, image.shape[1], image.shape[0]) if save_video else None
        else:
            cap = open_capture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open source: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = None
            if save_video:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = create_video_writer(run_dir / f"{source_id}_{architecture}.mp4", fps, width, height)
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
            slots_window = None

        window_size = self.settings.default_sequence_window_size
        slot_buffer = deque(maxlen=window_size)
        frame_count = 0
        for frame_index, frame in frames:
            frame_count = frame_index + 1
            detections = extractor.extract_detections(frame)
            slots = extractor.detections_to_slots(detections, frame.shape)
            slot_buffer.append(slots)
            rendered = frame.copy()
            for detection in detections:
                x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
                track_id = detection.get("track_id", -1)
                cv2.rectangle(rendered, (x1, y1), (x2, y2), tuple(self.settings.visualization.box_color), 2)
                cv2.putText(rendered, f"ID {track_id} {detection['score']:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tuple(self.settings.visualization.box_color), 2, cv2.LINE_AA)
                for joint in detection["keypoints"]:
                    x, y, conf = joint
                    if conf < 0.3:
                        continue
                    cv2.circle(rendered, (int(x), int(y)), 3, (255, 255, 255), -1)
            positive_score = 0.0
            label = "warming-up"
            if len(slot_buffer) > 0 and source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                window_array = slots_window
            elif len(slot_buffer) == window_size:
                window_array = np.stack(slot_buffer, axis=0)
            else:
                window_array = None
            if window_array is not None:
                features = self.build_features(window_array)
                tensor = self.prepare_input(architecture, features)
                with torch.no_grad():
                    positive_score = float(torch.sigmoid(model(tensor)).item())
                label = task_type if positive_score >= threshold else "normal"
            rendered = draw_hud(
                rendered,
                [
                    f"model: {architecture}",
                    f"task: {task_type}",
                    f"source: {source.name}",
                    f"frame: {frame_index}",
                    f"score: {positive_score:.2f}",
                    f"label: {label}",
                ],
            )
            predictions_rows.append(
                {
                    "frame_index": frame_index,
                    "track_id": -1,
                    "label": label,
                    "positive_score": positive_score,
                    "score": positive_score,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "metadata_json": json.dumps({"weights_path": str(resolved_weights)}),
                }
            )
            if writer is not None:
                writer.write(rendered)
            if preview:
                display = resize_for_display(rendered, self.settings.default_preview_width, self.settings.default_preview_height)
                cv2.imshow(f"{architecture.upper()} Preview", display)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                    break
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
        frame = pd.DataFrame(predictions_rows)
        if frame.empty:
            frame = empty_predictions_frame()
        run_id = run_dir.name
        frame.insert(0, "run_id", run_id)
        frame.insert(1, "source_id", source_id)
        frame.insert(2, "source_path", str(source))
        frame.insert(3, "task_type", task_type)
        frame.insert(4, "model_name", architecture)
        frame.insert(5, "architecture", architecture)
        if "timestamp_sec" not in frame.columns:
            frame.insert(7, "timestamp_sec", frame["frame_index"].map(lambda value: value / max(fps, 1.0)))
        predictions_path = save_predictions(run_dir / "predictions.csv", frame)
        write_json(
            run_dir / "metadata.json",
            {
                "task_type": task_type,
                "architecture": architecture,
                "source": str(source),
                "weights_path": str(resolved_weights),
                "fps": fps,
                "frame_count": frame_count,
            },
        )
        return InferenceArtifacts(
            run_dir=run_dir,
            predictions_path=predictions_path,
            video_path=(run_dir / f"{source_id}_{architecture}.mp4" if save_video else None),
            frame_count=frame_count,
            fps=fps,
            source_id=source_id,
        )
