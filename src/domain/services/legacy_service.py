from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO

from src.core.logs import get_logger
from src.core.settings import load_settings
from src.core.utils.fs import ensure_dir, slugify, timestamp_slug
from src.core.utils.video import create_video_writer, draw_hud, frame_to_seconds, image_to_bgr, open_capture, resize_for_display
from src.dal.local.predictions import empty_predictions_frame, save_predictions


LOGGER = get_logger(__name__)


class FightClassifierLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)


class LegacyRunners:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_fight_pretrained(
        self,
        source: Path,
        preview: bool,
        save_video: bool,
        run_dir: Path,
    ) -> Tuple[pd.DataFrame, Optional[Path], int, float]:
        pose_weights = self.settings.resolve_legacy_asset("fight_pose_weights")
        fight_weights = self.settings.resolve_legacy_asset("fight_classifier_weights")
        if not pose_weights.exists() or not fight_weights.exists():
            raise FileNotFoundError("Missing fight legacy weights")

        pose_model = YOLO(str(pose_weights))
        classifier = FightClassifierLSTM(input_dim=17 * 7).to(self.device)
        classifier.load_state_dict(torch.load(fight_weights, map_location=self.device))
        classifier.eval()

        cap = open_capture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = None
        if save_video:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = create_video_writer(run_dir / f"{slugify(source.stem)}_legacy_fight.mp4", fps, width, height)

        window_size = self.settings.default_sequence_window_size
        pose_min_mean_conf = 0.40
        fight_threshold = 0.50
        smooth_frames = 2
        pose_buffers = defaultdict(lambda: deque(maxlen=window_size))
        prev_xy = {}
        dist_hist = defaultdict(lambda: deque(maxlen=2))
        votes = defaultdict(lambda: deque(maxlen=smooth_frames))
        frame_rows = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = pose_model.track(source=frame, persist=True, imgsz=640, conf=0.50, iou=0.70, verbose=False)
            rendered = frame.copy()
            scene_positive = 0.0
            if results and results[0].boxes is not None and results[0].keypoints is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                keypoints = results[0].keypoints
                centers = {}
                for index in range(len(boxes)):
                    track_id = int(boxes.id[index].item())
                    centers[track_id] = keypoints.xy[index].cpu().numpy().mean(axis=0)
                for index in range(len(boxes)):
                    track_id = int(boxes.id[index].item())
                    x1, y1, x2, y2 = boxes.xyxy[index].cpu().numpy().astype(int)
                    kp_xy = keypoints.xy[index].cpu().numpy().astype(np.float32)
                    kp_conf = keypoints.conf[index].cpu().numpy().astype(np.float32)
                    if float(kp_conf.mean()) < pose_min_mean_conf:
                        continue
                    if track_id in prev_xy:
                        dxy = kp_xy - prev_xy[track_id]
                    else:
                        dxy = np.zeros_like(kp_xy, dtype=np.float32)
                    prev_xy[track_id] = kp_xy.copy()
                    center_a = centers[track_id]
                    nearest_dist = 999.0
                    for other_id, center in centers.items():
                        if other_id == track_id:
                            continue
                        nearest_dist = min(nearest_dist, float(np.linalg.norm(center_a - center)))
                    dist_hist[track_id].append(nearest_dist)
                    dist_delta = float(dist_hist[track_id][0] - dist_hist[track_id][1]) if len(dist_hist[track_id]) == 2 else 0.0
                    interaction = np.repeat(np.array([[nearest_dist, dist_delta]], dtype=np.float32), 17, axis=0)
                    features = np.concatenate([kp_xy, dxy, kp_conf[:, None], interaction], axis=1).astype(np.float32)
                    pose_buffers[track_id].append(features)
                    probability = 0.0
                    label = "NORMAL"
                    color = tuple(self.settings.visualization.negative_color)
                    if len(pose_buffers[track_id]) == window_size:
                        window = np.stack(pose_buffers[track_id], axis=0)
                        tensor = torch.from_numpy(window.reshape(1, window_size, 17 * 7)).float().to(self.device)
                        with torch.no_grad():
                            probability = float(torch.sigmoid(classifier(tensor)).item())
                        votes[track_id].append(probability > fight_threshold)
                        scene_positive = max(scene_positive, probability)
                        if sum(votes[track_id]) >= smooth_frames:
                            label = "FIGHT"
                            color = tuple(self.settings.visualization.positive_color)
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(rendered, f"ID {track_id} | {label} {probability:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    frame_rows.append(
                        {
                            "frame_index": frame_index,
                            "track_id": track_id,
                            "label": "fight" if probability >= fight_threshold else "normal",
                            "positive_score": probability,
                            "score": probability,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "metadata_json": json.dumps({"detector": "legacy-fight-pretrained"}),
                        }
                    )
            rendered = draw_hud(
                rendered,
                [
                    "model: legacy-fight-pretrained",
                    f"source: {source.name}",
                    f"frame: {frame_index}",
                    f"scene positive: {scene_positive:.2f}",
                ],
            )
            if writer is not None:
                writer.write(rendered)
            if preview:
                display = resize_for_display(rendered, self.settings.default_preview_width, self.settings.default_preview_height)
                cv2.imshow("Legacy Fight Preview", display)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                    break
            frame_index += 1
        cap.release()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
        frame = pd.DataFrame(frame_rows)
        if frame.empty:
            frame = empty_predictions_frame()
        return frame, (run_dir / f"{slugify(source.stem)}_legacy_fight.mp4" if save_video else None), frame_index, fps

    def run_fall_detector(
        self,
        source: Path,
        preview: bool,
        save_video: bool,
        run_dir: Path,
    ) -> Tuple[pd.DataFrame, Optional[Path], int, float]:
        fall_weights = self.settings.resolve_legacy_asset("fall_detector_weights")
        if not fall_weights.exists():
            raise FileNotFoundError(f"Missing fall detector weights: {fall_weights}")
        model = YOLO(str(fall_weights))

        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            frame = image_to_bgr(source)
            frames = [(0, frame)]
            fps = 1.0
            writer = create_video_writer(run_dir / f"{slugify(source.stem)}_legacy_fall.mp4", 1.0, frame.shape[1], frame.shape[0]) if save_video else None
        else:
            cap = open_capture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open source: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = None
            if save_video:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = create_video_writer(run_dir / f"{slugify(source.stem)}_legacy_fall.mp4", fps, width, height)
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

        frame_rows = []
        last_index = 0
        for frame_index, frame in frames:
            last_index = frame_index + 1
            rendered = frame.copy()
            results = model.predict(source=frame, imgsz=640, conf=0.25, iou=0.70, verbose=False)[0]
            scene_positive = 0.0
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    label_name = results.names[class_id]
                    normalized_label = "fall" if label_name == "fallen" else "normal"
                    positive_score = score if normalized_label == "fall" else 1.0 - score
                    scene_positive = max(scene_positive, positive_score if normalized_label == "fall" else 0.0)
                    color = tuple(self.settings.visualization.positive_color if normalized_label == "fall" else self.settings.visualization.box_color)
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(rendered, f"{label_name.upper()} {score:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                    frame_rows.append(
                        {
                            "frame_index": frame_index,
                            "track_id": -1,
                            "label": normalized_label,
                            "positive_score": scene_positive if normalized_label == "fall" else 0.0,
                            "score": score,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "metadata_json": json.dumps({"raw_label": label_name}),
                        }
                    )
            rendered = draw_hud(
                rendered,
                [
                    "model: legacy-fall-detector",
                    f"source: {source.name}",
                    f"frame: {frame_index}",
                    f"scene positive: {scene_positive:.2f}",
                ],
            )
            if writer is not None:
                writer.write(rendered)
            if preview:
                display = resize_for_display(rendered, self.settings.default_preview_width, self.settings.default_preview_height)
                cv2.imshow("Legacy Fall Preview", display)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                    break
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
        frame = pd.DataFrame(frame_rows)
        if frame.empty:
            frame = empty_predictions_frame()
        return frame, (run_dir / f"{slugify(source.stem)}_legacy_fall.mp4" if save_video else None), last_index, fps
