from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logs import get_logger
from src.core.settings import load_settings
from src.core.utils.fs import discover_sources, ensure_dir, slugify, timestamp_slug
from src.dal.local.io import read_npz, write_dataframe, write_npz
from src.dal.local.labels import load_labels
from src.domain.services.metrics_service import POSITIVE_ALIASES
from src.domain.services.pose_service import PoseExtractor


LOGGER = get_logger(__name__)


@dataclass
class DatasetArtifacts:
    dataset_path: Path
    metadata_path: Path
    keypoint_cache_paths: List[Path]


class DatasetService:
    def __init__(self) -> None:
        self.settings = load_settings()

    def positive_flag(self, task_type: str, label: str) -> int:
        normalized = str(label).strip().lower()
        return 1 if normalized in POSITIVE_ALIASES.get(task_type, {task_type, "positive", "1", "true"}) else 0

    def extract_or_load_keypoints(
        self,
        source: Path,
        force: bool = False,
        backend: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, int, Path]:
        cache_path = self.settings.keypoints_dir / f"{slugify(source.stem)}_pose_slots.npz"
        if cache_path.exists() and not force:
            cached = read_npz(cache_path)
            return cached["slots"], float(cached["fps"][0]), int(cached["frame_count"][0]), cache_path
        extractor = PoseExtractor(backend=backend)
        slots, fps, frame_count, _ = extractor.extract_cached_slots(source)
        payload = {
            "slots": slots.astype(np.float32),
            "fps": np.array([fps], dtype=np.float32),
            "frame_count": np.array([frame_count], dtype=np.int32),
        }
        write_npz(cache_path, **payload)
        return slots, fps, frame_count, cache_path

    def build_feature_tensor(self, slots: np.ndarray) -> np.ndarray:
        deltas = np.zeros_like(slots[..., :2])
        deltas[1:] = slots[1:, :, :, :2] - slots[:-1, :, :, :2]
        features = np.concatenate([slots, deltas], axis=-1)
        return features.astype(np.float32)

    def assign_splits(self, source_ids: List[str], val_ratio: float, test_ratio: float) -> Dict[str, str]:
        ordered = sorted(set(source_ids))
        total = len(ordered)
        test_count = int(round(total * test_ratio))
        val_count = int(round(total * val_ratio))
        split_map = {}
        for index, source_id in enumerate(ordered):
            if index < test_count:
                split_map[source_id] = "test"
            elif index < test_count + val_count:
                split_map[source_id] = "val"
            else:
                split_map[source_id] = "train"
        return split_map

    def build_frame_truth(self, labels: pd.DataFrame, source_id: str, task_type: str, frame_count: int) -> np.ndarray:
        truth = np.zeros(frame_count, dtype=np.int64)
        scoped = labels[(labels["source_id"] == source_id) & (labels["task_type"] == task_type)]
        for _, row in scoped.iterrows():
            if self.positive_flag(task_type, row["label"]) != 1:
                continue
            start = max(0, int(row["frame_start"]))
            end = min(frame_count - 1, int(row["frame_end"]))
            truth[start : end + 1] = 1
        return truth

    def prepare_dataset(
        self,
        labels_path: Path,
        task_type: str,
        source_root: Path,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        backend: Optional[str] = None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        force_pose: bool = False,
    ) -> DatasetArtifacts:
        labels = load_labels(labels_path)
        sources = discover_sources(source_root)
        if not sources:
            raise RuntimeError(f"No sources found at {source_root}")
        window_size = window_size or self.settings.default_sequence_window_size
        stride = stride or self.settings.default_sequence_stride
        split_map = self.assign_splits([slugify(source.stem) for source in sources], val_ratio, test_ratio)

        samples = []
        metadata_rows = []
        cache_paths = []
        for source in sources:
            source_id = slugify(source.stem)
            slots, fps, frame_count, cache_path = self.extract_or_load_keypoints(source, force=force_pose, backend=backend)
            cache_paths.append(cache_path)
            features = self.build_feature_tensor(slots)
            truth = self.build_frame_truth(labels, source_id, task_type, frame_count)
            for start in range(0, max(frame_count - window_size + 1, 1), stride):
                end = min(start + window_size, frame_count)
                window = features[start:end]
                if window.shape[0] < window_size:
                    pad = np.zeros((window_size - window.shape[0],) + window.shape[1:], dtype=np.float32)
                    window = np.concatenate([window, pad], axis=0)
                positive_ratio = float(truth[start:end].mean()) if end > start else 0.0
                label = 1 if positive_ratio >= 0.2 else 0
                samples.append(window.transpose(3, 0, 2, 1))
                metadata_rows.append(
                    {
                        "source_id": source_id,
                        "source_path": str(source),
                        "task_type": task_type,
                        "split": split_map[source_id],
                        "frame_start": start,
                        "frame_end": end - 1,
                        "label": label,
                        "fps": fps,
                        "positive_ratio": positive_ratio,
                    }
                )
        x = np.stack(samples, axis=0).astype(np.float32)
        y = np.array([row["label"] for row in metadata_rows], dtype=np.float32)
        metadata = pd.DataFrame(metadata_rows)
        stamp = timestamp_slug()
        dataset_path = self.settings.manifests_dir / f"{slugify(task_type)}_{stamp}_dataset.npz"
        metadata_path = self.settings.manifests_dir / f"{slugify(task_type)}_{stamp}_metadata.csv"
        write_npz(dataset_path, x=x, y=y)
        write_dataframe(metadata_path, metadata)
        return DatasetArtifacts(dataset_path=dataset_path, metadata_path=metadata_path, keypoint_cache_paths=cache_paths)
