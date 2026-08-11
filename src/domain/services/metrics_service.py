from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.core.settings import load_settings
from src.core.utils.fs import ensure_dir, timestamp_slug, write_json
from src.dal.local.labels import load_labels
from src.dal.local.reports import save_metrics_frame, save_metrics_json


POSITIVE_ALIASES = {
    "fall": {"fall", "fallen", "positive", "1", "true"},
    "fight": {"fight", "fighting", "positive", "1", "true"},
}


@dataclass
class MetricsArtifacts:
    metrics_json: Optional[Path]
    metrics_csv: Optional[Path]
    confusion_csv: Optional[Path]


class MetricsService:
    def normalize_binary_label(self, task_type: str, label: str) -> int:
        normalized = str(label).strip().lower()
        return 1 if normalized in POSITIVE_ALIASES.get(task_type, {task_type, "positive", "1", "true"}) else 0

    def frame_ground_truth(self, labels: pd.DataFrame, source_id: str, task_type: str, frame_count: int) -> np.ndarray:
        truth = np.zeros(frame_count, dtype=int)
        scoped = labels[(labels["source_id"] == source_id) & (labels["task_type"] == task_type)]
        for _, row in scoped.iterrows():
            if self.normalize_binary_label(task_type, row["label"]) != 1:
                continue
            start = max(0, int(row["frame_start"]))
            end = min(frame_count - 1, int(row["frame_end"]))
            truth[start : end + 1] = 1
        return truth

    def frame_predictions(self, predictions: pd.DataFrame, frame_count: int) -> np.ndarray:
        grouped = predictions.groupby("frame_index")["positive_score"].max()
        frame_scores = np.zeros(frame_count, dtype=float)
        for frame_index, score in grouped.items():
            if 0 <= int(frame_index) < frame_count:
                frame_scores[int(frame_index)] = float(score)
        return (frame_scores >= 0.5).astype(int)

    def generate_summary(self, predictions: pd.DataFrame) -> Dict:
        if predictions.empty:
            return {
                "frame_predictions": 0,
                "positive_frames": 0,
                "mean_score": 0.0,
                "max_score": 0.0,
            }
        grouped = predictions.groupby("frame_index")["positive_score"].max().fillna(0.0)
        return {
            "frame_predictions": int(grouped.shape[0]),
            "positive_frames": int((grouped >= 0.5).sum()),
            "mean_score": float(grouped.mean()),
            "max_score": float(grouped.max()),
        }

    def evaluate_predictions(
        self,
        predictions: pd.DataFrame,
        labels_path: Optional[Path],
        output_dir: Path,
        source_id: str,
        task_type: str,
        frame_count: int,
    ) -> MetricsArtifacts:
        ensure_dir(output_dir)
        summary = self.generate_summary(predictions)
        metrics_json_path = output_dir / "metrics_summary.json"
        save_metrics_json(metrics_json_path, summary)
        if labels_path is None or not labels_path.exists():
            return MetricsArtifacts(metrics_json=metrics_json_path, metrics_csv=None, confusion_csv=None)

        labels = load_labels(labels_path)
        truth = self.frame_ground_truth(labels, source_id, task_type, frame_count)
        pred = self.frame_predictions(predictions, frame_count)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth,
            pred,
            average="binary",
            zero_division=0,
        )
        accuracy = accuracy_score(truth, pred)
        cm = confusion_matrix(truth, pred, labels=[0, 1])
        metrics_frame = pd.DataFrame(
            [
                {
                    "task_type": task_type,
                    "source_id": source_id,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "accuracy": accuracy,
                    "support_positive": int(truth.sum()),
                    "support_total": int(frame_count),
                }
            ]
        )
        confusion_frame = pd.DataFrame(
            [
                {"truth": 0, "pred": 0, "count": int(cm[0, 0])},
                {"truth": 0, "pred": 1, "count": int(cm[0, 1])},
                {"truth": 1, "pred": 0, "count": int(cm[1, 0])},
                {"truth": 1, "pred": 1, "count": int(cm[1, 1])},
            ]
        )
        metrics_csv_path = save_metrics_frame(output_dir / "metrics.csv", metrics_frame)
        confusion_csv_path = save_metrics_frame(output_dir / "confusion_matrix.csv", confusion_frame)
        payload = summary | metrics_frame.iloc[0].to_dict() | {"labels_path": str(labels_path)}
        save_metrics_json(metrics_json_path, payload)
        return MetricsArtifacts(metrics_json=metrics_json_path, metrics_csv=metrics_csv_path, confusion_csv=confusion_csv_path)
