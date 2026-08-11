from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.utils.fs import ensure_dir


PREDICTION_COLUMNS = [
    "run_id",
    "source_id",
    "source_path",
    "task_type",
    "model_name",
    "architecture",
    "frame_index",
    "timestamp_sec",
    "track_id",
    "label",
    "positive_score",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "metadata_json",
]


def empty_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def save_predictions(path: Path, frame: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    prepared = frame.copy()
    for column in PREDICTION_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = ""
    prepared[PREDICTION_COLUMNS].to_csv(path, index=False)
    return path


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in PREDICTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[PREDICTION_COLUMNS]
