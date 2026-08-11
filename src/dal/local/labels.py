from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.utils.fs import ensure_dir, slugify


LABEL_COLUMNS = [
    "source_id",
    "source_path",
    "task_type",
    "split",
    "frame_start",
    "frame_end",
    "label",
    "subject_id",
    "review_status",
    "reviewer",
    "notes",
]


def empty_labels_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_labels_frame()
    frame = pd.read_csv(path)
    for column in LABEL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[LABEL_COLUMNS]


def save_labels(path: Path, frame: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    prepared = frame.copy()
    for column in LABEL_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = ""
    prepared[LABEL_COLUMNS].to_csv(path, index=False)
    return path


def default_label_path(labels_dir: Path, source_path: Path, task_type: str) -> Path:
    return labels_dir / f"{slugify(source_path.stem)}_{slugify(task_type)}_labels.csv"


def resolve_labels_path(
    explicit_path: Optional[Path], labels_dir: Path, source_path: Path, task_type: str
) -> Path:
    if explicit_path is not None:
        return explicit_path
    return default_label_path(labels_dir, source_path, task_type)
