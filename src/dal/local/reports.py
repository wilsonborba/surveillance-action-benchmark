from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.core.utils.fs import ensure_dir, write_json


def save_metrics_json(path: Path, metrics: Dict) -> Path:
    write_json(path, metrics)
    return path


def save_metrics_frame(path: Path, frame: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
    return path
