from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.core.utils.fs import ensure_dir, write_json


def write_dataframe(path: Path, frame: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
    return path


def read_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_npz(path: Path, **arrays: Any) -> Path:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    return path


def read_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_report_json(path: Path, payload: dict) -> Path:
    write_json(path, payload)
    return path
