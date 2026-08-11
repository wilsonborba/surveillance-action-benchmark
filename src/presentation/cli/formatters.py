from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from rich.console import Console
from rich.table import Table


console = Console()


def print_assets_table(report: Dict[str, dict]) -> None:
    table = Table(title="Legacy Asset Inspection")
    table.add_column("Key")
    table.add_column("Exists")
    table.add_column("Size MB")
    table.add_column("Path")
    for key, payload in report.items():
        table.add_row(key, "yes" if payload["exists"] else "no", str(payload["size_mb"] or "-"), payload["path"])
    console.print(table)


def print_metrics_table(frame: pd.DataFrame, title: str = "Metrics") -> None:
    table = Table(title=title)
    for column in frame.columns:
        table.add_column(column)
    for _, row in frame.iterrows():
        table.add_row(*[str(row[column]) for column in frame.columns])
    console.print(table)


def print_json_summary(path: Path) -> None:
    payload = json.loads(path.read_text())
    table = Table(title=f"Summary: {path.name}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    console.print(table)


def print_paths(title: str, paths: Iterable[Path]) -> None:
    table = Table(title=title)
    table.add_column("Path")
    for path in paths:
        table.add_row(str(path))
    console.print(table)
