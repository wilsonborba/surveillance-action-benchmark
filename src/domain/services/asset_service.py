from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.core.settings import load_settings


class AssetInspector:
    def __init__(self) -> None:
        self.settings = load_settings()

    def inspect(self) -> Dict[str, dict]:
        report = {}
        for key in self.settings.legacy_assets:
            path = self.settings.resolve_legacy_asset(key)
            report[key] = {
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
            }
        return report
