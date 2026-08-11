from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "project.yaml"


class VisualizationSettings(BaseModel):
    font_scale: float = 0.7
    line_thickness: int = 2
    hud_alpha: float = 0.85
    positive_color: List[int] = Field(default_factory=lambda: [0, 0, 255])
    negative_color: List[int] = Field(default_factory=lambda: [80, 200, 120])
    neutral_color: List[int] = Field(default_factory=lambda: [255, 200, 0])
    box_color: List[int] = Field(default_factory=lambda: [64, 160, 255])


class LabelingSettings(BaseModel):
    positive_keys: Dict[str, str] = Field(default_factory=dict)
    negative_key: str = "0"
    uncertain_key: str = "u"


class ProjectSettings(BaseModel):
    project_name: str
    legacy_source_root: str
    default_pose_backend: str = "torchvision"
    default_sequence_window_size: int = 32
    default_sequence_stride: int = 8
    default_max_persons: int = 2
    default_positive_threshold: float = 0.55
    default_preview_width: int = 1280
    default_preview_height: int = 720
    legacy_assets: Dict[str, str] = Field(default_factory=dict)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    labeling: LabelingSettings = Field(default_factory=LabelingSettings)

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def legacy_root(self) -> Path:
        return Path(self.legacy_source_root)

    @property
    def inputs_dir(self) -> Path:
        return REPO_ROOT / "inputs"

    @property
    def input_videos_dir(self) -> Path:
        return self.inputs_dir / "videos"

    @property
    def input_images_dir(self) -> Path:
        return self.inputs_dir / "images"

    @property
    def artifacts_dir(self) -> Path:
        return REPO_ROOT / "artifacts"

    @property
    def keypoints_dir(self) -> Path:
        return self.artifacts_dir / "keypoints"

    @property
    def manifests_dir(self) -> Path:
        return self.artifacts_dir / "manifests"

    @property
    def labels_dir(self) -> Path:
        return self.artifacts_dir / "labels"

    @property
    def outputs_dir(self) -> Path:
        return REPO_ROOT / "outputs"

    @property
    def output_runs_dir(self) -> Path:
        return self.outputs_dir / "runs"

    @property
    def output_videos_dir(self) -> Path:
        return self.outputs_dir / "videos"

    @property
    def output_reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    def resolve_legacy_asset(self, key: str) -> Path:
        relative = self.legacy_assets[key]
        return self.legacy_root / relative


@lru_cache(maxsize=1)
def load_settings() -> ProjectSettings:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    return ProjectSettings(**raw)
