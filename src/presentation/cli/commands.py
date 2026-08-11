from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console

from src.core.logs import configure_logging, get_logger
from src.core.settings import load_settings
from src.core.utils.fs import discover_sources, ensure_dir, slugify, timestamp_slug
from src.dal.local.labels import resolve_labels_path
from src.dal.local.predictions import save_predictions
from src.domain.services.asset_service import AssetInspector
from src.domain.services.dataset_service import DatasetService
from src.domain.services.inference_service import SequenceInferenceService
from src.domain.services.labeling_service import LabelingService
from src.domain.services.legacy_service import LegacyRunners
from src.domain.services.metrics_service import MetricsService
from src.domain.services.training_service import TrainingService
from src.presentation.cli.formatters import print_assets_table, print_json_summary, print_metrics_table, print_paths


console = Console()
LOGGER = get_logger(__name__)
settings = load_settings()
app = typer.Typer(help="Surveillance Action Benchmark CLI")
assets_app = typer.Typer(help="Inspect legacy assets")
dataset_app = typer.Typer(help="Prepare benchmark datasets")
train_app = typer.Typer(help="Train LSTM or ST-GCN models")
run_app = typer.Typer(help="Run inference for legacy or sequence models")
metrics_app = typer.Typer(help="Inspect metrics and reports")
label_app = typer.Typer(help="Manual review and labeling")
benchmark_app = typer.Typer(help="Run multiple models and compare outputs")

app.add_typer(assets_app, name="assets")
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")
app.add_typer(run_app, name="run")
app.add_typer(metrics_app, name="metrics")
app.add_typer(label_app, name="label")
app.add_typer(benchmark_app, name="benchmark")


def _prepare_predictions_frame(
    frame: pd.DataFrame,
    run_id: str,
    source: Path,
    task_type: str,
    model_name: str,
    architecture: str,
    fps: float,
) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["run_id"] = run_id
    prepared["source_id"] = slugify(source.stem)
    prepared["source_path"] = str(source)
    prepared["task_type"] = task_type
    prepared["model_name"] = model_name
    prepared["architecture"] = architecture
    if "frame_index" not in prepared.columns:
        prepared["frame_index"] = 0
    prepared["timestamp_sec"] = prepared["frame_index"].map(lambda value: value / max(fps, 1.0))
    preferred_order = [
        "run_id",
        "source_id",
        "source_path",
        "task_type",
        "model_name",
        "architecture",
        "frame_index",
        "timestamp_sec",
    ]
    remaining = [column for column in prepared.columns if column not in preferred_order]
    return prepared[preferred_order + remaining]


def _evaluate_run(
    run_dir: Path,
    predictions_frame: pd.DataFrame,
    task_type: str,
    source: Path,
    frame_count: int,
    labels_path: Optional[Path],
) -> None:
    metrics_service = MetricsService()
    resolved_labels = None
    if labels_path is not None:
        resolved_labels = labels_path
    else:
        candidate = resolve_labels_path(None, settings.labels_dir, source, task_type)
        if candidate.exists():
            resolved_labels = candidate
    metrics_service.evaluate_predictions(
        predictions=predictions_frame,
        labels_path=resolved_labels,
        output_dir=run_dir,
        source_id=slugify(source.stem),
        task_type=task_type,
        frame_count=frame_count,
    )


def _run_preview_flags(headless: bool, preview: bool) -> bool:
    return False if headless else preview


def _save_video_flag(headless: bool, save_video: bool) -> bool:
    return save_video or headless


@assets_app.command("inspect")
def inspect_assets() -> None:
    """Check the currently referenced legacy assets."""
    configure_logging()
    report = AssetInspector().inspect()
    print_assets_table(report)


@dataset_app.command("prepare")
def prepare_dataset(
    labels_path: Path = typer.Option(..., exists=True, readable=True),
    task_type: str = typer.Option(..., help="fall or fight"),
    source_root: Path = typer.Option(settings.input_videos_dir, exists=True),
    window_size: int = typer.Option(settings.default_sequence_window_size),
    stride: int = typer.Option(settings.default_sequence_stride),
    val_ratio: float = typer.Option(0.2),
    test_ratio: float = typer.Option(0.1),
    backend: str = typer.Option(settings.default_pose_backend),
    force_pose: bool = typer.Option(False),
) -> None:
    """Extract pose windows and build a trainable dataset."""
    configure_logging()
    artifacts = DatasetService().prepare_dataset(
        labels_path=labels_path,
        task_type=task_type,
        source_root=source_root,
        window_size=window_size,
        stride=stride,
        backend=backend,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        force_pose=force_pose,
    )
    print_paths("Dataset Artifacts", [artifacts.dataset_path, artifacts.metadata_path, *artifacts.keypoint_cache_paths])


@train_app.command("model")
def train_model(
    architecture: str = typer.Option(..., help="lstm or stgcn"),
    task_type: str = typer.Option(..., help="fall or fight"),
    dataset_path: Path = typer.Option(..., exists=True, readable=True),
    metadata_path: Path = typer.Option(..., exists=True, readable=True),
    epochs: int = typer.Option(8),
    batch_size: int = typer.Option(8),
    learning_rate: float = typer.Option(1e-3),
) -> None:
    """Train a binary sequence model for a task."""
    configure_logging()
    artifacts = TrainingService().train(
        architecture=architecture,
        task_type=task_type,
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    print_paths("Training Artifacts", [artifacts.run_dir, artifacts.weights_path, artifacts.history_path, artifacts.summary_path])


@run_app.command("legacy-fight")
def run_legacy_fight(
    source: Path = typer.Option(..., exists=True),
    preview: bool = typer.Option(True),
    headless: bool = typer.Option(False),
    save_video: bool = typer.Option(False),
    labels_path: Optional[Path] = typer.Option(None),
) -> None:
    """Run the available pretrained legacy fight model."""
    configure_logging()
    preview_flag = _run_preview_flags(headless, preview)
    save_video_flag = _save_video_flag(headless, save_video)
    run_dir = settings.output_runs_dir / "inference" / "fight" / "legacy-fight-pretrained" / timestamp_slug()
    ensure_dir(run_dir)
    predictions, video_path, frame_count, fps = LegacyRunners().run_fight_pretrained(source, preview_flag, save_video_flag, run_dir)
    prepared = _prepare_predictions_frame(predictions, run_dir.name, source, "fight", "legacy-fight-pretrained", "legacy-fight-pretrained", fps)
    predictions_path = save_predictions(run_dir / "predictions.csv", prepared)
    _evaluate_run(run_dir, prepared, "fight", source, frame_count, labels_path)
    print_paths("Run Artifacts", [run_dir, predictions_path] + ([video_path] if video_path else []))


@run_app.command("legacy-fall")
def run_legacy_fall(
    source: Path = typer.Option(..., exists=True),
    preview: bool = typer.Option(True),
    headless: bool = typer.Option(False),
    save_video: bool = typer.Option(False),
    labels_path: Optional[Path] = typer.Option(None),
) -> None:
    """Run the available legacy fall detector weights."""
    configure_logging()
    preview_flag = _run_preview_flags(headless, preview)
    save_video_flag = _save_video_flag(headless, save_video)
    run_dir = settings.output_runs_dir / "inference" / "fall" / "legacy-fall-detector" / timestamp_slug()
    ensure_dir(run_dir)
    predictions, video_path, frame_count, fps = LegacyRunners().run_fall_detector(source, preview_flag, save_video_flag, run_dir)
    prepared = _prepare_predictions_frame(predictions, run_dir.name, source, "fall", "legacy-fall-detector", "legacy-fall-detector", fps)
    predictions_path = save_predictions(run_dir / "predictions.csv", prepared)
    _evaluate_run(run_dir, prepared, "fall", source, frame_count, labels_path)
    print_paths("Run Artifacts", [run_dir, predictions_path] + ([video_path] if video_path else []))


@run_app.command("sequence")
def run_sequence(
    task_type: str = typer.Option(..., help="fall or fight"),
    architecture: str = typer.Option(..., help="lstm or stgcn"),
    source: Path = typer.Option(..., exists=True),
    weights_path: Optional[Path] = typer.Option(None),
    preview: bool = typer.Option(True),
    headless: bool = typer.Option(False),
    save_video: bool = typer.Option(False),
    labels_path: Optional[Path] = typer.Option(None),
    threshold: float = typer.Option(settings.default_positive_threshold),
) -> None:
    """Run a trained sequence model."""
    configure_logging()
    preview_flag = _run_preview_flags(headless, preview)
    save_video_flag = _save_video_flag(headless, save_video)
    artifacts = SequenceInferenceService().run(
        task_type=task_type,
        architecture=architecture,
        source=source,
        weights_path=weights_path,
        preview=preview_flag,
        save_video=save_video_flag,
        threshold=threshold,
    )
    predictions = pd.read_csv(artifacts.predictions_path)
    _evaluate_run(artifacts.run_dir, predictions, task_type, source, artifacts.frame_count, labels_path)
    print_paths("Run Artifacts", [artifacts.run_dir, artifacts.predictions_path] + ([artifacts.video_path] if artifacts.video_path else []))


@metrics_app.command("show")
def show_metrics(path: Path = typer.Option(..., exists=True)) -> None:
    """Show metrics from a JSON or CSV report."""
    configure_logging()
    if path.suffix.lower() == ".json":
        print_json_summary(path)
    elif path.suffix.lower() == ".csv":
        print_metrics_table(pd.read_csv(path), title=path.name)
    else:
        raise typer.BadParameter("Unsupported report format. Use JSON or CSV.")


@label_app.command("review")
def review_labels(
    source: Path = typer.Option(..., exists=True),
    task_type: str = typer.Option(..., help="fall or fight"),
    labels_path: Optional[Path] = typer.Option(None),
    predictions_path: Optional[Path] = typer.Option(None),
) -> None:
    """Open a lightweight OpenCV-based manual labeling session."""
    configure_logging()
    artifacts = LabelingService().review(source, task_type, labels_path=labels_path, predictions_path=predictions_path)
    print_paths("Labeling Artifacts", [artifacts.labels_path])


@benchmark_app.command("compare")
def compare_models(
    task_type: str = typer.Option(..., help="fall or fight"),
    source: Path = typer.Option(..., exists=True),
    labels_path: Optional[Path] = typer.Option(None),
    models: List[str] = typer.Option(..., help="Comma-separated repeated option. Example: --models legacy-fall --models stgcn"),
    sequence_weights_lstm: Optional[Path] = typer.Option(None),
    sequence_weights_stgcn: Optional[Path] = typer.Option(None),
) -> None:
    """Run multiple models headless and aggregate comparison metrics."""
    configure_logging()
    metrics_service = MetricsService()
    comparison_rows = []
    created_reports = []
    for model_name in models:
        normalized = model_name.strip().lower()
        if normalized == "legacy-fight":
            run_dir = settings.output_runs_dir / "inference" / "fight" / "legacy-fight-pretrained" / timestamp_slug()
            ensure_dir(run_dir)
            predictions, _, frame_count, fps = LegacyRunners().run_fight_pretrained(source, False, True, run_dir)
            prepared = _prepare_predictions_frame(predictions, run_dir.name, source, "fight", "legacy-fight-pretrained", "legacy-fight-pretrained", fps)
            save_predictions(run_dir / "predictions.csv", prepared)
            _evaluate_run(run_dir, prepared, "fight", source, frame_count, labels_path)
            metrics_path = run_dir / "metrics_summary.json"
        elif normalized == "legacy-fall":
            run_dir = settings.output_runs_dir / "inference" / "fall" / "legacy-fall-detector" / timestamp_slug()
            ensure_dir(run_dir)
            predictions, _, frame_count, fps = LegacyRunners().run_fall_detector(source, False, True, run_dir)
            prepared = _prepare_predictions_frame(predictions, run_dir.name, source, "fall", "legacy-fall-detector", "legacy-fall-detector", fps)
            save_predictions(run_dir / "predictions.csv", prepared)
            _evaluate_run(run_dir, prepared, "fall", source, frame_count, labels_path)
            metrics_path = run_dir / "metrics_summary.json"
        elif normalized in {"lstm", "stgcn"}:
            artifacts = SequenceInferenceService().run(
                task_type=task_type,
                architecture=normalized,
                source=source,
                weights_path=sequence_weights_lstm if normalized == "lstm" else sequence_weights_stgcn,
                preview=False,
                save_video=True,
            )
            predictions = pd.read_csv(artifacts.predictions_path)
            _evaluate_run(artifacts.run_dir, predictions, task_type, source, artifacts.frame_count, labels_path)
            metrics_path = artifacts.run_dir / "metrics_summary.json"
            run_dir = artifacts.run_dir
        else:
            raise typer.BadParameter(f"Unsupported model option: {model_name}")
        payload = json.loads(metrics_path.read_text())
        payload["model_name"] = normalized
        payload["run_dir"] = str(run_dir)
        comparison_rows.append(payload)
        created_reports.append(metrics_path)
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_path = settings.output_reports_dir / f"comparison_{task_type}_{timestamp_slug()}.csv"
    comparison_frame.to_csv(comparison_path, index=False)
    print_metrics_table(comparison_frame, title=f"Comparison - {task_type}")
    print_paths("Comparison Artifacts", [comparison_path, *created_reports])
