from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.logs import get_logger
from src.core.settings import load_settings
from src.core.utils.fs import ensure_dir, timestamp_slug
from src.dal.local.io import read_npz, write_report_json
from src.domain.models.sequence_lstm import SequenceLSTMClassifier
from src.domain.models.stgcn import STGCNBinaryClassifier


LOGGER = get_logger(__name__)


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class TrainingArtifacts:
    run_dir: Path
    weights_path: Path
    history_path: Path
    summary_path: Path


class TrainingService:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, architecture: str, x: np.ndarray) -> nn.Module:
        architecture = architecture.lower()
        if architecture == "lstm":
            _, c, t, v, m = x.shape
            return SequenceLSTMClassifier(input_dim=c * v * m)
        if architecture == "stgcn":
            return STGCNBinaryClassifier(in_channels=x.shape[1])
        raise ValueError(f"Unsupported architecture: {architecture}")

    def prepare_tensor(self, architecture: str, batch_x: torch.Tensor) -> torch.Tensor:
        if architecture == "lstm":
            batch_x = batch_x.permute(0, 2, 3, 4, 1).reshape(batch_x.shape[0], batch_x.shape[2], -1)
        return batch_x

    def train(
        self,
        architecture: str,
        task_type: str,
        dataset_path: Path,
        metadata_path: Path,
        epochs: int = 8,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
    ) -> TrainingArtifacts:
        loaded = read_npz(dataset_path)
        metadata = pd.read_csv(metadata_path)
        x = loaded["x"]
        y = loaded["y"]
        train_mask = metadata["split"] == "train"
        val_mask = metadata["split"].isin(["val", "test"])
        if train_mask.sum() == 0:
            train_mask = np.ones(len(metadata), dtype=bool)
        if val_mask.sum() == 0:
            val_mask = ~train_mask
        if val_mask.sum() == 0:
            val_mask = train_mask
        train_dataset = WindowDataset(x[train_mask.to_numpy()], y[train_mask.to_numpy()])
        val_dataset = WindowDataset(x[val_mask.to_numpy()], y[val_mask.to_numpy()])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = self.build_model(architecture, x).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        run_dir = self.settings.output_runs_dir / "training" / task_type / architecture / timestamp_slug()
        ensure_dir(run_dir)
        history = []
        best_val_loss = float("inf")
        best_weights = run_dir / "best.pt"
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = self.prepare_tensor(architecture, batch_x.to(self.device))
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = self.prepare_tensor(architecture, batch_x.to(self.device))
                    batch_y = batch_y.to(self.device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_losses.append(float(loss.item()))
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    correct += int((preds == batch_y).sum().item())
                    total += int(batch_y.numel())
            epoch_stats = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
                "val_accuracy": float(correct / max(total, 1)),
            }
            history.append(epoch_stats)
            if epoch_stats["val_loss"] <= best_val_loss:
                best_val_loss = epoch_stats["val_loss"]
                torch.save(model.state_dict(), best_weights)
        history_path = run_dir / "history.json"
        summary_path = run_dir / "summary.json"
        write_report_json(history_path, {"history": history})
        write_report_json(
            summary_path,
            {
                "architecture": architecture,
                "task_type": task_type,
                "dataset_path": str(dataset_path),
                "metadata_path": str(metadata_path),
                "device": str(self.device),
                "best_val_loss": best_val_loss,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
        )
        return TrainingArtifacts(run_dir=run_dir, weights_path=best_weights, history_path=history_path, summary_path=summary_path)
