from __future__ import annotations

import torch
from torch import nn


def coco_adjacency() -> torch.Tensor:
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]
    adjacency = torch.eye(17)
    for src, dst in edges:
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
    adjacency = adjacency / degree
    return adjacency


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(9, 1),
            stride=(stride, 1),
            padding=(4, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x_graph = torch.einsum("nctv,vw->nctw", x, adjacency)
        out = self.temporal(x_graph)
        out = self.bn(out)
        out = out + self.residual(x)
        return self.relu(out)


class STGCNBinaryClassifier(nn.Module):
    def __init__(self, in_channels: int = 5) -> None:
        super().__init__()
        self.register_buffer("adjacency", coco_adjacency())
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        self.block1 = STGCNBlock(in_channels, 64)
        self.block2 = STGCNBlock(64, 128, stride=2)
        self.block3 = STGCNBlock(128, 256, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N, C, T, V, M
        x = x.mean(dim=-1)
        n, c, t, v = x.shape
        x = x.permute(0, 2, 1, 3).reshape(n, t, c * v).transpose(1, 2)
        x = self.data_bn(x)
        x = x.transpose(1, 2).reshape(n, t, c, v).permute(0, 2, 1, 3)
        x = self.block1(x, self.adjacency)
        x = self.block2(x, self.adjacency)
        x = self.block3(x, self.adjacency)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        return self.head(x).squeeze(1)
