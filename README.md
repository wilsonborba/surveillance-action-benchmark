# Surveillance Action Benchmark

Benchmark workspace for comparing **legacy LSTM-based video event detection** against a new **pose + GNN pipeline** for surveillance scenarios.

## Scope

The repository focuses on two event families:

- `fall detection`
- `fight detection`

And two modeling approaches:

- `legacy / baseline`: existing LSTM-based approach and available trained assets
- `new / target`: pose-based graph model with higher-precision pose extraction and temporal reasoning

## Objectives

- Run and audit the currently available baseline assets.
- Build a comparable pose + GNN pipeline.
- Produce reproducible metrics for both approaches.
- Support both automated evaluation and optional human-reviewed labeling.
- Keep the implementation lightweight enough for an MVP handoff.

## Evaluation Modes

### 1. Automated benchmark mode

Use a fixed labeled evaluation set to compute:

- precision
- recall
- F1-score
- confusion matrix
- per-class support
- latency / throughput summaries

### 2. Human review mode

Use a lightweight review workflow to:

- create or correct ground truth on unlabeled videos
- validate disagreement cases
- inspect false positives and false negatives

## Current Findings

From the source material reviewed on August 11, 2026:

- the fighting baseline script points directly to existing weights
- the fighting baseline video path is missing in the source package
- the falling cascade script points to missing detector and video paths
- an existing fall detector weight exists, but the current cascade script does not load it directly

## Repository Layout

- `inputs/videos/` raw input videos
- `inputs/images/` raw still images for frame-level tests
- `artifacts/keypoints/` extracted pose sequences
- `artifacts/manifests/` train / validation / test manifests
- `artifacts/labels/` benchmark labels and review exports
- `outputs/runs/` training outputs and checkpoints
- `outputs/videos/` rendered inference videos
- `outputs/reports/` metrics, comparison tables, and summaries
- `src/` implementation scripts

## Branch Strategy

This repository follows a lightweight GitFlow-style branch model:

- `main`
- `development`
- `feature/main`
- `feature/*`
- `fix/*`
- `release/*`

Use Conventional Commits for tracked changes.

## Immediate Build Order

1. stabilize legacy fall and fight runners
2. locate or replace missing runtime assets
3. define benchmark dataset structure and labels
4. implement pose extraction with a higher-precision stack
5. implement a lightweight ST-GCN-style baseline
6. generate side-by-side metrics and artifacts

## Notes

Project-facing context stays in this repository.
Execution-heavy handoff notes for agents are kept locally in `docs/` and intentionally ignored by Git.
