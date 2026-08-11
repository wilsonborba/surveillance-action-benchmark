# Surveillance Action Benchmark

Benchmark workspace for comparing **legacy surveillance event detectors** against **trainable pose-sequence models** for:

- `fall detection`
- `fight detection`

The repository provides one CLI-driven workflow for:

- running legacy models
- extracting pose sequences
- preparing trainable datasets
- training `LSTM` and `ST-GCN` models
- running preview or headless inference
- generating metrics automatically
- comparing model families side by side
- performing optional human review / manual labeling

## What is implemented

### Legacy model family

- `legacy-fall-detector`
  - wraps the available `falling_detector.pt`
  - runs directly on video or image inputs
  - overlays detections and posture labels

- `legacy-fight-pretrained`
  - wraps the available `yolo11n-pose.pt + fight_classifier_lstm_v2.pt`
  - runs tracking + temporal LSTM classification
  - overlays per-track fight predictions

### Trainable sequence family

- `lstm`
  - binary sequence classifier over extracted pose windows
  - supports both `fall` and `fight`

- `stgcn`
  - graph-based skeleton classifier using a lightweight ST-GCN-style model
  - supports both `fall` and `fight`

### Evaluation and review

- automated metrics generation on headless runs
- metrics saved per run in machine-readable files
- comparison command for multiple models on the same source
- OpenCV-based manual label review workflow

## Repository layout

- `config/project.yaml` repository configuration and external asset references
- `inputs/videos/` raw videos dropped by the user
- `inputs/images/` raw still images
- `artifacts/keypoints/` cached pose extraction artifacts
- `artifacts/manifests/` dataset manifests and trainable windows
- `artifacts/labels/` evaluation labels and review outputs
- `outputs/runs/` training and inference run artifacts
- `outputs/videos/` optional rendered outputs
- `outputs/reports/` comparison reports
- `src/` implementation

## Setup

### 1. Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### 2. Check legacy assets

```bash
python -m src.main assets inspect
```

This verifies whether the repository can see the external legacy weight files configured in `config/project.yaml`.

## Input expectations

### Video and image inputs

Drop your media into:

- `inputs/videos/`
- `inputs/images/`

Or pass a direct path with `--source`.

### Label format

Ground truth labels are simple CSV files with these columns:

- `source_id`
- `source_path`
- `task_type`
- `split`
- `frame_start`
- `frame_end`
- `label`
- `subject_id`
- `review_status`
- `reviewer`
- `notes`

For binary evaluation, any positive segment is marked with one of the positive task labels:

- `fall` or `fallen` for fall detection
- `fight` or `fighting` for fight detection

Frames not covered by positive segments are treated as negative by default.

## CLI workflow

## 1. Presentation mode

Default behavior is preview-oriented: the command opens a window, overlays detections, and does not require saving outputs.

### Legacy fall preview

```bash
python -m src.main run legacy-fall --source inputs/videos/example.mp4
```

### Legacy fight preview

```bash
python -m src.main run legacy-fight --source inputs/videos/example.mp4
```

### Sequence model preview

```bash
python -m src.main run sequence --task-type fall --architecture stgcn --source inputs/videos/example.mp4
```

## 2. Headless mode

Headless mode saves predictions and metrics automatically.

### Legacy fall headless

```bash
python -m src.main run legacy-fall \
  --source inputs/videos/example.mp4 \
  --headless \
  --labels-path artifacts/labels/example_fall_labels.csv
```

### Legacy fight headless

```bash
python -m src.main run legacy-fight \
  --source inputs/videos/example.mp4 \
  --headless \
  --labels-path artifacts/labels/example_fight_labels.csv
```

### Sequence model headless

```bash
python -m src.main run sequence \
  --task-type fall \
  --architecture stgcn \
  --source inputs/videos/example.mp4 \
  --headless \
  --labels-path artifacts/labels/example_fall_labels.csv
```

## 3. Prepare a trainable dataset

```bash
python -m src.main dataset prepare \
  --labels-path artifacts/labels/example_fall_labels.csv \
  --task-type fall \
  --source-root inputs/videos \
  --window-size 32 \
  --stride 8
```

This will:

- extract pose slots
- cache keypoints
- build windowed datasets
- save dataset artifacts into `artifacts/manifests/`

## 4. Train a sequence model

### Train LSTM

```bash
python -m src.main train model \
  --architecture lstm \
  --task-type fall \
  --dataset-path artifacts/manifests/fall_YYYYMMDD_HHMMSS_dataset.npz \
  --metadata-path artifacts/manifests/fall_YYYYMMDD_HHMMSS_metadata.csv
```

### Train ST-GCN

```bash
python -m src.main train model \
  --architecture stgcn \
  --task-type fight \
  --dataset-path artifacts/manifests/fight_YYYYMMDD_HHMMSS_dataset.npz \
  --metadata-path artifacts/manifests/fight_YYYYMMDD_HHMMSS_metadata.csv
```

Training outputs are stored under `outputs/runs/training/`.

## 5. Compare multiple models

### Fall comparison

```bash
python -m src.main benchmark compare \
  --task-type fall \
  --source inputs/videos/example.mp4 \
  --labels-path artifacts/labels/example_fall_labels.csv \
  --models legacy-fall \
  --models lstm \
  --models stgcn
```

### Fight comparison

```bash
python -m src.main benchmark compare \
  --task-type fight \
  --source inputs/videos/example.mp4 \
  --labels-path artifacts/labels/example_fight_labels.csv \
  --models legacy-fight \
  --models lstm \
  --models stgcn
```

Comparison reports are saved into `outputs/reports/`.

## 6. Show metrics in the terminal

```bash
python -m src.main metrics show --path outputs/reports/comparison_fall_YYYYMMDD_HHMMSS.csv
```

Or show a per-run JSON summary:

```bash
python -m src.main metrics show --path outputs/runs/inference/fall/stgcn/RUN_ID/metrics_summary.json
```

## 7. Manual review / labeling

```bash
python -m src.main label review \
  --source inputs/videos/example.mp4 \
  --task-type fall \
  --predictions-path outputs/runs/inference/fall/stgcn/RUN_ID/predictions.csv
```

Keyboard workflow:

- `[` mark segment start
- `]` save segment ending at current frame
- `1` set current label to the positive class
- `0` set current label to `normal`
- `u` set current label to `uncertain`
- `space` pause / resume
- `q` quit and save

## Output behavior

### Per-run artifacts

Each inference run stores:

- `predictions.csv`
- `metrics_summary.json`
- `metrics.csv` when labels exist
- `confusion_matrix.csv` when labels exist
- rendered video when saved

### Training artifacts

Each training run stores:

- `best.pt`
- `history.json`
- `summary.json`

## Model and architecture notes

### Legacy fall

The available fall baseline uses the existing `falling_detector.pt` asset found in the legacy source package.

### Legacy fight

The available fight baseline uses the existing `yolo11n-pose.pt` and `fight_classifier_lstm_v2.pt` assets found in the legacy source package.

### Sequence family

The trainable `lstm` and `stgcn` models share one pose-window dataset format, which allows easier apples-to-apples comparison between architectures.

## Branch strategy

This repository follows a lightweight GitFlow-style approach:

- `main`
- `development`
- `feature/main`
- `feature/*`
- `fix/*`
- `release/*`

Use Conventional Commits for tracked changes.

## Notes

- Local agent memory and working notes stay in `docs/` and are intentionally not tracked.
- Legacy assets remain external and are referenced through `config/project.yaml`.
- On systems without ground-truth labels, runs still produce summaries; full benchmark metrics require labels.
