# Bottle Cap Anomaly Detection

Industrial bottle cap removal verification using **PatchCore** + fixed ROI.

**Scenario**: Fixed camera, fixed lighting, bottle moves along track. Detect whether bottle cap has been removed.

## Pipeline

```
Video → Fixed ROI → Yellow Filter → PatchCore → Anomaly Score → Alarm
```

| Stage | Description |
|-------|-------------|
| ROI Selection | Interactive mouse-drag on first frame |
| Frame Extraction | Yellow pixel filter keeps only frames with bottle present |
| Training | One-class PatchCore (normal samples only) |
| Inference | Stable-station trigger, latched single-detection per bottle |
| Alarm | Score > threshold with diagnostics panel |

## Quick Start

### 1. Select ROI

```bash
python roi_selector.py --video data/videos/normal_training.mp4
```

### 2. Extract Training Frames

```bash
python extract_roi_frames.py \
  --video data/videos/normal_training.mp4 \
  --output data/bottle_cap/train/good \
  --step 10
```

### 3. Train

```bash
python train_patchcore.py --data data/bottle_cap --output checkpoints
```

### 4. Debug Single Frame

```bash
python debug_infer_one_frame.py \
  --checkpoint checkpoints/Patchcore/bottle_cap/latest/weights/lightning/model.ckpt \
  --image path/to/roi_frame.jpg --no-crop
```

### 5. Video Inference

```bash
python infer_video.py \
  --video data/videos/abnormal_test.mp4 \
  --checkpoint checkpoints/Patchcore/bottle_cap/latest/weights/lightning/model.ckpt
```

## Configuration

Edit `roi_config.yaml`:

```yaml
roi:
  x1: 821
  y1: 536
  x2: 990
  y2: 684
threshold: 50.0
stable_seconds: 0.5
decision_frames: 5
exit_frames: 15
```

## Results

| Video | Detections | Result |
|-------|-----------|--------|
| `abnormal_test.mp4` | 4 bottles | 2 NORMAL, 2 ABNORMAL (cap not removed) |

See `output/annotated_result.mp4` for the full annotated video with diagnostics panel.

## Requirements

- Python 3.10+
- anomalib (installed via `pip install -e .` from anomalib source)
- opencv-python, pyyaml, torch

## File Structure

```
├── roi_selector.py          # Interactive ROI selection
├── extract_roi_frames.py    # ROI frame extraction with yellow filter
├── train_patchcore.py       # One-class PatchCore training
├── debug_infer_one_frame.py # Single-frame inference validation
├── infer_video.py           # Video inference with diagnostics panel
├── roi_config.yaml          # Configuration
├── data/videos/             # Sample videos
└── output/                  # Annotated results
```
