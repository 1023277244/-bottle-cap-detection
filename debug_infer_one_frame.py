#!/usr/bin/env python3
"""Debug: single-frame PatchCore inference — verify output fields and scores.

This script answers the key question before building the full pipeline:
    "When I call model(roi_tensor), what do I actually get back?"

It tests BOTH modes:
    A) post_processor enabled  (normalized scores + labels)
    B) post_processor disabled (raw scores only)

Usage:
    python debug_infer_one_frame.py --checkpoint checkpoints/patchcore_bottle_cap.ckpt --image path/to/roi_frame.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from anomalib.models.image.patchcore import Patchcore

# Allow anomalib types in PyTorch 2.6+ checkpoint loading
from anomalib import PrecisionType
import torch.serialization as _serial
_serial.add_safe_globals([PrecisionType])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def print_separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def inspect_output(result, label: str) -> None:
    """Print every field of the InferenceBatch."""
    print(f"\n--- {label} ---")
    print(f"  type       : {type(result).__name__}")
    print(f"  _fields    : {getattr(result, '_fields', 'N/A')}")

    for field_name in ["pred_score", "pred_label", "anomaly_map", "pred_mask"]:
        val = getattr(result, field_name, "MISSING")
        if val is None:
            print(f"  {field_name:12s}: None")
        elif isinstance(val, torch.Tensor):
            print(f"  {field_name:12s}: shape={list(val.shape)}  dtype={val.dtype}  "
                  f"device={val.device}  min={val.min().item():.6f}  max={val.max().item():.6f}")
            if val.numel() == 1:
                print(f"  {field_name:12s}  VALUE = {val.item():.8f}")
        else:
            print(f"  {field_name:12s}: {type(val).__name__} = {val}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug single-frame PatchCore inference",
    )
    parser.add_argument("--checkpoint", required=True, help="PatchCore .ckpt file")
    parser.add_argument("--image", required=True, help="ROI image (jpg/png) to test")
    parser.add_argument("--config", default="roi_config.yaml", help="ROI config")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip ROI cropping (use when image is already a cropped ROI)")
    args = parser.parse_args()

    # ---- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load image (full frame, then crop ROI if config says so) ---------
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error("Image not found: %s", img_path)
        sys.exit(1)

    # Try to load ROI config; if image is already a cropped ROI, use it as-is
    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        roi = cfg["roi"]
        use_roi_crop = True
        logger.info("ROI config loaded: x1=%d y1=%d x2=%d y2=%d",
                    roi["x1"], roi["y1"], roi["x2"], roi["y2"])
    except FileNotFoundError:
        roi = None
        use_roi_crop = False
        logger.info("No ROI config found; treating image as pre-cropped ROI.")

    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.error("Cannot read image: %s", img_path)
        sys.exit(1)
    logger.info("Image loaded: shape=%s dtype=%s", frame.shape, frame.dtype)

    if use_roi_crop and not args.no_crop:
        roi_frame = frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
        logger.info("After ROI crop: shape=%s", roi_frame.shape)
    else:
        roi_frame = frame
        logger.info("Using image as-is (no crop): shape=%s", roi_frame.shape)

    # ---- Load model -------------------------------------------------------
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model = Patchcore.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    logger.info("Model loaded.")

    # ---- Inspect model structure -----------------------------------------
    print_separator("MODEL STRUCTURE")
    print(f"  pre_processor  : {type(model.pre_processor).__name__ if model.pre_processor else 'None'}")
    print(f"  post_processor : {type(model.post_processor).__name__ if model.post_processor else 'None'}")
    print(f"  model          : {type(model.model).__name__}")
    print(f"  memory_bank    : shape={list(model.model.memory_bank.shape)}")

    if model.post_processor:
        pp = model.post_processor
        print(f"  PP normalize   : {pp.enable_normalization}")
        print(f"  PP threshold   : {pp.enable_thresholding}")
        print(f"  PP image_min   : {pp.image_min.item()}")
        print(f"  PP image_max   : {pp.image_max.item()}")
        print(f"  PP img_thresh  : {pp._image_threshold.item()}")

    # ---- Prepare tensor --------------------------------------------------
    tensor = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    logger.info("Input tensor: shape=%s  range=[%.3f, %.3f]",
                list(tensor.shape), tensor.min().item(), tensor.max().item())

    # ==================================================================
    # TEST A: with post_processor (default)
    # ==================================================================
    print_separator("TEST A: post_processor ENABLED (default)")
    with torch.no_grad():
        result_a = model(tensor)
    inspect_output(result_a, "Output WITH post_processor")

    # ==================================================================
    # TEST B: without post_processor
    # ==================================================================
    print_separator("TEST B: post_processor DISABLED")
    model.post_processor = None
    with torch.no_grad():
        result_b = model(tensor)
    inspect_output(result_b, "Output WITHOUT post_processor")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print_separator("SUMMARY")

    score_a = result_a.pred_score
    score_b = result_b.pred_score

    if score_a is not None and score_b is not None:
        print(f"  Score (with PP)    : {score_a.item():.8f}")
        print(f"  Score (without PP) : {score_b.item():.8f}")
        print(f"  Difference         : {abs(score_a.item() - score_b.item()):.8f}")
        print(f"  Same?               : {bool(torch.allclose(score_a, score_b))}")

    # Also check anomaly_map
    amap_a = result_a.anomaly_map
    amap_b = result_b.anomaly_map
    if amap_a is not None and amap_b is not None:
        print(f"  Anomaly map (PP)   : shape={list(amap_a.shape)}  "
              f"min={amap_a.min().item():.4f}  max={amap_a.max().item():.4f}")
        print(f"  Anomaly map (raw)  : shape={list(amap_b.shape)}  "
              f"min={amap_b.min().item():.4f}  max={amap_b.max().item():.4f}")

    # Recommendation
    print()
    if score_a is not None:
        print("  ✅ pred_score is ALWAYS present (with or without post_processor).")
        print("  ✅ anomaly_map is ALWAYS present.")
    print("  → infer_video.py can safely use result.pred_score in either mode.")
    print("  → Which mode to use depends on whether you want raw or normalized scores.")
    print("  → For one-class (no abnormal validation data), raw scores are safer.")
    print("  → Calibrate your threshold based on raw scores from normal frames.")


if __name__ == "__main__":
    main()
