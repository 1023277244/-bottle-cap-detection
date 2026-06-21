#!/usr/bin/env python3
"""Extract ROI frames from a normal video to build a one-class training dataset.

PatchCore is one-class: only normal (good) images are needed for training.
This script crops every Nth frame to the fixed ROI and saves to the
MVTec-compatible directory structure:

    data/bottle_cap/train/good/
        frame_000000.jpg
        frame_000005.jpg
        ...

A content filter skips empty/irrelevant ROIs (e.g. empty fixture, track, no bottle)
so the training set stays clean.

Usage:
    # Extract with default filter
    python extract_roi_frames.py --video normal.mp4 --output data/bottle_cap/train/good --step 10

    # Tune filter threshold with debug mode
    python extract_roi_frames.py --video normal.mp4 --output data/bottle_cap/train/good \
        --step 5 --filter laplacian --filter-threshold 20 --debug

    # Time range + filter
    python extract_roi_frames.py --video normal.mp4 --output data/bottle_cap/train/good \
        --start 20 --end 50 --step 5
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# ROI content filters — all return (valid, score)
# ---------------------------------------------------------------------------


def filter_yellow(roi_bgr: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Detect yellow bottle cap area in HSV space.

    Yellow in HSV:  H ∈ [15, 45]  S ≥ 60  V ≥ 60
    Score = yellow_pixel_ratio (0.0 – 1.0).
    Valid if ratio ≥ threshold.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 60, 60], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / mask.size
    return ratio >= threshold, ratio


def filter_laplacian(roi_bgr: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Blur → Laplacian → variance. High variance = texture present."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    score = float(lap.var())
    return score >= threshold, score


FILTERS = {
    "yellow": filter_yellow,
    "laplacian": filter_laplacian,
}
FILTER_THRESHOLDS = {
    "yellow": 0.03,      # 3% yellow pixels = bottle present
    "laplacian": 15.0,    # reasonable default
}


def is_valid_roi(roi_bgr: np.ndarray, method: str, threshold: float) -> tuple[bool, float]:
    """Return (valid, score)."""
    return FILTERS[method](roi_bgr, threshold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ROI frames from normal video for PatchCore one-class training",
    )
    parser.add_argument("--video", required=True, help="Path to normal video")
    parser.add_argument("--output", required=True,
                        help="Output directory (e.g. data/bottle_cap/train/good)")
    parser.add_argument("--config", default="roi_config.yaml", help="ROI config file")
    parser.add_argument("--step", type=int, default=10, help="Extract every Nth frame")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=-1.0, help="End time in seconds (-1 = end)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to save (0 = unlimited)")

    # Filter args
    parser.add_argument("--filter", choices=list(FILTERS), default="yellow",
                        help="ROI content filter (default: yellow)")
    parser.add_argument("--filter-threshold", type=float, default=None,
                        help="Score threshold. Frames below are skipped. "
                             "For 'yellow': ratio of yellow pixels (0.0–1.0). "
                             "If not set, auto-calibrates.")
    parser.add_argument("--debug", action="store_true",
                        help="Save rejected ROIs to output/rejected/ for inspection")

    args = parser.parse_args()

    # ---- ROI config -------------------------------------------------------
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    roi = cfg["roi"]
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

    # ---- Output dirs ------------------------------------------------------
    good_dir = Path(args.output)
    good_dir.mkdir(parents=True, exist_ok=True)
    reject_dir = good_dir.parent / "rejected"
    if args.debug:
        reject_dir.mkdir(parents=True, exist_ok=True)

    # ---- Video ------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start * fps)
    end_frame = int(args.end * fps) if args.end >= 0 else total_frames
    end_frame = min(end_frame, total_frames)

    logger.info("Video: %s | %.1f fps | %d frames | %.1fs", args.video, fps, total_frames,
                total_frames / fps)
    logger.info("Time: %.1fs–%.1fs | frames: %d–%d | step: %d",
                args.start, args.end if args.end >= 0 else total_frames / fps,
                start_frame, end_frame, args.step)
    logger.info("Filter: %s (threshold=%s)", args.filter,
                args.filter_threshold if args.filter_threshold else "auto")
    logger.info("Output: %s", good_dir)

    # Seek
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ---- First pass: auto-calibrate threshold if not provided ------------
    threshold = args.filter_threshold

    if threshold is None:
        default = FILTER_THRESHOLDS.get(args.filter, 0.0)
        logger.info("Auto-calibrating filter threshold...")
        sample_scores: list[float] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(60):
            ret, frame = cap.read()
            if not ret:
                break
            roi_frame = frame[y1:y2, x1:x2]
            _, s = is_valid_roi(roi_frame, args.filter, 0.0)
            sample_scores.append(s)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if sample_scores:
            sample_scores.sort()
            nonzero = [s for s in sample_scores if s > 0.001]
            if args.filter == "yellow" and nonzero:
                # Bimodal: empty ≈ 0, bottle >> 0.
                # Threshold = half the median of non-zero scores.
                threshold = float(np.median(nonzero)) * 0.5
                threshold = max(threshold, 0.01)  # floor at 1%
            else:
                p10 = np.percentile(sample_scores, 10)
                median = np.median(sample_scores)
                threshold = (p10 + median) / 2
            logger.info(
                "Auto threshold: %.4f  (min=%.4f  median=%.4f  max=%.4f  default=%.4f)",
                threshold, np.min(sample_scores), np.median(sample_scores),
                np.max(sample_scores), default,
            )

    # ---- Frame loop ------------------------------------------------------
    frame_idx = start_frame
    saved = 0
    skipped = 0
    all_scores: list[float] = []

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.step == 0:
            roi_frame = frame[y1:y2, x1:x2]
            valid, score = is_valid_roi(roi_frame, args.filter, threshold)
            all_scores.append(score)

            if valid:
                out_path = good_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), roi_frame)
                saved += 1
                if saved % 50 == 0:
                    logger.info("  saved %d  (frame %d/%d)  score=%.2f",
                                saved, frame_idx, end_frame, score)
            else:
                skipped += 1
                if args.debug:
                    rp = reject_dir / f"reject_{frame_idx:06d}_score{score:.1f}.jpg"
                    cv2.imwrite(str(rp), roi_frame)

        frame_idx += 1
        if args.max_frames > 0 and saved >= args.max_frames:
            break

    cap.release()

    # ---- Stats -----------------------------------------------------------
    total_checked = saved + skipped
    logger.info("=" * 50)
    logger.info("Done.")
    logger.info("  Total frames checked : %d", total_checked)
    logger.info("  Saved (valid ROI)    : %d  (%.1f%%)",
                saved, saved / max(total_checked, 1) * 100)
    logger.info("  Skipped (empty ROI)  : %d  (%.1f%%)",
                skipped, skipped / max(total_checked, 1) * 100)
    if all_scores:
        logger.info("  Score range          : %.2f – %.2f", np.min(all_scores), np.max(all_scores))
        logger.info("  Threshold            : %.2f", threshold)
    logger.info("  Output → %s", good_dir)
    if args.debug:
        logger.info("  Rejected → %s", reject_dir)
    logger.info("Next: python train_patchcore.py --data %s --output checkpoints",
                str(good_dir.parent.parent))


if __name__ == "__main__":
    main()
