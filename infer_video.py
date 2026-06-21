#!/usr/bin/env python3
"""Fixed-station PatchCore bottle-cap anomaly detection.

Logic:
    video stream
    → fixed ROI crop every frame
    → detect if bottle is present (yellow check)
    → wait until bottle stays continuously for stable_seconds
    → take decision_frames from the middle of the stable window
    → PatchCore → avg score → NORMAL / ABNORMAL
    → report once, then wait for bottle to leave before next detection
    → entry / exit / empty / motion phases: skip

Usage:
    python infer_video.py --video test.mp4 --checkpoint model.ckpt
"""

import argparse
import csv
import logging
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from anomalib.models.image.patchcore import Patchcore
from anomalib import PrecisionType
import torch.serialization as _serial
_serial.add_safe_globals([PrecisionType])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Bottle presence
# ---------------------------------------------------------------------------
YELLOW_LOWER = np.array([15, 60, 60], dtype=np.uint8)
YELLOW_UPPER = np.array([45, 255, 255], dtype=np.uint8)


def bottle_in_roi(roi_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    return float(np.count_nonzero(mask)) / mask.size >= 0.03


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def crop(frame: np.ndarray, roi: dict) -> np.ndarray:
    return frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]


def to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Diagnostics panel (right side, detection-only)
# ---------------------------------------------------------------------------
PANEL_W = 400
BAR_H = 18
HEATMAP_GLOBAL_MAX = 80.0   # fixed upper bound for anomaly_map colour mapping


def make_canvas(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    return canvas


def draw_panel(canvas: np.ndarray, roi_bgr: np.ndarray, anomaly_map: np.ndarray | None,
               score: float, yellow_ratio: float, saturation: float,
               result: str, threshold: float, detect_time: str) -> np.ndarray:
    """Draw diagnostics panel. NORMAL → blue heatmap, ABNORMAL → red hot spots."""
    h, full_w = canvas.shape[:2]
    w = full_w - PANEL_W
    x0 = w + 10
    color = (0, 0, 255) if result == "ABNORMAL" else (0, 255, 0)

    # Panel border
    cv2.rectangle(canvas, (w, 0), (full_w - 1, h - 1), color, 3)

    # ---- Header ----
    cv2.putText(canvas, "PATCHCORE DIAGNOSTICS", (x0, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    badge = "CAP NOT REMOVED" if result == "ABNORMAL" else "NORMAL"
    cv2.putText(canvas, badge, (x0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    y = 68

    # ---- ROI crop (top-left of panel, keep original aspect) ----
    rh, rw = roi_bgr.shape[:2]
    thumb_w = PANEL_W - 20
    thumb_h = int(rh * thumb_w / rw)
    thumb = cv2.resize(roi_bgr, (thumb_w, thumb_h))
    cv2.putText(canvas, "ROI Crop", (x0, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    canvas[y:y + thumb_h, x0:x0 + thumb_w] = thumb
    cv2.rectangle(canvas, (x0, y), (x0 + thumb_w, y + thumb_h), color, 1)
    y += thumb_h + 8

    # ---- Anomaly map (same size as ROI, fixed global range) ----
    if anomaly_map is not None:
        cv2.putText(canvas, "Anomaly Map", (x0, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        amap = anomaly_map.squeeze()
        # Fixed global range: 0 → blue, HEATMAP_GLOBAL_MAX → red
        amap_clipped = np.clip(amap, 0, HEATMAP_GLOBAL_MAX) / HEATMAP_GLOBAL_MAX * 255
        amap_u8 = amap_clipped.astype(np.uint8)
        # Resize to match ROI crop exactly
        amap_resized = cv2.resize(amap_u8, (rw, rh), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)
        # Display at same size as ROI thumbnail
        heat_disp = cv2.resize(heat, (thumb_w, thumb_h))
        canvas[y:y + thumb_h, x0:x0 + thumb_w] = heat_disp
        cv2.rectangle(canvas, (x0, y), (x0 + thumb_w, y + thumb_h), color, 1)
        y += thumb_h + 12

    # ---- Metrics (text only, no progress bars) ----
    bar_w = PANEL_W - 30
    font_scale = 0.5
    row_h = 18

    metrics = [
        ("Score", f"{score:.2f}"),
        ("Threshold", f"{threshold:.2f}"),
        ("Yellow Ratio", f"{yellow_ratio:.1%}"),
        ("Saturation", f"{saturation:.0f}"),
        ("", ""),
        ("Detection Time", detect_time),
    ]

    for label, value in metrics:
        if label:
            cv2.putText(canvas, f"{label}:", (x0, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1)
            cv2.putText(canvas, value, (x0 + 140, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        y += row_h

    # ---- Score gauge bar ----
    cv2.putText(canvas, "Score", (x0, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1)
    y += 16
    bx, by = x0, y
    # Background
    cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + BAR_H), (40, 40, 40), -1)
    # Threshold marker
    t_ratio = min(threshold / HEATMAP_GLOBAL_MAX, 1.0)
    tx = bx + int(bar_w * t_ratio)
    cv2.line(canvas, (tx, by), (tx, by + BAR_H), (0, 255, 255), 2)
    # Score bar (green→yellow→red)
    s_ratio = min(score / HEATMAP_GLOBAL_MAX, 1.0)
    if s_ratio < t_ratio * 0.7:
        bar_color = (0, 200, 0)
    elif s_ratio < t_ratio:
        bar_color = (0, 200, 255)
    else:
        bar_color = (0, 0, 255)
    cv2.rectangle(canvas, (bx, by), (bx + int(bar_w * s_ratio), by + BAR_H), bar_color, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + BAR_H), (100, 100, 100), 1)

    # Clear the rest
    y += BAR_H + 5
    canvas[y:, w:] = (0, 0, 0)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_inference(video_path: str, checkpoint_path: str,
                  config_path: str = "roi_config.yaml", device: str = "auto") -> None:
    # Config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    roi = cfg["roi"]
    threshold = float(cfg["threshold"])
    stable_sec = float(cfg.get("stable_seconds", 2.0))
    decision_n = int(cfg.get("decision_frames", 5))
    out_video = Path(cfg["output"]["video_path"])
    alarm_dir = Path(cfg["output"]["alarm_dir"])
    log_file = Path(cfg["output"]["log_file"])

    # Device + model
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       device if device != "auto" else "cpu")
    logger.info("Device: %s", dev)
    model = Patchcore.load_from_checkpoint(str(checkpoint_path), map_location=dev)
    model.post_processor = None
    model.eval()
    model.to(dev)

    # Video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %dx%d %.1ffps %d frames", w, h, fps, total)

    stable_frames_needed = int(stable_sec * fps)
    exit_frames = int(cfg.get("exit_frames", 15))  # frames of absence to confirm bottle left

    # Output
    out_video.parent.mkdir(parents=True, exist_ok=True)
    alarm_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w + PANEL_W, h))
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "detect_frame", "avg_score", "threshold", "decision"])

    # State machine: EMPTY → WAITING → DONE (latched) → EMPTY
    state = "EMPTY"
    present_count = 0
    absent_count = 0
    roi_buffer: deque = deque(maxlen=stable_frames_needed + decision_n + 10)

    # Latched result — persists until bottle truly leaves
    last_result: dict = {}

    detect_count = 0
    frame_idx = 0
    logger.info("Station: stable>=%.1fs exit>=%df threshold=%.1f",
                stable_sec, exit_frames, threshold)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = crop(frame, roi)
        present = bottle_in_roi(roi_frame)

        if present:
            absent_count = 0
            present_count += 1
            roi_buffer.append((frame_idx, roi_frame.copy()))

            if state == "EMPTY":
                state = "WAITING"
                present_count = 1
                roi_buffer.clear()
                roi_buffer.append((frame_idx, roi_frame.copy()))

            elif state == "WAITING" and present_count >= stable_frames_needed:
                # ---- DETECT (once per visit) ----
                state = "DONE"
                buf_list = list(roi_buffer)
                picks = buf_list[-decision_n:] if len(buf_list) >= decision_n else buf_list

                scores = []
                amaps = []
                for _fidx, _roi in picks:
                    t = to_tensor(_roi, dev)
                    with torch.no_grad():
                        out = model(t)
                        scores.append(float(out.pred_score.cpu().item()))
                        if out.anomaly_map is not None:
                            amaps.append(out.anomaly_map.cpu().numpy())

                avg = float(np.mean(scores))
                result_text = "ABNORMAL" if avg > threshold else "NORMAL"
                detect_count += 1

                diag_roi = picks[len(picks) // 2][1].copy()
                diag_amap = amaps[len(amaps) // 2] if amaps else None
                hsv = cv2.cvtColor(diag_roi, cv2.COLOR_BGR2HSV)
                ymask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
                diag_yellow = float(np.count_nonzero(ymask)) / ymask.size
                diag_sat = float(np.mean(hsv[:, :, 1]))

                last_result = {
                    "text": result_text,
                    "score": avg,
                    "roi_bgr": diag_roi,
                    "anomaly_map": diag_amap,
                    "yellow": diag_yellow,
                    "sat": diag_sat,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }

                ts = datetime.now().isoformat(timespec="seconds")
                detect_frame = picks[len(picks) // 2][0]
                with open(log_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([ts, detect_frame, f"{avg:.4f}", f"{threshold:.2f}", result_text])

                if result_text == "ABNORMAL":
                    ap = alarm_dir / f"detect{detect_count:02d}_f{detect_frame}_score{avg:.2f}.jpg"
                    cv2.imwrite(str(ap), frame)
                    logger.warning("ALARM #%d  frame=%d  score=%.2f",
                                   detect_count, detect_frame, avg)
                else:
                    logger.info("OK    #%d  frame=%d  score=%.2f",
                                detect_count, detect_frame, avg)

            # DONE: bottle still present → keep result, do NOT re-detect

        else:
            present_count = 0
            absent_count += 1

            if absent_count >= exit_frames:
                # Bottle truly gone — full reset
                state = "EMPTY"
                last_result = {}
                absent_count = 0
                roi_buffer.clear()

        # Build canvas: show panel as long as bottle is present + result exists
        canvas = make_canvas(frame)
        show_panel = (state == "DONE" and last_result)
        if show_panel:
            lr = last_result
            canvas = draw_panel(canvas, lr["roi_bgr"], lr["anomaly_map"],
                                lr["score"], lr["yellow"], lr["sat"],
                                lr["text"], threshold, lr["time"])
            # ROI box on main frame
            x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            pcolor = (0, 0, 255) if lr["text"] == "ABNORMAL" else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), pcolor, 2)
            cv2.putText(canvas, "CAP NOT REMOVED" if lr["text"] == "ABNORMAL" else "NORMAL",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pcolor, 2)
        writer.write(canvas)

        if frame_idx % 100 == 0:
            logger.info("  frame %d/%d  state=%s", frame_idx, total, state)

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    logger.info("Done. %d frames, %d detections → %s", frame_idx, detect_count, out_video)


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-station PatchCore bottle cap detection")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="roi_config.yaml")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    run_inference(args.video, args.checkpoint, args.config, args.device)


if __name__ == "__main__":
    main()
