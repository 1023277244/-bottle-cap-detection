#!/usr/bin/env python3
"""Interactive ROI selector — click & drag on the first frame, auto-save to roi_config.yaml.

Usage:
    python roi_selector.py --video test.mp4

Controls:
    - Mouse drag  → draw ROI rectangle
    - r           → reset selection
    - ENTER / y   → confirm and save to roi_config.yaml
    - ESC / q     → quit without saving
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml

WIN_NAME = "ROI Selector — drag to select, ENTER to save, ESC to quit"
CONFIG_PATH = Path(__file__).resolve().parent / "roi_config.yaml"


class ROISelector:
    def __init__(self, frame: cv2.Mat) -> None:
        self._frame = frame.copy()
        self._display = frame.copy()
        self._drawing = False
        self._x1 = self._y1 = self._x2 = self._y2 = -1

    @property
    def roi(self) -> dict | None:
        if -1 in (self._x1, self._y1, self._x2, self._y2):
            return None
        x1, x2 = sorted((self._x1, self._x2))
        y1, y2 = sorted((self._y1, self._y2))
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _redraw(self) -> None:
        self._display = self._frame.copy()
        roi = self.roi
        if roi:
            cv2.rectangle(
                self._display,
                (roi["x1"], roi["y1"]),
                (roi["x2"], roi["y2"]),
                (0, 255, 0), 2,
            )
            w, h = roi["x2"] - roi["x1"], roi["y2"] - roi["y1"]
            cv2.putText(
                self._display,
                f"ROI: ({roi['x1']},{roi['y1']}) → ({roi['x2']},{roi['y2']})  {w}x{h}",
                (10, self._frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )

    @staticmethod
    def _mouse_cb(event: int, x: int, y: int, flags: int, self_) -> None:
        del flags
        if event == cv2.EVENT_LBUTTONDOWN:
            self_._drawing = True
            self_._x1, self_._y1 = x, y
            self_._x2, self_._y2 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self_._drawing:
            self_._x2, self_._y2 = x, y
            self_._redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self_._drawing = False
            self_._x2, self_._y2 = x, y
            self_._redraw()

    def run(self) -> dict | None:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN_NAME, self._mouse_cb, self)

        print("  Drag mouse → draw ROI")
        print("  ENTER     → save & exit")
        print("  r         → reset")
        print("  ESC / q   → quit without saving\n")

        self._redraw()
        while True:
            cv2.imshow(WIN_NAME, self._display)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord("y"), ord("Y")):  # ENTER or y
                roi = self.roi
                if roi is None:
                    print("  ⚠ No ROI selected. Drag to draw a rectangle first.")
                    continue
                cv2.destroyAllWindows()
                return roi

            if key in (27, ord("q"), ord("Q")):  # ESC or q
                cv2.destroyAllWindows()
                return None

            if key == ord("r"):  # reset
                self._x1 = self._y1 = self._x2 = self._y2 = -1
                self._redraw()
                print("  ↺ ROI reset.")


# ---------------------------------------------------------------------------
def load_existing_config(path: Path) -> dict:
    """Load existing config or return a default skeleton."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(roi: dict, path: Path) -> None:
    """Update only the roi section of the config, preserving everything else."""
    cfg = load_existing_config(path)

    cfg["roi"] = {
        "x1": roi["x1"],
        "y1": roi["y1"],
        "x2": roi["x2"],
        "y2": roi["y2"],
    }

    # Keep defaults for missing keys
    cfg.setdefault("threshold", 0.5)
    cfg.setdefault("window_size", 10)
    cfg.setdefault("alarm_consecutive_frames", 5)
    cfg.setdefault("output", {
        "video_path": "output/result_video.mp4",
        "alarm_dir": "output/alarms",
        "log_file": "output/alarm_log.csv",
    })

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive ROI selector for video anomaly detection",
    )
    parser.add_argument("--video", required=True, help="Video file to select ROI from")
    parser.add_argument(
        "--output", default=str(CONFIG_PATH),
        help=f"Config output path (default: {CONFIG_PATH})",
    )
    args = parser.parse_args()

    # Open video, grab first frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Cannot read first frame from video.")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Video: {args.video}")
    print(f"Resolution: {w} x {h}")
    print()

    selector = ROISelector(frame)
    roi = selector.run()

    if roi is None:
        print("Cancelled. No changes saved.")
        sys.exit(0)

    output_path = Path(args.output)
    save_config(roi, output_path)
    print(f"✅ ROI saved → {output_path}")
    print(f"   x1={roi['x1']}  y1={roi['y1']}  x2={roi['x2']}  y2={roi['y2']}")
    print(f"   size = {roi['x2'] - roi['x1']} x {roi['y2'] - roi['y1']}")


if __name__ == "__main__":
    main()
