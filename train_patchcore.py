#!/usr/bin/env python3
"""Train PatchCore for bottle cap anomaly detection (one-class).

PatchCore is a one-class model: it only needs NORMAL images for training.
The model learns "what normal looks like" and flags anything different as abnormal.

Expected data structure (MVTec-compatible):

    data/bottle_cap/train/good/
        frame_000000.jpg
        frame_000005.jpg
        ...

Build this dataset with:
    python extract_roi_frames.py --video normal_video.mp4 --output data/bottle_cap/train/good

Usage:
    python train_patchcore.py --data data/bottle_cap --output checkpoints
"""

import argparse
import logging
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PatchCore (one-class) for bottle cap detection",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Root folder containing train/good/ (e.g. data/bottle_cap)",
    )
    parser.add_argument(
        "--output", type=str, default="./checkpoints",
        help="Output directory for model checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--coreset-sampling-ratio", type=float, default=0.1,
        help="Coreset sampling ratio, 0.0~1.0 (default: 0.1)",
    )
    parser.add_argument(
        "--num-neighbors", type=int, default=9,
        help="Number of nearest neighbors for scoring (default: 9)",
    )
    args = parser.parse_args()

    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate data directory
    normal_dir = data_root / "train" / "good"
    if not normal_dir.exists():
        logger.error("Normal directory not found: %s", normal_dir)
        logger.error(
            "Expected structure: %s/train/good/  with normal ROI images.\n"
            "Run: python extract_roi_frames.py --video normal_video.mp4 --output %s/train/good",
            data_root, data_root,
        )
        raise FileNotFoundError(normal_dir)

    num_images = len(list(normal_dir.glob("*.jpg"))) + len(list(normal_dir.glob("*.png")))
    logger.info("Found %d normal images in %s", num_images, normal_dir)

    # ---- Data module (one-class: normal only) -------------------------------
    datamodule = Folder(
        name="bottle_cap",
        root=str(data_root),
        normal_dir="train/good",
        abnormal_dir=None,                  # ← No abnormal images for one-class
        normal_split_ratio=0.2,             # ← 20% normal held out for validation
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=4,
    )

    # ---- Model ---------------------------------------------------------------
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
    )

    # ---- Engine --------------------------------------------------------------
    engine = Engine(
        max_epochs=1,
        default_root_dir=str(output_dir),
    )

    # ---- Train ---------------------------------------------------------------
    logger.info("Training PatchCore (one-class, normal-only)...")
    engine.fit(datamodule=datamodule, model=model)

    # ---- Test (on held-out normal images) ------------------------------------
    logger.info("Running evaluation on held-out normal images...")
    test_results = engine.test(datamodule=datamodule, model=model)
    logger.info("Test results: %s", test_results)

    # ---- Save checkpoint -----------------------------------------------------
    ckpt_path = output_dir / "patchcore_bottle_cap.ckpt"
    engine.trainer.save_checkpoint(str(ckpt_path))
    logger.info("Model saved → %s", ckpt_path)
    logger.info("")
    logger.info("Next step — run inference:")
    logger.info(
        "  python infer_video.py --video test.mp4 --checkpoint %s",
        ckpt_path,
    )


if __name__ == "__main__":
    main()
