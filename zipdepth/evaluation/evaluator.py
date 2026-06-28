"""Zero-shot depth evaluation loop for ZipDepth.

ZipDepth predicts affine-invariant inverse depth. For each sample we:
  1. predict with the real ZipDepth inference path,
  2. align prediction to ground truth in disparity space (least-squares scale/shift),
  3. convert back to depth and compute metrics.
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from zipdepth.evaluation.metrics import (
    METRIC_FUNCTIONS,
    MetricTracker,
    align_depth_least_square,
    compute_depth_metrics,
    depth2disparity,
    disparity2depth,
)
from zipdepth.evaluation.datasets import (
    get_eval_mask,
    kitti_benchmark_crop,
    load_gt,
)

logger = logging.getLogger(__name__)


def evaluate(
    predictor,
    samples: List[dict],
    cfg: dict,
    alignment: str = 'least_square_disparity',
    alignment_max_res: Optional[int] = None,
    apply_kitti_crop: bool = True,
    eval_mask_type: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_predictions: bool = False,
) -> Dict[str, float]:
    """Evaluate ``predictor`` over ``samples`` and return the mean metrics.

    Args:
        predictor: object exposing ``infer_image(bgr) -> np.ndarray`` (inverse depth).
        samples: list of sample dicts from ``discover_samples``.
        cfg: dataset configuration (depth range, crops, masks).
        alignment: ``least_square_disparity`` (default) or ``least_square``.
        alignment_max_res: downsample resolution for the least-squares solve.
        apply_kitti_crop: apply the KITTI benchmark crop when the config asks for it.
        eval_mask_type: ``garg`` / ``eigen`` / ``None`` evaluation mask override.
        output_dir: where to write the per-sample CSV (and predictions if requested).
        save_predictions: also dump raw predictions as ``.npy``.
    """
    min_depth, max_depth = cfg['min_depth'], cfg['max_depth']
    metric_names = list(METRIC_FUNCTIONS.keys())
    tracker = MetricTracker(metric_names)
    per_sample_rows = []

    pred_dir = None
    if save_predictions and output_dir:
        pred_dir = Path(output_dir) / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            depth_gt, valid_mask = load_gt(sample, cfg)
        except Exception as e:
            logger.warning(f"Skipping {sample['name']}: cannot load GT ({e})")
            continue

        # ---- Prediction (inverse depth at original resolution) ----
        image_bgr = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
        if image_bgr is None:
            logger.warning(f"Skipping {sample['name']}: cannot load image")
            continue
        pred_raw = predictor.infer_image(image_bgr)

        # Match prediction to GT resolution (e.g. ETH3D RGB != GT size).
        if pred_raw.shape[:2] != depth_gt.shape[:2]:
            pred_raw = cv2.resize(pred_raw, (depth_gt.shape[1], depth_gt.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

        if save_predictions and pred_dir is not None:
            np.save(str(pred_dir / (sample['name'].replace('/', '_') + '.npy')), pred_raw)

        # ---- KITTI benchmark crop ----
        if apply_kitti_crop and cfg.get('kitti_bm_crop'):
            depth_gt = kitti_benchmark_crop(depth_gt)
            valid_mask = kitti_benchmark_crop(valid_mask)
            pred_raw = kitti_benchmark_crop(pred_raw)

        # ---- Evaluation mask (Garg/Eigen) ----
        if eval_mask_type:
            h, w = depth_gt.shape[:2]
            valid_mask = valid_mask & get_eval_mask(h, w, eval_mask_type)

        # ---- Affine alignment ----
        if alignment == 'least_square_disparity':
            gt_disp, gt_nonneg = depth2disparity(depth_gt, return_mask=True)
            fit_mask = valid_mask & gt_nonneg & (pred_raw > 0)
            pred_disp = align_depth_least_square(gt_disp, pred_raw, fit_mask, alignment_max_res)
            pred_depth = disparity2depth(np.clip(pred_disp, a_min=1e-3, a_max=None))
        elif alignment == 'least_square':
            pred_depth = align_depth_least_square(depth_gt, pred_raw, valid_mask, alignment_max_res)
        else:
            raise ValueError(f"Unknown alignment: {alignment}")

        pred_depth = np.clip(pred_depth, min_depth, max_depth)

        # ---- Metrics ----
        metrics = compute_depth_metrics(pred_depth, depth_gt, valid_mask)
        tracker.update(metrics)
        per_sample_rows.append({'filename': sample['name'], **metrics})

    if output_dir and per_sample_rows:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / 'per_sample.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename'] + metric_names)
            writer.writeheader()
            writer.writerows(per_sample_rows)
        logger.info(f"Per-sample metrics: {csv_path}")

    return tracker.result()


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: Dict[str, float], dataset: str, n_samples: int):
    """Print a compact horizontal results table to the console."""
    names = list(results.keys())
    col_w = max(9, max((len(n) for n in names), default=0) + 2)
    header = "".join(f"{n:>{col_w}}" for n in names)
    values = "".join(f"{results[n]:>{col_w}.4f}" for n in names)
    width = max(len(header), 40)
    print(f"\n{'=' * width}")
    print(f"  Results — {dataset}  ({n_samples} samples)")
    print('-' * width)
    print(header)
    print(values)
    print(f"{'=' * width}\n")


def save_results(results: Dict[str, float], dataset: str, output_dir: str,
                 alignment: str, checkpoint: str):
    """Write the mean metrics + run metadata to ``accuracy_<dataset>.json``."""
    data = {
        'metadata': {
            'dataset': dataset,
            'checkpoint': checkpoint,
            'alignment': alignment,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': {k: round(v, 6) for k, v in results.items()},
    }
    path = Path(output_dir) / f'accuracy_{dataset}.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved: {path}")
    return path
