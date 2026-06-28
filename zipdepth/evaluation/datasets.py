"""Zero-shot evaluation datasets: configuration, sample discovery, and GT loading.

Supported benchmarks (Marigold protocol): NYUv2, KITTI (Eigen split), ETH3D,
ScanNet, and DIODE. Each entry in :data:`DATASET_CONFIGS` carries the depth
range and protocol options (crops, evaluation masks) for that dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_CONFIGS: Dict[str, dict] = {
    'nyuv2': {
        'gt_type': 'nyu',
        'min_depth': 1e-3,
        'max_depth': 10.0,
        'eigen_crop': True,          
    },
    'kitti': {
        'gt_type': 'kitti',         
        'min_depth': 1e-3,
        'max_depth': 80.0,
        'kitti_bm_crop': True,       
        'eval_mask': 'eigen',        
    },
    'eth3d': {
        'gt_type': 'eth3d',         
        'min_depth': 1e-5,
        'max_depth': 60.0,
        'eth3d_height': 4032,
        'eth3d_width': 6048,
    },
    'scannet': {
        'gt_type': 'scannet',        
        'min_depth': 1e-3,
        'max_depth': 10.0,
    },
    'diode': {
        'gt_type': 'diode',          
        'domain': 'all',
        'min_depth': 0.6,
        'max_depth': 80.0,
    },
}


# =============================================================================
# CROPS AND EVALUATION MASKS
# =============================================================================

KB_CROP_HEIGHT, KB_CROP_WIDTH = 352, 1216


def kitti_benchmark_crop(img: np.ndarray) -> np.ndarray:
    """KITTI benchmark center crop to 352 x 1216."""
    h, w = img.shape[:2]
    top = h - KB_CROP_HEIGHT
    left = (w - KB_CROP_WIDTH) // 2
    return img[top:top + KB_CROP_HEIGHT, left:left + KB_CROP_WIDTH]


def get_eval_mask(height: int, width: int, mask_type: str) -> np.ndarray:
    """Garg / Eigen evaluation crop mask used on KITTI."""
    mask = np.zeros((height, width), dtype=bool)
    if mask_type == 'garg':
        mask[int(0.40810811 * height):int(0.99189189 * height),
             int(0.03594771 * width):int(0.96405229 * width)] = True
    elif mask_type == 'eigen':
        mask[int(0.3324324 * height):int(0.91351351 * height),
             int(0.0359477 * width):int(0.96405229 * width)] = True
    return mask


# =============================================================================
# GROUND-TRUTH LOADING
# =============================================================================

def load_gt(sample: dict, cfg: dict):
    """Load ground-truth depth and validity mask for one sample.

    Returns:
        depth_gt: [H, W] float32 metric depth (0 where invalid).
        valid_mask: [H, W] bool.
    """
    gt_type = cfg['gt_type']
    min_d, max_d = cfg['min_depth'], cfg['max_depth']

    if gt_type == 'nyu':
        depth_gt = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        valid_mask = (depth_gt > min_d) & (depth_gt < max_d)
        if cfg.get('eigen_crop'):
            eigen = np.zeros_like(valid_mask)
            eigen[45:471, 41:601] = True
            valid_mask &= eigen

    elif gt_type == 'scannet':
        depth_gt = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        valid_mask = (depth_gt > min_d) & (depth_gt < max_d)

    elif gt_type == 'kitti':
        depth_png = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise FileNotFoundError(sample['depth_path'])
        depth_gt = depth_png.astype(np.float32) / 256.0
        valid_mask = (depth_gt > min_d) & (depth_gt < max_d)

    elif gt_type == 'eth3d':
        with open(sample['depth_path'], 'rb') as f:
            depth_gt = np.frombuffer(f.read(), dtype=np.float32).copy()
        depth_gt = depth_gt.reshape((cfg['eth3d_height'], cfg['eth3d_width']))
        valid_mask = np.isfinite(depth_gt) & (depth_gt > min_d) & (depth_gt < max_d)
        depth_gt[~valid_mask] = 0.0

    elif gt_type == 'diode':
        depth_gt = np.load(sample['depth_path']).squeeze().astype(np.float32)
        valid_mask = np.load(sample['mask_path']).squeeze().astype(bool)
        valid_mask &= (depth_gt > min_d) & (depth_gt < max_d)

    else:
        raise ValueError(f"Unknown gt_type: {gt_type}")

    return depth_gt, valid_mask


# =============================================================================
# SAMPLE DISCOVERY
# =============================================================================

def discover_samples(dataset: str, data_dir: str, domain: str = 'all') -> List[dict]:
    """Dispatch to the right discovery routine for ``dataset``."""
    finders = {
        'nyuv2': _discover_nyuv2,
        'kitti': _discover_kitti_eigen,
        'eth3d': _discover_eth3d,
        'scannet': _discover_scannet,
        'diode': lambda d: _discover_diode(d, domain),
    }
    if dataset not in finders:
        raise ValueError(f"Unknown dataset: {dataset}")
    samples = finders[dataset](data_dir)
    logger.info(f"Found {len(samples)} {dataset} samples")
    return samples


def _discover_nyuv2(data_dir: str) -> List[dict]:
    """NYUv2: ``{scene}/rgb_XXXX.png`` + ``depth_XXXX.png``."""
    data_dir = Path(data_dir)
    samples = []
    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for rgb_file in sorted(scene_dir.glob('rgb_*.png')):
            sample_id = rgb_file.stem.replace('rgb_', '')
            depth_file = scene_dir / f'depth_{sample_id}.png'
            if depth_file.exists():
                samples.append({
                    'image_path': str(rgb_file),
                    'depth_path': str(depth_file),
                    'name': f"{scene_dir.name}/{sample_id}",
                })
    return samples


def _discover_kitti_eigen(data_dir: str) -> List[dict]:
    """KITTI Eigen split: ``{date}/{drive}/image_02/data`` + projected GT depth."""
    data_dir = Path(data_dir)
    samples = []
    for date_dir in sorted(data_dir.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith('2011_'):
            continue
        for drive_dir in sorted(date_dir.iterdir()):
            image_dir = drive_dir / 'image_02' / 'data'
            gt_dir = drive_dir / 'proj_depth' / 'groundtruth' / 'image_02'
            if not image_dir.exists() or not gt_dir.exists():
                continue
            gt_files = {f.name for f in gt_dir.glob('*.png')}
            for img_file in sorted(image_dir.glob('*.png')):
                if img_file.name in gt_files:
                    samples.append({
                        'image_path': str(img_file),
                        'depth_path': str(gt_dir / img_file.name),
                        'name': f"{date_dir.name}/{drive_dir.name}/{img_file.stem}",
                    })
    return samples


def _discover_eth3d(data_dir: str) -> List[dict]:
    """ETH3D: ``depth/{scene}/ground_truth_depth/dslr_images`` + undistorted RGB."""
    data_dir = Path(data_dir)
    depth_root, image_root = data_dir / 'depth', data_dir / 'images'
    if not depth_root.exists() or not image_root.exists():
        raise FileNotFoundError(f"ETH3D requires depth/ and images/ under {data_dir}")
    samples = []
    for scene_dir in sorted(depth_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        gt_dir = scene_dir / 'ground_truth_depth' / 'dslr_images'
        img_dir = image_root / scene_dir.name / 'images' / 'dslr_images_undistorted'
        if not gt_dir.exists() or not img_dir.exists():
            logger.warning(f"ETH3D: skipping scene {scene_dir.name} (missing gt or images)")
            continue
        rgb_map = {f.stem: f for f in img_dir.iterdir()
                   if f.suffix.lower() in ('.jpg', '.jpeg', '.png')}
        for gt_file in sorted(gt_dir.iterdir()):
            if gt_file.is_file() and gt_file.stem in rgb_map:
                samples.append({
                    'image_path': str(rgb_map[gt_file.stem]),
                    'depth_path': str(gt_file),
                    'name': f"{scene_dir.name}/{gt_file.stem}",
                })
    return samples


def _discover_scannet(data_dir: str) -> List[dict]:
    """ScanNet: ``sceneXXXX_XX/color/XXXXXX.jpg`` + ``depth/XXXXXX.png``."""
    data_dir = Path(data_dir)
    samples = []
    for scene_dir in sorted(data_dir.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith('scene'):
            continue
        color_dir, depth_dir = scene_dir / 'color', scene_dir / 'depth'
        if not color_dir.exists() or not depth_dir.exists():
            continue
        for rgb_file in sorted(color_dir.glob('*.jpg')):
            depth_file = depth_dir / f'{rgb_file.stem}.png'
            if depth_file.exists():
                samples.append({
                    'image_path': str(rgb_file),
                    'depth_path': str(depth_file),
                    'name': f"{scene_dir.name}/{rgb_file.stem}",
                })
    return samples


def _discover_diode(data_dir: str, domain: str = 'all') -> List[dict]:
    """DIODE: ``{domain}/scene_*/scan_*/<stem>.png`` + ``<stem>_depth(.npy|_mask.npy)``."""
    data_dir = Path(data_dir)
    domains = ['indoors', 'outdoor'] if domain == 'all' else [domain]
    samples = []
    for dom in domains:
        dom_dir = data_dir / dom
        if not dom_dir.exists():
            logger.warning(f"DIODE domain directory not found: {dom_dir}")
            continue
        for scene_dir in sorted(dom_dir.glob('scene_*')):
            for scan_dir in sorted(scene_dir.glob('scan_*')):
                for img_file in sorted(scan_dir.glob('*.png')):
                    stem = img_file.stem
                    depth_file = scan_dir / f'{stem}_depth.npy'
                    mask_file = scan_dir / f'{stem}_depth_mask.npy'
                    if depth_file.exists() and mask_file.exists():
                        samples.append({
                            'image_path': str(img_file),
                            'depth_path': str(depth_file),
                            'mask_path': str(mask_file),
                            'name': f"{dom}/{scene_dir.name}/{scan_dir.name}/{img_file.name}",
                        })
    return samples
