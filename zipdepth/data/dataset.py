"""Dataset loaders for depth estimation"""

import torch
from torch.utils.data import Dataset
import numpy as np
from turbojpeg import TurboJPEG, TJPF_RGB
from pathlib import Path
from typing import Optional, List
import cv2
cv2.setNumThreads(0)
import json


def _as_str(value) -> str:
    """Return a Python str from a memmap index entry.

    Index arrays may be stored as byte-strings (dtype ``S``, the format written
    by ``prepare_index.py``) or as unicode strings (dtype ``U``). Decode the
    former, pass the latter through unchanged.
    """
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


class LargeScaleDepthDataset(Dataset):

    def __init__(self,
                 index_file: str,
                 domains: Optional[List[str]] = None,
                 transform=None,
                 max_samples: Optional[int] = None):

        self.transform = transform
        self._jpeg = TurboJPEG()

        index_path = Path(index_file)
        prefix = str(index_path.with_suffix(''))
        rgb_file = f'{prefix}_rgb.npy'
        metadata_file = f'{prefix}_metadata.json'

        if Path(rgb_file).exists():
            print(f"Loading index: {prefix}_*.npy")
            self._load_numpy(prefix, metadata_file)
        else:
            raise FileNotFoundError(
                f"Converted index not found: {rgb_file}\n"
                f"Run: python scripts/prepare_index.py convert --input {index_file}"
            )

        if domains is not None:
            print(f"Filtering by domains: {domains}")
            self._filter_by_domains(domains)
        else:
            self.valid_indices = list(range(len(self.rgb_paths)))

        if max_samples is not None:
            self.valid_indices = self.valid_indices[:max_samples]
            print(f"Limited to {max_samples:,} samples")

        self._print_summary()

    def _load_numpy(self, prefix, metadata_file):
        # Fixed-length byte-string dtype → the kernel maps these arrays read-only
        # and shares physical pages across DataLoader workers (no copy-on-write).
        # mmap_mode='r' only works correctly with fixed-size dtypes (S*, U*, numeric);
        # dtype=object would load everything into RAM, defeating the purpose.
        self.rgb_paths   = np.load(f'{prefix}_rgb.npy',    mmap_mode='r')
        self.depth_paths = np.load(f'{prefix}_depth.npy',  mmap_mode='r')
        self.domains     = np.load(f'{prefix}_domain.npy', mmap_mode='r')

        if Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                self.index_metadata = json.load(f)
        else:
            self.index_metadata = {
                'version': 'unknown',
                'created_at': 'unknown',
                'total_samples': len(self.rgb_paths),
            }

        print(f"  {len(self.rgb_paths):,} samples (memory-mapped)")

    def _filter_by_domains(self, domains):
        wanted = set(domains)
        original_count = len(self.rgb_paths)
        print("  Filtering domains...")
        self.valid_indices = [
            i for i in range(len(self.domains))
            if _as_str(self.domains[i]) in wanted
        ]
        print(f"  Filtered: {original_count:,} -> {len(self.valid_indices):,}")

    def _print_summary(self):
        domain_counts = {}
        for idx in self.valid_indices:
            domain = _as_str(self.domains[idx])
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print(f"\nDataset summary:")
        print(f"  Total samples: {len(self.valid_indices):,}")
        print(f"  Domains: {len(domain_counts)}")
        for domain in sorted(domain_counts.keys()):
            print(f"    {domain:20s}: {domain_counts[domain]:>10,} samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        MAX_RETRIES = 5

        for attempt in range(MAX_RETRIES):
            sample_idx = (idx + attempt) % len(self.valid_indices)
            real_idx = self.valid_indices[sample_idx]

            try:
                rgb_path   = _as_str(self.rgb_paths[real_idx])
                depth_path = _as_str(self.depth_paths[real_idx])
                domain     = _as_str(self.domains[real_idx])

                rgb = self._load_rgb(rgb_path)

                depth = None
                if depth_path:
                    depth = self._load_depth(depth_path)

                if self.transform:
                    rgb, depth = self.transform(rgb, depth)

                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

                output = {
                    'image': rgb_tensor,
                    'domain': domain,
                    'path': rgb_path,
                }

                if depth is not None:
                    depth_tensor = torch.from_numpy(depth).unsqueeze(0).contiguous()
                    output['depth'] = depth_tensor

                del rgb, depth
                return output

            except Exception as e:
                if attempt == 0:
                    print(f"WARNING: Failed to load sample {real_idx} ({rgb_path}): {e}")
                continue

        print(f"ERROR: Failed to load valid sample after {MAX_RETRIES} attempts starting from idx {idx}")

        if self.transform and hasattr(self.transform, 'height'):
            h, w = self.transform.height, self.transform.width
        else:
            h, w = 512, 512

        return {
            'image': torch.zeros(3, h, w, dtype=torch.uint8).contiguous(),
            'depth': torch.zeros(1, h, w, dtype=torch.uint16).contiguous(),
            'domain': 'corrupted',
            'path': f'dummy_sample_{idx}',
        }

    def _load_rgb(self, path: str) -> np.ndarray:
        if path.lower().endswith(('.jpg', '.jpeg')):
            with open(path, 'rb') as f:
                return self._jpeg.decode(f.read(), pixel_format=TJPF_RGB)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_depth(self, path: str) -> np.ndarray:
        path_lower = path.lower()

        if path_lower.endswith('.png'):
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Cannot load: {path}")

        elif path_lower.endswith(('.npy', '.npz')):
            if path_lower.endswith('.npz'):
                with np.load(path) as data:
                    depth = data['depth'] if 'depth' in data else data[data.files[0]]
            else:
                depth = (np.load(path) * 256.0).astype(np.uint16)

        else:
            raise ValueError(f"Unsupported format: {path}")

        if depth.ndim == 3:
            depth = depth.squeeze()

        return depth


class BalancedDomainSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: LargeScaleDepthDataset,
        num_samples: Optional[int] = None,
        temperature: float = 0.2,
        max_repeat: int = 30,
        min_coverage: float = 0.25,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        self.domain_indices = {}
        for i, valid_idx in enumerate(dataset.valid_indices):
            domain = _as_str(dataset.domains[valid_idx])
            if domain not in self.domain_indices:
                self.domain_indices[domain] = []
            self.domain_indices[domain].append(i)

        self.domains = list(self.domain_indices.keys())
        counts = np.array([len(self.domain_indices[d]) for d in self.domains])

        total = num_samples if num_samples else len(dataset)

        weights = (1.0 / counts) ** temperature
        probs_raw = weights / weights.sum()

        raw_sampled = probs_raw * total
        max_sampled = counts * max_repeat
        min_sampled = counts * min_coverage
        effective = np.clip(raw_sampled, min_sampled, max_sampled)
        self.domain_probs = effective / effective.sum()

        self.num_samples = (total // world_size) * world_size
        self.num_per_rank = self.num_samples // world_size

        print(f"\nBalancedDomainSampler (t={temperature}, max={max_repeat}x, min={min_coverage*100:.0f}%):")
        for d, p, c in sorted(zip(self.domains, self.domain_probs, counts), key=lambda x: -x[2]):
            sampled = p * self.num_samples
            rep = sampled / c
            print(f"  {d:20s}: {c:>10,} -> {sampled:>10,.0f} sampled ({rep:.1f}x)")
        print(f"  Total per epoch: {self.num_samples:,} ({self.num_per_rank:,} per rank)\n")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        chosen_domains = rng.choice(
            len(self.domains), size=self.num_samples, p=self.domain_probs
        )

        indices = np.empty(self.num_samples, dtype=np.int64)
        for i, d_idx in enumerate(chosen_domains):
            domain = self.domains[d_idx]
            pool = self.domain_indices[domain]
            indices[i] = pool[rng.randint(len(pool))]

        rank_indices = indices[self.rank::self.world_size]
        return iter(rank_indices.tolist())

    def __len__(self):
        return self.num_per_rank
