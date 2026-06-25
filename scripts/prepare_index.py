"""
Build and convert a ZipDepth dataset index.

Workflow
--------
1. Scan one or more RGB/depth directory pairs and produce a JSON index.
2. Convert that JSON index to numpy memmap files (fast I/O at millions of samples).

Both steps can be run separately or together.

Step 1 — build JSON from directory pairs
-----------------------------------------
    python scripts/prepare_index.py build \
        --domains MyDataset /path/to/rgb /path/to/depth \
        --domains AnotherSet /path/to/rgb2 /path/to/depth2 \
        --output dataset_index.json

    # Or pass a YAML/JSON config file:
    python scripts/prepare_index.py build --config domains.yaml --output dataset_index.json

    Config file format (YAML or JSON):
        MyDataset:
            rgb:   /path/to/rgb
            depth: /path/to/depth
        AnotherSet:
            rgb:   /path/to/rgb2
            depth: /path/to/depth2

    Folder structure expected under each root:
        rgb_root/
            image001.jpg
            subdir/image002.png
            ...
        depth_root/         <- same relative structure, any depth extension
            image001.png    <- PNG uint16
            subdir/image002.npy

Step 2 — convert JSON index to numpy memmaps
---------------------------------------------
    python scripts/prepare_index.py convert --input dataset_index.json

    Produces:
        dataset_index_rgb.npy
        dataset_index_depth.npy
        dataset_index_domain.npy
        dataset_index_metadata.json

    The ZipDepth dataloader auto-detects these files from the JSON path.

Both steps at once
------------------
    python scripts/prepare_index.py build \
        --domains MyDataset /rgb /depth \
        --output dataset_index.json \
        --convert
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm


RGB_EXTENSIONS  = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.webp'}
DEPTH_EXTENSIONS = {'.png', '.npy', '.npz'}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _scan_domain(domain: str, rgb_root: Path, depth_root: Path):
    if not rgb_root.exists():
        print(f"  [SKIP] RGB root not found: {rgb_root}")
        return [], 0, 0
    if not depth_root.exists():
        print(f"  [SKIP] Depth root not found: {depth_root}")
        return [], 0, 0

    samples, missing = [], 0
    n_root = len(rgb_root.parts)

    for rgb_path in tqdm(rgb_root.rglob('*'), desc=domain, unit='img'):
        if rgb_path.suffix not in RGB_EXTENSIONS:
            continue
        rel = rgb_path.parts[n_root:]
        depth_path = None
        for ext in DEPTH_EXTENSIONS:
            candidate = depth_root.joinpath(*rel).with_suffix(ext)
            if candidate.exists():
                depth_path = candidate
                break
        if depth_path is None:
            missing += 1
            continue
        samples.append({'rgb': str(rgb_path), 'depth': str(depth_path), 'domain': domain})

    print(f"  {domain}: {len(samples):,} pairs  ({missing:,} missing depth)")
    return samples, len(samples), missing


def build_index(domain_roots: dict, output: Path, seed: int = 42):
    random.seed(seed)
    all_samples, per_domain = [], {}

    for domain, roots in domain_roots.items():
        samples, valid, missing = _scan_domain(
            domain, Path(roots['rgb']), Path(roots['depth'])
        )
        all_samples.extend(samples)
        per_domain[domain] = {'valid': valid, 'missing': missing}

    random.shuffle(all_samples)

    index = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'total_samples': len(all_samples),
        'domains': list(domain_roots.keys()),
        'per_domain': per_domain,
        'samples': all_samples,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(index, f)

    print(f"\nIndex written: {output}  ({len(all_samples):,} total samples)")
    return index


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

def convert_index(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    if not samples:
        print("ERROR: 'samples' list is empty.")
        sys.exit(1)

    print(f"Converting {len(samples):,} samples to numpy memmap...")

    rgb_list    = [s['rgb']                 for s in samples]
    depth_list  = [(s.get('depth') or '')   for s in samples]
    domain_list = [s['domain']              for s in samples]

    max_rgb    = max(len(p) for p in rgb_list)    + 1
    max_depth  = max(len(p) for p in depth_list)  + 1
    max_domain = max(len(d) for d in domain_list) + 1

    prefix = str(json_path.with_suffix(''))
    np.save(f'{prefix}_rgb.npy',    np.array(rgb_list,    dtype=f'S{max_rgb}'))
    np.save(f'{prefix}_depth.npy',  np.array(depth_list,  dtype=f'S{max_depth}'))
    np.save(f'{prefix}_domain.npy', np.array(domain_list, dtype=f'S{max_domain}'))

    meta = {
        'version':       data.get('version', '1.0'),
        'created_at':    data.get('created_at', ''),
        'total_samples': len(samples),
    }
    with open(f'{prefix}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {prefix}_{{rgb,depth,domain}}.npy + _metadata.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_domain_config(path: str) -> dict:
    p = Path(path)
    if p.suffix in {'.yaml', '.yml'}:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f)
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Build and convert ZipDepth dataset index.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # -- build --
    p_build = sub.add_parser('build', help='Scan directories and build JSON index.')
    src = p_build.add_mutually_exclusive_group(required=True)
    src.add_argument('--domains', metavar=('NAME', 'RGB_DIR', 'DEPTH_DIR'),
                     nargs=3, action='append',
                     help='Domain name + rgb dir + depth dir. Repeatable.')
    src.add_argument('--config', metavar='FILE',
                     help='YAML or JSON file mapping domain names to {rgb, depth} roots.')
    p_build.add_argument('--output', required=True, metavar='FILE',
                         help='Output JSON index path.')
    p_build.add_argument('--seed', type=int, default=42)
    p_build.add_argument('--convert', action='store_true',
                         help='Also convert to numpy memmaps after building.')

    # -- convert --
    p_conv = sub.add_parser('convert', help='Convert an existing JSON index to numpy memmaps.')
    p_conv.add_argument('--input', required=True, metavar='FILE',
                        help='Path to dataset_index.json.')

    args = parser.parse_args()

    if args.cmd == 'build':
        if args.config:
            domain_roots = _load_domain_config(args.config)
        else:
            domain_roots = {name: {'rgb': rgb, 'depth': depth}
                            for name, rgb, depth in args.domains}
        build_index(domain_roots, Path(args.output), seed=args.seed)
        if args.convert:
            convert_index(Path(args.output))

    elif args.cmd == 'convert':
        convert_index(Path(args.input))


if __name__ == '__main__':
    main()
