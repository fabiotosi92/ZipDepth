"""
ZipDepth model export.

Exports a trained checkpoint to ONNX or TorchScript (trace / script).

Usage:
    # ONNX  (default 384x384)
    python scripts/export.py --ckpt checkpoints/model.pth --format onnx

    # ONNX at a specific resolution
    python scripts/export.py --ckpt checkpoints/model.pth --format onnx --height 512 --width 512

    # TorchScript — traced
    python scripts/export.py --ckpt checkpoints/model.pth --format torchscript

    # TorchScript — frozen (smaller, faster on CPU)
    python scripts/export.py --ckpt checkpoints/model.pth --format torchscript-frozen
"""

import sys
import copy
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from zipdepth.model.architecture import create_model
from zipdepth.utils.model_utils import fuse_remaining_conv_bn


# =============================================================================
# HELPERS
# =============================================================================

def load_model(ckpt_path: str, variant: str, global_mode: str, device: str,
               upsample_unfold: bool = True) -> nn.Module:
    model = create_model(variant=variant, global_mode=global_mode, upsample_unfold=upsample_unfold)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = ckpt.get('model_state_dict', ckpt)
    if next(iter(sd)).startswith('module.'):
        sd = {k[7:]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  Ignored training-only keys: {unexpected}")
    if missing:
        print(f"  Warning — missing keys: {missing}")
    model = model.to(device).eval()
    model.fuse_for_inference()
    fuse_remaining_conv_bn(model)
    return model


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_onnx(model: nn.Module, input_shape: tuple, output_path: str, opset: int = 17):
    """Export to ONNX with static shapes."""
    import torch.nn.functional as F

    import types
    H, W = input_shape[2], input_shape[3]
    s16 = (H // 16, W // 16)

    for m in model.modules():
        if type(m).__name__ == 'GlobalContextBlock':
            def _fwd(self_m, x, _h=s16[0], _w=s16[1]):
                ctx = F.avg_pool2d(x, kernel_size=(_h, _w))
                return x + self_m.transform(ctx)
            m.forward = types.MethodType(_fwd, m)

        elif type(m).__name__ == 'StripPoolingAttention':
            def _fwd(self_m, x):
                B, C, H, W = x.shape
                gate = self_m.gate_conv(
                    F.adaptive_avg_pool2d(x, (H, 1)) +
                    F.adaptive_avg_pool2d(x, (1, W))
                )
                return x * gate
            m.forward = types.MethodType(_fwd, m)

    cs = model.encoder.cross_scale
    def _cs_fwd(self_cs, x_high, x_low, _s=s16):
        lo = F.interpolate(self_cs.low_to_high(x_low), size=_s, mode='nearest')
        hi = F.avg_pool2d(self_cs.high_to_low(x_high), 2, 2)
        return x_high + lo * 0.3, x_low + hi * 0.3
    cs.forward = types.MethodType(_cs_fwd, cs)

    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Pre-export sanity check: output shape {out.shape}")

    raw_path = output_path.replace('.onnx', '_raw.onnx')
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, raw_path,
            input_names=['image'], output_names=['depth'],
            opset_version=opset,
            do_constant_folding=True,
        )
    print(f"  Raw ONNX: {Path(raw_path).stat().st_size / 1e6:.1f} MB")

    try:
        import onnx
        from onnxsim import simplify as _sim
        simplified, ok = _sim(onnx.load(raw_path))
        if ok:
            onnx.save(simplified, output_path)
            Path(raw_path).unlink()
            print(f"  Simplified ONNX: {Path(output_path).stat().st_size / 1e6:.1f} MB")
            return
    except ImportError:
        pass

    Path(raw_path).rename(output_path)
    print(f"  ONNX saved (install onnxsim for a smaller file)")


def export_torchscript(model: nn.Module, input_shape: tuple, output_path: str, frozen: bool):
    """Export to TorchScript via tracing."""
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        traced = torch.jit.trace(model.cpu(), dummy)
    if frozen:
        traced = torch.jit.freeze(traced)
        with torch.no_grad():
            for _ in range(3):
                traced(dummy)
    torch.jit.save(traced, output_path)
    print(f"  TorchScript {'(frozen) ' if frozen else ''}saved: "
          f"{Path(output_path).stat().st_size / 1e6:.1f} MB")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ZipDepth model export')
    parser.add_argument('--ckpt',         type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--format',       type=str, required=True,
                        choices=['onnx', 'torchscript', 'torchscript-frozen'],
                        help='Export format')
    parser.add_argument('--variant',      type=str, default='base',
                        choices=['small', 'base', 'large', 'giant'])
    parser.add_argument('--global-mode',  type=str, default='balanced',
                        choices=['none', 'balanced', 'full'])
    parser.add_argument('--height',       type=int, default=384)
    parser.add_argument('--width',        type=int, default=384)
    parser.add_argument('--output',       type=str, default=None,
                        help='Output file path (auto-named if omitted)')
    parser.add_argument('--opset',        type=int, default=17,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--npu',          action='store_true',
                        help='Use NPU-compatible upsampling (required for zipdepth_base_npu.pth). '
                             'Replaces torch.nn.Unfold with an NPU-friendly equivalent.')
    args = parser.parse_args()

    fmt = args.format
    ext = '.onnx' if fmt == 'onnx' else '.pt'
    suffix = '' if fmt == 'onnx' else ('_frozen' if fmt == 'torchscript-frozen' else '')
    default_name = f"zipdepth_{args.variant}_{args.width}x{args.height}{suffix}{ext}"
    output_path = args.output or str(Path(args.ckpt).parent / default_name)

    device = 'cpu'  # export always on CPU for maximum portability
    input_shape = (1, 3, args.height, args.width)

    print(f"\nExporting ZipDepth-{args.variant} → {fmt}  ({args.width}x{args.height})")
    model = load_model(args.ckpt, args.variant, args.global_mode, device,
                       upsample_unfold=not args.npu)

    if fmt == 'onnx':
        export_onnx(model, input_shape, output_path, opset=args.opset)
    else:
        export_torchscript(model, input_shape, output_path,
                           frozen=(fmt == 'torchscript-frozen'))

    print(f"\nDone → {output_path}\n")


if __name__ == '__main__':
    main()
