"""
ZipDepth benchmark — parameters, FLOPs, and latency across deployment backends.

Usage:
    python scripts/benchmark.py --height 384 --width 384
    python scripts/benchmark.py --ckpt checkpoints/model.pth
    python scripts/benchmark.py --fp16
    python scripts/benchmark.py --fp16 --channels-last
    python scripts/benchmark.py --compile-mode max-autotune
"""

import sys
import copy
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from zipdepth.model.architecture import create_model
from zipdepth.utils.model_utils import fuse_remaining_conv_bn, strip_state_dict_prefixes


# =============================================================================
# HELPERS
# =============================================================================

def _measure(model, x, warmup: int, measure: int) -> dict:
    device = x.device.type
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
    lats = []
    if device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            for _ in range(measure):
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                lats.append(start.elapsed_time(end))
    else:
        with torch.inference_mode():
            for _ in range(measure):
                t0 = time.perf_counter()
                model(x)
                lats.append((time.perf_counter() - t0) * 1000)
    arr = np.array(lats)
    q1, q3 = np.percentile(arr, [25, 75])
    arr = arr[(arr >= q1 - 1.5 * (q3 - q1)) & (arr <= q3 + 1.5 * (q3 - q1))]
    mean = float(arr.mean())
    return dict(mean_ms=mean, std_ms=float(arr.std(ddof=1)),
                median_ms=float(np.median(arr)), p95_ms=float(np.percentile(arr, 95)),
                fps=1000.0 / mean, n=len(arr))


def _count_flops(model, x):
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model, x)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    except Exception:
        pass
    try:
        import thop
        flops, _ = thop.profile(model, inputs=(x,), verbose=False)
        return flops / 1e9
    except Exception:
        return None


def _count_flops_profiler(model, x) -> float:
    """FLOPs via torch.profiler (event-based, independent of fvcore/thop)."""
    from torch.profiler import profile as torch_profile, ProfilerActivity
    with torch.no_grad():
        with torch_profile(
            activities=[ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=True,
        ) as prof:
            model(x)
    return sum(getattr(e, 'flops', 0) for e in prof.key_averages()) / 1e9


def _flops_consistency_check(model, x, label: str) -> None:
    """Cross-check fvcore vs torch.profiler FLOPs.
    """
    fv = _count_flops(model, x)
    try:
        pr = _count_flops_profiler(model, x)
    except Exception as e:
        print(f"    [profiler] could not run: {e}")
        return

    if fv is None:
        print(f"    [check] fvcore/thop unavailable — profiler: {pr:.3f} G")
        return

    ratio = pr / fv if fv > 0 else float('inf')
    ok = abs(ratio - 2.0) <= 0.15 * 2.0

    if ok:
        status = f"OK  (ratio ~2x expected: fvcore=GMACs, profiler=GFLOPs)"
    else:
        status = f"SUSPICIOUS — ratio deviates from expected 2x; some ops may be missed"

    print(f"  FLOPs consistency check ({label})")
    print(f"    fvcore/thop    : {fv:.3f} G  (GMACs)")
    print(f"    torch.profiler : {pr:.3f} G  (GFLOPs, MACs x2)")
    print(f"    ratio          : {ratio:.3f}x  [{status}]")
    if not ok:
        print(f"    *** investigate: ops counted by profiler but not fvcore (or vice versa) ***")


def _print_model_info(model, model_fused, input_size, device, skip_flops_check=False):
    x = torch.randn(*input_size).to(device)

    total   = sum(p.numel() for p in model.parameters())
    fused_n = sum(p.numel() for p in model_fused.parameters())

    print(f"\n  Parameters")
    print(f"  {'─'*54}")
    print(f"  {'Unfused':<20s}  {total/1e6:>7.2f} M  "
          f"(FP32 {total*4/1024**2:.0f} MB  FP16 {total*2/1024**2:.0f} MB)")
    print(f"  {'Fused':<20s}  {fused_n/1e6:>7.2f} M  "
          f"(FP32 {fused_n*4/1024**2:.0f} MB  FP16 {fused_n*2/1024**2:.0f} MB)")

    # per-section breakdown
    try:
        enc,   dec   = model.encoder,       model.decoder
        enc_f, dec_f = model_fused.encoder,  model_fused.decoder

        def _cp(*mods): return sum(p.numel() for m in mods for p in m.parameters())

        sections = [
            ("Stem + Stage 1 (RepVGG)", _cp(enc.stem_half, enc.stem_quarter, enc.stage1),
                                        _cp(enc_f.stem_half, enc_f.stem_quarter, enc_f.stage1)),
            ("Stage 2 (+SPA)",          _cp(enc.down2, enc.stage2),
                                        _cp(enc_f.down2, enc_f.stage2)),
            ("Stage 3 (+SE, GCB)",      _cp(enc.down3, enc.stage3),
                                        _cp(enc_f.down3, enc_f.stage3)),
            ("Stage 4",                 _cp(enc.down4, enc.stage4),
                                        _cp(enc_f.down4, enc_f.stage4)),
            ("SPPF + Cross-Scale",      _cp(enc.spp, enc.cross_scale),
                                        _cp(enc_f.spp, enc_f.cross_scale)),
            ("FPN Fusion",              _cp(dec.proj4, dec.fuse3, dec.fuse2, dec.fuse1, dec.fuse_half),
                                        _cp(dec_f.proj4, dec_f.fuse3, dec_f.fuse2, dec_f.fuse1, dec_f.fuse_half)),
            ("Head + Convex Upsample",  _cp(dec.head_half, dec.convex_up),
                                        _cp(dec_f.head_half, dec_f.convex_up)),
        ]

        print(f"\n  {'Component':<33s} {'Unfused':>9s}  {'Fused':>9s}  {'Share':>6s}")
        print(f"  {'─'*65}")
        printed_dec = False
        for name, pt, pf in sections:
            if name in ("FPN Fusion", "Head + Convex Upsample") and not printed_dec:
                print(f"  --- Decoder ---")
                printed_dec = True
            share = f"{pf / fused_n * 100:.1f}%" if fused_n > 0 else "-"
            print(f"  {name:<33s} {pt/1e6:>8.2f}M  {pf/1e6:>8.2f}M  {share:>6s}")
        print(f"  {'─'*65}")
        print(f"  {'Total':<33s} {total/1e6:>8.2f}M  {fused_n/1e6:>8.2f}M")
    except Exception:
        pass

    # FLOPs — unfused and fused
    print(f"\n  GFLOPs  (conv/matmul MACs — standard CV convention)")
    print(f"  {'─'*54}")
    mf = copy.deepcopy(model).to(device).eval()
    uf = _count_flops(mf, x); del mf
    fu = _count_flops(model_fused, x)
    if uf is not None:
        print(f"  {'Unfused':<20s}  {uf:>7.2f} G")
    if fu is not None:
        print(f"  {'Fused':<20s}  {fu:>7.2f} G")
    if uf is None and fu is None:
        print("  install fvcore or thop to measure")

    if not skip_flops_check:
        print(f"\n  FLOPs cross-check  (fvcore vs torch.profiler, CPU)")
        print(f"  {'─'*54}")
        cpu_model = copy.deepcopy(model_fused).cpu().eval()
        x_cpu = x.cpu()
        _flops_consistency_check(cpu_model, x_cpu, label='fused, CPU')
        del cpu_model, x_cpu


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ZipDepth benchmark')
    parser.add_argument('--variant',      type=str, default='base',
                        choices=['nano', 'small', 'base', 'large', 'xlarge'])
    parser.add_argument('--global-mode',  type=str, default='balanced',
                        choices=['none', 'balanced', 'full'])
    parser.add_argument('--height',       type=int, default=384)
    parser.add_argument('--width',        type=int, default=384)
    parser.add_argument('--warmup',       type=int, default=50)
    parser.add_argument('--measure',      type=int, default=200)
    parser.add_argument('--ckpt',         type=str, default=None)
    parser.add_argument('--fp16',         action='store_true',
                        help='Also benchmark FP16 (CUDA only)')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--no-compile',     action='store_true',
                        help='Skip torch.compile')
    parser.add_argument('--channels-last',  action='store_true',
                        help='Use channels_last memory format (NHWC) — matches production predictor')
    parser.add_argument('--cpu-only',       action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    input_size = (1, 3, args.height, args.width)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # ── Load model ────────────────────────────────────────────────────────────
    model = create_model(variant=args.variant, global_mode=args.global_mode,
                         upsample_unfold=True)
    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)
        sd = strip_state_dict_prefixes(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"  Ignored training-only keys: {unexpected}")
        if missing:
            print(f"  Warning — missing keys (random init): {missing}")
    else:
        print("  WARNING: no checkpoint — random weights. Latency is valid; depth output is meaningless.")
    model = model.to(device).eval()

    x = torch.randn(*input_size).to(device)
    if args.channels_last and device == 'cuda':
        x = x.to(memory_format=torch.channels_last)

    # ── Header + model info (m_fused created temporarily, then freed) ─────────
    print(f"\n{'='*60}")
    print(f"  ZipDepth-{args.variant}  |  {args.width}x{args.height}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    m_fused_info = copy.deepcopy(model).to(device).eval()
    m_fused_info.fuse_for_inference()
    fuse_remaining_conv_bn(m_fused_info)
    _print_model_info(model, m_fused_info, input_size, device, skip_flops_check=True)
    del m_fused_info
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ── Benchmarks ────────────────────────────────────────────────────────────
    results = []

    def _run(label, m, x_in=None):
        r = _measure(m, x_in if x_in is not None else x, args.warmup, args.measure)
        results.append((label, r))
        print(f"  {label:<40s}  {r['mean_ms']:6.2f} ± {r['std_ms']:.2f} ms"
              f"  p95={r['p95_ms']:.2f}  {r['fps']:.1f} FPS")

    print(f"\n  Latency  ({args.warmup} warmup / {args.measure} measured, IQR-filtered)")
    print(f"  {'─'*66}")
    print(f"  {'Backend':<40s}  {'Mean':>6}   {'Std':>5}   {'p95':>5}   FPS")
    print(f"  {'─'*66}")

    cl = args.channels_last and device == 'cuda'
    cl_suffix = " +CL" if cl else ""

    def _to_cl(m):
        return m.to(memory_format=torch.channels_last) if cl else m

    # 1. Eager FP32
    m_eager = _to_cl(copy.deepcopy(model).to(device).eval())
    if device == 'cuda':
        with torch.inference_mode():
            for _ in range(100):
                m_eager(x); torch.cuda.synchronize()
    _run(f"Eager FP32{cl_suffix}", m_eager)
    del m_eager
    if device == 'cuda':
        torch.cuda.empty_cache()

    # 2. Fused FP32 — create m_fused now for all remaining backends
    m_fused = copy.deepcopy(model).to(device).eval()
    m_fused.fuse_for_inference()
    fuse_remaining_conv_bn(m_fused)
    m_fused = _to_cl(m_fused)

    _run(f"Fused FP32{cl_suffix}", m_fused)

    # 3. Fused FP16
    x_fp16 = None
    if args.fp16 and device == 'cuda':
        x_fp16 = x.half()
        m_fp16 = _to_cl(copy.deepcopy(m_fused).half())
        _run(f"Fused FP16{cl_suffix}", m_fp16, x_in=x_fp16)
        del m_fp16
        torch.cuda.empty_cache()

    # 4. torch.compile
    if not args.no_compile and device == 'cuda':
        print(f"\n  Compiling ({args.compile_mode}) — warming up ...")
        mc = torch.compile(_to_cl(copy.deepcopy(m_fused)), mode=args.compile_mode, fullgraph=False)
        with torch.inference_mode():
            for _ in range(max(args.warmup, 30)):
                mc(x); torch.cuda.synchronize()
        print(f"  {'─'*66}")
        print(f"  {'Backend':<40s}  {'Mean':>6}   {'Std':>5}   {'p95':>5}   FPS")
        print(f"  {'─'*66}")
        _run(f"compile FP32 ({args.compile_mode}){cl_suffix}", mc)

        if args.fp16:
            if x_fp16 is None:
                x_fp16 = x.half()
            mc_fp16 = torch.compile(_to_cl(copy.deepcopy(m_fused).half()), mode=args.compile_mode, fullgraph=False)
            with torch.inference_mode():
                for _ in range(max(args.warmup, 50)):
                    mc_fp16(x_fp16); torch.cuda.synchronize()
            _run(f"compile FP16 ({args.compile_mode}){cl_suffix}", mc_fp16, x_in=x_fp16)

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        best = min(results, key=lambda t: t[1]['mean_ms'])
        base_ms = results[0][1]['mean_ms']
        print(f"\n  Best: {best[0]}  →  {best[1]['mean_ms']:.2f} ms  "
              f"({best[1]['fps']:.0f} FPS,  {base_ms/best[1]['mean_ms']:.1f}x vs Eager)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
