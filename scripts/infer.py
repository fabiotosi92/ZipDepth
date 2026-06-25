"""
ZipDepth inference — single image, folder, or video.

Usage:
    # Single image
    python scripts/infer.py --checkpoint checkpoints/final_model.pth --input image.jpg

    # Folder of images
    python scripts/infer.py --checkpoint checkpoints/final_model.pth --input /path/to/images/ --output /path/to/output/

    # Video
    python scripts/infer.py --checkpoint checkpoints/final_model.pth --input video.mp4

    # FP16 for speed
    python scripts/infer.py --checkpoint checkpoints/final_model.pth --input image.jpg --fp16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from zipdepth.inference.predictor import DepthInference


def main():
    parser = argparse.ArgumentParser(description='ZipDepth Inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input: image file, folder of images, or video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (auto-named if omitted)')
    parser.add_argument('--variant', type=str, default='base',
                        choices=['small', 'base', 'large', 'giant'],
                        help='Model variant (must match checkpoint)')
    parser.add_argument('--input-size', type=int, default=384,
                        help='Minimum shorter-side size for model input')
    parser.add_argument('--ensure-multiple-of', type=int, default=32,
                        help='Round input dimensions to multiples of this')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision (CUDA only, faster)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster inference (CUDA only, slow first run)')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--warmup-iters', type=int, default=3,
                        help='Warmup iterations to avoid CUDA cold-start (0 = skip, default: 3)')
    parser.add_argument('--save-raw', action='store_true',
                        help='Also save raw depth as .npy')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit number of video frames to process')
    parser.add_argument('--output-size', type=int, default=None,
                        help='Output video height in pixels (default: model input size). '
                             'Use e.g. 1080 for 1080p output.')
    parser.add_argument('--no-colormap', action='store_true',
                        help='Skip colorized visualization (no JPEG output). '
                             'Use with --save-raw to get only raw .npy depth files.')
    parser.add_argument('--npu', action='store_true',
                        help='Use NPU-compatible upsampling (required for zipdepth_base_npu.pth). '
                             'Replaces torch.nn.Unfold with an NPU-friendly equivalent.')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
                        help='Image file extensions to scan for (folder mode)')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    predictor = DepthInference(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        device=args.device,
        use_half=args.fp16,
        use_compile=args.compile,
        compile_mode=args.compile_mode,
        input_size=args.input_size,
        ensure_multiple_of=args.ensure_multiple_of,
        warmup_iters=args.warmup_iters,
        upsample_unfold=not args.npu,
    )

    if input_path.is_dir():
        predictor.predict_batch(
            input_dir=str(input_path),
            output_dir=args.output,
            save_raw=args.save_raw,
            colorize=not args.no_colormap,
            extensions=tuple(args.extensions),
        )
    elif input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        predictor.predict_video(
            video_path=str(input_path),
            output_path=args.output,
            max_frames=args.max_frames,
            output_height=args.output_size,
            save_raw=args.save_raw,
            colorize=not args.no_colormap,
        )
    else:
        predictor.predict_image(
            image_path=str(input_path),
            output_path=args.output,
            save_raw=args.save_raw,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
