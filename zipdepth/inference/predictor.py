"""
ZipDepth inference — single image, folder, or video.
Aspect ratio preserving.

"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional

import threading
from queue import Queue

try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False

from zipdepth.model.architecture import create_model
from zipdepth.utils.colormap import depth_to_colormap
from zipdepth.utils.model_utils import strip_state_dict_prefixes


@dataclass
class InferenceStats:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


def make_divisible(value: float, divisor: int) -> int:
    """Round to nearest multiple of divisor (at least divisor)."""
    return max(divisor, int(round(value / divisor) * divisor))


class DepthInference:
    def __init__(
        self,
        checkpoint_path: str,
        variant: str = 'base',
        device: str = 'cuda',
        use_half: bool = False,
        use_compile: bool = False,
        compile_mode: str = 'reduce-overhead',
        input_size: int = 384,
        ensure_multiple_of: int = 32,
        warmup_iters: int = 3,
        upsample_unfold: bool = True,
    ):
        self.device = device
        self.variant = variant
        self.use_half = use_half and device == 'cuda'
        self.use_compile = use_compile and device == 'cuda'
        self.compile_mode = compile_mode
        self._warmup_iters = warmup_iters
        self.input_size = input_size
        self.ensure_multiple_of = ensure_multiple_of
        self.dtype = torch.float16 if self.use_half else torch.float32

        print(f"Loading ZipDepth-{variant}...")
        self.model = create_model(variant=variant, upsample_unfold=upsample_unfold)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        sd = checkpoint.get('model_state_dict', checkpoint)
        sd = strip_state_dict_prefixes(sd)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"  Ignored training-only keys: {unexpected}")
        if missing:
            print(f"  Warning — missing keys (random init): {missing}")

        self.model = self.model.to(device).eval()

        print("Fusing model for inference...")
        self.model.fuse_for_inference()

        if self.use_half:
            self.model = self.model.half()

        if device == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
            torch.set_float32_matmul_precision('high')  # TF32 Tensor Cores for FP32 matmul

        if self.use_compile:
            print(f"Compiling model (mode={compile_mode}) ...")
            self.model = torch.compile(self.model, mode=compile_mode, fullgraph=False)

        # TurboJPEG
        self.jpeg_decoder = None
        if TURBOJPEG_AVAILABLE:
            try:
                self.jpeg_decoder = TurboJPEG()
                print("Using TurboJPEG")
            except Exception:
                pass

        # Pre-allocated buffers
        self._resize_buf: Optional[np.ndarray] = None
        self._resize_buf_shape: Optional[Tuple[int, int]] = None
        self._pin_buf_u8: Optional[torch.Tensor] = None      # pinned uint8 for H2D
        self._pin_buf_u8_shape: Optional[Tuple[int, int]] = None
        self._gpu_buf_u8: Optional[torch.Tensor] = None       # GPU uint8
        self._gpu_buf_u8_shape: Optional[Tuple[int, int]] = None
        self._gpu_buf_float: Optional[torch.Tensor] = None    # GPU float [1,3,H,W]
        self._gpu_buf_float_shape: Optional[Tuple[int, int]] = None

        precision  = "FP16" if self.use_half else "FP32"
        mode_parts = [f"fused {precision}"]
        if self.use_compile:
            mode_parts.append(f"compiled/{compile_mode}")
        loader_name = "TurboJPEG" if self.jpeg_decoder else "cv2"
        print(f"  Device:   {device}  |  {', '.join(mode_parts)}  |  loader: {loader_name}")
        print(f"  Input:    shorter side >= {input_size}, aligned to {ensure_multiple_of}")

        self._warmup(num_iters=30 if self.use_compile else self._warmup_iters)

    def _warmup(self, num_iters: int):
        if num_iters == 0:
            print("  Warmup:   skipped")
            return
        h = make_divisible(self.input_size, self.ensure_multiple_of)
        w = make_divisible(self.input_size * 16 / 9, self.ensure_multiple_of)
        dummy = torch.randn(1, 3, h, w, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            for _ in range(num_iters):
                self.model(dummy)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        print(f"  Warmup:   {num_iters} iters at {w}×{h}  — ready")

    def _compute_target_size(self, h: int, w: int) -> Tuple[int, int]:
        """Compute model input dimensions preserving aspect ratio."""
        scale = self.input_size / min(h, w)
        new_h = make_divisible(h * scale, self.ensure_multiple_of)
        new_w = make_divisible(w * scale, self.ensure_multiple_of)
        return new_h, new_w

    def _ensure_buffers(self, new_h: int, new_w: int):
        """Allocate/reallocate all buffers if resolution changed."""
        shape = (new_h, new_w)
        if self._resize_buf_shape == shape:
            return  # All buffers already correct size

        self._resize_buf = np.empty((new_h, new_w, 3), dtype=np.uint8)
        self._resize_buf_shape = shape

        if self.device == 'cuda':
            # Pinned CPU buffer: [H, W, 3] uint8, contiguous, for fast H2D
            self._pin_buf_u8 = torch.empty(new_h, new_w, 3, dtype=torch.uint8, pin_memory=True)
            self._pin_buf_u8_shape = shape

        # GPU uint8 staging buffer: [H, W, 3]
        self._gpu_buf_u8 = torch.empty(new_h, new_w, 3, dtype=torch.uint8, device=self.device)
        self._gpu_buf_u8_shape = shape

        # GPU float model input: [1, 3, H, W] — channels_last on CUDA for Tensor Core efficiency
        mf = torch.channels_last if self.device == 'cuda' else torch.contiguous_format
        self._gpu_buf_float = torch.empty(1, 3, new_h, new_w, dtype=self.dtype,
                                          device=self.device).to(memory_format=mf)
        self._gpu_buf_float_shape = shape

    def image2tensor(self, raw_image: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            raw_image: BGR uint8 from cv2.imread

        Returns:
            tensor: [1, 3, H_new, W_new] in [0, 1]
            h: original height
            w: original width
        """
        h, w = raw_image.shape[:2]
        new_h, new_w = self._compute_target_size(h, w)
        self._ensure_buffers(new_h, new_w)

        # 1. Resize BGR uint8 into pre-allocated buffer
        cv2.resize(raw_image, (new_w, new_h), dst=self._resize_buf, interpolation=cv2.INTER_LINEAR)

        # 2. Upload uint8 to GPU — zero intermediate float copies on CPU
        if self.device == 'cuda':
            self._pin_buf_u8.numpy()[:] = self._resize_buf
            self._gpu_buf_u8.copy_(self._pin_buf_u8, non_blocking=True)
        else:
            self._gpu_buf_u8.copy_(torch.from_numpy(self._resize_buf))

        # 3. All conversions on GPU
        rgb_gpu = self._gpu_buf_u8[:, :, [2, 1, 0]]
        self._gpu_buf_float[0] = rgb_gpu.permute(2, 0, 1).to(dtype=self.dtype)
        self._gpu_buf_float.div_(255.0)

        return self._gpu_buf_float, h, w

    @torch.no_grad()
    def infer_image(self, raw_image: np.ndarray) -> np.ndarray:
        """
        Predict depth from BGR image.

        Args:
            raw_image: BGR uint8 from cv2.imread

        Returns:
            depth [H, W] at original resolution, float32 numpy
        """
        image, h, w = self.image2tensor(raw_image)

        depth = self.model(image)

        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)

        return depth[0, 0].cpu().float().numpy()

    def _load_bgr(self, path: str) -> np.ndarray:
        if self.jpeg_decoder and path.lower().endswith(('.jpg', '.jpeg')):
            with open(path, 'rb') as f:
                jpeg_data = f.read()
            return self.jpeg_decoder.decode(jpeg_data, pixel_format=TJPF_BGR)

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load: {path}")
        return img

    def predict_image(
        self,
        image_path: str,
        output_path: str = None,
        save_raw: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, InferenceStats]:
        stats = InferenceStats()

        raw_image = self._load_bgr(image_path)
        h, w = raw_image.shape[:2]
        new_h, new_w = self._compute_target_size(h, w)

        if self.device == 'cuda' and self._warmup_iters > 0:
            _dummy = torch.randn(1, 3, new_h, new_w, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                self.model(_dummy)
            torch.cuda.synchronize()
            del _dummy

        total_start = time.time()

        if verbose:
            scale = self.input_size / min(h, w)
            loader_name = "TurboJPEG" if self.jpeg_decoder else "cv2"
            sep = "  " + "─" * 54
            print(f"\nProcessing {Path(image_path).name}")
            print(f"  Input:          {w}×{h}  →  model input: {new_w}×{new_h}"
                  f"  (scale {scale:.2f}×, loader: {loader_name})")
            print(sep)

        # Preprocess
        pre_start = time.time()
        image_tensor, _, _ = self.image2tensor(raw_image)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        stats.preprocess_ms = (time.time() - pre_start) * 1000

        # Inference
        if self.device == 'cuda':
            torch.cuda.synchronize()
        inf_start = time.time()
        with torch.no_grad():
            depth = self.model(image_tensor)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        stats.inference_ms = (time.time() - inf_start) * 1000

        # Postprocess
        post_start = time.time()
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        depth_np = depth.cpu().float().numpy()
        if self.device == 'cuda':
            torch.cuda.synchronize()
        stats.postprocess_ms = (time.time() - post_start) * 1000

        stats.total_ms = (time.time() - total_start) * 1000

        if verbose:
            precision = "FP16" if self.use_half else "FP32"
            backend = ("compiled/" + self.compile_mode) if self.use_compile else "fused"
            sep = "  " + "─" * 54
            gpu_fps = 1000.0 / stats.inference_ms if stats.inference_ms > 0 else 0.0
            print(f"  Preprocessing:  {stats.preprocess_ms:6.1f} ms"
                  f"  (resize → H2D → normalize)")
            fwd_device = "GPU" if self.device == 'cuda' else "CPU"
            print(f"  Inference:      {stats.inference_ms:6.1f} ms  →  {gpu_fps:.0f} FPS"
                  f"  ({fwd_device} forward, {backend} {precision}, synchronized)")
            print(f"  Postprocessing: {stats.postprocess_ms:6.1f} ms"
                  f"  (bilinear upsample → D2H)")
            print(sep)
            print(f"  Total:          {stats.total_ms:6.1f} ms  →  {stats.fps:.0f} FPS  (end-to-end)")
            print(f"  Note: single-image timing — run benchmark.py for steady-state throughput")
            print(sep)
            print(f"  Depth range:    [{depth_np.min():.3f}, {depth_np.max():.3f}]")

        # Save
        if output_path is None:
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_depth.jpg"

        depth_colored = depth_to_colormap(depth_np, cmap='Spectral')
        cv2.imwrite(str(output_path), depth_colored)

        if verbose:
            print(f"  Saved to: {output_path}")

        if save_raw:
            raw_path = Path(output_path).with_suffix('.npy')
            np.save(raw_path, depth_np)
            if verbose:
                print(f"  Raw saved to: {raw_path}")

        return depth_np, stats

    def predict_batch(
        self,
        input_dir: str,
        output_dir: str = None,
        save_raw: bool = False,
        colorize: bool = True,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
    ):
        input_path = Path(input_dir)
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        image_files = sorted(set(image_files))

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images")

        if output_dir is None:
            output_dir = input_path / 'depth_output'
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        failed = []

        # ── Prefetch: load images on a background thread ─────────────────────
        prefetch_queue: Queue = Queue(maxsize=2)

        def _prefetch_worker(files):
            for p in files:
                try:
                    img = self._load_bgr(str(p))
                    prefetch_queue.put((p, img, None))
                except Exception as e:
                    prefetch_queue.put((p, None, e))
            prefetch_queue.put(None)

        loader_thread = threading.Thread(target=_prefetch_worker, args=(image_files,), daemon=True)
        loader_thread.start()

        # ── Writer: colormap + disk write on a background thread ─────────────
        write_queue: Queue = Queue(maxsize=2)
        t_colormap = []
        t_write = []

        def _writer_worker():
            while True:
                item = write_queue.get()
                if item is None:
                    break
                depth_cpu, out_file = item
                try:
                    depth_np = depth_cpu.numpy()  # blocks until async D2H finishes
                    if colorize:
                        _t = time.time()
                        depth_colored = depth_to_colormap(depth_np, cmap='Spectral')
                        t_colormap.append((time.time() - _t) * 1000)
                        _t = time.time()
                        cv2.imwrite(str(out_file), depth_colored)
                        t_write.append((time.time() - _t) * 1000)
                    if save_raw:
                        np.save(str(out_file.with_suffix('.npy')), depth_np)
                except Exception as e:
                    failed.append((out_file.name, str(e)))

        writer_thread = threading.Thread(target=_writer_worker, daemon=True)
        writer_thread.start()

        # ── GPU loop: no blocking syncs ───────
        gpu_events = []   # (start_ev, end_ev) per frame for model-only timing (CUDA)
        cpu_times = []    # per-frame model time in ms (CPU fallback)
        frame_count = 0
        t_wall_start = time.time()
        t_stamps = []     

        pbar = tqdm(total=len(image_files), desc="Processing")
        with torch.no_grad():
            while True:
                item = prefetch_queue.get()
                if item is None:
                    break
                img_path, raw_image, load_err = item
                if load_err is not None:
                    failed.append((img_path.name, str(load_err)))
                    pbar.update(1)
                    continue

                try:
                    h, w = raw_image.shape[:2]

                    image_tensor, _, _ = self.image2tensor(raw_image)

                    if self.device == 'cuda':
                        ev_start = torch.cuda.Event(enable_timing=True)
                        ev_end   = torch.cuda.Event(enable_timing=True)
                        ev_start.record()
                        depth = self.model(image_tensor)
                        ev_end.record()
                        gpu_events.append((ev_start, ev_end))
                    else:
                        _t0 = time.perf_counter()
                        depth = self.model(image_tensor)
                        cpu_times.append((time.perf_counter() - _t0) * 1000)

                    if depth.dim() == 2:
                        depth = depth.unsqueeze(0).unsqueeze(0)
                    elif depth.dim() == 3:
                        depth = depth.unsqueeze(1)
                    depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)

                    depth_cpu = depth[0, 0].float().to('cpu', non_blocking=True)

                    out_file = output_path / f"{img_path.stem}_depth.jpg"
                    write_queue.put((depth_cpu, out_file)) 

                    frame_count += 1
                    t_stamps.append(time.time())
                    if len(t_stamps) > 30:
                        t_stamps.pop(0)
                    fps = (len(t_stamps) - 1) / (t_stamps[-1] - t_stamps[0] + 1e-9) if len(t_stamps) > 1 else 0
                    pbar.set_postfix({'FPS': f'{fps:.0f}'})

                except Exception as e:
                    failed.append((img_path.name, str(e)))

                pbar.update(1)

        pbar.close()
        loader_thread.join()
        write_queue.put(None)
        writer_thread.join()

        t_wall = time.time() - t_wall_start

        if self.device == 'cuda':
            if gpu_events:
                torch.cuda.synchronize()
            inf_times = [s.elapsed_time(e) for s, e in gpu_events] if gpu_events else [0.0]
            timing_src = "GPU, CUDA events"
        else:
            inf_times = cpu_times or [0.0]
            timing_src = "CPU, perf_counter"
        avg_gpu = float(np.mean(inf_times))
        wall_fps = frame_count / t_wall if t_wall > 0 else 0

        print(f"\n{'='*60}")
        print(f"Processed: {frame_count}/{len(image_files)}")
        print(f"Failed:    {len(failed)}")
        print(f"\n  PROFILING BREAKDOWN (avg ms, async pipeline):")
        print(f"  {'Load (disk->CPU):':<28} [prefetched in background]")
        print(f"  {'Model inference:':<28} {avg_gpu:>6.1f} ms  →  {1000/avg_gpu:.0f} FPS  ({timing_src})")
        if colorize and t_colormap:
            print(f"  {'Colormap:':<28} {np.mean(t_colormap):>6.1f} ms  (writer thread, overlapped)")
        if t_write:
            print(f"  {'Write (disk):':<28} {np.mean(t_write):>6.1f} ms  (writer thread, overlapped)")
        print(f"  {'─'*44}")
        print(f"  {'Wall-clock throughput:':<28}          {wall_fps:.0f} FPS  (end-to-end)")
        print(f"\nOutput: {output_path}")
        if failed:
            for name, err in failed[:5]:
                print(f"  x {name}: {err}")

    def predict_video(
        self,
        video_path: str,
        output_path: str = None,
        max_frames: int = None,
        output_height: int = None,
        save_raw: bool = False,
        colorize: bool = True,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        target_h, target_w = self._compute_target_size(height, width)

        if output_height is not None:
            scale = output_height / height
            out_h = output_height
            out_w = make_divisible(width * scale, 2)
        else:
            out_h, out_w = target_h, target_w

        print(f"Video: {Path(video_path).name}")
        print(f"  Original:    {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"  Model input: {target_w}x{target_h}")
        print(f"  Output res:  {out_w}x{out_h}" + (f"  (combined: {out_w*2}x{out_h})" if colorize else ""))

        if output_path is None:
            output_path = Path(video_path).parent / f"{Path(video_path).stem}_depth.mp4"

        frames_dir = Path(output_path).parent / f"{Path(video_path).stem}_depth_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Frames dir:  {frames_dir}")

        self._ensure_buffers(target_h, target_w)
        frame_vis_buf = np.empty((out_h, out_w, 3), dtype=np.uint8)

        # ── Writer thread: colormap + JPEG/npy save, overlapped with GPU ──────
        write_queue: Queue = Queue(maxsize=2)

        def _writer_worker():
            combined_buf = np.empty((out_h, out_w * 2, 3), dtype=np.uint8) if colorize else None
            while True:
                item = write_queue.get()
                if item is None:
                    break
                depth_cpu, frame_vis, fidx = item
                depth_np = depth_cpu.numpy()  # blocks until async D2H finishes
                if colorize:
                    depth_colored = depth_to_colormap(depth_np, cmap='Spectral')
                    combined_buf[:, :out_w] = frame_vis
                    combined_buf[:, out_w:] = depth_colored
                    cv2.imwrite(str(frames_dir / f"{fidx:06d}.jpg"), combined_buf,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                if save_raw:
                    np.save(str(frames_dir / f"{fidx:06d}.npy"), depth_np)

        writer_thread = threading.Thread(target=_writer_worker, daemon=True)
        writer_thread.start()

        # ── Pass 1: GPU loop, async D2H, no blocking syncs ────────────────────
        gpu_events = []
        cpu_times = []    # CPU fallback timing (no CUDA events on CPU)
        frame_idx = 0
        t_stamps = []
        t_wall_start = time.time()

        try:
            pbar = tqdm(total=total_frames, desc="Inference")
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret or (max_frames and frame_idx >= max_frames):
                        break

                    image_tensor, _, _ = self.image2tensor(frame)

                    if self.device == 'cuda':
                        ev_start = torch.cuda.Event(enable_timing=True)
                        ev_end   = torch.cuda.Event(enable_timing=True)
                        ev_start.record()
                        depth = self.model(image_tensor)
                        ev_end.record()
                        gpu_events.append((ev_start, ev_end))
                    else:
                        _t0 = time.perf_counter()
                        depth = self.model(image_tensor)
                        cpu_times.append((time.perf_counter() - _t0) * 1000)

                    if depth.dim() == 2:
                        depth = depth.unsqueeze(0).unsqueeze(0)
                    elif depth.dim() == 3:
                        depth = depth.unsqueeze(1)
                    depth = F.interpolate(depth, (out_h, out_w), mode="bilinear", align_corners=True)

                    depth_cpu = depth[0, 0].float().to('cpu', non_blocking=True)

                    if colorize:
                        cv2.resize(frame, (out_w, out_h), dst=frame_vis_buf, interpolation=cv2.INTER_LINEAR)
                        frame_vis_copy = frame_vis_buf.copy()
                    else:
                        frame_vis_copy = None

                    t_stamps.append(time.time())
                    if len(t_stamps) > 30:
                        t_stamps.pop(0)

                    write_queue.put((depth_cpu, frame_vis_copy, frame_idx))

                    frame_idx += 1
                    pbar.update(1)
                    fps_val = (len(t_stamps) - 1) / (t_stamps[-1] - t_stamps[0] + 1e-9) if len(t_stamps) > 1 else 0
                    pbar.set_postfix({'FPS': f'{fps_val:.0f}'})

            pbar.close()
        finally:
            cap.release()

        write_queue.put(None)
        writer_thread.join()

        t_wall = time.time() - t_wall_start
        if self.device == 'cuda':
            if gpu_events:
                torch.cuda.synchronize()
            inf_times = [s.elapsed_time(e) for s, e in gpu_events] if gpu_events else [0.0]
            timing_src = "GPU, CUDA events"
        else:
            inf_times = cpu_times or [0.0]
            timing_src = "CPU, perf_counter"
        avg_gpu = float(np.mean(inf_times))
        wall_fps = frame_idx / t_wall if t_wall > 0 else 0

        print(f"\n{'='*50}")
        print(f"Frames: {frame_idx}")
        print(f"  Model inference:  {avg_gpu:.1f} ms  →  {1000/avg_gpu:.0f} FPS  ({timing_src})")
        print(f"  Wall-clock:       {1000/wall_fps:.1f} ms  →  {wall_fps:.0f} FPS  (end-to-end, async)")

        # ── Pass 2: encode video from saved frames ────────────────────────────
        if colorize:
            frame_files = sorted(frames_dir.glob("*.jpg"))
            if frame_files:
                first = cv2.imread(str(frame_files[0]))
                enc_h, enc_w = first.shape[:2]
                print(f"\nEncoding video ({len(frame_files)} frames @ {enc_w}x{enc_h}) ...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (enc_w, enc_h))
                try:
                    out.write(first)
                    for fpath in tqdm(frame_files[1:], desc="Encoding video"):
                        img = cv2.imread(str(fpath))
                        if img is not None:
                            out.write(img)
                finally:
                    out.release()
            print(f"Output: {output_path}")

        print(f"Frames: {frames_dir}")
