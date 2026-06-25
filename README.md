<div align="center">

<h1> ⚡ ZipDepth ⚡ </h1>

<h2>
  Bringing Lightweight Zero-Shot Monocular Depth <br> Anywhere, on Any Device
</h2>

# 🏛️ ECCV 2026

**[Fabio Tosi](https://fabiotosi92.github.io/) · [Luca Bartolomei](https://bartn8.github.io/) · [Matteo Poggi](https://mattpoggi.github.io/) · [Stefano Mattoccia](https://stefanomattoccia.github.io/)**

University of Bologna

<br/>

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](#)
&nbsp;&nbsp;
[![Supplementary](https://img.shields.io/badge/PDF-Supplementary-f97316?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](#)
&nbsp;&nbsp;
[![Project Page](https://img.shields.io/badge/Project-Page-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white)](https://zipdepth.github.io/)

</div>

---

## 📢 News

- **[Jul 2026]** — 🚀 Code and pretrained model released.
- **[Jun 2026]** — 🎉 ZipDepth accepted at ECCV 2026.

---

ZipDepth is a lightweight zero-shot monocular depth estimation model that achieves the best accuracy–efficiency trade-off among lightweight methods, approaching transformer-based foundation models at a fraction of their cost. It combines reparameterizable convolutions (RepVGG), efficient global attention with learnable tokens, and a compact FPN decoder — all designed to run fast on any device, from edge hardware to server GPUs.

<div align="center">
<img src="assets/figures/pareto.png" width="640"/>
</div>

---

## 🖼️ Qualitative Results

ZipDepth generalizes across diverse domains without any fine-tuning — nighttime driving, outdoor objects, close-up textures, and synthetic content.

<table align="center">
  <tr>
    <td align="center"><b>RGB</b></td>
    <td align="center"><b>Depth</b></td>
    <td align="center"><b>RGB</b></td>
    <td align="center"><b>Depth</b></td>
  </tr>
  <tr>
    <td><img src="assets/qualitative/driving_night_rgb.jpg" width="320"/></td>
    <td><img src="assets/qualitative/driving_night_depth.jpg" width="320"/></td>
    <td><img src="assets/qualitative/car_show_rgb.jpg" width="320"/></td>
    <td><img src="assets/qualitative/car_show_depth.jpg" width="320"/></td>
  </tr>
  <tr>
    <td><img src="assets/qualitative/close_up_rgb.jpg" width="320"/></td>
    <td><img src="assets/qualitative/close_up_depth.jpg" width="320"/></td>
    <td><img src="assets/qualitative/synthetic_indoor_rgb.jpg" width="320"/></td>
    <td><img src="assets/qualitative/synthetic_indoor_depth.jpg" width="320"/></td>
  </tr>
</table>

---

## 📊 Quantitative Results

<div align="center">
<img src="assets/figures/table_accuracy.png" width="860"/>
</div>

ZipDepth achieves state-of-the-art accuracy among lightweight embedded models on NYUv2, KITTI, ETH3D, ScanNet, and DIODE, while being significantly more efficient than large pretrained models.

---

## 🏗️ Architecture

<div align="center">
<img src="assets/figures/framework.png" width="860"/>
</div>

The encoder is organized in four hierarchical stages. Stages 1–2 use **RepVGG** reparameterizable blocks (3×3 + 1×1 + identity branches fused into a single 3×3 at inference) augmented with **Strip Pooling Attention** for horizontal/vertical context. Stage 3 adds **Squeeze-and-Excitation** channel attention and a **Global Context Block**. Stage 4 applies **Efficient Global Attention** via learnable global tokens.

The neck combines **SPPF** multi-scale pooling with a **Cross-Scale Fusion** module. The decoder is a lightweight FPN with a **Convex Upsampling** head for sub-pixel-accurate depth maps.

---

## 🔧 Installation

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.4, CUDA (recommended)

```bash
git clone https://github.com/fabiotosi92/ZipDepth
cd ZipDepth
pip install -r requirements.txt
pip install -e .
```

<details>
<summary>Optional dependencies</summary>

```bash
# Faster JPEG decoding (recommended)
pip install PyTurboJPEG

# FLOPs measurement in benchmark
pip install fvcore thop

# ONNX graph simplification during export
pip install onnx onnxsim
```
</details>

---

## 🗂️ Checkpoints

Pretrained checkpoints are included in the repository under `checkpoints/`:

| File | Upsampling | Params (fused) | Recommended for |
|------|-----------|---------------|-----------------|
| [`zipdepth_base.pth`](https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth) | Convex (unfold) | ~6.1 M | GPU / server |
| [`zipdepth_base_npu.pth`](https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base_npu.pth) | Convex (unfold-free) | ~6.1 M | NPU / mobile / CPU |

Both variants share identical encoder and decoder weights. The only difference is the final upsampling step:

- **`zipdepth_base.pth`** — uses `torch.nn.Unfold` for sub-pixel convex upsampling. Best accuracy on GPU.
- **`zipdepth_base_npu.pth`** — replaces the unfold operation with an NPU-friendly equivalent. Use this for ONNX export targeting mobile or edge devices.

---

## 🚀 Quick Start

### Single image

```bash
python scripts/infer.py \
  --checkpoint checkpoints/zipdepth_base.pth \
  --input assets/examples/im0.jpg \
  --input-size 384
```

### Folder of images

```bash
python scripts/infer.py \
  --checkpoint checkpoints/zipdepth_base.pth \
  --input assets/examples/imgs/ \
  --output output/depth/
```

### Video

```bash
python scripts/infer.py \
  --checkpoint checkpoints/zipdepth_base.pth \
  --input assets/examples/clip.mp4
```

---

## ⚡ Inference

```bash
python scripts/infer.py --checkpoint <ckpt> --input <path> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Path to `.pth` checkpoint |
| `--input` | required | Image file, folder, or video |
| `--output` | auto | Output path (auto-named if omitted) |
| `--input-size` | `512` | Shorter-side length for model input. Aspect ratio is preserved; the longer side scales proportionally. Rounded to the nearest multiple of 32 (e.g. `384`, `512`, `768`) |
| `--fp16` | off | FP16 precision (CUDA only) |
| `--compile` | off | `torch.compile` for faster steady-state throughput |
| `--npu` | off | Use NPU-compatible upsampling — required when loading `zipdepth_base_npu.pth` |
| `--save-raw` | off | Also save depth map as `.npy` |
| `--no-colormap` | off | Skip colorized JPEG — use with `--save-raw` for raw depth only |
| `--output-size` | model input | Output video height in pixels (e.g. `1080` for 1080p) |
| `--max-frames` | all | Limit number of video frames to process |

Batch and video inference use an **asynchronous pipeline**: GPU inference and CPU colormap/write are overlapped, keeping the GPU continuously busy. Timing is measured with CUDA events.

### More examples

```bash
# Raw depth maps only — maximum speed, no visualization
python scripts/infer.py \
  --checkpoint checkpoints/zipdepth_base.pth \
  --input assets/examples/imgs/ \
  --no-colormap --save-raw

# FP16 + torch.compile — highest throughput
python scripts/infer.py \
  --checkpoint checkpoints/zipdepth_base.pth \
  --input assets/examples/imgs/ \
  --fp16 --compile

# 4K video → 1080p output
python scripts/infer.py \
  --checkpoint checkpoints/zipdepth_base.pth \
  --input assets/examples/clip.mp4 \
  --output-size 1080
```

---

## 📦 Export

### ONNX

```bash
# GPU checkpoint (default)
python scripts/export.py \
  --ckpt checkpoints/zipdepth_base.pth \
  --format onnx \
  --height 384 --width 384

# NPU / mobile checkpoint — add --npu
python scripts/export.py \
  --ckpt checkpoints/zipdepth_base_npu.pth \
  --format onnx \
  --height 384 --width 384 --npu
```

Install `onnxsim` for automatic graph simplification (applied transparently if available).

### TorchScript

```bash
# Traced
python scripts/export.py \
  --ckpt checkpoints/zipdepth_base.pth \
  --format torchscript

# Frozen — smaller and faster on CPU
python scripts/export.py \
  --ckpt checkpoints/zipdepth_base.pth \
  --format torchscript-frozen
```

The output is saved next to the checkpoint by default. Pass `--output` to override the path.

---

## 📈 Benchmark

Measures parameters, GFLOPs, and latency across backends with IQR-filtered statistics.

```bash
python scripts/benchmark.py --height 384 --width 384
python scripts/benchmark.py --height 384 --width 384 --fp16
```

Sample output on an **RTX 3090** (ZipDepth-base, 384×384):

```text
  Backend                            Mean     Std     p95    FPS
  ──────────────────────────────────────────────────────────────
  Eager FP32                         5.3 ms  ±0.1   5.0    207
  Fused FP32                         3.1 ms  ±0.0   3.1    323
  Fused FP16                         3.8 ms  ±0.0   3.8    266
  compile FP32 (reduce-overhead)     1.3 ms  ±0.0   1.3    768
  compile FP16 (reduce-overhead)     1.3 ms  ±0.0   1.3    771
```

---

## 🧪 Evaluation

We adopt the evaluation protocol and zero-shot benchmarks from [Marigold](https://github.com/prs-eth/marigold). Please refer to their repository for dataset preparation and evaluation scripts.

---

## 🎓 Training

### Training Data

ZipDepth was trained via knowledge distillation using pseudo depth maps generated by [Depth Anything V2 Large](https://github.com/DepthAnything/Depth-Anything-V2). The training set spans **17 domains** and contains approximately **14.1 million** RGB–depth pairs:

`ACDC` · `ADE20K` · `bdd100k` · `Cityscapes` · `COCO` · `DrivingStereo` · `Flickr1024` · `Gated2Depth` · `GoogleLandmarks` · `HRWSI` · `HoloPix50K` · `Mapillary` · `MegaDepth` · `Object365` · `OpenImagesv7` · `SA-1B` · `Trans10K`

### Dataset Index

The dataloader expects a JSON index of RGB–depth pairs. To build one from your own data, organize images into parallel RGB and depth directories with matching relative paths, then run:

```bash
# Single domain
python scripts/prepare_index.py build \
    --domains MyDataset /path/to/rgb /path/to/depth \
    --output dataset_index.json

# Multiple domains via a YAML config
python scripts/prepare_index.py build \
    --config domains.yaml --output dataset_index.json
```

YAML config format:

```yaml
MyDataset:
    rgb:   /path/to/rgb
    depth: /path/to/depth
AnotherSet:
    rgb:   /path/to/rgb2
    depth: /path/to/depth2
```

The depth maps can be PNG (uint16) or `.npy`/`.npz` files. Before training, convert the JSON index to numpy memmap format (much faster I/O at millions of samples):

```bash
python scripts/prepare_index.py convert --input dataset_index.json
```

Or do both in one shot with `--convert`:

```bash
python scripts/prepare_index.py build \
    --domains MyDataset /path/to/rgb /path/to/depth \
    --output dataset_index.json --convert
```

Set `index_file` in `configs/default.json` to the JSON path — the dataloader auto-detects the converted `.npy` files.

### Launch

Edit `configs/default.json` to set your dataset path and hyperparameters, then launch:

```bash
# Single GPU
python scripts/train.py --config configs/default.json

# Multi-GPU with DDP (e.g. 2 GPUs)
torchrun --nproc_per_node=2 scripts/train.py --config configs/default.json

# Resume from a checkpoint
python scripts/train.py \
  --config configs/default.json \
  --resume checkpoints/base_384x384/epoch_3.pth
```

Key configuration fields:

```json
{
  "model":    { "variant": "base" },
  "data":     { "index_file": "/path/to/dataset_index.json", "height": 384, "width": 384 },
  "training": { "epochs": 5, "batch_size": 96 },
  "optimizer":{ "lr": 1e-3, "weight_decay": 0.05 },
  "amp":      { "enabled": true, "dtype": "bfloat16" }
}
```

Training uses a scale-and-shift invariant loss with gradient regularization, AdamW with OneCycleLR scheduling, and bfloat16 mixed precision.

---

## 🙏 Acknowledgements

We thank the authors of [Marigold](https://github.com/prs-eth/marigold) and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for their excellent work and for releasing code and evaluation protocols that made this research possible.

---

## 📝 Citation

```bibtex
@inproceedings{tosi2026zipdepth,
  title     = {ZipDepth: Bringing Lightweight Zero-Shot Monocular Depth Anywhere, on Any Device},
  author    = {Tosi, Fabio and Bartolomei, Luca and Poggi, Matteo and Mattoccia, Stefano},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
