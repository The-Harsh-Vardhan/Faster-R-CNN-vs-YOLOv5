# Faster R-CNN on COCO 2017 — Comprehensive Technical Report

> **Source Notebook:** `faster-rcnn-oncocodataset.ipynb`
> **Execution Platform:** Kaggle (NVIDIA Tesla T4 GPU, CUDA 12.8)
> **Execution Date:** 2026-04-07
> **Status:** Training ran through Epoch 3/5; Epochs 4–5 did NOT complete (notebook was still running / timed out).

---

## A. Introduction

### Objective
Train and evaluate a **Faster R-CNN** object detection model on the full **COCO 2017** dataset using PyTorch and Torchvision, running on Kaggle GPU infrastructure.

### Problem Statement
General-purpose **multi-class object detection**: given an input image, predict bounding boxes and class labels for every object instance present. COCO 2017 is the standard benchmark for this task, containing 80 object categories across diverse real-world scenes.

### Why Faster R-CNN
The notebook uses `torchvision.models.detection.fasterrcnn_resnet50_fpn` — the canonical two-stage detector:

- **Two-stage pipeline** (RPN → ROI Head) provides higher localization accuracy than single-stage detectors at the cost of speed
- **COCO-pretrained weights** (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`) give a strong initialization — the notebook fine-tunes rather than training from scratch
- **Native Torchvision support** means minimal boilerplate: the model handles anchor generation, NMS, and multi-task loss internally
- Compared to YOLO (single-stage, speed-optimized) or DETR (transformer-based, no NMS), Faster R-CNN remains the go-to baseline for detection research

---

## B. Dataset Analysis (COCO 2017)

### Description
The COCO (Common Objects in Context) 2017 dataset is loaded from Kaggle: `/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017`.

### Split Statistics (from notebook output)

| Split | Images Used | Batches |
|-------|-------------|---------|
| Train | **117,266** | 29,317 (batch size 4) |
| Val   | **4,952**   | 1,238 (batch size 4) |

- **80 object categories** (person, car, dog, etc.)
- Images with **zero annotations are filtered out** in the dataset class (`img_ids = [id for id in img_ids if len(coco.getAnnIds(...)) > 0]`)
- `MAX_TRAIN_IMAGES` and `MAX_VAL_IMAGES` are set to `None` → **full dataset** is used

### Annotation Format (COCO JSON)
The notebook reads standard COCO JSON via `pycocotools.coco.COCO`:
- **Bounding boxes** are stored as `[x, y, width, height]` (COCO format), converted to `[x1, y1, x2, y2]` (Pascal VOC / xyxy format) in the dataset class
- Each annotation includes: `bbox`, `category_id`, `area`, `iscrowd`, `image_id`
- Tiny boxes (`w <= 1` or `h <= 1`) are explicitly **discarded** in `__getitem__`

### Preprocessing Steps (from code)

1. **Resize** to fixed `640×640` via `torchvision.transforms.functional.resize`
2. **To tensor** via `torchvision.transforms.functional.to_tensor` (scales pixels to [0, 1], reorders to CHW)
3. **Bounding box rescaling** — boxes are linearly scaled to match the resized image dimensions using `w_scale = IMAGE_SIZE / orig_width` and `h_scale = IMAGE_SIZE / orig_height`

> [!WARNING]
> **No data augmentation** is applied. Both `get_train_transforms` and `get_val_transforms` perform identical operations (resize + to_tensor). This is a significant limitation (see Section H/I).

---

## C. Model Architecture — Faster R-CNN Deep Dive

The notebook loads `torchvision.models.detection.fasterrcnn_resnet50_fpn`. Here is how each component works internally:

### 1. Backbone Network: ResNet-50 + FPN

**Confirmed from code:** `fasterrcnn_resnet50_fpn(weights=weights)` → ResNet-50 backbone with Feature Pyramid Network.

- **ResNet-50** extracts hierarchical feature maps at 4 stages (C2, C3, C4, C5) with strides {4, 8, 16, 32}
- **FPN (Feature Pyramid Network)** builds a top-down pathway with lateral connections, producing 5 feature map levels (P2–P6) at strides {4, 8, 16, 32, 64}
- Each FPN level has **256 channels** — this multi-scale representation is critical for detecting objects of varying sizes

### 2. Region Proposal Network (RPN)

The RPN operates on each FPN level to generate region proposals:

#### Anchor Generation
- **Sizes:** The default Torchvision config uses sizes `((32,), (64,), (128,), (256,), (512,))` — one size per FPN level
- **Aspect ratios:** `((0.5, 1.0, 2.0),)` per level → **3 anchors per spatial location per level**
- At a 640×640 input, this generates approximately **160×160 + 80×80 + 40×40 + 20×20 + 10×10 = ~34,000** anchor locations × 3 ratios ≈ **~100,000 anchors total**
- The notebook includes a `plot_rpn_anchor_boxes()` function that visualizes these anchors per FPN level (defined but **not executed** due to notebook timeout)

#### Objectness Score
- For each anchor, a **3×3 conv → 1×1 conv** predicts a binary objectness score (object vs. background)
- During training, anchors are labeled positive (IoU ≥ 0.7 with any GT box), negative (IoU < 0.3), or ignored

#### Bounding Box Regression
- A parallel **1×1 conv** head predicts 4 regression deltas (dx, dy, dw, dh) to refine each anchor to a tighter proposal
- NMS (Non-Maximum Suppression) is applied post-RPN to reduce ~100K anchors to ~2000 proposals for training / 300 for inference

### 3. ROI Align
- Proposals from RPN are mapped back to the appropriate FPN level based on proposal area
- **ROI Align** (not ROI Pooling) extracts fixed-size 7×7 feature maps using bilinear interpolation — avoids quantization artifacts from ROI Pooling

### 4. Classification + Bounding Box Regression Head
- Two fully-connected layers (1024-d each) process the 7×7×256 pooled features
- **Classification head:** FC → 91-class softmax (80 COCO classes + background + reserved slots)
- **Box regression head:** FC → 91×4 outputs (per-class box deltas)

### 5. Loss Functions
Faster R-CNN uses a **multi-task loss** computed internally by Torchvision when the model is in `.train()` mode and targets are provided:

| Loss Component | Formula | Purpose |
|----------------|---------|---------|
| RPN Classification | Binary Cross-Entropy | Object vs. background at anchor level |
| RPN Box Regression | Smooth L1 Loss | Refine anchor → proposal |
| ROI Classification | Cross-Entropy | Classify proposals into 91 classes |
| ROI Box Regression | Smooth L1 Loss | Refine proposal → final box |

The total loss is `sum(loss_dict.values())` — all four components are summed with equal weight. This is confirmed in the `train_one_epoch` function:
```python
loss_dict = model(images, targets)
losses = sum(loss for loss in loss_dict.values())
```

---

## D. Implementation Details

### Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | **2.10.0+cu128** | Deep learning framework |
| `torchvision` | **0.25.0+cu128** | Faster R-CNN model + transforms |
| `pycocotools` | installed via pip | COCO annotation parsing |
| `torchmetrics` | installed via pip | Mean Average Precision computation |
| `matplotlib` | (system) | Visualization |
| `PIL` | (system) | Image loading |
| `tqdm` | (system) | Progress bars |
| `numpy` | (system) | Numerical operations |

**Python version:** 3.12.12 | **Device:** CUDA (Tesla T4)

### Model Loading
```python
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
```
- Uses **COCO-pretrained weights** (downloaded: `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`, 160MB)
- The model head is **NOT modified** — all 91 output classes are retained (standard COCO setup)
- This is **fine-tuning**, not transfer learning to a new task

### Dataset Loader (`CocoDetectionForTorchvision`)
Custom `torch.utils.data.Dataset` subclass:
- Initializes `pycocotools.coco.COCO` for annotation access
- Filters images with zero annotations
- Supports optional `max_images` cap (unused here — set to `None`)
- Converts COCO `[x, y, w, h]` → `[x1, y1, x2, y2]` format
- Returns dict targets with keys: `boxes`, `labels`, `image_id`, `area`, `iscrowd`
- Custom `collate_fn` returns lists (not stacked tensors) — required by Torchvision detection models

### Training Loop
```python
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    for images, targets in tqdm(loader):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=True):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += losses.item()
    return running_loss / len(loader), elapsed
```
Key features:
- **Mixed precision** via `torch.amp.autocast` + `GradScaler` (FP16 forward, FP32 gradients)
- `zero_grad(set_to_none=True)` — memory optimization
- Loss is averaged over batches per epoch

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image Size | **640×640** | Fixed resize, no aspect ratio preservation |
| Batch Size | **4** | Limited by T4 GPU memory (16GB) |
| Epochs | **5** (only 3 completed) | Insufficient for convergence |
| Learning Rate | **5e-4** | AdamW optimizer |
| Weight Decay | **1e-4** | L2 regularization |
| Optimizer | **AdamW** | Decoupled weight decay |
| LR Scheduler | **StepLR** | `step_size=2, gamma=0.1` → LR drops 10× at epochs 2, 4 |
| Num Workers | **2** | DataLoader workers |
| Seed | **42** | Reproducibility |

---

## E. Training Analysis

### Epoch-wise Training Logs (Extracted from Notebook Output)

| Epoch | Train Loss | mAP | mAP@0.50 | mAP@0.75 | Train Time | Eval Time | Checkpoint |
|-------|-----------|-----|----------|----------|------------|-----------|------------|
| 1/5 | **0.9156** | 0.0671 | 0.1437 | 0.0537 | 12,888.8s (~3.6h) | 611.8s (~10.2m) | `fasterrcnn_epoch_01.pth` |
| 2/5 | **nan** | 0.0630 | 0.1414 | 0.0469 | 12,886.8s (~3.6h) | 607.7s (~10.1m) | `fasterrcnn_epoch_02.pth` |
| 3/5 | **0.7987** | 0.1270 | 0.2413 | 0.1199 | 12,876.9s (~3.6h) | 608.8s (~10.1m) | `fasterrcnn_epoch_03.pth` |
| 4/5 | — | — | — | — | — | — | **Not completed** |
| 5/5 | — | — | — | — | — | — | **Not completed** |

> [!CAUTION]
> **Epoch 2 reports `Train Loss: nan`** — this is a critical numerical instability issue. Despite this, mAP metrics were still computed (the model recovered by epoch 3, likely due to the LR drop from StepLR at epoch 2).

### Observations on Convergence

1. **NaN loss at Epoch 2** is a red flag. Likely causes:
   - Learning rate too high for fine-tuning a pretrained model (5e-4 with AdamW can be aggressive)
   - Mixed precision (FP16) overflow in loss computation
   - The StepLR decay at epoch 2 (LR: 5e-4 → 5e-5) "rescued" training by epoch 3

2. **Improving trend visible:** mAP@0.50 went from 0.1437 → 0.2413 (epoch 1→3), showing the model is learning despite the NaN episode

3. **Far from convergence:** Standard COCO training requires 12–24 epochs with warmup. The pretrained model's mAP@0.50 should be ~0.37 without fine-tuning; after 3 epochs of fine-tuning with issues, we're at 0.24 — the model is **performing worse than the pretrained baseline** due to the NaN corruption

4. **Training is extremely slow:** ~3.6 hours per epoch on T4 for 117K images at batch size 4. Total estimated time for 5 epochs: **~20 hours** (exceeds Kaggle's 12h GPU limit, explaining why epochs 4–5 never completed)

5. **No validation loss** is tracked — only training loss and mAP metrics on the validation set

---

## F. Performance Evaluation

### Metrics Used
The notebook uses `torchmetrics.detection.MeanAveragePrecision`:
- **mAP** (averaged over IoU thresholds 0.50:0.05:0.95) — the primary COCO metric
- **mAP@0.50** — AP at IoU ≥ 0.50 (PASCAL VOC style)
- **mAP@0.75** — AP at IoU ≥ 0.75 (strict localization)
- **Per-class AP** — enabled via `class_metrics=True` (available but not plotted due to timeout)
- **Box format:** `xyxy` | **IoU type:** `bbox`

### Best Results Achieved (Epoch 3 — last completed)

| Metric | Value | COCO Baseline (pretrained, no fine-tune) |
|--------|-------|------------------------------------------|
| mAP | **0.1270** | ~0.37 |
| mAP@0.50 | **0.2413** | ~0.58 |
| mAP@0.75 | **0.1199** | ~0.39 |

### Analysis

- **Results are significantly below expectations.** A pretrained Faster R-CNN ResNet-50 FPN should achieve ~37% mAP on COCO val2017 out of the box. After fine-tuning, results should improve — but here they degraded.
- **Root cause:** The NaN loss at epoch 2 corrupted model weights. Even though the LR drop stabilized training, the model lost information from the pretrained weights.
- **mAP@0.50 at 0.24** means roughly 1 in 4 detections at IoU ≥ 0.5 are correct — poor for production use.
- The gap between mAP@0.50 (0.24) and mAP@0.75 (0.12) indicates **localization quality is poor** — the model finds objects roughly but doesn't tightly fit bounding boxes.

---

## G. Visualization Results

### Implemented Visualizations (in code, but NOT executed due to notebook timeout)

The notebook defines a comprehensive visualization suite:

| Function | Purpose | Status |
|----------|---------|--------|
| `plot_training_dashboard(history)` | Loss curve + mAP curves + epoch speed | **Not executed** |
| `plot_rpn_anchor_boxes(model)` | Visualize RPN anchor shapes per FPN level | **Not executed** |
| `plot_coco_class_distribution(dataset)` | Top-K class frequency bar chart | **Not executed** |
| `plot_bbox_size_distribution(dataset)` | Width/height scatter + histograms | **Not executed** |
| `plot_confidence_histogram(loader, model)` | Score distribution of predictions | **Not executed** |
| `plot_per_class_map(metrics)` | Per-class AP horizontal bar chart | **Not executed** |
| `show_ground_truth(dataset, idx)` | Display GT boxes on image | **Not executed** |
| `show_predictions(model, dataset, device)` | Display predicted boxes with scores ≥ 0.5 | **Not executed** |

> [!NOTE]
> All visualization cells have `execution_count: null` and `outputs: []` — confirming the notebook timed out during the training loop (cell 10) before reaching these cells.

### What the Visualizations Would Show (based on code)

- **Ground truth viewer** draws cyan rectangles with class names in black labels
- **Prediction viewer** draws lime-green rectangles with `class: score` labels, filtered by `score_thr=0.5`
- **Training dashboard** is a 2×2 grid: loss curve (top), mAP curves (bottom-left), timing (bottom-right)
- **Class distribution** shows top-30 classes by annotation count (horizontal bar chart)
- **Anchor visualization** shows scaled anchor boxes centered at (0.5, 0.5) for each FPN level

---

## H. Limitations

### Dataset Limitations
- **Fixed 640×640 resize without aspect ratio preservation** — distorts non-square images, which degrades detection accuracy (COCO images have varied aspect ratios)
- **No data augmentation** — no horizontal flip, color jitter, random crop, or mosaic. Train and val transforms are identical. This severely limits generalization
- **Full COCO is too large for Kaggle's time limit** — 117K training images at batch size 4 = 29,317 iterations per epoch × ~0.44s/iter ≈ 3.6 hours/epoch. Five epochs require ~18 hours, but Kaggle limits GPU sessions to 12 hours

### Model Constraints
- **91 output classes retained** — the notebook doesn't modify the classification head even though only 80 classes have data. The extra classes waste computation
- **No backbone freezing** — all parameters are trainable (`params = [p for p in model.parameters() if p.requires_grad]`). Fine-tuning the entire backbone with a high LR led to the NaN instability
- **No gradient clipping** — could have prevented the NaN loss explosion

### Training Inefficiencies
- **No learning rate warmup** — jumping straight to LR=5e-4 on a pretrained model caused instability
- **StepLR is too aggressive** — `gamma=0.1` drops LR by 10× every 2 epochs, which is too drastic for only 5 epochs of training
- **No early stopping** — even after NaN loss, training continued blindly
- **Mixed precision issues** — FP16 can underflow/overflow with large loss values; no loss scaling safeguards beyond the default GradScaler

### Hardware Bottlenecks
- **NVIDIA Tesla T4 (16GB VRAM)** — limits batch size to 4 at 640×640. Larger batches would improve gradient stability
- **Kaggle 12-hour GPU limit** — insufficient for full COCO training at this configuration. Only 3 of 5 epochs completed
- **2 DataLoader workers** — low parallelism; disk I/O may bottleneck training speed
- **No multi-GPU support** — single GPU training only

---

## I. Improvements & Optimization

### Accuracy Improvements

1. **Add learning rate warmup** — use `LinearLR` for the first 500–1000 iterations before the main schedule. This prevents weight corruption from large early gradients
2. **Freeze backbone initially** — freeze ResNet layers for 1–2 epochs, then unfreeze with a lower LR (discriminative fine-tuning)
3. **Add gradient clipping** — `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` to prevent NaN
4. **Data augmentation** — at minimum add:
   - Random horizontal flip (almost free, ~1-2% mAP gain)
   - Color jitter (brightness, contrast, saturation)
   - Random scale/crop
   - Mosaic augmentation (as in YOLOv4/v5) for small object detection
5. **Use CosineAnnealingLR** instead of StepLR for smoother decay
6. **Lower initial LR** to 1e-4 or use different LRs for backbone vs. head

### Training Speed Improvements

1. **Subsample the dataset** — use `MAX_TRAIN_IMAGES = 20000` for faster iteration
2. **Increase batch size** — use gradient accumulation to simulate batch size 8-16
3. **Increase `num_workers` to 4** — improve data loading throughput
4. **Use a smaller image size** (e.g., 512) for initial experiments
5. **Use `torch.compile(model)`** — can give 10-30% speedup on PyTorch 2.x

### Alternative Architectures

| Model | mAP@0.50 (COCO) | Speed (T4) | Key Advantage |
|-------|-----------------|------------|---------------|
| Faster R-CNN (this) | ~58% (pretrained) | ~7 FPS | High accuracy, well-understood |
| **YOLOv8-L** | ~53% | ~40 FPS | 6× faster, easier training |
| **RT-DETR** | ~54% | ~35 FPS | No NMS, end-to-end |
| **DINO (DETR)** | ~63% | ~5 FPS | SOTA accuracy |
| **Co-DETR** | ~66% | ~3 FPS | Best accuracy |

- **YOLOv8** would be the pragmatic choice: trains faster, better augmentation pipeline built-in (mosaic, mixup), single-stage simplicity
- **DETR/RT-DETR** eliminates NMS and anchor design entirely via transformer-based set prediction

---

## J. Conclusion

### What Was Achieved
- A complete, end-to-end Faster R-CNN training pipeline on full COCO 2017 was implemented
- The code is well-structured: clean dataset class, config dataclass, checkpoint saving, mAP evaluation, and comprehensive (though unexecuted) visualization suite
- 3 out of 5 epochs completed, producing checkpoint files and mAP evaluations

### What Failed
- **NaN training loss at Epoch 2** — the most critical failure, caused by aggressive fine-tuning without warmup or gradient clipping
- **Only 3/5 epochs completed** — Kaggle's 12-hour GPU limit was hit (~10.8 hours of training + ~30 min of evaluation = ~11.3 hours)
- **Model degraded below pretrained baseline** — mAP of 0.127 vs. pretrained 0.37; fine-tuning made the model worse
- **All visualization cells failed to execute** — no visual output was produced

### Practical Usability
In its current state, the model is **not usable for production**:
- mAP@0.50 of 0.24 is too low for reliable object detection
- The NaN episode suggests weight corruption
- The best checkpoint (`fasterrcnn_best.pth`) corresponds to epoch 1 (mAP@0.50 = 0.1437) or epoch 3 (0.2413) — either way, far below the pretrained baseline

**To make this notebook production-ready:**
1. Add LR warmup + gradient clipping (fixes NaN)
2. Add data augmentation (fixes generalization)
3. Use cosine LR schedule over 12+ epochs
4. Subsample to ~20K images or use a faster architecture to fit within Kaggle time limits
5. Evaluate the pretrained model BEFORE fine-tuning to establish a proper baseline
