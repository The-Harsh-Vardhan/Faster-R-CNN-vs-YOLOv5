# 🔍 Faster R-CNN vs YOLOv5: A Comparative Study on COCO and Pascal VOC

> **CVDL Project — Group 5 | CSL 313 | 6th Semester**  
> Indian Institute of Information Technology, Nagpur

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-00BFFF?logo=github)](https://github.com/ultralytics/yolov5)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project presents a **structured empirical comparison** of two dominant object detection paradigms:

| Model | Paradigm | Strength |
|-------|----------|----------|
| **Faster R-CNN** | Two-stage detector | High localization precision |
| **YOLOv5** | One-stage detector | Real-time inference speed |

Both models are trained and evaluated on:
- **COCO-style** dataset (primary analysis)
- **Pascal VOC** dataset (supplementary cross-validation)

Key evaluation axes: **mAP**, **inference speed (FPS)**, **bounding box quality**, and **accuracy–speed trade-off**.

---

## 🏆 Key Results

### COCO — Primary Analysis

| Model | mAP@0.50 | mAP@0.50:0.95 | mAP@0.75 | Latency (ms/img) | FPS |
|-------|----------|----------------|----------|-----------------|-----|
| Faster R-CNN | **0.2741** | **0.1475** | **0.1425** | 150.00 | 6.67 |
| YOLOv5 | 0.2240 | 0.1170 | ~0.1010 | **12.88** | **77.64** |

### Pascal VOC — Cross-Validation

| Model | mAP@0.50 | FPS |
|-------|----------|-----|
| Faster R-CNN | 0.6934 | 5.81 |
| YOLOv5 | **0.7459** | **50.12** |

> **Takeaway**: Faster R-CNN delivers stronger strict-IoU localization; YOLOv5 is ~11× faster and suitable for real-time pipelines.

---

## 📁 Repository Structure

```
.
├── notebooks/                    # Jupyter notebooks for training & evaluation
│   ├── faster-rcnn-oncocodataset.ipynb       # Faster R-CNN on COCO (Part 1)
│   ├── faster-rcnn-oncocodataset-Part2.ipynb # Faster R-CNN on COCO (Part 2)
│   ├── yolo-on-coco.ipynb                    # YOLOv5 on COCO
│   ├── notebook1-rcnn_Pascal_VOC_dataset.ipynb # Faster R-CNN on Pascal VOC
│   ├── notebook2-yolo__Pascal_VOC_dataset.ipynb # YOLOv5 on Pascal VOC
│   ├── comparison-rcnn-yolo_Pascal_VOC_dataset.ipynb # Side-by-side VOC comparison
│   ├── graph.ipynb               # RCNN training curves & graphs
│   ├── Report_Graph.ipynb        # Combined report visualizations
│   └── yolo-result-figures.ipynb # YOLO result figure extraction
│
├── results/
│   └── figures/                  # Generated evaluation plots
│       ├── 01_loss_vs_epoch.png
│       ├── 02_train_time_per_epoch.png
│       ├── 03_map_vs_epoch.png
│       ├── 04_map_vs_iou_threshold.png
│       ├── 05_fps_comparison.png
│       ├── 06_inference_latency.png
│       ├── 07_precision_recall_curve.png
│       ├── 08_fp_vs_fn.png
│       ├── 09_confidence_distribution.png
│       ├── 10_per_class_map.png
│       ├── 11_sample_detections.png
│       └── 12_side_by_side_gt_vs_pred.png
│
├── report/                       # Final report sources
│   ├── v7.tex                    # Final LaTeX source (IEEE format)
│   ├── CVDL_Report_v7_6th_Semester.pdf  # Latest compiled report
│   └── Images/                   # Figures used in the report
│
├── docs/                         # Technical documentation & notes
│   ├── faster_rcnn_technical_report.md
│   ├── faster_rcnn_pascal.md
│   └── RCNN_on_YOLO.md
│
├── Extras/                       # Archive of intermediate/auxiliary files
└── CVDL_Report_6th_Semester.pdf  # Final submitted report (root copy)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU fallback
- Jupyter Notebook / JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/The-Harsh-Vardhan/CVDL-Project---6th-sem.git
cd CVDL-Project---6th-sem

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics pycocotools matplotlib seaborn pandas numpy jupyter
```

### Running Notebooks

```bash
jupyter notebook
```

Open notebooks in the following order for full reproducibility:

1. `notebooks/faster-rcnn-oncocodataset.ipynb` — Train Faster R-CNN on COCO
2. `notebooks/yolo-on-coco.ipynb` — Train YOLOv5 on COCO
3. `notebooks/graph.ipynb` — Generate RCNN training curves
4. `notebooks/Report_Graph.ipynb` — Generate final comparison plots

---

## 📊 Evaluation Metrics

### Mean Average Precision (mAP)

$$\text{mAP} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{1}{C} \sum_{c=1}^{C} \text{AP}_{c,\tau}$$

with $\mathcal{T} = \{0.50, 0.55, \ldots, 0.95\}$ for COCO-standard evaluation.

### Inference Speed

$$\text{FPS} = \frac{1000}{T_{\text{ms}}}$$

---

## 🧠 Model Architectures

### Faster R-CNN (Two-Stage)
1. **Backbone** — ResNet-50 / ResNet-101 with FPN
2. **Region Proposal Network (RPN)** — generates candidate bounding boxes
3. **RoI Pooling + Classifier** — refines proposals and classifies objects

### YOLOv5 (One-Stage)
- Single forward pass predicts bounding boxes, objectness scores, and class probabilities
- CSPDarknet backbone with PANet neck
- CIoU localization loss for stable training

---

## 📈 Visual Results

| Metric | Figure |
|--------|--------|
| Loss vs Epoch | `results/figures/01_loss_vs_epoch.png` |
| mAP vs Epoch | `results/figures/03_map_vs_epoch.png` |
| mAP vs IoU Threshold | `results/figures/04_map_vs_iou_threshold.png` |
| Inference Latency | `results/figures/06_inference_latency.png` |
| FPS Comparison | `results/figures/05_fps_comparison.png` |
| Precision-Recall Curve | `results/figures/07_precision_recall_curve.png` |
| Sample Detections | `results/figures/11_sample_detections.png` |

---

## 👥 Team — Group 5

| Name | Roll Number |
|------|-------------|
| Adil Khan | bt23csd... |
| Rakshit Modanwal | bt23csd062 |
| **Harsh Vardhan** | bt23csd041 |
| Piyush Jain | bt23csd067 |
| Yash Vikram | bt23csd031 |

**Department of Computer Science and Engineering**  
Indian Institute of Information Technology, Nagpur, Maharashtra, India

---

## 📄 Report

The full IEEE-format technical report is available at:
- [`CVDL_Report_6th_Semester.pdf`](./CVDL_Report_6th_Semester.pdf) — Final submitted version
- [`report/v7.tex`](./report/v7.tex) — LaTeX source

---

## 📚 References

1. Ren et al., *Faster R-CNN*, IEEE TPAMI, 2017.
2. Redmon et al., *You Only Look Once*, CVPR, 2016.
3. Girshick et al., *R-CNN*, CVPR, 2014.
4. Girshick, *Fast R-CNN*, ICCV, 2015.
5. Liu et al., *SSD*, ECCV, 2016.
6. Lin et al., *Focal Loss / RetinaNet*, ICCV, 2017.
7. Bochkovskiy et al., *YOLOv4*, arXiv, 2020.
8. Jocher et al., *YOLOv5*, GitHub, 2020.
9. Lin et al., *Microsoft COCO*, ECCV, 2014.

---

## 📝 License

This project is for academic purposes under **CSL 313 — Computer Vision and Deep Learning**, IIIT Nagpur.
