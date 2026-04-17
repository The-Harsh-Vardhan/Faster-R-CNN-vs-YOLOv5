# Audit: Group 5 Requirements vs v3 Report

Date: 2026-04-17

## Scope

- Assignment requirements source: [G5.md](G5.md)
- Report audited: [v3.tex](v3.tex)
- Deliverable artifacts checked in workspace: notebooks under [..](..)

Status legend:

- MET: Requirement is clearly satisfied with explicit evidence.
- PARTIAL: Requirement is touched but incomplete, indirect, or has placeholders/estimates.
- MISSING: No clear evidence in the report/workspace.

## Executive Summary

- MET: 14
- PARTIAL: 9
- MISSING: 7

Overall: The report is strong on comparative results, speed/accuracy discussion, and visualization requirements, but incomplete on dataset protocol details, class-wise error analysis, and some mandatory deliverables (notably saved model weights).

## Requirement-by-Requirement Audit

### 1) Dataset Preparation

1. Select an appropriate dataset (COCO/VOC/custom): PARTIAL

- Evidence: [v3.tex](v3.tex) uses COCO main analysis with VOC appendix.
- Gap: Requirement expects one common dataset protocol for fair training/evaluation; current report mixes contexts.

2. Split dataset 70/20/10: MISSING

- Evidence: No explicit split ratio statement in [v3.tex](v3.tex).

3. Correct format for Faster R-CNN (COCO JSON/VOC XML): MISSING

- Evidence: No explicit annotation-format confirmation in [v3.tex](v3.tex).

4. Correct format for YOLOv5 (YOLO TXT): MISSING

- Evidence: No explicit annotation-format confirmation in [v3.tex](v3.tex).

5. Custom annotation workflow if custom dataset used: N/A / PARTIAL

- Evidence: [v3.tex](v3.tex) does not claim custom annotation.
- Note: If non-custom datasets were used only, this can be considered not applicable; still undocumented.

### 2) Model Selection

1. Use pre-trained Faster R-CNN and fine-tune: PARTIAL

- Evidence: [v3.tex](v3.tex) describes model usage and training logs, but does not explicitly state pretrained checkpoint name and fine-tuning setup.

2. Explain Faster R-CNN two-stage pipeline (RPN + classifier/regressor): MET

- Evidence: [v3.tex](v3.tex) includes two-stage explanation and references.

3. Use pre-trained YOLOv5 and fine-tune: PARTIAL

- Evidence: [v3.tex](v3.tex) describes YOLOv5 workflow but not explicit pretrained variant/checkpoint and fine-tuning config.

4. Explain YOLO single-stage prediction: MET

- Evidence: [v3.tex](v3.tex) explains single-stage behavior.

### 3) Model Training

1. Training pipelines + hyperparameters (batch size, LR, optimizer, epochs): PARTIAL

- Evidence: [v3.tex](v3.tex) references epoch logs and model optimization discussion.
- Gap: No explicit hyperparameter table or exact values.

2. Train both on same dataset for fair comparison: PARTIAL

- Evidence: COCO comparative tables exist in [v3.tex](v3.tex).
- Gap: Report also mixes VOC appendix and does not explicitly assert same train/val/test protocol for both in main experiment section.

3. Monitor training loss and performance across epochs: PARTIAL

- Evidence: Notebook dashboards and epoch-based reporting are included in [v3.tex](v3.tex).
- Gap: Loss curves/epoch metrics are not systematically tabulated in report text.

### 4) Performance Evaluation & Comparison

#### A. mAP

1. Compute mAP@0.5 and mAP@0.5:0.95: MET

- Evidence: COCO and VOC tables in [v3.tex](v3.tex).

2. Analyze IoU threshold impact: MET

- Evidence: mAP-vs-IoU visualization and discussion in [v3.tex](v3.tex).

#### B. Inference Time

1. Measure inference time per image: MET

- Evidence: speed table in [v3.tex](v3.tex).

2. Compute FPS: MET

- Evidence: speed table and FPS formula in [v3.tex](v3.tex).

#### C. Detection Accuracy

1. Compare false positives and false negatives: MISSING

- Evidence: No FP/FN analysis in [v3.tex](v3.tex).

2. Analyze class-wise precision/recall and bounding-box precision by class: MISSING

- Evidence: No per-class precision/recall section in [v3.tex](v3.tex).

3. Visualize sample detections for box quality: MET

- Evidence: sample detection figures in [v3.tex](v3.tex).

### 5) Final Report & Analysis Content

1. Comparison of mAP scores: MET

- Evidence: quantitative comparison tables in [v3.tex](v3.tex).

2. Inference time analysis (which model is faster): MET

- Evidence: latency/FPS table and text in [v3.tex](v3.tex).

3. Bounding box precision / better localization discussion: PARTIAL

- Evidence: qualitative localization discussion in [v3.tex](v3.tex).
- Gap: no explicit localization precision metric beyond mAP proxy.

4. Accuracy vs Speed trade-off discussion: MET

- Evidence: dedicated subsection in [v3.tex](v3.tex).

5. Graphical analysis

- mAP vs IoU threshold plot: MET
- Inference time comparison plot: MET
- Sample detections with boxes: MET

## Deliverables Audit

1. Code implementation (training/eval scripts): PARTIAL

- Evidence: multiple training/evaluation notebooks found in workspace.
- Gap: requirement says scripts; implementation is notebook-based, which is usually acceptable but should be explicitly acknowledged.

2. Model weights (saved trained models): MISSING

- Evidence: no .pt/.pth/.ckpt/.onnx files found in workspace scan.

3. Evaluation metrics (mAP, inference time, accuracy comparison): MET

- Evidence: metrics tables and figures in [v3.tex](v3.tex).

4. Performance analysis report: MET

- Evidence: full report in [v3.tex](v3.tex).

5. Visualizations: MET

- Evidence: plots and sample detections in [v3.tex](v3.tex) and Images directory.

## High-Priority Gaps To Fix

1. Add an explicit dataset protocol subsection in [v3.tex](v3.tex): dataset name/version, split ratio (70/20/10), and annotation formats for both models.
2. Add class-wise evaluation: precision/recall per class and at least one confusion/error table.
3. Add FP/FN analysis with counts and representative failure cases.
4. Replace placeholder/estimated values with measured values where possible.
5. Add/attach trained model weight artifacts and document paths.

## Suggested Minimal Patch Plan

1. Insert a "Dataset and Experimental Protocol" table with split and format details.
2. Insert a "Class-wise Error Analysis" subsection with FP/FN and per-class P/R.
3. Add a short "Reproducibility and Artifacts" subsection listing notebooks and saved weights.
4. Update conclusion wording to distinguish measured vs estimated metrics.
