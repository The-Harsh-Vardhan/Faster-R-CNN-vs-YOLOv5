# RCNN on Pascal VOC: Presentation + VIVA Ready Notes

## 1) One-line Intro (Start your section with this)

In this part, I implemented and analyzed Faster R-CNN on Pascal VOC, then compared it with YOLO on the same dataset to explain the accuracy-speed trade-off in object detection.

---

## 2) 60-second Project Context (Transition from previous speaker)

Our team studied four combinations:

- RCNN on COCO
- YOLO on COCO
- RCNN on Pascal VOC (my contribution)
- YOLO on Pascal VOC

My focus is Pascal VOC because it is a classic benchmark with 20 object classes and clear XML annotations, making it ideal for understanding detector behavior.

---

## 3) Problem Statement (What you were solving)

Goal: Detect and localize objects in Pascal VOC images using Faster R-CNN and compare against YOLO outputs on the same dataset.

Why this matters:

- Detection quality is not only classification; localization quality matters too.
- In real applications, we must choose between strict box quality and real-time speed.

---

## 4) Dataset and Setup (Say this clearly in presentation)

Dataset used: Pascal VOC (VOC-style annotations)

- Annotation format: XML bounding boxes
- Classes: 20 foreground classes

Pipeline highlights from implementation:

- Framework: PyTorch + Torchvision detection API
- Model: `fasterrcnn_resnet50_fpn_v2` with pretrained backbone
- Head modification: Final predictor changed to 21 classes (20 VOC + background)
- Training run in notebook: 5 epochs, batch size 2
- Evaluation metric tool: COCO-style mAP computation (`torchmetrics` MeanAveragePrecision)

---

## 5) Faster R-CNN in Simple but Technical Terms

Faster R-CNN is a two-stage detector:

1. Stage 1 (RPN): Proposes candidate object regions.
2. Stage 2 (ROI Head): Classifies each region and refines box coordinates.

Why two-stage helps:

- It spends more computation per candidate region.
- Usually gives stronger localization reliability, especially in cluttered scenes.

Trade-off:

- Higher latency than one-stage methods like YOLO.

---

## 6) Key Pascal VOC Comparison Results (Use this table on slide)

Based on the project report appendix notebook summaries:

| Model                     | mAP@0.50 |   FPS |
| ------------------------- | -------: | ----: |
| Faster R-CNN (Pascal VOC) |   0.6934 |  5.81 |
| YOLO (Pascal VOC)         |   0.7459 | 50.12 |

Quick derived comparison:

- YOLO speedup over RCNN is about 8.6x (50.12 / 5.81).
- In this Pascal run, YOLO also had higher mAP@0.50.

Important interpretation line:
Even when YOLO wins on mAP@0.50 here, RCNN remains valuable for two-stage detection analysis and can still be preferred in precision-critical use-cases, especially when stricter IoU behavior and proposal quality are prioritized.

---

## 7) How to Explain Your Graphs/Outputs

When showing sample detections:

- Point out box tightness around object boundaries.
- Mention missed detections (false negatives) and extra detections (false positives).
- Explain confidence threshold effects: higher threshold reduces false positives but may increase misses.

When showing speed graph:

- Convert latency to FPS intuition: higher FPS means better real-time capability.
- State practical boundary: ~30 FPS is commonly considered real-time.

---

## 8) 2-minute Speaking Script (Ready to read)

In my part, I worked on Faster R-CNN using the Pascal VOC dataset and compared it with YOLO results on the same dataset. Pascal VOC has 20 object classes and structured XML annotations, so it is a strong benchmark for detector comparison.

Technically, I used a pretrained Faster R-CNN with ResNet50-FPN backbone and modified the prediction head for 20 VOC classes plus background. The training and evaluation were done in PyTorch notebooks, and performance was measured using mAP and inference speed.

The key observation is the accuracy-speed trade-off. In our Pascal VOC results, Faster R-CNN achieved mAP@0.50 of 0.6934 with 5.81 FPS, while YOLO achieved mAP@0.50 of 0.7459 with 50.12 FPS. So YOLO is significantly faster in deployment terms and also strong on mAP in this setup.

Still, Faster R-CNN is important academically and practically because two-stage detection provides strong region proposal reasoning and often improves localization quality in complex scenes. So the model choice depends on deployment goals: if real-time response is mandatory, YOLO is preferred; if detailed proposal-based analysis is needed, RCNN remains a strong option.

---

## 9) VIVA Ready Q&A (High-probability questions)

### Q1. What is the difference between R-CNN, Fast R-CNN, and Faster R-CNN?

- R-CNN: external region proposals + CNN per proposal (very slow).
- Fast R-CNN: shared convolution features, faster ROI processing.
- Faster R-CNN: adds RPN for learnable proposals, fully integrated two-stage pipeline.

### Q2. Why is Faster R-CNN called a two-stage detector?

Because it first proposes regions (RPN) and then classifies/refines them in a second stage.

### Q3. Why did you choose Pascal VOC?

It is a standard benchmark with clean annotations and 20 classes, useful for controlled comparison and education-oriented analysis.

### Q4. Why did you use a pretrained backbone?

Transfer learning reduces training time and improves convergence when dataset size is moderate.

### Q5. Why does YOLO usually run faster?

YOLO predicts boxes and classes in one forward pass without a separate proposal stage.

### Q6. What does mAP@0.50 mean?

Mean Average Precision computed with IoU threshold 0.50. A detection is correct if IoU >= 0.50 and class is correct.

### Q7. Why can mAP@0.50 be misleading alone?

It is lenient; stricter localization quality is better reflected with higher IoU thresholds (e.g., 0.75) or mAP@0.50:0.95.

### Q8. What are false positives and false negatives?

- False positive: model predicts object where there is none or wrong class.
- False negative: model misses a true object.

### Q9. How do confidence thresholds affect results?

Higher threshold reduces false positives but can increase false negatives.

### Q10. What is NMS and why is it needed?

Non-Maximum Suppression removes overlapping duplicate predictions, keeping the most confident box.

### Q11. Why is FPS important?

It indicates throughput. Real-time systems require sufficiently high FPS for smooth operation.

### Q12. In your results, YOLO is both faster and better on mAP@0.50. Then why study RCNN?

RCNN provides strong two-stage detection fundamentals, richer proposal-based reasoning, and often better strict localization behavior in challenging settings.

### Q13. What is the role of anchors in Faster R-CNN?

Anchors are reference boxes of multiple scales/aspect ratios used by RPN to propose candidate objects.

### Q14. What did you modify in the model for VOC?

The final classification head was replaced to output 21 classes (20 VOC + background).

### Q15. Why include background as a class in RCNN heads?

The classifier must distinguish object categories from background regions.

### Q16. If given more compute, what would you improve first?

More epochs, stronger augmentation, hyperparameter tuning, and class-wise error analysis.

### Q17. How would you make RCNN faster?

Use lighter backbones, mixed precision, pruning, quantization, and optimized inference runtimes.

### Q18. Which model would you deploy for CCTV real-time analytics?

YOLO, because high FPS is usually the primary requirement.

### Q19. Which model for offline forensic analysis or high-quality annotation support?

Faster R-CNN, due to stronger proposal-based analysis and robust two-stage refinement.

### Q20. One limitation of your current experiment?

Comparison is based on a specific training setup and limited epoch budget; broader tuning may change absolute numbers.

---

## 10) Rapid Revision Sheet (30 sec before viva)

- Detector type: Faster R-CNN = two-stage, YOLO = one-stage.
- My dataset: Pascal VOC, 20 classes, XML annotations.
- My RCNN setup: pretrained FasterRCNN-ResNet50-FPN, head changed to 21 classes.
- Core results (VOC): RCNN mAP@0.50 = 0.6934, FPS = 5.81; YOLO mAP@0.50 = 0.7459, FPS = 50.12.
- Final conclusion: choose model by deployment need (precision-focused analysis vs real-time speed).

---

## 11) Last-slide Closing Statement

My RCNN-on-VOC contribution demonstrates the two-stage detection pipeline and provides a direct empirical benchmark against YOLO on the same dataset. The final takeaway is practical: model selection should be requirement-driven, not architecture-driven.
