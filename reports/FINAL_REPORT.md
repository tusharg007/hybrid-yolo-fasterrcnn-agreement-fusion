# Hybrid Object Detection Report

## Project Overview

This project evaluates a hybrid object detector that combines YOLO and Faster R-CNN. The original crop-refinement cascade was implemented first, but evaluation showed a structural problem: it reduced false positives while sharply increasing false negatives and made inference substantially slower than Faster R-CNN alone. The project was then redesigned around a detector agreement-fusion strategy, where YOLO and Faster R-CNN both run on the full image and the final detections are fused or filtered based on cross-detector agreement.

## Objective

The target was to achieve a better speed-accuracy trade-off than standalone detectors:

- `YOLO` as the high-speed detector
- `Faster R-CNN` as the high-accuracy detector
- `Hybrid` as a precision-improving model that preserves most of Faster R-CNN's accuracy without the severe runtime penalty of the crop cascade

## Methodology

### 1. Standalone baselines

- `YOLO` was evaluated directly on the full image.
- `Faster R-CNN` was evaluated directly on the full image.

### 2. Initial hybrid design: crop refinement

The first hybrid design used YOLO to generate proposals, cropped those regions, and then ran Faster R-CNN on each crop. This design had two major flaws:

- repeated Faster R-CNN execution over many crops made runtime much worse than standalone Faster R-CNN
- tight or missing proposals caused recall loss, which translated directly into higher false negatives and lower `mAP`

### 3. Final hybrid design: agreement fusion

The final hybrid design uses:

- YOLO full-image detections
- Faster R-CNN full-image detections
- class-aware agreement matching based on IoU
- score fusion and optional YOLO fallback for unmatched confident detections
- final class-aware NMS

This preserves full-image context and avoids re-running Faster R-CNN for every proposal crop.

## Evaluation Scope

- Dataset: `VOC 2012`
- Scope: `100-image validation subset`
- Selected final run: `voc100_agreement_variant_b`

## Final Selected Hybrid Configuration

```json
{
  "strategy": "agreement_fusion",
  "proposal_conf": 0.05,
  "proposal_iou": 0.7,
  "max_proposals": 100,
  "frcnn_score_thresh": 0.05,
  "agreement_iou": 0.45,
  "unmatched_frcnn_score_thresh": 0.55,
  "unmatched_yolo_score_thresh": 0.6,
  "score_fusion": "frcnn",
  "final_score_thresh": 0.2,
  "final_nms_iou": 0.5,
  "fusion": "nms"
}
```

## Final Model Comparison

| model | mAP@0.5 | mAP@0.5:0.95 | mean_inf_time_s | fps | false_positives | false_negatives | speedup_vs_frcnn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo | 0.7303 | 0.5633 | 0.0233 | 42.8669 | 6523 | 16 | 6.3530 |
| frcnn | 0.8809 | 0.6821 | 0.1482 | 6.7476 | 1305 | 4 | 1.0000 |
| hybrid | 0.8599 | 0.6873 | 0.1587 | 6.3026 | 226 | 15 | 0.9341 |

## Experimental Progression

| run_id | model | mAP@0.5 | mAP@0.5:0.95 | mean_inf_time_s | fps | false_positives | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| voc100_agreement_default | yolo | 0.7303 | 0.5633 | 0.0241 | 41.5727 | 6523 | 16 |
| voc100_agreement_default | frcnn | 0.8809 | 0.6821 | 0.1400 | 7.1423 | 1305 | 4 |
| voc100_agreement_default | hybrid | 0.8475 | 0.6849 | 0.1513 | 6.6092 | 175 | 18 |
| voc50_crop_refine_baseline | yolo | 0.7646 | 0.6110 | 0.0332 | 30.1206 | 3462 | 11 |
| voc50_crop_refine_baseline | frcnn | 0.9198 | 0.7513 | 0.1427 | 7.0082 | 592 | 1 |
| voc50_crop_refine_baseline | hybrid | 0.7868 | 0.6155 | 0.5721 | 1.7480 | 486 | 9 |
| voc100_agreement_variant_b | yolo | 0.7303 | 0.5633 | 0.0233 | 42.8669 | 6523 | 16 |
| voc100_agreement_variant_b | frcnn | 0.8809 | 0.6821 | 0.1482 | 6.7476 | 1305 | 4 |
| voc100_agreement_variant_b | hybrid | 0.8599 | 0.6873 | 0.1587 | 6.3026 | 226 | 15 |
| voc100_agreement_variant_c | yolo | 0.7303 | 0.5633 | 0.0232 | 43.1829 | 6523 | 16 |
| voc100_agreement_variant_c | frcnn | 0.8809 | 0.6821 | 0.1504 | 6.6499 | 1305 | 4 |
| voc100_agreement_variant_c | hybrid | 0.8582 | 0.6861 | 0.1619 | 6.1770 | 159 | 18 |

## Analysis

### YOLO

- Highest speed: `42.87 FPS`
- Lowest latency: `0.0233s/image`
- Accuracy lower than Faster R-CNN and hybrid
- Very high false-positive count because it is optimized for detection throughput and broad recall

### Faster R-CNN

- Best `mAP@0.5`: `0.8809`
- Strong `mAP@0.5:0.95`: `0.6821`
- Better recall than the hybrid
- Slower than YOLO, but still the strongest pure accuracy baseline

### Final Hybrid

- `mAP@0.5`: `0.8599`
- `mAP@0.5:0.95`: `0.6873`
- `FPS`: `6.30`
- False positives reduced sharply to `226`
- False negatives remained above Faster R-CNN at `15`

The final hybrid significantly improved over the original crop-refinement cascade. It no longer suffered the extreme runtime penalty of repeated crop processing, and it preserved localization quality much better. The best hybrid run slightly exceeded Faster R-CNN on `mAP@0.5:0.95`, which suggests that the agreement-fusion strategy improved box ranking and localization stability. However, Faster R-CNN still remained better on `mAP@0.5`, which indicates that recall was still slightly lower in the hybrid system.

## Key Findings

1. The original crop-refinement hybrid was not a good production design.
   It was structurally too slow and too sensitive to proposal recall loss.
2. The agreement-fusion redesign is the correct hybrid direction.
   It keeps latency close to Faster R-CNN while retaining the hybrid's precision advantage.
3. The hybrid is best interpreted as a precision-oriented detector.
   It suppresses false positives much better than Faster R-CNN, but still loses some true detections.
4. The final hybrid is a valid research result even without winning every metric.
   It demonstrates a meaningful trade-off rather than a universal improvement.

## Limitations

- The current final comparison is based on a VOC subset rather than a full COCO evaluation.
- COCO was not finalized in the current experiment set because the Kaggle input was not mounted correctly during the reported runs.
- False-positive and false-negative counts depend on thresholding choices and should be interpreted alongside `mAP`, not in isolation.

## Recommended Final Conclusion

The final agreement-fusion hybrid is the strongest hybrid design explored in this project. It substantially improves over the original crop-refinement approach by preserving full-image context and avoiding repeated Faster R-CNN execution on cropped proposals. Compared with standalone Faster R-CNN, the final hybrid offers a large reduction in false positives and competitive `mAP@0.5:0.95`, with only a small runtime penalty. Compared with YOLO, it sacrifices throughput but gains a much stronger precision-accuracy profile. For applications where false positives are costly and near-Faster-R-CNN latency is acceptable, the redesigned hybrid provides a useful trade-off.

## Generated Assets

- `reports/tables/final_model_comparison.csv`
- `reports/tables/experiment_runs.csv`
- `reports/tables/stage_timing_breakdown.csv`
- `reports/figures/inference_time_comparison.png`
- `reports/figures/fps_comparison.png`
- `reports/figures/accuracy_comparison.png`
- `reports/figures/error_comparison.png`
- `reports/figures/performance_heatmap.png`
- `reports/figures/performance_radar.png`
- `reports/figures/stage_timing_heatmap.png`
