# Hybrid Object Detection: YOLO Proposals + Faster R-CNN Refinement

This was an academic research project under **Dr. Jagdish Chakole**.

## Contributors

- Tushar Ghosh (BT23CSD043)
- Ritesh Singh (BT23CSD003)
- Satyam Deo (BT23CSD004)
- Arjit Tiwari (BT23CSD008)
- Rajdeep Banerjee (BT23CSD029)

---

This project implements a corrected hybrid object detection pipeline that uses:

- `YOLO` for fast region proposal generation
- `Faster R-CNN` for crop-level classification and box refinement
- class-aware post-processing with `NMS` or optional `Weighted Box Fusion`
- unified evaluation for `mAP`, latency, `FPS`, false positives, and false negatives

## Why your current hybrid metrics look wrong

Our reported pattern is consistent with a recall bottleneck in the proposal stage:

- `false positives` are much lower because YOLO proposals are filtering many candidates early
- `false negatives` are much higher because anything YOLO fails to propose can never be recovered by Faster R-CNN
- `FPS` is close to Faster R-CNN because you are still running a heavy second stage on too many regions, or serially classifying crops with high overhead
- `mAP` drops below both standalone models because the pipeline is losing recall before refinement

That is not a paradox. It is what happens when stage 1 is used as a hard gate without recall-preserving controls.

## Fixes implemented here

1. Lower proposal confidence and use top-k proposal selection to preserve recall.
2. Expand proposal boxes with configurable context padding before refinement.
3. Run Faster R-CNN only on proposal crops, then map refined boxes back to image space.
4. Apply class-aware final NMS or optional weighted box fusion.
5. Optionally fuse YOLO detections back into the final output as a recall safety net.
6. Track stage-wise timing so the speed bottleneck is measurable instead of guessed.

## Project layout

```text
hybrid_detector/
  config.py
  geometry.py
  evaluation.py
  datasets.py
  models/
    yolo_wrapper.py
    faster_rcnn_wrapper.py
    hybrid.py
evaluate.py
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Example usage

### Full comparison on one dataset

```bash
python compare_models.py ^
  --dataset voc ^
  --dataset-root C:\path\to\VOCdevkit\VOC2012 ^
  --yolo-weights yolo11n.pt ^
  --frcnn-weights DEFAULT ^
  --device cuda ^
  --max-images 500
```

### VOC 2012

```bash
python evaluate.py ^
  --mode hybrid ^
  --dataset voc ^
  --dataset-root C:\path\to\VOCdevkit\VOC2012 ^
  --yolo-weights yolo11n.pt ^
  --frcnn-weights DEFAULT ^
  --device cuda ^
  --max-images 500
```

### COCO 2017 val5k

```bash
python evaluate.py ^
  --mode hybrid ^
  --dataset coco ^
  --dataset-root C:\path\to\coco ^
  --annotation-file C:\path\to\coco\annotations\instances_val2017.json ^
  --image-dir C:\path\to\coco\val2017 ^
  --yolo-weights yolo11n.pt ^
  --frcnn-weights DEFAULT ^
  --device cuda ^
  --max-images 5000
```

## What to expect

- If the hybrid pipeline is implemented correctly, it will usually reduce false positives.
- It will not automatically beat standalone Faster R-CNN in `mAP`.
- It can outperform naive Faster R-CNN on speed only if:
  - YOLO proposals are high recall but not too numerous
  - crop refinement is batched efficiently
  - post-processing is lightweight
- If `proposal recall` is low, hybrid `mAP` will collapse even when refinement quality is good.

## Recommended ablations for your report

Run the hybrid model with these settings and compare:

1. Proposal confidence: `0.05`, `0.10`, `0.25`
2. Max proposals per image: `25`, `50`, `100`
3. Context padding ratio: `0.05`, `0.10`, `0.20`
4. Final fusion mode: `nms`, `wbf`
5. YOLO recall fallback: `off`, `on`

The key plot is:

- proposal recall vs final mAP

If proposal recall is low, the rest of the hybrid pipeline cannot recover.

## Interpreting your current results

Our table already suggests:

1. The hybrid stage is over-pruning detections early, which explains the very low false positives.
2. Proposal recall is too low, which explains the large increase in false negatives and the `mAP` drop.
3. The Faster R-CNN stage is still dominating runtime, which explains why hybrid `FPS` is still close to Faster R-CNN.

That combination is not impossible. It is exactly what a mis-tuned cascade looks like.

## Reporting Assets

To generate CSV summaries, comparison plots, and a markdown report from experiment outputs:

```bash
python generate_report_assets.py
```

This writes:

- `reports/tables/final_model_comparison.csv`
- `reports/tables/experiment_runs.csv`
- `reports/tables/stage_timing_breakdown.csv`
- `reports/figures/*.png`
- `reports/FINAL_REPORT.md`
