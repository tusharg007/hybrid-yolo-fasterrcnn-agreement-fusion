from __future__ import annotations

from statistics import mean

from tqdm import tqdm

from hybrid_detector.config import EvalConfig
from hybrid_detector.geometry import count_fp_fn


def evaluate_detector(detector, dataset, config: EvalConfig) -> dict:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(
        iou_type="bbox",
        max_detection_thresholds=[100, 300, 1000],
    )
    metric.warn_on_many_detections = False
    total_fp = 0
    total_fn = 0
    timings = {"yolo": [], "crop": [], "frcnn": [], "post": [], "total": []}

    num_images = len(dataset) if config.max_images is None else min(len(dataset), config.max_images)
    for idx in tqdm(range(num_images), desc="Evaluating"):
        sample = dataset[idx]
        output = detector.predict(sample.image)

        preds = [
            {
                "boxes": output.boxes,
                "scores": output.scores,
                "labels": output.labels,
            }
        ]
        target = [
            {
                "boxes": sample.target["boxes"],
                "labels": sample.target["labels"],
            }
        ]
        metric.update(preds, target)

        fp, fn = count_fp_fn(
            pred_boxes=output.boxes,
            pred_labels=output.labels,
            pred_scores=output.scores,
            gt_boxes=sample.target["boxes"],
            gt_labels=sample.target["labels"],
            iou_threshold=config.iou_match_threshold,
        )
        total_fp += fp
        total_fn += fn

        for name, value in output.timings.items():
            timings[name].append(value)

    result = metric.compute()
    mean_total = mean(timings["total"]) if timings["total"] else 0.0
    fps = (1.0 / mean_total) if mean_total > 0 else 0.0
    return {
        "mAP@0.5": float(result["map_50"].item()),
        "mAP@0.5:0.95": float(result["map"].item()),
        "mean_inf_time_s": mean_total,
        "fps": fps,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "stage_timings": {name: mean(values) if values else 0.0 for name, values in timings.items()},
    }
