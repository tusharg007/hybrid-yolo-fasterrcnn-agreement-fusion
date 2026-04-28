from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from hybrid_detector.config import HybridConfig
from hybrid_detector.geometry import (
    class_aware_nms,
    clip_boxes_xyxy,
    expand_boxes_xyxy,
    filter_small_boxes,
    weighted_box_fusion,
)


@dataclass
class DetectionOutput:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    timings: dict[str, float]


class HybridDetector:
    def __init__(self, yolo_model, frcnn_model, config: HybridConfig) -> None:
        self.yolo_model = yolo_model
        self.frcnn_model = frcnn_model
        self.config = config

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> DetectionOutput:
        _, height, width = image.shape

        t0 = time.perf_counter()
        yolo = self.yolo_model.predict(
            image=image,
            conf=self.config.proposal_conf,
            iou=self.config.proposal_iou,
            max_det=self.config.max_proposals,
        )
        t1 = time.perf_counter()

        proposal_boxes = expand_boxes_xyxy(yolo["boxes"], width, height, self.config.proposal_padding)
        proposal_scores = yolo["scores"]
        proposal_labels = yolo["labels"]

        valid = filter_small_boxes(proposal_boxes)
        proposal_boxes = proposal_boxes[valid]
        proposal_scores = proposal_scores[valid]
        proposal_labels = proposal_labels[valid]

        crops = []
        crop_scores = []
        crop_origins = []
        for box, score, label in zip(proposal_boxes.tolist(), proposal_scores.tolist(), proposal_labels.tolist()):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            crop = image[:, y1:y2, x1:x2]
            if crop.numel() == 0:
                continue
            crops.append(crop)
            crop_scores.append(float(score))
            crop_origins.append((x1, y1))

        t2 = time.perf_counter()
        refined = self.frcnn_model.predict_crops(crops, batch_size=self.config.frcnn_batch_size)
        t3 = time.perf_counter()

        final_boxes = []
        final_scores = []
        final_labels = []

        for origin, crop_out, proposal_score in zip(
            crop_origins,
            refined,
            crop_scores,
        ):
            if crop_out["boxes"].numel() == 0:
                continue

            keep = crop_out["scores"] >= self.config.refinement_score_thresh
            boxes = crop_out["boxes"][keep]
            scores = crop_out["scores"][keep]
            labels = crop_out["labels"][keep]
            if boxes.numel() == 0:
                continue

            ox, oy = origin
            boxes[:, 0::2] += ox
            boxes[:, 1::2] += oy
            boxes = clip_boxes_xyxy(boxes, width, height)

            for box, score, label in zip(boxes, scores, labels):
                if int(label.item()) <= 0:
                    continue
                fused_score = float(score.item()) * float(proposal_score)
                final_boxes.append(box)
                final_scores.append(torch.tensor(fused_score, dtype=torch.float32))
                final_labels.append(label)

        if self.config.use_yolo_fallback:
            yolo_keep = proposal_scores >= self.config.yolo_fallback_score_thresh
            for box, score, label in zip(proposal_boxes[yolo_keep], proposal_scores[yolo_keep], proposal_labels[yolo_keep]):
                final_boxes.append(box.clone())
                final_scores.append(score.clone())
                final_labels.append(label.clone())

        if not final_boxes:
            timings = {
                "yolo": t1 - t0,
                "crop": t2 - t1,
                "frcnn": t3 - t2,
                "post": 0.0,
                "total": t3 - t0,
            }
            return DetectionOutput(
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.int64),
                timings=timings,
            )

        boxes = torch.stack(final_boxes)
        scores = torch.stack(final_scores)
        labels = torch.stack(final_labels)

        score_keep = scores >= self.config.final_score_thresh
        boxes = boxes[score_keep]
        scores = scores[score_keep]
        labels = labels[score_keep]
        valid_labels = labels > 0
        boxes = boxes[valid_labels]
        scores = scores[valid_labels]
        labels = labels[valid_labels]

        if boxes.numel() == 0:
            timings = {
                "yolo": t1 - t0,
                "crop": t2 - t1,
                "frcnn": t3 - t2,
                "post": 0.0,
                "total": t3 - t0,
            }
            return DetectionOutput(
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.int64),
                timings=timings,
            )

        t4 = time.perf_counter()
        if self.config.fusion == "wbf":
            boxes, scores, labels = weighted_box_fusion(
                boxes=boxes,
                scores=scores,
                labels=labels,
                iou_threshold=self.config.wbf_iou,
                skip_box_thresh=self.config.wbf_skip_box_thresh,
            )
        keep = class_aware_nms(boxes, scores, labels, self.config.final_nms_iou)
        t5 = time.perf_counter()

        timings = {
            "yolo": t1 - t0,
            "crop": t2 - t1,
            "frcnn": t3 - t2,
            "post": t5 - t4,
            "total": t5 - t0,
        }
        return DetectionOutput(boxes=boxes[keep], scores=scores[keep], labels=labels[keep], timings=timings)
