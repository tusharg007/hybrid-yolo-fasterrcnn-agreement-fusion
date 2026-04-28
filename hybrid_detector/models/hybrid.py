from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torchvision.ops import box_iou

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
        if self.config.strategy == "crop_refine":
            return self._predict_crop_refine(image)
        return self._predict_agreement_fusion(image)

    def _predict_agreement_fusion(self, image: torch.Tensor) -> DetectionOutput:
        t0 = time.perf_counter()
        yolo = self.yolo_model.predict(
            image=image,
            conf=self.config.proposal_conf,
            iou=self.config.proposal_iou,
            max_det=self.config.max_proposals,
        )
        t1 = time.perf_counter()

        frcnn = self.frcnn_model.predict_full_image(image)
        t2 = time.perf_counter()

        frcnn_keep = frcnn["scores"] >= self.config.frcnn_score_thresh
        frcnn_boxes = frcnn["boxes"][frcnn_keep]
        frcnn_scores = frcnn["scores"][frcnn_keep]
        frcnn_labels = frcnn["labels"][frcnn_keep]

        yolo_boxes = yolo["boxes"]
        yolo_scores = yolo["scores"]
        yolo_labels = yolo["labels"]

        final_boxes: list[torch.Tensor] = []
        final_scores: list[torch.Tensor] = []
        final_labels: list[torch.Tensor] = []
        matched_yolo = torch.zeros((len(yolo_boxes),), dtype=torch.bool)

        for box, score, label in zip(frcnn_boxes, frcnn_scores, frcnn_labels):
            if int(label.item()) <= 0:
                continue

            same_label = torch.nonzero(yolo_labels == label, as_tuple=False).squeeze(1)
            matched = False
            if same_label.numel() > 0:
                overlaps = box_iou(box.unsqueeze(0), yolo_boxes[same_label])[0]
                best_local = int(torch.argmax(overlaps).item())
                best_iou = float(overlaps[best_local].item())
                if best_iou >= self.config.agreement_iou:
                    yolo_idx = int(same_label[best_local].item())
                    matched_yolo[yolo_idx] = True
                    fused_box = self._fuse_boxes(
                        frcnn_box=box,
                        frcnn_score=float(score.item()),
                        yolo_box=yolo_boxes[yolo_idx],
                        yolo_score=float(yolo_scores[yolo_idx].item()),
                    )
                    fused_score = self._fuse_scores(
                        frcnn_score=float(score.item()),
                        yolo_score=float(yolo_scores[yolo_idx].item()),
                    )
                    final_boxes.append(fused_box)
                    final_scores.append(torch.tensor(fused_score, dtype=torch.float32))
                    final_labels.append(label)
                    matched = True

            if not matched and float(score.item()) >= self.config.unmatched_frcnn_score_thresh:
                final_boxes.append(box.clone())
                final_scores.append(score.clone())
                final_labels.append(label.clone())

        if self.config.use_yolo_fallback:
            for idx, (box, score, label) in enumerate(zip(yolo_boxes, yolo_scores, yolo_labels)):
                if matched_yolo[idx]:
                    continue
                if float(score.item()) < self.config.unmatched_yolo_score_thresh or int(label.item()) <= 0:
                    continue
                final_boxes.append(box.clone())
                final_scores.append(score.clone())
                final_labels.append(label.clone())

        return self._finalize(
            final_boxes=final_boxes,
            final_scores=final_scores,
            final_labels=final_labels,
            timings={
                "yolo": t1 - t0,
                "crop": 0.0,
                "frcnn": t2 - t1,
                "post": 0.0,
                "total": t2 - t0,
            },
        )

    def _predict_crop_refine(self, image: torch.Tensor) -> DetectionOutput:
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
                fused_score = self._fuse_scores(
                    frcnn_score=float(score.item()),
                    yolo_score=float(proposal_score),
                )
                final_boxes.append(box)
                final_scores.append(torch.tensor(fused_score, dtype=torch.float32))
                final_labels.append(label)

        if self.config.use_yolo_fallback:
            yolo_keep = proposal_scores >= self.config.yolo_fallback_score_thresh
            for box, score, label in zip(proposal_boxes[yolo_keep], proposal_scores[yolo_keep], proposal_labels[yolo_keep]):
                final_boxes.append(box.clone())
                final_scores.append(score.clone())
                final_labels.append(label.clone())

        return self._finalize(
            final_boxes=final_boxes,
            final_scores=final_scores,
            final_labels=final_labels,
            timings={
                "yolo": t1 - t0,
                "crop": t2 - t1,
                "frcnn": t3 - t2,
                "post": 0.0,
                "total": t3 - t0,
            },
        )

    def _fuse_boxes(
        self,
        frcnn_box: torch.Tensor,
        frcnn_score: float,
        yolo_box: torch.Tensor,
        yolo_score: float,
    ) -> torch.Tensor:
        total = max(frcnn_score + yolo_score, 1e-6)
        return ((frcnn_box * frcnn_score) + (yolo_box * yolo_score)) / total

    def _fuse_scores(self, frcnn_score: float, yolo_score: float) -> float:
        if self.config.score_fusion == "frcnn":
            return frcnn_score
        if self.config.score_fusion == "max":
            return max(frcnn_score, yolo_score)
        if self.config.score_fusion == "geometric_mean":
            return float((frcnn_score * yolo_score) ** 0.5)
        return float((0.75 * frcnn_score) + (0.25 * yolo_score))

    def _finalize(
        self,
        final_boxes: list[torch.Tensor],
        final_scores: list[torch.Tensor],
        final_labels: list[torch.Tensor],
        timings: dict[str, float],
    ) -> DetectionOutput:
        if not final_boxes:
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
            return DetectionOutput(
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.int64),
                timings=timings,
            )

        t_post_start = time.perf_counter()
        if self.config.fusion == "wbf":
            boxes, scores, labels = weighted_box_fusion(
                boxes=boxes,
                scores=scores,
                labels=labels,
                iou_threshold=self.config.wbf_iou,
                skip_box_thresh=self.config.wbf_skip_box_thresh,
            )
        keep = class_aware_nms(boxes, scores, labels, self.config.final_nms_iou)
        t_post_end = time.perf_counter()
        timings = dict(timings)
        timings["post"] = t_post_end - t_post_start
        timings["total"] += timings["post"]
        return DetectionOutput(boxes=boxes[keep], scores=scores[keep], labels=labels[keep], timings=timings)
