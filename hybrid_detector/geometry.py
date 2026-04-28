from __future__ import annotations

from collections import defaultdict

import torch
from torchvision.ops import box_iou, nms


def clip_boxes_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(0, width)
    clipped[:, 1::2] = clipped[:, 1::2].clamp(0, height)
    return clipped


def expand_boxes_xyxy(boxes: torch.Tensor, width: int, height: int, ratio: float) -> torch.Tensor:
    if boxes.numel() == 0 or ratio <= 0:
        return clip_boxes_xyxy(boxes, width, height)

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    w = x2 - x1
    h = y2 - y1
    dx = w * ratio
    dy = h * ratio
    expanded = torch.stack((x1 - dx, y1 - dy, x2 + dx, y2 + dy), dim=1)
    return clip_boxes_xyxy(expanded, width, height)


def filter_small_boxes(boxes: torch.Tensor, min_size: float = 2.0) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes.device)
    wh = boxes[:, 2:] - boxes[:, :2]
    return (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    keep_chunks = []
    for label in labels.unique():
        idx = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        label_keep = nms(boxes[idx], scores[idx], iou_threshold)
        keep_chunks.append(idx[label_keep])

    keep = torch.cat(keep_chunks)
    order = scores[keep].argsort(descending=True)
    return keep[order]


def weighted_box_fusion(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    skip_box_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels

    keep = scores >= skip_box_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if boxes.numel() == 0:
        return boxes, scores, labels

    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for label in labels.unique():
        idxs = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        label_boxes = boxes[idxs]
        label_scores = scores[idxs]
        order = label_scores.argsort(descending=True)
        label_boxes = label_boxes[order]
        label_scores = label_scores[order]

        used = torch.zeros(len(label_boxes), dtype=torch.bool, device=boxes.device)
        for i in range(len(label_boxes)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            ious = box_iou(label_boxes[i].unsqueeze(0), label_boxes)[0]
            matches = torch.nonzero((ious >= iou_threshold) & (~used), as_tuple=False).squeeze(1)
            for match in matches.tolist():
                used[match] = True
                cluster.append(match)

            cluster_boxes = label_boxes[cluster]
            cluster_scores = label_scores[cluster]
            weights = cluster_scores / cluster_scores.sum().clamp_min(1e-6)
            fused_box = (cluster_boxes * weights[:, None]).sum(dim=0)
            fused_score = cluster_scores.mean()
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(label)

    return (
        torch.stack(fused_boxes, dim=0),
        torch.stack(fused_scores, dim=0),
        torch.stack(fused_labels, dim=0),
    )


def count_fp_fn(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int]:
    if pred_boxes.numel() == 0:
        return 0, int(gt_boxes.shape[0])
    if gt_boxes.numel() == 0:
        return int(pred_boxes.shape[0]), 0

    false_positives = 0
    false_negatives = 0
    matched_gt = defaultdict(set)

    score_order = pred_scores.argsort(descending=True)
    for pred_idx in score_order.tolist():
        label = int(pred_labels[pred_idx].item())
        gt_idx = torch.nonzero(gt_labels == label, as_tuple=False).squeeze(1)
        if gt_idx.numel() == 0:
            false_positives += 1
            continue

        ious = box_iou(pred_boxes[pred_idx].unsqueeze(0), gt_boxes[gt_idx])[0]
        best_local = int(torch.argmax(ious).item())
        best_iou = float(ious[best_local].item())
        best_gt = int(gt_idx[best_local].item())
        if best_iou >= iou_threshold and best_gt not in matched_gt[label]:
            matched_gt[label].add(best_gt)
        else:
            false_positives += 1

    total_matched = sum(len(v) for v in matched_gt.values())
    false_negatives = int(gt_boxes.shape[0] - total_matched)
    return false_positives, false_negatives
