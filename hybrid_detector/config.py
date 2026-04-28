from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class HybridConfig:
    strategy: Literal["agreement_fusion", "crop_refine"] = "agreement_fusion"
    proposal_conf: float = 0.10
    proposal_iou: float = 0.70
    max_proposals: int = 50
    frcnn_batch_size: int = 4
    proposal_padding: float = 0.10
    frcnn_score_thresh: float = 0.05
    refinement_score_thresh: float = 0.25
    final_score_thresh: float = 0.25
    final_nms_iou: float = 0.50
    fusion: Literal["nms", "wbf"] = "nms"
    use_yolo_fallback: bool = True
    yolo_fallback_score_thresh: float = 0.35
    agreement_iou: float = 0.50
    unmatched_frcnn_score_thresh: float = 0.60
    unmatched_yolo_score_thresh: float = 0.55
    score_fusion: Literal["frcnn", "max", "weighted_sum", "geometric_mean"] = "weighted_sum"
    wbf_iou: float = 0.55
    wbf_skip_box_thresh: float = 0.001


@dataclass(slots=True)
class EvalConfig:
    iou_match_threshold: float = 0.50
    device: str = "cuda"
    max_images: int | None = None
    warmup_images: int = 10
