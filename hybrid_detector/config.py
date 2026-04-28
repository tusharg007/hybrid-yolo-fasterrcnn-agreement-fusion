from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class HybridConfig:
    proposal_conf: float = 0.10
    proposal_iou: float = 0.70
    max_proposals: int = 50
    proposal_padding: float = 0.10
    refinement_score_thresh: float = 0.25
    final_score_thresh: float = 0.25
    final_nms_iou: float = 0.50
    fusion: Literal["nms", "wbf"] = "nms"
    use_yolo_fallback: bool = True
    yolo_fallback_score_thresh: float = 0.35
    wbf_iou: float = 0.55
    wbf_skip_box_thresh: float = 0.001


@dataclass(slots=True)
class EvalConfig:
    iou_match_threshold: float = 0.50
    device: str = "cuda"
    max_images: int | None = None
    warmup_images: int = 10
