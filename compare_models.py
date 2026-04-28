from __future__ import annotations

import argparse
import json

from hybrid_detector.config import EvalConfig, HybridConfig
from hybrid_detector.datasets import COCODataset, VOCDataset
from hybrid_detector.evaluation import evaluate_detector
from hybrid_detector.label_maps import (
    COCO80_CLASS_NAMES,
    TORCHVISION_COCO_CATEGORIES,
    VOC_CLASS_NAMES,
    build_source_to_target_map,
)
from hybrid_detector.models.faster_rcnn_wrapper import FasterRCNNRefiner, StandaloneFasterRCNNDetector
from hybrid_detector.models.hybrid import HybridDetector
from hybrid_detector.models.yolo_wrapper import YOLOProposalModel, StandaloneYOLODetector


def build_dataset(args):
    if args.dataset == "voc":
        return VOCDataset(root=args.dataset_root, split=args.split)
    if args.dataset == "coco":
        return COCODataset(image_dir=args.image_dir, annotation_file=args.annotation_file)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_class_maps(dataset) -> tuple[dict[int, int], dict[int, int]]:
    if dataset.class_names == list(VOC_CLASS_NAMES):
        return (
            build_source_to_target_map(COCO80_CLASS_NAMES, VOC_CLASS_NAMES),
            build_source_to_target_map(TORCHVISION_COCO_CATEGORIES, VOC_CLASS_NAMES),
        )
    return (
        build_source_to_target_map(COCO80_CLASS_NAMES, dataset.class_names),
        build_source_to_target_map(TORCHVISION_COCO_CATEGORIES, dataset.class_names),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLO, Faster R-CNN, and Hybrid on one dataset")
    parser.add_argument("--dataset", choices=["voc", "coco"], required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--yolo-weights", type=str, required=True)
    parser.add_argument("--frcnn-weights", type=str, default="DEFAULT")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--hybrid-strategy", choices=["agreement_fusion", "crop_refine"], default="agreement_fusion")
    parser.add_argument("--proposal-conf", type=float, default=0.10)
    parser.add_argument("--proposal-iou", type=float, default=0.70)
    parser.add_argument("--max-proposals", type=int, default=50)
    parser.add_argument("--frcnn-batch-size", type=int, default=4)
    parser.add_argument("--proposal-padding", type=float, default=0.10)
    parser.add_argument("--frcnn-score-thresh", type=float, default=0.05)
    parser.add_argument("--refinement-score-thresh", type=float, default=0.25)
    parser.add_argument("--final-score-thresh", type=float, default=0.25)
    parser.add_argument("--final-nms-iou", type=float, default=0.50)
    parser.add_argument("--agreement-iou", type=float, default=0.50)
    parser.add_argument("--unmatched-frcnn-score-thresh", type=float, default=0.60)
    parser.add_argument("--unmatched-yolo-score-thresh", type=float, default=0.55)
    parser.add_argument("--score-fusion", choices=["frcnn", "max", "weighted_sum", "geometric_mean"], default="weighted_sum")
    parser.add_argument("--fusion", choices=["nms", "wbf"], default="nms")
    parser.add_argument("--disable-yolo-fallback", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(args)
    yolo_map, frcnn_map = build_class_maps(dataset)
    eval_config = EvalConfig(device=args.device, max_images=args.max_images)
    hybrid_config = HybridConfig(
        strategy=args.hybrid_strategy,
        proposal_conf=args.proposal_conf,
        proposal_iou=args.proposal_iou,
        max_proposals=args.max_proposals,
        frcnn_batch_size=args.frcnn_batch_size,
        proposal_padding=args.proposal_padding,
        frcnn_score_thresh=args.frcnn_score_thresh,
        refinement_score_thresh=args.refinement_score_thresh,
        final_score_thresh=args.final_score_thresh,
        final_nms_iou=args.final_nms_iou,
        fusion=args.fusion,
        use_yolo_fallback=not args.disable_yolo_fallback,
        agreement_iou=args.agreement_iou,
        unmatched_frcnn_score_thresh=args.unmatched_frcnn_score_thresh,
        unmatched_yolo_score_thresh=args.unmatched_yolo_score_thresh,
        score_fusion=args.score_fusion,
    )

    yolo_detector = StandaloneYOLODetector(weights=args.yolo_weights, device=args.device, class_map=yolo_map)
    frcnn_detector = StandaloneFasterRCNNDetector(weights=args.frcnn_weights, device=args.device, class_map=frcnn_map)
    hybrid_detector = HybridDetector(
        yolo_model=YOLOProposalModel(weights=args.yolo_weights, device=args.device, class_map=yolo_map),
        frcnn_model=FasterRCNNRefiner(weights=args.frcnn_weights, device=args.device, class_map=frcnn_map),
        config=hybrid_config,
    )

    results = {
        "yolo": evaluate_detector(yolo_detector, dataset, eval_config),
        "frcnn": evaluate_detector(frcnn_detector, dataset, eval_config),
        "hybrid": evaluate_detector(hybrid_detector, dataset, eval_config),
    }
    frcnn_time = results["frcnn"]["mean_inf_time_s"]
    for result in results.values():
        result["speedup_vs_frcnn"] = (frcnn_time / result["mean_inf_time_s"]) if result["mean_inf_time_s"] > 0 else 0.0

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
