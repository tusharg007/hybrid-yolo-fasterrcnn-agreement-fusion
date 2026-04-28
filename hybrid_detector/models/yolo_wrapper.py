from __future__ import annotations

import torch

from hybrid_detector.models.hybrid import DetectionOutput


class YOLOProposalModel:
    def __init__(self, weights: str, device: str, class_map: dict[int, int] | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.class_map = class_map or {}

    def predict(
        self,
        image: torch.Tensor,
        conf: float,
        iou: float,
        max_det: int,
    ) -> dict[str, torch.Tensor]:
        image_uint8 = (image.clamp(0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy()
        results = self.model.predict(
            source=image_uint8,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            device=self.device,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        boxes = results.boxes.xyxy.detach().cpu().float()
        scores = results.boxes.conf.detach().cpu().float()
        labels = results.boxes.cls.detach().cpu().long()
        if self.class_map:
            mapped = torch.tensor([self.class_map.get(int(label), 0) for label in labels.tolist()], dtype=torch.long)
            labels = mapped

        valid = labels > 0
        return {
            "boxes": boxes[valid],
            "scores": scores[valid],
            "labels": labels[valid],
        }


class StandaloneYOLODetector:
    def __init__(self, weights: str, device: str, class_map: dict[int, int] | None = None) -> None:
        self.model = YOLOProposalModel(weights=weights, device=device, class_map=class_map)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> DetectionOutput:
        import time

        t0 = time.perf_counter()
        pred = self.model.predict(image=image, conf=0.001, iou=0.7, max_det=300)
        t1 = time.perf_counter()
        return DetectionOutput(
            boxes=pred["boxes"],
            scores=pred["scores"],
            labels=pred["labels"],
            timings={"yolo": t1 - t0, "crop": 0.0, "frcnn": 0.0, "post": 0.0, "total": t1 - t0},
        )
