from __future__ import annotations

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from hybrid_detector.models.hybrid import DetectionOutput


class FasterRCNNRefiner:
    def __init__(
        self,
        weights: str,
        device: str,
        num_classes: int | None = None,
        class_map: dict[int, int] | None = None,
    ) -> None:
        if weights == "DEFAULT":
            model_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=model_weights)
        else:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)
            state = torch.load(weights, map_location="cpu")
            self.model.load_state_dict(state)

        self.model.eval()
        self.model.to(device)
        self.device = device
        self.class_map = class_map or {}

    @torch.inference_mode()
    def predict_crops(self, crops: list[torch.Tensor], batch_size: int = 4) -> list[dict[str, torch.Tensor]]:
        if not crops:
            return []
        refined = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(crops), batch_size):
            batch = [crop.to(self.device, non_blocking=True) for crop in crops[start : start + batch_size]]
            outputs = self.model(batch)
            for out in outputs:
                labels = out["labels"].detach().cpu()
                if self.class_map:
                    labels = torch.tensor([self.class_map.get(int(label), 0) for label in labels.tolist()], dtype=torch.long)
                refined.append(
                    {
                        "boxes": out["boxes"].detach().cpu(),
                        "scores": out["scores"].detach().cpu(),
                        "labels": labels,
                    }
                )
            del batch
            del outputs
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return refined


class StandaloneFasterRCNNDetector:
    def __init__(self, weights: str, device: str, class_map: dict[int, int] | None = None) -> None:
        self.model = FasterRCNNRefiner(weights=weights, device=device, class_map=class_map)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> DetectionOutput:
        import time

        t0 = time.perf_counter()
        pred = self.model.model([image.to(self.model.device)])[0]
        t1 = time.perf_counter()
        labels = pred["labels"].detach().cpu()
        if self.model.class_map:
            labels = torch.tensor([self.model.class_map.get(int(label), 0) for label in labels.tolist()], dtype=torch.long)

        valid = labels > 0
        return DetectionOutput(
            boxes=pred["boxes"].detach().cpu()[valid],
            scores=pred["scores"].detach().cpu()[valid],
            labels=labels[valid],
            timings={"yolo": 0.0, "crop": 0.0, "frcnn": t1 - t0, "post": 0.0, "total": t1 - t0},
        )
