from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset

from hybrid_detector.label_maps import VOC_CLASS_NAMES


@dataclass(frozen=True)
class DatasetSample:
    image: torch.Tensor
    target: dict
    image_id: str


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    channels = len(image.getbands())
    tensor = data.view(image.size[1], image.size[0], channels).permute(2, 0, 1).float() / 255.0
    return tensor


class VOCDataset(Dataset):
    VOC_CLASSES = VOC_CLASS_NAMES
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    def __init__(self, root: str, split: str = "val") -> None:
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")
        self.class_names = list(self.VOC_CLASSES)
        split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file, "r", encoding="utf-8") as fh:
            self.ids = [line.strip() for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> DatasetSample:
        image_id = self.ids[index]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        image = Image.open(image_path).convert("RGB")
        image_tensor = pil_to_tensor(image)

        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            difficult = int(obj.findtext("difficult", default="0"))
            if difficult:
                continue
            name = obj.findtext("name")
            bnd = obj.find("bndbox")
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.CLASS_TO_IDX[name])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return DatasetSample(image=image_tensor, target=target, image_id=image_id)


class COCODataset(Dataset):
    def __init__(self, image_dir: str, annotation_file: str) -> None:
        from pycocotools.coco import COCO

        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = sorted(self.coco.imgs.keys())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = sorted(cat["id"] for cat in cats)
        self.cat_to_idx = {cat_id: idx + 1 for idx, cat_id in enumerate(self.category_ids)}
        self.class_names = ["__background__"] + [cat["name"] for cat in sorted(cats, key=lambda cat: cat["id"])]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> DatasetSample:
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        image_path = os.path.join(self.image_dir, info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = pil_to_tensor(image)

        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_to_idx[ann["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return DatasetSample(image=image_tensor, target=target, image_id=str(image_id))
