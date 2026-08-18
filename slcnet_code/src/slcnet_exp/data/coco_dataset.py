from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from slcnet_exp.common.io import read_json
from slcnet_exp.models.interfaces import DetectionTarget


class COCOHBBDetectionDataset(Dataset[tuple[torch.Tensor, DetectionTarget]]):
    """COCO HBB dataset reader with SAR-aware channel conversion and letterboxing."""

    def __init__(
        self,
        annotation_path: str | Path,
        image_root: str | Path,
        split_path: str | Path | None = None,
        input_size: int = 640,
        in_channels: int = 1,
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if in_channels not in {1, 3}:
            raise ValueError("The built-in COCO reader supports one- or three-channel image tensors.")
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.input_size = input_size
        self.in_channels = in_channels
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        coco = read_json(self.annotation_path)
        self.category_ids = sorted(int(category["id"]) for category in coco.get("categories", []))
        if not self.category_ids:
            raise ValueError(f"COCO annotations have no categories: {self.annotation_path}")
        self.category_to_label = {category_id: index for index, category_id in enumerate(self.category_ids)}
        images = {int(image["id"]): image for image in coco.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
        for annotation in coco.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id in annotations_by_image and int(annotation.get("iscrowd", 0)) == 0:
                annotations_by_image[image_id].append(annotation)
        if split_path is None:
            selected_ids = sorted(images)
        else:
            split = read_json(split_path)
            selected_ids = [int(image["id"]) for image in split.get("images", [])]
        missing_ids = [image_id for image_id in selected_ids if image_id not in images]
        if missing_ids:
            raise ValueError(f"Split references image IDs absent from annotations: {missing_ids[:5]}")
        self.records = [(images[image_id], annotations_by_image[image_id]) for image_id in selected_ids]

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        if np.issubdtype(image.dtype, np.integer):
            divisor = float(np.iinfo(image.dtype).max)
        else:
            divisor = 1.0 if float(np.nanmax(image, initial=0.0)) <= 1.0 else 255.0
        return np.nan_to_num(image.astype(np.float32) / max(divisor, 1.0), nan=0.0, posinf=1.0, neginf=0.0)

    def _read_image(self, file_name: str) -> np.ndarray:
        path = Path(file_name)
        image_path = path if path.is_absolute() else self.image_root / path
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        if self.in_channels == 1:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image[..., None]
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _letterbox(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        original_height, original_width = image.shape[:2]
        scale = min(self.input_size / original_height, self.input_size / original_width)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[..., None]
        canvas = np.zeros((self.input_size, self.input_size, resized.shape[2]), dtype=resized.dtype)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        transformed = boxes.copy()
        if transformed.size:
            transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_x
            transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_y
            transformed[:, 0::2] = np.clip(transformed[:, 0::2], 0, self.input_size)
            transformed[:, 1::2] = np.clip(transformed[:, 1::2], 0, self.input_size)
        return canvas, transformed, {
            "scale": float(scale),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
            "original_width": float(original_width),
            "original_height": float(original_height),
            "input_size": float(self.input_size),
            "flipped": 0.0,
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget]:
        image_row, annotations = self.records[index]
        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id not in self.category_to_label:
                continue
            x, y, width, height = (float(value) for value in annotation["bbox"])
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_to_label[category_id])
        image = self._read_image(str(image_row["file_name"]))
        box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        image, box_array, letterbox = self._letterbox(image, box_array)
        if self.training and box_array.size and random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            x1 = box_array[:, 0].copy()
            box_array[:, 0] = self.input_size - box_array[:, 2]
            box_array[:, 2] = self.input_size - x1
            letterbox["flipped"] = 1.0
        normalized = self._normalize(image)
        tensor = torch.from_numpy(np.ascontiguousarray(normalized)).permute(2, 0, 1)
        target = DetectionTarget(
            boxes=torch.from_numpy(box_array),
            labels=torch.as_tensor(labels, dtype=torch.long),
            image_id=int(image_row["id"]),
            original_size=(int(letterbox["original_height"]), int(letterbox["original_width"])),
            letterbox=letterbox,
        )
        return tensor, target


def collate_detection_batch(samples: list[tuple[torch.Tensor, DetectionTarget]]) -> tuple[torch.Tensor, list[DetectionTarget]]:
    if not samples:
        raise ValueError("Cannot collate an empty detection batch.")
    images, targets = zip(*samples, strict=True)
    return torch.stack(images), list(targets)


def invert_letterbox_boxes(boxes: torch.Tensor, letterbox: dict[str, float] | None) -> torch.Tensor:
    """Map detector-coordinate xyxy boxes back to original image coordinates."""

    if letterbox is None or boxes.numel() == 0:
        return boxes
    restored = boxes.clone()
    if letterbox.get("flipped", 0.0):
        x1 = restored[:, 0].clone()
        input_width = float(letterbox.get("input_size", boxes.shape[-1]))
        restored[:, 0] = input_width - restored[:, 2]
        restored[:, 2] = input_width - x1
    scale = float(letterbox["scale"])
    restored[:, 0::2] = (restored[:, 0::2] - float(letterbox["pad_x"])) / scale
    restored[:, 1::2] = (restored[:, 1::2] - float(letterbox["pad_y"])) / scale
    restored[:, 0::2] = restored[:, 0::2].clamp(0, float(letterbox["original_width"]))
    restored[:, 1::2] = restored[:, 1::2].clamp(0, float(letterbox["original_height"]))
    return restored
