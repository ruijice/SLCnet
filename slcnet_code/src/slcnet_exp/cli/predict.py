from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from slcnet_exp.common.io import read_config, write_json
from slcnet_exp.data import COCOHBBDetectionDataset, collate_detection_batch, invert_letterbox_boxes
from slcnet_exp.metrics.coco_eval import evaluate_coco
from slcnet_exp.models import SLCNetConfig, SLCNetDetector


def _model_config(payload: dict[str, Any], checkpoint: dict[str, Any]) -> SLCNetConfig:
    source = checkpoint.get("model_config", payload.get("model", {}))
    allowed = {item.name for item in fields(SLCNetConfig)}
    config = SLCNetConfig(**{key: value for key, value in source.items() if key in allowed})
    config.validate()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone SLCNet inference and export COCO HBB predictions.")
    parser.add_argument("--config", required=True, help="JSON or YAML config with data paths.")
    parser.add_argument("--checkpoint", required=True, help="SLCNet checkpoint produced by the train command.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", required=True, help="COCO prediction JSON destination.")
    parser.add_argument("--metrics-out", default=None, help="Optional metrics JSON path when --gt is supplied.")
    parser.add_argument("--gt", default=None, help="Optional COCO ground-truth JSON for evaluation.")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = read_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = _model_config(payload, checkpoint)
    data = payload["data"]
    annotation_path = data.get(f"{args.split}_annotations", data.get("annotations"))
    image_root = data.get(f"{args.split}_images", data.get("image_root"))
    split_path = data.get(f"{args.split}_split")
    if not annotation_path or not image_root:
        raise ValueError("The selected split requires annotations and image_root data paths.")
    dataset = COCOHBBDetectionDataset(
        annotation_path=annotation_path,
        image_root=image_root,
        split_path=split_path,
        input_size=model_config.input_size,
        in_channels=model_config.in_channels,
        training=False,
        horizontal_flip_probability=0.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_detection_batch)
    model = SLCNetDetector(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets in loader:
            results = model.predict(
                images.to(device),
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )
            for result, target in zip(results, targets, strict=True):
                boxes = invert_letterbox_boxes(result.boxes.cpu(), target.letterbox)
                for box, score, label in zip(boxes, result.scores.cpu(), result.labels.cpu(), strict=True):
                    x1, y1, x2, y2 = (float(value) for value in box)
                    predictions.append(
                        {
                            "image_id": int(target.image_id),
                            "category_id": dataset.category_ids[int(label)],
                            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            "score": float(score),
                        }
                    )
    output_path = Path(args.out)
    write_json(output_path, predictions)
    print(f"Wrote {len(predictions)} predictions to {output_path}")
    if args.gt:
        metrics = evaluate_coco(args.gt, output_path)
        if args.metrics_out:
            write_json(args.metrics_out, metrics)
        print(metrics)


if __name__ == "__main__":
    main()
