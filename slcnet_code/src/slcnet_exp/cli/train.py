from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from slcnet_exp.common.io import ensure_dir, read_config, write_json
from slcnet_exp.data import COCOHBBDetectionDataset, collate_detection_batch
from slcnet_exp.models import SLCNetConfig, SLCNetDetector


def _model_config(payload: dict[str, Any]) -> SLCNetConfig:
    allowed = {item.name for item in fields(SLCNetConfig)}
    values = {key: value for key, value in payload.get("model", {}).items() if key in allowed}
    config = SLCNetConfig(**values)
    config.validate()
    return config


def _dataset(payload: dict[str, Any], split: str, model_config: SLCNetConfig, training: bool) -> COCOHBBDetectionDataset:
    data = payload["data"]
    annotation_path = data.get(f"{split}_annotations", data.get("annotations"))
    image_root = data.get(f"{split}_images", data.get("image_root"))
    split_path = data.get(f"{split}_split")
    if not annotation_path or not image_root:
        raise ValueError(f"data.{split}_annotations/annotations and data.{split}_images/image_root are required.")
    return COCOHBBDetectionDataset(
        annotation_path=annotation_path,
        image_root=image_root,
        split_path=split_path,
        input_size=model_config.input_size,
        in_channels=model_config.in_channels,
        training=training,
        horizontal_flip_probability=float(data.get("horizontal_flip_probability", 0.5 if training else 0.0)),
    )


def _mean_loss(model: SLCNetDetector, loader: DataLoader[Any], device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    totals: dict[str, float] = {}
    batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        output = model(images, return_aux=False)
        losses = model.loss(output, targets)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
    if not batches:
        raise ValueError("The data loader produced no batches.")
    return {name: value / batches for name, value in totals.items()}


def _checkpoint(
    path: Path,
    epoch: int,
    model: SLCNetDetector,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_validation: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_validation": best_validation,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the standalone PyTorch SLCNet HBB detector.")
    parser.add_argument("--config", required=True, help="JSON or YAML config with model, data, and training mappings.")
    parser.add_argument("--output-dir", default=None, help="Overrides training.output_dir in the config.")
    parser.add_argument("--resume", default=None, help="Checkpoint to restore before continuing training.")
    parser.add_argument("--device", default=None, help="Explicit torch device, e.g. cpu or cuda:0.")
    args = parser.parse_args()

    payload = read_config(args.config)
    training = payload.get("training", {})
    model_config = _model_config(payload)
    output_dir = ensure_dir(args.output_dir or training.get("output_dir", "runs/slcnet"))
    device = torch.device(args.device or training.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda is unavailable.")
    train_dataset = _dataset(payload, "train", model_config, training=True)
    validation_dataset = _dataset(payload, "val", model_config, training=False)
    batch_size = int(training.get("batch_size", 4))
    workers = int(training.get("workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
    )
    model = SLCNetDetector(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    start_epoch = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = _mean_loss(model, train_loader, device, optimizer)
        model.eval()
        with torch.no_grad():
            validation_losses = _mean_loss(model, validation_loader, device, optimizer=None)
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_losses,
            "validation": validation_losses,
        }
        history.append(record)
        improved = validation_losses["total"] < best_validation
        if improved:
            best_validation = validation_losses["total"]
            _checkpoint(output_dir / "best.pt", epoch, model, optimizer, scheduler, best_validation)
        _checkpoint(output_dir / "last.pt", epoch, model, optimizer, scheduler, best_validation)
        write_json(output_dir / "history.json", history)
        print(record)


if __name__ == "__main__":
    main()
