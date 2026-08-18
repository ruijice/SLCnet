from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional

from slcnet_exp.models.interfaces import (
    DetectionResult,
    DetectionTarget,
    ScalePrediction,
    SLCNetConfig,
    SLCNetOutput,
)


def _conv_norm_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = _conv_norm_act(channels, channels)
        self.second = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(features + self.second(self.first(features)))


class HierarchicalFeatureExtractor(nn.Module):
    """Independent SAR backbone emitting spatial P3, contextual P4, and semantic P5."""

    def __init__(self, config: SLCNetConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.out_channels = (base * 4, base * 6, base * 8)
        self.stem = _conv_norm_act(config.in_channels, base, stride=2)
        self.stage2 = nn.Sequential(_conv_norm_act(base, base * 2, stride=2), ResidualBlock(base * 2))
        self.stage3 = nn.Sequential(_conv_norm_act(base * 2, base * 4, stride=2), ResidualBlock(base * 4))
        self.stage4 = nn.Sequential(_conv_norm_act(base * 4, base * 6, stride=2), ResidualBlock(base * 6))
        self.stage5 = nn.Sequential(_conv_norm_act(base * 6, base * 8, stride=2), ResidualBlock(base * 8))

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stage1 = self.stem(images)
        stage2 = self.stage2(stage1)
        p3 = self.stage3(stage2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class CoastalCueEncoder(nn.Module):
    """Image-only coastal-risk and cue-reliability estimators without mask supervision."""

    def __init__(self, in_channels: int, cue_kernel: int) -> None:
        super().__init__()
        self.cue_kernel = cue_kernel
        hidden = 16
        self.risk_encoder = nn.Sequential(_conv_norm_act(4, hidden), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
        self.reliability_encoder = nn.Sequential(_conv_norm_act(3, hidden), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
        self.in_channels = in_channels

    def _cue_stacks(self, images: Tensor) -> tuple[Tensor, Tensor]:
        gray = images.mean(dim=1, keepdim=True)
        kernel = self.cue_kernel
        local_mean = functional.avg_pool2d(gray, kernel, stride=1, padding=kernel // 2)
        local_variance = functional.avg_pool2d((gray - local_mean).square(), kernel, stride=1, padding=kernel // 2)
        dispersion = local_variance.clamp_min(0.0).sqrt()
        gradient_x = functional.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
        gradient_y = functional.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
        directional_transition = 0.5 * (gradient_x + gradient_y)
        local_peak = functional.max_pool2d(gray, kernel, stride=1, padding=kernel // 2)
        peak_contrast = (local_peak - local_mean).clamp_min(0.0)
        risk_stack = torch.cat((local_mean, dispersion, directional_transition, peak_contrast), dim=1)
        stability_stack = torch.cat(
            (
                (1.0 + dispersion).reciprocal(),
                (1.0 + directional_transition).reciprocal(),
                (1.0 + peak_contrast).reciprocal(),
            ),
            dim=1,
        )
        return risk_stack, stability_stack

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        risk_stack, stability_stack = self._cue_stacks(images.float())
        return {
            "risk": self.risk_encoder(risk_stack),
            "reliability": self.reliability_encoder(stability_stack),
            "risk_stack": risk_stack,
            "stability_stack": stability_stack,
        }


class ReliabilityCalibratedGuidance(nn.Module):
    """Feature-conditioned soft guidance with the paper's neutral reliability contraction."""

    def __init__(self, channels: int, alpha_init: float) -> None:
        super().__init__()
        initial_logit = log(alpha_init / (1.0 - alpha_init))
        self.prior_projection = nn.Conv2d(1, channels, 1, bias=False)
        self.gate_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate_norm = nn.BatchNorm2d(channels)
        self.alpha_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def forward(self, features: Tensor, prior: Tensor, reliability: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        aligned_prior = functional.interpolate(prior, size=features.shape[-2:], mode="bilinear", align_corners=False)
        aligned_reliability = functional.interpolate(
            reliability, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        raw_gate = torch.sigmoid(self.gate_norm(self.gate_projection(features * self.prior_projection(aligned_prior))))
        gate = 0.5 + aligned_reliability * (raw_gate - 0.5)
        alpha = self.alpha_logit.sigmoid()
        delta = features * (2.0 * gate - 1.0)
        guided = features + alpha * delta
        return guided, {
            "prior": aligned_prior,
            "reliability": aligned_reliability,
            "raw_gate": raw_gate,
            "gate": gate,
            "delta": delta,
            "alpha": alpha,
        }


class LocalContextAggregation(nn.Module):
    """Depthwise local scattering-neighborhood verification after soft guidance."""

    def __init__(self, channels: int, context_kernel: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            context_kernel,
            padding=context_kernel // 2,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, guided_features: Tensor) -> Tensor:
        context = self.pointwise(functional.silu(self.norm(self.depthwise(guided_features))))
        return guided_features + context


class GuidedContextBlock(nn.Module):
    def __init__(self, channels: int, config: SLCNetConfig) -> None:
        super().__init__()
        self.guidance = ReliabilityCalibratedGuidance(channels, config.alpha_init)
        self.context = LocalContextAggregation(channels, config.context_kernel)

    def forward(self, features: Tensor, prior: Tensor, reliability: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        guided, evidence = self.guidance(features, prior, reliability)
        context = self.context(guided)
        evidence["guided"] = guided
        evidence["context"] = context
        return context, evidence


class MultiEvidenceSmallObjectBranch(nn.Module):
    """Three complementary high-resolution routes conditioned on P3 context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.detail_stem = nn.Sequential(_conv_norm_act(channels, channels), ResidualBlock(channels))
        self.local_detail = nn.Sequential(_conv_norm_act(channels, channels), _conv_norm_act(channels, channels))
        self.weak_target = nn.Sequential(_conv_norm_act(channels, channels, kernel_size=1), _conv_norm_act(channels, channels))
        self.fine_edge = nn.Sequential(
            _conv_norm_act(channels, channels, kernel_size=3, groups=channels),
            _conv_norm_act(channels, channels, kernel_size=1),
        )
        self.fuse = _conv_norm_act(channels * 3, channels, kernel_size=1)
        self.merge = nn.Sequential(_conv_norm_act(channels * 2, channels, kernel_size=1), ResidualBlock(channels))

    def forward(self, p3_features: Tensor, context_features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        detail = self.detail_stem(p3_features) + p3_features
        local = self.local_detail(detail)
        weak = self.weak_target(detail)
        edge = self.fine_edge(detail)
        sob = self.fuse(torch.cat((local, weak, edge), dim=1))
        merged = self.merge(torch.cat((sob, context_features), dim=1)) + context_features
        return merged, {"detail": detail, "local_detail": local, "weak_target": weak, "fine_edge": edge, "sob": sob}


class SLCNeck(nn.Module):
    """Bidirectional fusion with paper-aligned guidance at P3/P4 only."""

    def __init__(self, backbone_channels: tuple[int, int, int], config: SLCNetConfig) -> None:
        super().__init__()
        channels = config.neck_channels
        p3_channels, p4_channels, p5_channels = backbone_channels
        self.lateral3 = nn.Conv2d(p3_channels, channels, 1, bias=False)
        self.lateral4 = nn.Conv2d(p4_channels, channels, 1, bias=False)
        self.lateral5 = nn.Conv2d(p5_channels, channels, 1, bias=False)
        self.top_down4 = nn.Sequential(_conv_norm_act(channels, channels), ResidualBlock(channels))
        self.top_down3 = nn.Sequential(_conv_norm_act(channels, channels), ResidualBlock(channels))
        self.guided4 = GuidedContextBlock(channels, config)
        self.guided3 = GuidedContextBlock(channels, config)
        self.sob = MultiEvidenceSmallObjectBranch(channels)
        self.down3 = _conv_norm_act(channels, channels, stride=2)
        self.down4 = _conv_norm_act(channels, channels, stride=2)
        self.bottom_up4 = nn.Sequential(_conv_norm_act(channels, channels), ResidualBlock(channels))
        self.bottom_up5 = nn.Sequential(_conv_norm_act(channels, channels), ResidualBlock(channels))

    def forward(
        self,
        features: tuple[Tensor, Tensor, Tensor],
        cue_maps: dict[str, Tensor],
        return_aux: bool,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], dict[str, Tensor]]:
        p3, p4, p5 = features
        lateral5 = self.lateral5(p5)
        lateral4 = self.lateral4(p4)
        lateral3 = self.lateral3(p3)
        fused4 = self.top_down4(lateral4 + functional.interpolate(lateral5, size=lateral4.shape[-2:], mode="nearest"))
        fused3 = self.top_down3(lateral3 + functional.interpolate(fused4, size=lateral3.shape[-2:], mode="nearest"))
        context4, evidence4 = self.guided4(fused4, cue_maps["risk"], cue_maps["reliability"])
        context3, evidence3 = self.guided3(fused3, cue_maps["risk"], cue_maps["reliability"])
        output3, sob_evidence = self.sob(fused3, context3)
        output4 = self.bottom_up4(context4 + self.down3(output3))
        output5 = self.bottom_up5(lateral5 + self.down4(output4))
        if not return_aux:
            return (output3, output4, output5), {}
        auxiliary: dict[str, Tensor] = {
            "risk": cue_maps["risk"],
            "reliability": cue_maps["reliability"],
            "risk_stack": cue_maps["risk_stack"],
            "stability_stack": cue_maps["stability_stack"],
            "p3_fused": fused3,
            "p4_fused": fused4,
            "p3": output3,
            "p4": output4,
            "p5": output5,
        }
        auxiliary.update({f"p3_{name}": value for name, value in evidence3.items()})
        auxiliary.update({f"p4_{name}": value for name, value in evidence4.items()})
        auxiliary.update({f"sob_{name}": value for name, value in sob_evidence.items()})
        return (output3, output4, output5), auxiliary


class _PredictionTower(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(_conv_norm_act(channels, channels), _conv_norm_act(channels, channels))

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class DecoupledHBBHead(nn.Module):
    """Independent confidence/classification and distributional HBB regression branches."""

    def __init__(self, config: SLCNetConfig) -> None:
        super().__init__()
        channels = config.neck_channels
        self.strides = config.strides
        self.reg_max = config.reg_max
        self.num_classes = config.num_classes
        self.class_towers = nn.ModuleList(_PredictionTower(channels) for _ in self.strides)
        self.box_towers = nn.ModuleList(_PredictionTower(channels) for _ in self.strides)
        self.confidence_layers = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in self.strides)
        self.class_layers = nn.ModuleList(nn.Conv2d(channels, self.num_classes, 1) for _ in self.strides)
        self.distance_layers = nn.ModuleList(nn.Conv2d(channels, 4 * (self.reg_max + 1), 1) for _ in self.strides)

    def forward(self, feature_levels: Iterable[Tensor]) -> tuple[ScalePrediction, ...]:
        predictions: list[ScalePrediction] = []
        for feature, stride, class_tower, box_tower, confidence_layer, class_layer, distance_layer in zip(
            feature_levels,
            self.strides,
            self.class_towers,
            self.box_towers,
            self.confidence_layers,
            self.class_layers,
            self.distance_layers,
            strict=True,
        ):
            classification = class_tower(feature)
            regression = box_tower(feature)
            predictions.append(
                ScalePrediction(
                    confidence_logits=confidence_layer(classification),
                    class_logits=class_layer(classification),
                    distance_logits=distance_layer(regression),
                    stride=stride,
                )
            )
        return tuple(predictions)


def _grid_points(height: int, width: int, stride: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack(((x_coords + 0.5) * stride, (y_coords + 0.5) * stride), dim=-1)


def distance_distribution_expectation(distance_logits: Tensor, reg_max: int) -> Tensor:
    batch, _, height, width = distance_logits.shape
    logits = distance_logits.reshape(batch, 4, reg_max + 1, height, width)
    bins = torch.arange(reg_max + 1, device=distance_logits.device, dtype=distance_logits.dtype)
    return (logits.softmax(dim=2) * bins.view(1, 1, -1, 1, 1)).sum(dim=2)


def decode_distance_boxes(prediction: ScalePrediction, reg_max: int) -> Tensor:
    distances = distance_distribution_expectation(prediction.distance_logits, reg_max) * prediction.stride
    batch, _, height, width = distances.shape
    points = _grid_points(height, width, prediction.stride, distances.device, distances.dtype).view(1, height, width, 2)
    distances = distances.permute(0, 2, 3, 1)
    boxes = torch.stack(
        (
            points[..., 0] - distances[..., 0],
            points[..., 1] - distances[..., 1],
            points[..., 0] + distances[..., 2],
            points[..., 1] + distances[..., 3],
        ),
        dim=-1,
    )
    return boxes.reshape(batch, height * width, 4)


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bottom_right = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (boxes_a[..., 2:] - boxes_a[..., :2]).clamp_min(0).prod(dim=-1)
    area_b = (boxes_b[..., 2:] - boxes_b[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-6)


def generalized_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bottom_right = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (boxes_a[..., 2:] - boxes_a[..., :2]).clamp_min(0).prod(dim=-1)
    area_b = (boxes_b[..., 2:] - boxes_b[..., :2]).clamp_min(0).prod(dim=-1)
    union = (area_a + area_b - intersection).clamp_min(1e-6)
    enclosure = (
        (torch.maximum(boxes_a[..., 2:], boxes_b[..., 2:]) - torch.minimum(boxes_a[..., :2], boxes_b[..., :2]))
        .clamp_min(0)
        .prod(dim=-1)
        .clamp_min(1e-6)
    )
    return intersection / union - (enclosure - union) / enclosure


def hbb_nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    kept: list[Tensor] = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        current_box = boxes[current].expand(remaining.shape[0], -1)
        overlaps = box_iou(current_box, boxes[remaining])
        order = remaining[overlaps <= iou_threshold]
    return torch.stack(kept)


def class_aware_hbb_nms(boxes: Tensor, scores: Tensor, labels: Tensor, iou_threshold: float) -> Tensor:
    unique_labels = labels.unique(sorted=True)
    source_indices: list[Tensor] = []
    for label in unique_labels:
        source = torch.nonzero(labels == label, as_tuple=False).flatten()
        source_indices.append(source[hbb_nms(boxes[source], scores[source], iou_threshold)])
    if not source_indices:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    indices = torch.cat(source_indices)
    return indices[scores[indices].argsort(descending=True)]


@dataclass
class _AssignedTargets:
    confidence: list[Tensor]
    classes: list[Tensor]
    boxes: list[Tensor]
    areas: list[Tensor]


class TargetAssigner:
    """Deterministic one-center anchor-free matching shared by every comparison."""

    def __init__(self, config: SLCNetConfig) -> None:
        self.config = config

    def _scale_index(self, max_side: float) -> int:
        for index, (low, high) in enumerate(self.config.scale_ranges):
            if low <= max_side < high:
                return index
        return len(self.config.scale_ranges) - 1

    def assign(self, scales: tuple[ScalePrediction, ...], targets: list[DetectionTarget]) -> _AssignedTargets:
        device = scales[0].confidence_logits.device
        confidence: list[Tensor] = []
        classes: list[Tensor] = []
        boxes: list[Tensor] = []
        areas: list[Tensor] = []
        for prediction in scales:
            batch, _, height, width = prediction.confidence_logits.shape
            confidence.append(torch.zeros((batch, height, width), device=device))
            classes.append(torch.full((batch, height, width), -1, dtype=torch.long, device=device))
            boxes.append(torch.zeros((batch, height, width, 4), device=device))
            areas.append(torch.full((batch, height, width), float("inf"), device=device))
        for batch_index, target in enumerate(targets):
            target_boxes = torch.as_tensor(target.boxes, dtype=torch.float32, device=device).reshape(-1, 4)
            target_labels = torch.as_tensor(target.labels, dtype=torch.long, device=device).reshape(-1)
            if target_boxes.shape[0] != target_labels.shape[0]:
                raise ValueError("Each target image must have one label per HBB.")
            for box, label in zip(target_boxes, target_labels, strict=True):
                width = float((box[2] - box[0]).clamp_min(0).item())
                height = float((box[3] - box[1]).clamp_min(0).item())
                if width <= 0 or height <= 0 or label < 0 or label >= self.config.num_classes:
                    continue
                scale_index = self._scale_index(max(width, height))
                stride = scales[scale_index].stride
                feature_height, feature_width = confidence[scale_index].shape[-2:]
                center_x = float(((box[0] + box[2]) * 0.5).item())
                center_y = float(((box[1] + box[3]) * 0.5).item())
                grid_x = min(max(int(center_x // stride), 0), feature_width - 1)
                grid_y = min(max(int(center_y // stride), 0), feature_height - 1)
                area = width * height
                if area >= float(areas[scale_index][batch_index, grid_y, grid_x].item()):
                    continue
                confidence[scale_index][batch_index, grid_y, grid_x] = 1.0
                classes[scale_index][batch_index, grid_y, grid_x] = label
                boxes[scale_index][batch_index, grid_y, grid_x] = box
                areas[scale_index][batch_index, grid_y, grid_x] = area
        return _AssignedTargets(confidence=confidence, classes=classes, boxes=boxes, areas=areas)


def _distribution_focal_loss(logits: Tensor, targets: Tensor, reg_max: int) -> Tensor:
    targets = targets.clamp(0, reg_max - 1e-4)
    lower = targets.floor().long()
    upper = (lower + 1).clamp(max=reg_max)
    upper_weight = targets - lower.to(targets.dtype)
    lower_weight = 1.0 - upper_weight
    flattened_logits = logits.reshape(-1, reg_max + 1)
    lower_loss = functional.cross_entropy(flattened_logits, lower.reshape(-1), reduction="none").reshape_as(targets)
    upper_loss = functional.cross_entropy(flattened_logits, upper.reshape(-1), reduction="none").reshape_as(targets)
    return (lower_loss * lower_weight + upper_loss * upper_weight).mean(dim=-1)


class SLCNetLoss(nn.Module):
    """Detection objective with paper-specific area-aware localization supervision."""

    def __init__(self, config: SLCNetConfig) -> None:
        super().__init__()
        self.config = config
        self.assigner = TargetAssigner(config)

    def _zero(self, scales: tuple[ScalePrediction, ...]) -> Tensor:
        return sum(
            (prediction.confidence_logits.sum() * 0.0 for prediction in scales),
            start=torch.zeros((), device=scales[0].confidence_logits.device),
        )

    def forward(self, output: SLCNetOutput, targets: list[DetectionTarget]) -> dict[str, Tensor]:
        if len(targets) != output.scales[0].confidence_logits.shape[0]:
            raise ValueError("The target list length must equal the image batch size.")
        assigned = self.assigner.assign(output.scales, targets)
        zero = self._zero(output.scales)
        confidence_losses: list[Tensor] = []
        class_losses: list[Tensor] = []
        localization_losses: list[Tensor] = []
        small_terms: list[Tensor] = []
        positive_count = 0
        small_count = 0
        for index, prediction in enumerate(output.scales):
            confidence_target = assigned.confidence[index]
            confidence_losses.append(
                functional.binary_cross_entropy_with_logits(prediction.confidence_logits.squeeze(1), confidence_target)
            )
            positive = confidence_target.bool()
            if not positive.any():
                continue
            positive_count += int(positive.sum().item())
            class_logits = prediction.class_logits.permute(0, 2, 3, 1)[positive]
            labels = assigned.classes[index][positive]
            class_target = functional.one_hot(labels, self.config.num_classes).to(class_logits.dtype)
            class_losses.append(functional.binary_cross_entropy_with_logits(class_logits, class_target))
            decoded_boxes = decode_distance_boxes(prediction, self.config.reg_max).reshape(
                prediction.confidence_logits.shape[0], prediction.confidence_logits.shape[-2], prediction.confidence_logits.shape[-1], 4
            )
            predicted_boxes = decoded_boxes[positive]
            target_boxes = assigned.boxes[index][positive]
            iou_loss = 1.0 - generalized_iou(predicted_boxes, target_boxes)
            batch, _, height, width = prediction.distance_logits.shape
            logits = prediction.distance_logits.reshape(batch, 4, self.config.reg_max + 1, height, width)
            logits = logits.permute(0, 3, 4, 1, 2)[positive]
            points = _grid_points(height, width, prediction.stride, logits.device, logits.dtype)
            point_targets = points.unsqueeze(0).expand(batch, -1, -1, -1)[positive]
            distances = torch.stack(
                (
                    point_targets[:, 0] - target_boxes[:, 0],
                    point_targets[:, 1] - target_boxes[:, 1],
                    target_boxes[:, 2] - point_targets[:, 0],
                    target_boxes[:, 3] - point_targets[:, 1],
                ),
                dim=-1,
            ).clamp_min(0.0) / prediction.stride
            dfl_loss = _distribution_focal_loss(logits, distances, self.config.reg_max)
            localization = iou_loss + dfl_loss
            localization_losses.append(localization.mean())
            positive_areas = assigned.areas[index][positive]
            small = positive_areas <= self.config.small_area_threshold
            if small.any():
                small_count += int(small.sum().item())
                weights = 1.0 + torch.exp(-positive_areas[small] / self.config.small_area_threshold)
                small_terms.append((weights * localization[small]).sum())
        confidence_loss = torch.stack(confidence_losses).mean() if confidence_losses else zero
        class_loss = torch.stack(class_losses).mean() if class_losses else zero
        localization_loss = torch.stack(localization_losses).mean() if localization_losses else zero
        small_loss = torch.stack(small_terms).sum() / small_count if small_count else zero
        detection_loss = confidence_loss + class_loss + localization_loss
        total = detection_loss + self.config.small_loss_weight * small_loss
        return {
            "total": total,
            "detection": detection_loss,
            "confidence": confidence_loss,
            "classification": class_loss,
            "localization": localization_loss,
            "small": small_loss,
            "positive_count": total.new_tensor(float(positive_count)),
            "small_count": total.new_tensor(float(small_count)),
        }


class SLCNetDetector(nn.Module):
    """Self-contained SLCNet single-stage detector for HBB tasks."""

    def __init__(self, config: SLCNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or SLCNetConfig()
        self.config.validate()
        self.backbone = HierarchicalFeatureExtractor(self.config)
        self.cues = CoastalCueEncoder(self.config.in_channels, self.config.cue_kernel)
        self.neck = SLCNeck(self.backbone.out_channels, self.config)
        self.head = DecoupledHBBHead(self.config)
        self.criterion = SLCNetLoss(self.config)

    def forward(self, images: Tensor, return_aux: bool = True) -> SLCNetOutput:
        if images.ndim != 4:
            raise ValueError("images must be a BxCxHxW tensor.")
        if images.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} image channel(s), received {images.shape[1]}.")
        height, width = images.shape[-2:]
        if height % max(self.config.strides) != 0 or width % max(self.config.strides) != 0:
            raise ValueError(f"Input height and width must be divisible by {max(self.config.strides)}.")
        feature_levels = self.backbone(images)
        cue_maps = self.cues(images)
        neck_features, auxiliary = self.neck(feature_levels, cue_maps, return_aux=return_aux)
        return SLCNetOutput(scales=self.head(neck_features), aux=auxiliary)

    def loss(self, output: SLCNetOutput, targets: list[DetectionTarget]) -> dict[str, Tensor]:
        return self.criterion(output, targets)

    def _decode_one(
        self,
        output: SLCNetOutput,
        batch_index: int,
        image_size: tuple[int, int],
        score_threshold: float,
        iou_threshold: float,
        max_detections: int,
    ) -> DetectionResult:
        all_boxes: list[Tensor] = []
        all_scores: list[Tensor] = []
        all_labels: list[Tensor] = []
        image_height, image_width = image_size
        for prediction in output.scales:
            boxes = decode_distance_boxes(prediction, self.config.reg_max)[batch_index]
            confidence = prediction.confidence_logits[batch_index].sigmoid().flatten()
            class_scores = prediction.class_logits[batch_index].sigmoid().permute(1, 2, 0).reshape(-1, self.config.num_classes)
            scores, labels = (confidence.unsqueeze(1) * class_scores).max(dim=1)
            selected = scores >= score_threshold
            if not selected.any():
                continue
            boxes = boxes[selected]
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            all_boxes.append(boxes[valid])
            all_scores.append(scores[selected][valid])
            all_labels.append(labels[selected][valid])
        if not all_boxes:
            device = output.scales[0].confidence_logits.device
            return DetectionResult(torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), dtype=torch.long, device=device))
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        kept = class_aware_hbb_nms(boxes, scores, labels, iou_threshold)[:max_detections]
        return DetectionResult(boxes=boxes[kept], scores=scores[kept], labels=labels[kept])

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_detections: int = 300,
    ) -> list[DetectionResult]:
        output = self.forward(images, return_aux=False)
        image_size = (images.shape[-2], images.shape[-1])
        return [
            self._decode_one(output, index, image_size, score_threshold, iou_threshold, max_detections)
            for index in range(images.shape[0])
        ]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def slcnet_model_summary(model: SLCNetDetector) -> dict[str, Any]:
    return {
        "model_class": type(model).__name__,
        "config": model.config.to_dict(),
        "trainable_parameters": count_trainable_parameters(model),
        "detection_strides": list(model.config.strides),
        "notes": [
            "P3 and P4 receive reliability-calibrated coastal guidance; P5 remains the deep semantic route.",
            "The detector is implemented only with PyTorch modules and native HBB decoding.",
            "Device-specific latency, throughput, and memory require a separate measurement run.",
        ],
    }


SLCNet = SLCNetDetector
