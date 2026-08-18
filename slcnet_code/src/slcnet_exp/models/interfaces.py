from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SLCNetConfig:
    """Configuration for the standalone single-stage SLCNet detector."""

    in_channels: int = 1
    num_classes: int = 1
    input_size: int = 640
    base_channels: int = 24
    neck_channels: int = 96
    strides: tuple[int, int, int] = (8, 16, 32)
    scale_ranges: tuple[tuple[float, float], ...] = ((0.0, 64.0), (64.0, 192.0), (192.0, float("inf")))
    reg_max: int = 8
    cue_kernel: int = 7
    context_kernel: int = 5
    alpha_init: float = 0.25
    small_area_threshold: float = 1024.0
    small_loss_weight: float = 1.0

    def validate(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if self.input_size <= 0 or self.input_size % max(self.strides) != 0:
            raise ValueError("input_size must be positive and divisible by the largest detection stride.")
        if self.base_channels < 8 or self.neck_channels < 8:
            raise ValueError("base_channels and neck_channels must be at least 8.")
        if len(self.strides) != 3 or tuple(sorted(self.strides)) != self.strides:
            raise ValueError("strides must contain three ascending feature strides.")
        if len(self.scale_ranges) != len(self.strides):
            raise ValueError("scale_ranges must contain one range per detection stride.")
        if any(low < 0 or high <= low for low, high in self.scale_ranges):
            raise ValueError("Each scale range must satisfy 0 <= low < high.")
        if self.reg_max < 2:
            raise ValueError("reg_max must be at least 2.")
        if self.cue_kernel < 3 or self.cue_kernel % 2 == 0:
            raise ValueError("cue_kernel must be an odd integer >= 3.")
        if self.context_kernel < 3 or self.context_kernel % 2 == 0:
            raise ValueError("context_kernel must be an odd integer >= 3.")
        if not 0.0 < self.alpha_init < 1.0:
            raise ValueError("alpha_init must lie strictly between 0 and 1.")
        if self.small_area_threshold <= 0 or self.small_loss_weight < 0:
            raise ValueError("Small-object loss settings must be non-negative with a positive area threshold.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class DetectionTarget:
    """One image's HBB labels in resized detector coordinates."""

    boxes: Any
    labels: Any
    image_id: int | None = None
    original_size: tuple[int, int] | None = None
    letterbox: dict[str, float] | None = None


@dataclass
class DetectorBatch:
    images: Any
    targets: list[DetectionTarget] | None = None


@dataclass
class ScalePrediction:
    """Raw predictions for one feature scale."""

    confidence_logits: Any
    class_logits: Any
    distance_logits: Any
    stride: int


@dataclass
class DetectionResult:
    boxes: Any
    scores: Any
    labels: Any


@dataclass
class SLCNetOutput:
    """Raw multi-scale predictions and optional paper-aligned intermediate maps."""

    scales: tuple[ScalePrediction, ...]
    aux: dict[str, Any] = field(default_factory=dict)
    detections: list[DetectionResult] | None = None


@dataclass
class DetectorOutput:
    """Compatibility container for callers that only consume decoded detections."""

    boxes: Any
    scores: Any
    labels: Any
    extra: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    def forward(self, images: Any, return_aux: bool = False) -> SLCNetOutput:
        ...

    def predict(self, images: Any, score_threshold: float = 0.25, iou_threshold: float = 0.5) -> list[DetectionResult]:
        ...
